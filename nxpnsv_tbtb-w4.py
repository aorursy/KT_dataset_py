# Imports

import os

from abc import ABCMeta, abstractmethod

from typing import Dict, List, Tuple, Union



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import mean_squared_log_error

from scipy.optimize.minpack import curve_fit

from scipy.optimize import curve_fit, OptimizeWarning

from scipy.optimize import least_squares

from xgboost import XGBRegressor
def RMSLE(actual: np.ndarray, prediction: np.ndarray) -> float:

    """Calculate Root Mean Square Log Error between actual and predicted values"""

    return np.sqrt(mean_squared_log_error(actual, np.maximum(0, prediction)))
def load_kaggle_csv(dataset: str, datadir: str) -> pd.DataFrame:

    """Load andt preprocess kaggle covid-19 csv dataset."""

    df = pd.read_csv(

        f"{os.path.join(datadir,dataset)}.csv", parse_dates=["Date"]

    )

    df['country'] = df["Country_Region"]

    if "Province_State" in df:

        df["Country_Region"] = np.where(

            df["Province_State"].isnull(),

            df["Country_Region"],

            df["Country_Region"] + "_" + df["Province_State"],

        )

        df.drop(columns="Province_State", inplace=True)

    if "ConfirmedCases" in df:

        df["ConfirmedCases"] = df.groupby("Country_Region")[

            "ConfirmedCases"

        ].cummax()

    if "Fatalities" in df:

        df["Fatalities"] = df.groupby("Country_Region")["Fatalities"].cummax()

    if "DayOfYear" not in df:

        df["DayOfYear"] = df["Date"].dt.dayofyear

    df["Date"] = df["Date"].dt.date

    return df



def dateparse(x): 

    try:

        return pd.datetime.strptime(x, '%Y-%m-%d')

    except:

        return pd.NaT



def prepare_lat_long(df):

    df["Country_Region"] = np.where(

            df["Province/State"].isnull(),

            df["Country/Region"],

            df["Country/Region"] + "_" + df["Province/State"],

        )

    return df[['Country_Region', 'Lat', 'Long']].drop_duplicates()



def get_extra_features(df): 

    df['school_closure_status_daily'] = np.where(df['school_closure'] < df['Date'], 1, 0)

    df['school_closure_first_fatality'] = np.where(df['school_closure'] < df['first_1Fatalities'], 1, 0)

    df['school_closure_first_10cases'] = np.where( df['school_closure'] < df['first_10ConfirmedCases'], 1, 0)

    #

    df['case_delta1_10'] = (df['first_10ConfirmedCases'] - df['first_1ConfirmedCases']).dt.days

    df['case_death_delta1'] = (df['first_1Fatalities'] - df['first_1ConfirmedCases']).dt.days

    df['case_delta1_100'] = (df['first_100ConfirmedCases'] - df['first_1ConfirmedCases']).dt.days

    df['days_since'] = df['DayOfYear']-df['case1_DayOfYear']

    df['weekday'] = pd.to_datetime(df['Date']).dt.weekday

    col = df.isnull().mean()

    rm_null_col = col[col > 0.2].index.tolist()

    return df
### Train data



# Take lat/long from week 1 data set

df_lat = prepare_lat_long(pd.read_csv("/kaggle/input/w1train/w1train.csv"))



# Get current train data

train = load_kaggle_csv("train", "/kaggle/input/covid19-global-forecasting-week-4")



# Insert augmentations



country_health_indicators = (

    (pd.read_csv("/kaggle/input/country-health-indicators/country_health_indicators_v3.csv", 

        parse_dates=['first_1ConfirmedCases', 'first_10ConfirmedCases', 

                     'first_50ConfirmedCases', 'first_100ConfirmedCases',

                     'first_1Fatalities', 'school_closure'], date_parser=dateparse)).rename(

        columns ={'Country_Region':'country'}))

# Merge augmentation to kaggle input

train = (pd.merge(train, country_health_indicators,

                  on="country",

                  how="left")).merge(df_lat, on='Country_Region', how='left')

train = get_extra_features(train)



# train=train.fillna(0)

train.head(3)
### TEST DATA

test = load_kaggle_csv("test", "/kaggle/input/covid19-global-forecasting-week-4")

test = (pd.merge(

    test, country_health_indicators, on="country", how="left")).merge(

    df_lat, on ='Country_Region', how='left')

test = get_extra_features(test)

del country_health_indicators
class Fitter(metaclass=ABCMeta):

    """

    Helper class for 1D fits using scipy fit.



    This version assumes y-data is positive and increasing.

    """



    def __init__(self, name):

        """Make fitter instance."""

        self.kwargs = {

            "method": "trf",

            "max_nfev": 20000,

            "x_scale": "jac",

            "loss": "linear",

            "jac": self.jacobian,

        }

        self.name = name

        self.rmsle = None

        self.fit_params = None

        self.fit_cov = None

        self.y_hat = None

        self.p0 = None

        self.bounds = None



    @abstractmethod

    def function(self, x: np.ndarray, *args) -> np.ndarray:

        """Mathematical function to fit."""

        pass



    @abstractmethod

    def jacobian(self, x: np.ndarray, *args) -> np.ndarray:

        """Jacobian of funciton."""

        pass



    @abstractmethod

    def guess(self) -> Tuple[List[float], List]:

        """First guess for fit optimium."""

        pass



    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Union[None, Tuple]:

        """Fit function to y over x."""

        # Update extra keywords for fit

        kwargs.update(self.kwargs)

            

        # Reset fit results

        self.rmsle = None

        self.fit_params = None

        self.fit_cov = None

        self.y_hat = None

        self.p0 = None

        self.bounds = None

        if len(x) <= 3:

            return



        # Guess params

        self.p0, self.bounds = self.guess(x, y)



        # Perform fit

        try:

            res = curve_fit(

                f=self.function,

                xdata=np.array(x, dtype=np.float128),

                ydata=np.array(y, dtype=np.float128),

                p0=self.p0,

                bounds=self.bounds,

                sigma=np.maximum(1, np.sqrt(y)),

                **kwargs,

            )

        except (ValueError, RuntimeError, OptimizeWarning) as e:

            print(e)

            return



        # Update fit results

        self.y_hat = self.function(x, *res[0])

        self.rmsle = np.sqrt(mean_squared_log_error(y, self.y_hat))

        self.fit_params = res[0]

        self.fit_cov = res[1]



    def plot_fit(self, x, y, ax=None, title=None, **kwargs):

        """Fit and plot."""

        self.fit(x, y, **kwargs)



        if self.fit_params is None:

            print("No result, cannot plot")

            return



        if ax is None:

            _, ax = plt.subplots()

        ax.set_title(f"{title or ''} {self.name}: rmsle={self.rmsle:.2f}")

        color = "g"

        ax.plot(x, y, "o", color=color, alpha=0.9)

        ax.plot(x, self.y_hat, "-", color="r")

        ax.set_ylabel("Counts", color=color)

        ax.set_xlabel("Day of Year")

        ax.tick_params(axis="y", labelcolor=color)

        ax2 = ax.twinx()

        color = "b"

        ax2.set_ylabel("Residual", color=color)

        ax2.plot(x, y - self.y_hat, ".", color=color)

        ax2.tick_params(axis="y", labelcolor=color)

        ax.text(

            0.05,

            0.95,

            "\n".join(

                [f"$p_{i}$={x:0.2f}" for i, x in enumerate(self.fit_params)]

            ),

            horizontalalignment="left",

            verticalalignment="top",

            transform=ax.transAxes,

        )



        

class Logistic(Fitter):

    def __init__(self):

        super().__init__(name="Logistic")



    def function(

        self, x: np.ndarray, K: float, B: float, M: float

    ) -> np.ndarray:

        return K / (1 + np.exp(-B * (x - M)))



    def jacobian(

        self, x: np.ndarray, K: float, B: float, M: float

    ) -> np.ndarray:

        dK = 1 / (1 + np.exp(-B * (x - M)))

        dB = (

            K

            * (x - M)

            * np.exp(-B * (x - M))

            / np.square(1 + np.exp(-B * (x - M)))

        )

        dM = K * B * np.exp(-B * (x - M)) / np.square(1 + np.exp(-B * (x - M)))

        return np.transpose([dK, dB, dM])



    def guess(

        self, x: np.ndarray, y: np.ndarray

    ) -> Tuple[List[float], List[float]]:

        K = y[-1]

        B = 0.1

        M = x[np.argmax(y >= 0.5 * K)]

        p0 = [K, B, M]

        bounds = [[y[-1], 1e-4, x[0]], [y[-1] * 8, 0.5, (1+x[-1]) * 2]]

        return p0, bounds



class GLF(Fitter):

    def __init__(self):

        super().__init__(name="GLF")



    def function(self, x, K, B, M, nu):

        return K / np.power((1 + np.exp(-B * (x - M))), 1 / nu)



    def jacobian(self, x, K, B, M, nu):

        nu1 = 1.0 / nu

        xM = x - M

        exp_BxM = np.exp(-B * xM)

        pow0 = np.power(1 + exp_BxM, -nu1)

        pow1 = K * exp_BxM / (nu * np.power(1 + exp_BxM, nu1 + 1))



        dK = pow0

        dB = xM * pow1

        dnu = K * np.log1p(exp_BxM) * pow0 / nu

        dM = B * pow1

        return np.transpose([dK, dB, dnu, dM])



    def guess(self, x, y):

        # Guess params and param bounds

        K = y[-1]

        B = 0.1

        M = x[np.argmax(y >= 0.5 * K)]

        nu = 0.5

        p0 = [K, B, M, nu]

        bounds = [[y[-1], 1e-3, x[0], 1e-2], [(y[-1]+1) * 10, 0.5, (x[-1]+1) * 2, 1.0]]

        return p0, bounds

    

class DiXGLF(Fitter):

    """Interpolation between 2 logistic function.



    First guess is split by y_max/2 so the first and second logistic

    start on different partitions of data.

    

    Uses 3-point estimator in place of explicit jacobian because of numeric stability.

    """



    def __init__(self):

        super().__init__(name="DiXGLF")

        self.glf = GLF()

        self.logistic = Logistic()

        self.kwargs.update({"jac": "3-point"})



    def function(self, x, B0, M0, K1, B1, M1, nu1, K2, B2, M2, nu2):

        alpha = self.logistic.function(x, 1, B0, M0)

        return alpha * self.glf.function(x, K1, B1, M1, nu1) + (

            1 - alpha

        ) * self.glf.function(x, K2, B2, M2, nu2)



    def jacobian(self, x, B0, M0, K1, B1, M1, nu1, K2, B2, M2, nu2):

        raise RuntimeError("%s jacobian not implemented", self.name)



    def guess(self, x, y):

        split = min(max(1, np.argmax(y >= 0.5 * y[-1])), len(x)-2)

        p01, bounds1 = self.glf.guess(x[:split], y[:split])

        p02, bounds2 = self.glf.guess(x[split:], y[split:])

        p0, bounds = self.logistic.guess(x, y)

        p0 = p0[1:]

        bounds = [bounds[0][1:], bounds[1][1:]]

        p0.extend(p01)

        p0.extend(p02)

        bounds[0].extend(bounds1[0])

        bounds[0].extend(bounds2[0])

        bounds[1].extend(bounds1[1])

        bounds[1].extend(bounds2[1])

        return p0, bounds
def apply_fitter(

    df: pd.DataFrame,

    fitter: Fitter,

    x_col: str = "DayOfYear",

    y_cols: List[str] = ["ConfirmedCases", "Fatalities"],

) -> pd.DataFrame:

    """Helper to apply fitter to dataframe groups"""

    x = df[x_col].astype(np.float128).to_numpy()

    result = {}

    for y_col in y_cols:

        y = df[y_col].astype(np.float128).to_numpy()

        fitter.fit(x, y)

        if fitter.rmsle is None:

            continue

        result[f"{y_col}_rmsle"] = fitter.rmsle

        df[f"y_hat_fitter_{y_col}"] = fitter.y_hat

        result.update({f"{y_col}_p_{i}": p for i, p in enumerate(fitter.fit_params)})

    return pd.DataFrame([result])
plt.style.use("seaborn-white")

sns.set_color_codes()

dixglf = DiXGLF()

train["y_hat_fitter_ConfirmedCases"]=0

train["y_hat_fitter_Fatalities"]=0

fig, ax = plt.subplots(2, 4, figsize=(16,8))

ax = ax.flatten()

for i, country in enumerate(("Italy", "Austria", "Korea, South", "Germany")):

    c = train[train["Country_Region"] == country]

    x = c["DayOfYear"].astype(np.float128).to_numpy()

    dixglf.plot_fit(x, c["ConfirmedCases"].astype(np.float128).to_numpy(), ax=ax[i], title=f"Cases {country}")

    dixglf.plot_fit(x, c["Fatalities"].astype(np.float128).to_numpy(), ax=ax[i+4], title=f"Deaths {country}")

fig.tight_layout()
train = pd.merge(

    train, train.groupby(

    ["Country_Region"], observed=True, sort=False

).apply(lambda x: apply_fitter(x, fitter=dixglf)).reset_index(), 

    on=["Country_Region"], how="left")
train["y_hat_fitter_ConfirmedCases"]=dixglf.function(

    train["DayOfYear"],

    train["ConfirmedCases_p_0"],

    train["ConfirmedCases_p_1"],

    train["ConfirmedCases_p_2"],

    train["ConfirmedCases_p_3"],

    train["ConfirmedCases_p_4"],

    train["ConfirmedCases_p_5"],

    train["ConfirmedCases_p_6"],

    train["ConfirmedCases_p_7"],

    train["ConfirmedCases_p_8"],

    train["ConfirmedCases_p_9"])

train["y_hat_fitter_Fatalities"]=dixglf.function(

    train["DayOfYear"],

    train["Fatalities_p_0"],

    train["Fatalities_p_1"],

    train["Fatalities_p_2"],

    train["Fatalities_p_3"],

    train["Fatalities_p_4"],

    train["Fatalities_p_5"],

    train["Fatalities_p_6"],

    train["Fatalities_p_7"],

    train["Fatalities_p_8"],

    train["Fatalities_p_9"])
train.head()
def apply_xgb_model(train, x_columns, y_column, xgb_params):

    X = train[x_columns].astype(np.float32).fillna(0).to_numpy()

    y = train[y_column].astype(np.float32).fillna(0).to_numpy()

    xgb_fit = XGBRegressor(**xgb_params).fit(X, y)

    y_hat = xgb_fit.predict(X)

    train[f"yhat_xgb_{y_column}"] = y_hat

    return RMSLE(y, y_hat), xgb_fit
xgb_params_c = dict(

    gamma=0.1,

    learning_rate=0.35,

    n_estimators=221,

    max_depth=15,

    min_child_weight=1,

    nthread=8,

    objective="reg:squarederror")



xgb_params_f = dict(

    gamma=0.1022,

    learning_rate=0.338,

    n_estimators=292,

    max_depth=14,

    min_child_weight=1,

    nthread=8,

    objective="reg:squarederror")



x_columns = ['DayOfYear', 

       'Diabetes, blood, & endocrine diseases (%)', 'Respiratory diseases (%)',

       'Diarrhea & common infectious diseases (%)',

       'Nutritional deficiencies (%)',

       'obesity - adult prevalence rate',

       'pneumonia-death-rates', 'animal_fats', 'animal_products', 'eggs',

       'offals', 'treenuts', 'vegetable_oils', 'nbr_surgeons',

       'nbr_anaesthesiologists', 'population',

       'school_shutdown_1case',

       'school_shutdown_10case', 'school_shutdown_50case',

       'school_shutdown_1death', 'case1_DayOfYear', 'case10_DayOfYear',

       'case50_DayOfYear',

       'school_closure_status_daily', 'case_delta1_10',

       'case_death_delta1', 'case_delta1_100', 'days_since','Lat','Long','weekday',

        'y_hat_fitter_ConfirmedCases', 'y_hat_fitter_Fatalities'

]



xgb_c_rmsle, xgb_c_fit = apply_xgb_model(train, x_columns, "ConfirmedCases", xgb_params_c)

xgb_f_rmsle, xgb_f_fit = apply_xgb_model(train, x_columns, "Fatalities", xgb_params_f)
def interpolate(alpha, x0, x1):

    return x0 * alpha + x1 * (1 - alpha)





def RMSLE_interpolate(alpha, y, x0, x1):

    return RMSLE(y, interpolate(alpha, x0, x1))





def fit_hybrid(

    train: pd.DataFrame, y_cols: List[str] = ["ConfirmedCases", "Fatalities"]

) -> pd.DataFrame:

    def fit_one(y_col: str):

        opt = least_squares(

            fun=RMSLE_interpolate,

            args=(

                train[y_col],

                train[f"y_hat_fitter_{y_col}"],

                train[f"yhat_xgb_{y_col}"],

            ),

            x0=(0.5,),

            bounds=((0.0), (1.0,)),

        )

        return {f"{y_col}_alpha": opt.x[0], f"{y_col}_cost": opt.cost}



    result = {}

    for y_col in y_cols:

        result.update(fit_one(y_col))

    return pd.DataFrame([result])





def predict_hybrid(

    df: pd.DataFrame,

    x_col: str = "DayOfYear",

    y_cols: List[str] = ["ConfirmedCases", "Fatalities"],

):

    def predict_one(col):

        df[f"yhat_hybrid_{col}"] = interpolate(

            df[f"{y_col}_alpha"].to_numpy(),

            df[f"y_hat_fitter_{y_col}"].to_numpy(),

            df[f"yhat_xgb_{y_col}"].to_numpy(),

        )



    for y_col in y_cols:

        predict_one(y_col)
train = pd.merge(

    train,

    train.groupby(["Country_Region"], observed=True, sort=False)

    .apply(lambda x: fit_hybrid(x))

    .reset_index(),

    on=["Country_Region"],

    how="left",

)
predict_hybrid(train)
print(

    "Confirmed:\n"

    f'Fitter\t{RMSLE(train["ConfirmedCases"], train["y_hat_fitter_ConfirmedCases"])}\n'

    f'XGBoost\t{RMSLE(train["ConfirmedCases"], train["yhat_xgb_ConfirmedCases"])}\n'

    f'Hybrid\t{RMSLE(train["ConfirmedCases"], train["yhat_hybrid_ConfirmedCases"])}\n'

    f"Fatalities:\n"

    f'Fitter\t{RMSLE(train["Fatalities"], train["y_hat_fitter_Fatalities"])}\n'

    f'XGBoost\t{RMSLE(train["Fatalities"], train["yhat_xgb_Fatalities"])}\n'

    f'Hybrid\t{RMSLE(train["Fatalities"], train["yhat_hybrid_Fatalities"])}\n'

)
# Merge logistic and hybrid fit into test

test = pd.merge(

    test, 

    train[["Country_Region"] +

          ['ConfirmedCases_p_0', 'ConfirmedCases_p_1', 'ConfirmedCases_p_2', 'ConfirmedCases_p_3', 'ConfirmedCases_p_4', 'ConfirmedCases_p_5', 'ConfirmedCases_p_6', 'ConfirmedCases_p_7', 'ConfirmedCases_p_8', 'ConfirmedCases_p_9']+

          ['Fatalities_p_0', 'Fatalities_p_1', 'Fatalities_p_2', 'Fatalities_p_3', 'Fatalities_p_4', 'Fatalities_p_5', 'Fatalities_p_6', 'Fatalities_p_7', 'Fatalities_p_8', 'Fatalities_p_9']+

          ["Fatalities_alpha"] + 

          ["ConfirmedCases_alpha"]].groupby(['Country_Region']).head(1), on="Country_Region", how="left")
# Test predictions

test["y_hat_fitter_ConfirmedCases"]=dixglf.function(

    test["DayOfYear"],

    test["ConfirmedCases_p_0"],

    test["ConfirmedCases_p_1"],

    test["ConfirmedCases_p_2"],

    test["ConfirmedCases_p_3"],

    test["ConfirmedCases_p_4"],

    test["ConfirmedCases_p_5"],

    test["ConfirmedCases_p_6"],

    test["ConfirmedCases_p_7"],

    test["ConfirmedCases_p_8"],

    test["ConfirmedCases_p_9"])

test["y_hat_fitter_Fatalities"]=dixglf.function(

    test["DayOfYear"],

    test["Fatalities_p_0"],

    test["Fatalities_p_1"],

    test["Fatalities_p_2"],

    test["Fatalities_p_3"],

    test["Fatalities_p_4"],

    test["Fatalities_p_5"],

    test["Fatalities_p_6"],

    test["Fatalities_p_7"],

    test["Fatalities_p_8"],

    test["Fatalities_p_9"])

test["yhat_xgb_ConfirmedCases"] = xgb_c_fit.predict(test[x_columns].to_numpy())

test["yhat_xgb_Fatalities"] = xgb_f_fit.predict(test[x_columns].to_numpy())

predict_hybrid(test)
submission = test[["ForecastId", "yhat_hybrid_ConfirmedCases", "yhat_hybrid_Fatalities"]].round(2).rename(

        columns={

            "yhat_hybrid_ConfirmedCases": "ConfirmedCases",

            "yhat_hybrid_Fatalities": "Fatalities",

        }

    )

submission["ConfirmedCases"] = np.maximum(0, submission["ConfirmedCases"])

submission["Fatalities"] = np.maximum(0, submission["Fatalities"])

submission.head()
submission.to_csv("submission.csv", index=False)