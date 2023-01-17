from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta

import os

from pprint import pprint

import warnings

from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib

from matplotlib.ticker import ScalarFormatter

%matplotlib inline

import numpy as np

import optuna

optuna.logging.disable_default_handler()

import pandas as pd

pd.plotting.register_matplotlib_converters()

import seaborn as sns

from scipy.integrate import solve_ivp

from scipy.optimize import curve_fit
plt.style.use("seaborn-ticks")

plt.rcParams["xtick.direction"] = "in"

plt.rcParams["ytick.direction"] = "in"

plt.rcParams["font.size"] = 11.0

plt.rcParams["figure.figsize"] = (9, 6)
def line_plot(df, title, ylabel="Cases", h=None, v=None,

              xlim=(None, None), ylim=(0, None), math_scale=True, y_logscale=False, y_integer=False):

    """

    Show chlonological change of the data.

    """

    ax = df.plot()

    if math_scale:

        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        ax.ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))

    if y_logscale:

        ax.set_yscale("log")

    if y_integer:

        fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)

        fmt.set_scientific(False)

        ax.yaxis.set_major_formatter(fmt)

    ax.set_title(title)

    ax.set_xlabel(None)

    ax.set_ylabel(ylabel)

    ax.set_xlim(*xlim)

    ax.set_ylim(*ylim)

    ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)

    if h is not None:

        ax.axhline(y=h, color="black", linestyle="--")

    if v is not None:

        if not isinstance(v, list):

            v = [v]

        for value in v:

            ax.axvline(x=value, color="black", linestyle="--")

    plt.tight_layout()

    plt.show()
for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_raw = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv")

test_raw = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv")

submission_sample_raw = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv")
submission_sample_raw.head()
df = pd.DataFrame(

    {

        "Nunique_train": train_raw.nunique(),

        "Nunique_test": test_raw.nunique(),

        "Null_Train": train_raw.isnull().sum(),

        "Null_Test": test_raw.isnull().sum(),

    }

)

df.fillna("-").T
train_raw.head()
df = train_raw.copy()

# Delete columns we will not use

df = df.drop(["Province/State", "Country/Region", "Lat", "Long", "Id"], axis=1)

# Type change

df["Date"] = pd.to_datetime(df["Date"])

df[["ConfirmedCases", "Fatalities"]] = df[["ConfirmedCases", "Fatalities"]].astype(np.int64)

# Only use confirmed > 0

df = df.loc[df["ConfirmedCases"] > 0, :]

df = df.groupby("Date").last().fillna(method="bfill").reset_index()

# Show data

train_df = df.copy()

train_df.head()
test_raw.head()
df = test_raw.drop(["Province/State", "Country/Region", "Lat", "Long"], axis=1)

df["Date"] = pd.to_datetime(df["Date"])

test_df = df.copy()

test_df.head()
train_df.describe(include="all").fillna("-").T
line_plot(train_df.set_index("Date"), "Cases over time", y_integer=True)
line_plot(train_df.set_index("Date").drop("ConfirmedCases", axis=1), "Cases over time", y_integer=True)
def show_trend(train_df, variable, n_changepoints=2):

    """

    Show trend of log10(@variable) using fbprophet package.

    @train_df <pd.DataFrame>: the cleaned train data

    @variable <str>: ConfirmedCases or Fatalities

    @n_changepoints <int>: max number of change points

    """

    # Data arrangement

    df = train_df.loc[:, ["Date", variable]]

    df.columns = ["ds", "y"]

    # Log10(x)

    warnings.resetwarnings()

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        df["y"] = np.log10(df["y"]).replace([np.inf, -np.inf], 0)

    # fbprophet

    model = Prophet(growth="linear", daily_seasonality=False, n_changepoints=n_changepoints)

    model.fit(df)

    future = model.make_future_dataframe(periods=0)

    forecast = model.predict(future)

    # Create figure

    fig = model.plot(forecast)

    _ = add_changepoints_to_plot(fig.gca(), model, forecast)

    plt.title(f"log10({variable}) over time and chainge points")

    plt.ylabel(f"log10(the number of cases)")

    plt.xlabel("")
show_trend(train_df, "ConfirmedCases")
show_trend(train_df, "Fatalities")
# From Wikipedia, https://en.wikipedia.org/wiki/California

# In 2019

total_population = int("39,512,223".replace(",", ""))

total_population
def create_target_df(ncov_df, total_population, start_date=None, date_format="%d%b%Y"):

    """

    Calculate the number of susceptible people,

     and calculate the elapsed time [day] from the start date of the target dataframe.

    @noc_df <pd.DataFrame>: the cleaned training data

    @total_population <int>: total population

    @start_date <str>: the start date or None

    @date_format <str>: format of @start_date

    @return <tuple(2 objects)>:

        - 1. start_date <pd.Timestamp>: the start date of the selected records

        - 2. target_df <pd.DataFrame>:

            - column T: elapsed time [min] from the start date of the dataset

            - column Susceptible: the number of patients who are in the palces but not infected/recovered/fatal

            - column Deaths: the number of death cases

    """

    df = ncov_df.copy()

    if start_date is not None:

        df = df.loc[df["Date"] > datetime.strptime(start_date, date_format), :]

    start_date = df.loc[df.index[0], "Date"]

    # column T

    df["T"] = ((df["Date"] - start_date).dt.total_seconds() / 60).astype(int)

    # coluns except T

    response_variables = ["Susceptible", "Infected", "Recovered", "Fatal"]

    df["Susceptible"] = total_population - df["ConfirmedCases"]

    df["Infected"] = 0

    df.loc[df.index[0], "Infected"] = df.loc[df.index[0], "ConfirmedCases"] - df.loc[df.index[0], "Fatalities"]

    df["Recovered"] = 0

    df["Fatal"] = df["Fatalities"]

    # Return

    target_df = df.loc[:, ["T", *response_variables]]

    return (start_date, target_df)
class ModelBase(object):

    NAME = "Model"

    VARIABLES = ["x"]

    PRIORITIES = np.array([1])



    @classmethod

    def param_dict(cls, train_df_divided=None, q_range=None):

        """

        Define parameters without tau. This function should be overwritten.

        @train_df_divided <pd.DataFrame>:

            - column: t and non-dimensional variables

        @q_range <list[float, float]>: quantile rage of the parameters calculated by the data

        @return <dict[name]=(type, min, max):

            @type <str>: "float" or "int"

            @min <float/int>: min value

            @max <float/int>: max value

        """

        param_dict = dict()

        return param_dict



    @staticmethod

    def calc_variables(df):

        """

        Calculate the variables of the model.

        This function should be overwritten.

        @df <pd.DataFrame>

        @return <pd.DataFrame>

        """

        return df



    @staticmethod

    def calc_variables_reverse(df):

        """

        Calculate measurable variables using the variables of the model.

        This function should be overwritten.

        @df <pd.DataFrame>

        @return <pd.DataFrame>

        """

        return df



    @classmethod

    def create_dataset(cls, ncov_df, total_population, start_date=None, date_format="%d%b%Y"):

        """

        Create dataset with the model-specific varibles.

        The variables will be divided by total population.

        The column names (not include T) will be lower letters.

        @params: See the function named create_target_df()

        @return <tuple(objects)>:

            - start_date <pd.Timestamp>

            - initials <tuple(float)>: the initial values

            - Tend <int>: the last value of T

            - df <pd.DataFrame>: the dataset

        """

        start_date, target_df = create_target_df(

            ncov_df, total_population, start_date=start_date, date_format=date_format

        )

        df = cls.calc_variables(target_df).set_index("T") / total_population

        df.columns = [n.lower() for n in df.columns]

        initials = df.iloc[0, :].values

        df = df.reset_index()

        Tend = df.iloc[-1, 0]

        return (start_date, initials, Tend, df)



    def calc_r0(self):

        """

        Calculate R0. This function should be overwritten.

        """

        return None



    def calc_days_dict(self, tau):

        """

        Calculate 1/beta [day] etc.

        This function should be overwritten.

        @param tau <int>: tau value [hour]

        """

        return dict()
class SIRF(ModelBase):

    NAME = "SIR-F"

    VARIABLES = ["x", "y", "z", "w"]

    PRIORITIES = np.array([10, 0, 0, 1])



    def __init__(self, theta, kappa, rho, sigma):

        super().__init__()

        self.theta = float(theta)

        self.kappa = float(kappa)

        self.rho = float(rho)

        self.sigma = float(sigma)



    def __call__(self, t, X):

        # x, y, z, w = [X[i] for i in range(len(self.VARIABLES))]

        # dxdt = - self.rho * x * y

        # dydt = self.rho * (1 - self.theta) * x * y - (self.sigma + self.kappa) * y

        # dzdt = self.sigma * y

        # dwdt = self.rho * self.theta * x * y + self.kappa * y

        dxdt = - self.rho * X[0] * X[1]

        dydt = self.rho * (1 - self.theta) * X[0] * X[1] - (self.sigma + self.kappa) * X[1]

        dzdt = self.sigma * X[1]

        dwdt = self.rho * self.theta * X[0] * X[1] + self.kappa * X[1]

        return np.array([dxdt, dydt, dzdt, dwdt])



    @classmethod

    def param_dict(cls, train_df_divided=None, q_range=None):

        param_dict = super().param_dict()

        param_dict["theta"] = ("float", 0, 1)

        param_dict["kappa"] = ("float", 0, 1)

        param_dict["rho"] = ("float", 0, 1)

        param_dict["sigma"] = ("float", 0, 1)

        return param_dict



    @staticmethod

    def calc_variables(df):

        df["X"] = df["Susceptible"]

        df["Y"] = df["Infected"]

        df["Z"] = df["Recovered"]

        df["W"] = df["Fatal"]

        return df.loc[:, ["T", "X", "Y", "Z", "W"]]



    @staticmethod

    def calc_variables_reverse(df):

        df["Susceptible"] = df["X"]

        df["Infected"] = df["Y"]

        df["Recovered"] = df["Z"]

        df["Fatal"] = df["W"]

        return df



    def calc_r0(self):

        try:

            r0 = self.rho * (1 - self.theta) / (self.sigma + self.kappa)

        except ZeroDivisionError:

            return np.nan

        return round(r0, 2)



    def calc_days_dict(self, tau):

        _dict = dict()

        _dict["alpha1 [-]"] = round(self.theta, 3)

        if self.kappa == 0:

            _dict["1/alpha2 [day]"] = 0

        else:

            _dict["1/alpha2 [day]"] = int(tau / 24 / 60 / self.kappa)

        _dict["1/beta [day]"] = int(tau / 24 / 60 / self.rho)

        if self.sigma == 0:

            _dict["1/gamma [day]"] = 0

        else:

            _dict["1/gamma [day]"] = int(tau / 24 / 60 / self.sigma)

        return _dict
def simulation(model, initials, step_n, **params):

    """

    Solve ODE of the model.

    @model <ModelBase>: the model

    @initials <tuple[float]>: the initial values

    @step_n <int>: the number of steps

    """

    tstart, dt, tend = 0, 1, step_n

    sol = solve_ivp(

        fun=model(**params),

        t_span=[tstart, tend],

        y0=np.array(initials, dtype=np.float64),

        t_eval=np.arange(tstart, tend + dt, dt),

        dense_output=True

    )

    t_df = pd.Series(data=sol["t"], name="t")

    y_df = pd.DataFrame(data=sol["y"].T.copy(), columns=model.VARIABLES)

    sim_df = pd.concat([t_df, y_df], axis=1)

    return sim_df
eg_initials = np.array([1, 0.0001, 0, 0])

eg_param_dict = {"theta": 0.08, "kappa": 0.0001, "sigma": 0.02, "rho": 0.2}

eg_df = simulation(SIRF, eg_initials, step_n=300, **eg_param_dict)

eg_df.tail()
line_plot(

    eg_df.set_index("t"),

    title=r"Example of SIR-F model: $R_0$={0}".format(SIRF(**eg_param_dict).calc_r0()),

    ylabel="",

    h=1

)
train_df.shape
df = train_df.copy()

_start_date = df["Date"].min()

df["Elapsed"] = (df["Date"] - _start_date).dt.total_seconds() / 60

_many_train_df = df.copy()

_many_train_df.head()
exp_f = lambda x, a, b: a * np.exp(b * x / 60 / 24)
x = _many_train_df["Elapsed"]

y1, y2 = _many_train_df["ConfirmedCases"], _many_train_df["Fatalities"]

t = np.arange(0, x.max(), 1)

param_c, _ = curve_fit(exp_f, x, y1)

param_f, _ = curve_fit(exp_f, x, y2)

df = pd.DataFrame(

    {

        "Elapsed": t,

        "ConfirmedCases": np.vectorize(exp_f)(t, a=param_c[0], b=param_c[1]),

        "Fatalities": np.vectorize(exp_f)(t, a=param_f[0], b=param_f[1])

    }

)

df["Date"] = _start_date + df["Elapsed"].apply(lambda x: timedelta(minutes=x))

many_train_df = df.loc[:, ["Date", "ConfirmedCases", "Fatalities"]]

many_train_df
many_train_df.shape
train_dataset = SIRF.create_dataset(many_train_df, total_population)

train_start_date, train_initials, train_Tend, transformed_train_df = train_dataset

pprint([train_start_date.strftime("%d%b%Y"), train_initials, train_Tend])
transformed_train_df.tail()
class Estimator(object):

    def __init__(self, model, ncov_df, total_population,

                 name=None, start_date=None, date_format="%d%b%Y", param_fold_range=(1, 1), **kwargs):

        """

        Set training data.

        @model <ModelBase>: the model

        @name <str>: name of the area

        @param_fold_range <tuple(float, float)>:

            if we have fixed parameters (as kwargs), paramater range will be

            from param_fold_range[0] * (fixed) to param_fold_range[1] * (fixed)

        @kwargs: fixed parameter of the model

        @the other params: See the function named create_target_df()

        """

        if param_fold_range == (1, 1):

            self.fixed_param_dict = kwargs.copy()

            self.range_param_dict = dict()

        else:

            self.fixed_param_dict = dict()

            fold_min, fold_max = param_fold_range

            self.range_param_dict = {

                name: (value * fold_min, value * fold_max)

                for (name, value) in kwargs.items()

            }

        dataset = model.create_dataset(

            ncov_df, total_population, start_date=start_date, date_format=date_format

        )

        self.start_time, self.initials, self.Tend, self.train_df = dataset

        self.total_population = total_population

        self.name = name

        self.model = model

        self.param_dict = dict()

        self.study = None

        self.optimize_df = None



    def run(self, n_trials=500):

        """

        Try estimation (optimization of parameters and tau).

        @n_trials <int>: the number of trials

        """

        if self.study is None:

            self.study = optuna.create_study(direction="minimize")

        self.study.optimize(

            lambda x: self.objective(x),

            n_trials=n_trials,

            n_jobs=-1

        )

        param_dict = self.study.best_params.copy()

        param_dict.update(self.fixed_param_dict)

        param_dict["R0"] = self.calc_r0()

        param_dict["score"] = self.score()

        param_dict.update(self.calc_days_dict())

        self.param_dict = param_dict.copy()

        return param_dict



    def history_df(self):

        """

        Return the hsitory of optimization.

        @return <pd.DataFrame>

        """

        optimize_df = self.study.trials_dataframe()

        optimize_df["time[s]"] = optimize_df["datetime_complete"] - optimize_df["datetime_start"]

        optimize_df["time[s]"] = optimize_df["time[s]"].dt.total_seconds()

        self.optimize_df = optimize_df.drop(["datetime_complete", "datetime_start"], axis=1)

        return self.optimize_df.sort_values("value", ascending=True)



    def history_graph(self):

        """

        Show the history of parameter search using pair-plot.

        """

        if self.optimize_df is None:

            self.history_df()

        df = self.optimize_df.copy()

        sns.pairplot(df.loc[:, df.columns.str.startswith("params_")], diag_kind="kde", markers="+")

        plt.show()



    def objective(self, trial):

        # Time

        if "tau" in self.fixed_param_dict.keys():

            tau = self.fixed_param_dict["tau"]

        else:

            tau = trial.suggest_int("tau", 1, 1440)

        train_df_divided = self.train_df.copy()

        train_df_divided["t"] = (train_df_divided["T"] / tau).astype(np.int64)

        # Parameters

        p_dict = dict()

        for (name, info) in self.model.param_dict(train_df_divided).items():

            if name in self.fixed_param_dict.keys():

                param = self.fixed_param_dict[name]

            else:

                value_min, value_max = info[1:]

                if name in self.range_param_dict.keys():

                    range_min, range_max = self.range_param_dict[name]

                    value_min = max(range_min, value_min)

                    value_max = min(range_max, value_max)

                if info[0] == "float":

                    param = trial.suggest_uniform(name, value_min, value_max)

                else:

                    param = trial.suggest_int(name, value_min, value_max)

            p_dict[name] = param

        # Simulation

        t_end = train_df_divided.loc[train_df_divided.index[-1], "t"]

        sim_df = simulation(self.model, self.initials, step_n=t_end, **p_dict)

        return self.error_f(train_df_divided, sim_df)



    def error_f(self, train_df_divided, sim_df):

        """

        We need to minimize the difference of the observed values and estimated values.

        This function calculate the difference of the estimated value and obsereved value.

        """

        df = pd.merge(train_df_divided, sim_df, on="t", suffixes=("_observed", "_estimated"))

        diffs = [

            # Weighted Average: the recent data is more important

            p * np.average(

                abs(df[f"{v}_observed"] - df[f"{v}_estimated"]) / (df[f"{v}_observed"] * self.total_population + 1),

                weights=df["t"]

            )

            for (p, v) in zip(self.model.PRIORITIES, self.model.VARIABLES)

            if p != 0

        ]

        return sum(diffs) * (self.total_population ** 2)



    def compare_df(self):

        """

        Show the taining data and simulated data in one dataframe.

        

        """

        est_dict = self.study.best_params.copy()

        est_dict.update(self.fixed_param_dict)

        tau = est_dict["tau"]

        est_dict.pop("tau")

        observed_df = self.train_df.drop("T", axis=1)

        observed_df["t"] = (self.train_df["T"] / tau).astype(int)

        t_end = observed_df.loc[observed_df.index[-1], "t"]

        sim_df = simulation(self.model, self.initials, step_n=t_end, **est_dict)

        df = pd.merge(observed_df, sim_df, on="t", suffixes=("_observed", "_estimated"))

        df = df.set_index("t")

        return df



    def compare_graph(self):

        """

        Compare obsereved and estimated values in graphs.

        """

        df = self.compare_df()

        use_variables = [

            v for (i, (p, v)) in enumerate(zip(self.model.PRIORITIES, self.model.VARIABLES))

            if p != 0

        ]

        val_len = len(use_variables) + 1

        fig, axes = plt.subplots(ncols=1, nrows=val_len, figsize=(9, 6 * val_len / 2))

        for (ax, v) in zip(axes.ravel()[1:],use_variables):

            df[[f"{v}_observed", f"{v}_estimated"]].plot.line(

                ax=ax, ylim=(None, None), sharex=True,

                title=f"{self.model.NAME}: Comparison of observed/estimated {v}(t)"

            )

            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

            ax.ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))

            ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)

        for v in use_variables:

            df[f"{v}_diff"] = df[f"{v}_observed"] - df[f"{v}_estimated"]

            df[f"{v}_diff"].plot.line(

                ax=axes.ravel()[0], sharex=True,

                title=f"{self.model.NAME}: observed - estimated"

            )

        axes.ravel()[0].axhline(y=0, color="black", linestyle="--")

        axes.ravel()[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        axes.ravel()[0].ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))

        axes.ravel()[0].legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)

        fig.tight_layout()

        fig.show()

    

    def calc_r0(self):

        """

        Calculate R0.

        """

        est_dict = self.study.best_params.copy()

        est_dict.update(self.fixed_param_dict)

        est_dict.pop("tau")

        model_instance = self.model(**est_dict)

        return model_instance.calc_r0()



    def calc_days_dict(self):

        """

        Calculate 1/beta etc.

        """

        est_dict = self.study.best_params.copy()

        est_dict.update(self.fixed_param_dict)

        tau = est_dict["tau"]

        est_dict.pop("tau")

        model_instance = self.model(**est_dict)

        return model_instance.calc_days_dict(tau)



    def predict_df(self, step_n):

        """

        Predict the values in the future.

        @step_n <int>: the number of steps

        @return <pd.DataFrame>: predicted data for measurable variables.

        """

        est_dict = self.study.best_params.copy()

        est_dict.update(self.fixed_param_dict)

        tau = est_dict["tau"]

        est_dict.pop("tau")

        df = simulation(self.model, self.initials, step_n=step_n, **est_dict)

        df["Time"] = (df["t"] * tau).apply(lambda x: timedelta(minutes=x)) + self.start_time

        df = df.set_index("Time").drop("t", axis=1)

        df = (df * self.total_population).astype(np.int64)

        upper_cols = [n.upper() for n in df.columns]

        df.columns = upper_cols

        df = self.model.calc_variables_reverse(df).drop(upper_cols, axis=1)

        return df



    def predict_graph(self, step_n, name=None, excluded_cols=None):

        """

        Predict the values in the future and create a figure.

        @step_n <int>: the number of steps

        @name <str>: name of the area

        @excluded_cols <list[str]>: the excluded columns in the figure

        """

        if self.name is not None:

            name = self.name

        else:

            name = str() if name is None else name

        df = self.predict_df(step_n=step_n)

        if excluded_cols is not None:

            df = df.drop(excluded_cols, axis=1)

        r0 = self.param_dict["R0"]

        title = f"Prediction in {name} with {self.model.NAME} model: R0 = {r0}"

        line_plot(df, title, v= datetime.today(), h=self.total_population)



    def rmsle(self, compare_df):

        """

        Return the value of RMSLE.

        @param compare_df <pd.DataFrame>

        """

        df = compare_df.set_index("t") * self.total_population

        score = 0

        for (priority, v) in zip(self.model.PRIORITIES, self.model.VARIABLES):

            if priority == 0:

                continue

            observed, estimated = df[f"{v}_observed"], df[f"{v}_estimated"]

            diff = (np.log(observed + 1) - np.log(estimated + 1))

            score += (diff ** 2).sum()

        rmsle = np.sqrt(score / len(df))

        return rmsle



    def score(self):

        """

        Return the value of RMSLE.

        """

        rmsle = self.rmsle(self.compare_df().reset_index("t"))

        return rmsle



    def info(self):

        """

        Return Estimater information.

        @return <tupple[object]>:

            - <ModelBase>: model

            - <dict[str]=str>: name, total_population, start_time, tau

            - <dict[str]=float>: values of parameters of model

        """

        param_dict = self.study.best_params.copy()

        param_dict.update(self.fixed_param_dict)

        info_dict = {

            "name": self.name,

            "total_population": self.total_population,

            "start_time": self.start_time,

            "tau": param_dict["tau"],

            "initials": self.initials

        }

        param_dict.pop("tau")

        return (self.model, info_dict, param_dict)
%%time

first_estimator = Estimator(SIRF, many_train_df, total_population, name="California", theta=0, sigma=0)

first_dict = first_estimator.run()
first_estimator.history_df().head()
first_estimator.history_graph()
first_estimator.compare_df()
first_estimator.compare_graph()
pd.DataFrame.from_dict({"First": first_dict}, orient="index")
_, info_dict, param_dict = first_estimator.info()

param_dict.pop("theta")

param_dict.pop("sigma")

param_dict["tau"] = info_dict["tau"]

pd.DataFrame.from_dict({"For second": param_dict}, orient="index")
%%time

second_estimator = Estimator(SIRF, many_train_df, total_population, name="California", **param_dict)

second_dict = second_estimator.run()
second_estimator.history_graph()
second_estimator.compare_graph()
pd.DataFrame.from_dict({"First": first_dict, "Second": second_dict}, orient="index")
_, info_dict, param_dict = second_estimator.info()

param_dict["tau"] = info_dict["tau"]

pd.DataFrame.from_dict({"For third": param_dict}, orient="index")
%%time

third_estimator = Estimator(

    SIRF, many_train_df, total_population, name="California", param_fold_range=(0.5, 1.5), **param_dict

)

third_dict = third_estimator.run()
third_estimator.history_graph()
third_estimator.compare_graph()
pd.DataFrame.from_dict({"First": first_dict, "Second": second_dict, "Third": third_dict}, orient="index")
first_score = first_dict["score"]

second_score = second_dict["score"]

third_score = third_dict["score"]

min_score = min(first_score, second_score, third_score)
if min_score == first_score:

    last_model, last_info_dict, last_param_dict = first_estimator.info()

if min_score == second_score:

    last_model, last_info_dict, last_param_dict = second_estimator.info()

else:

    last_model, last_info_dict, last_param_dict = third_estimator.info()
last_model.NAME
last_info_dict
pd.DataFrame.from_dict({"Info": last_param_dict}, orient="index")
class Predicter(object):

    """

    Predict the future using models.

    """

    def __init__(self, name, total_population, start_time, tau, initials, date_format="%d%b%Y"):

        """

        @name <str>: place name

        @total_population <int>: total population

        @start_time <datatime>: the start time

        @tau <int>: tau value (time step)

        @initials <list/tupple/np.array[float]>: initial values of the first model

        @date_format <str>: date format to display in figures

        """

        self.name = name

        self.total_population = total_population

        self.start_time = start_time

        self.tau = tau

        self.date_format = date_format

        # Un-fixed

        self.last_time = start_time

        self.axvlines = list()

        self.initials = initials

        self.df = pd.DataFrame()

        self.title_list = list()

        self.reverse_f = lambda x: x



    def add(self, model, end_day_n=None, count_from_last=False, vline=True, **param_dict):

        """

        @model <ModelBase>: the epidemic model

        @end_day_n <int/None>: day number of the end date (0, 1, 2,...), or None (now)

            - if @count_from_last <bool> is True, start point will be the last date registered to Predicter

        @vline <bool>: if True, vertical line will be shown at the end date

        @**param_dict <dict>: keyword arguments of the model

        """

        # Validate day nubber, and calculate step number

        if end_day_n is None:

            end_time = datetime.now()

        else:

            if count_from_last:

                end_time = self.last_time + timedelta(days=end_day_n)

            else:

                end_time = self.start_time + timedelta(days=end_day_n)

        if end_time <= self.last_time:

            raise Exception(f"Model on {end_time.strftime(self.date_format)} has been registered!")

        step_n = int((end_time - self.last_time).total_seconds() / 60 / self.tau)

        self.last_time = end_time

        # Perform simulation

        new_df = simulation(model, self.initials, step_n=step_n, **param_dict)

        new_df["t"] = new_df["t"] + len(self.df)

        self.df = pd.concat([self.df, new_df], axis=0).fillna(0)

        self.initials = new_df.set_index("t").iloc[-1, :]

        # For title

        if vline:

            self.axvlines.append(end_time)

            r0 = model(**param_dict).calc_r0()

            self.title_list.append(

                f"{model.NAME}({r0}, -{end_time.strftime(self.date_format)})"

            )

        # Update reverse function (X, Y,.. to Susceptible, Infected,...)

        self.reverse_f = model.calc_variables_reverse

        return self



    def restore_df(self):

        """

        Return the dimentional simulated data.

        @return <pd.DataFrame>

        """

        df = self.df.copy()

        df["Time"] = self.start_time + df["t"].apply(lambda x: timedelta(minutes=x * self.tau))

        df = df.drop("t", axis=1).set_index("Time") * self.total_population

        df = df.astype(np.int64)

        upper_cols = [n.upper() for n in df.columns]

        df.columns = upper_cols

        df = self.reverse_f(df).drop(upper_cols, axis=1)

        return df



    def restore_graph(self, drop_cols=None, **kwargs):

        """

        Show the dimentional simulate data as a figure.

        @drop_cols <list[str]>: the columns not to be shown

        @kwargs: keyword arguments of line_plot() function

        """

        df = self.restore_df()

        if drop_cols is not None:

            df = df.drop(drop_cols, axis=1)

        axvlines = [datetime.now(), *self.axvlines] if len(self.axvlines) == 1 else self.axvlines[:]

        line_plot(

            df,

            title=f"{self.name}: {', '.join(self.title_list)}",

            v=axvlines[:-1],

            h=self.total_population,

            **kwargs

        )
predicter_today = Predicter(**last_info_dict)

predicter_today.add(SIRF, end_day_n=None, count_from_last=False, vline=False, **last_param_dict)

predicter_today.restore_graph(drop_cols=["Susceptible"])
df = predicter_today.restore_df().reset_index()

df["Date"] = df["Time"].dt.date

df = df.drop("Time", axis=1).groupby("Date").last().reset_index()

df["Confirmed"] = df["Infected"] + df["Recovered"] + df["Fatal"]

df = pd.concat([train_df.drop("Date", axis=1), df], axis=1)

df.loc[:, ["Date", "ConfirmedCases", "Fatalities", "Confirmed", "Fatal"]]
days_to_predict = int((test_df["Date"].max() - datetime.today()).total_seconds() / 3600 / 24 + 1)

days_to_predict
predicter = Predicter(**last_info_dict)

predicter.add(SIRF, end_day_n=None, count_from_last=False, vline=False, **last_param_dict)

predicter.add(SIRF, end_day_n=days_to_predict, count_from_last=True, **last_param_dict)

predicter.restore_graph(drop_cols=["Susceptible"])
predicted_df = predicter.restore_df()

predicted_df.tail()
test_df.tail()
submission_sample_raw.head()
submission_sample_raw.tail()
submission_sample_raw.shape
# Predicted data

df = predicted_df.reset_index()

df["Date"] = df["Time"].dt.date.astype(str)

df["ConfirmedCases"] = df["Infected"] + df["Recovered"] + df["Fatal"]

df = df.rename({"Fatal": "Fatalities"}, axis=1)

df = df.loc[:, ["Date", "ConfirmedCases", "Fatalities"]]

df = df.groupby("Date").last().reset_index()

# Merge with test dataframe

_test_df = test_df.copy()

_test_df["Date"] = _test_df["Date"].astype(str)

df = pd.merge(_test_df, df, on="Date").drop("Date", axis=1)

submission_df = df.copy()

submission_df.head()
submission_df.tail()
submission_df.shape
submission_df.to_csv("submission.csv", index=False)