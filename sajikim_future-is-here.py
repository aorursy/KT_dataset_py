import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor

from sklearn.linear_model import PassiveAggressiveRegressor, ARDRegression, RidgeCV

from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR, LinearSVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.cross_decomposition import PLSRegression
reg_dict = {"LinearRegression": LinearRegression(),

            "Ridge": Ridge(),

            "Lasso": Lasso(),

            "ElasticNet": ElasticNet(), 

            "Polynomial_deg2": Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression())]),

            "Polynomial_deg3": Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression())]),

            "Polynomial_deg4": Pipeline([('poly', PolynomialFeatures(degree=4)),('linear', LinearRegression())]),

            "Polynomial_deg5": Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', LinearRegression())]),

            "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=3),

            "DecisionTreeRegressor": DecisionTreeRegressor(),

            "RandomForestRegressor": RandomForestRegressor(),

            "SVR_rbf": SVR(kernel='rbf', C=1e3, gamma=0.1, epsilon=0.1, degree=3),

            "SVR_linear": SVR(kernel='linear', C=1e3, gamma=0.1, epsilon=0.1, degree=3),

            "GaussianProcessRegressor": GaussianProcessRegressor(),

            "SGDRegressor": SGDRegressor(),

            "MLPRegressor": MLPRegressor(hidden_layer_sizes=(10,10), max_iter=100, early_stopping=True, n_iter_no_change=5),

            "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100), 

            "PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=100, tol=1e-3),

            "TheilSenRegressor": TheilSenRegressor(random_state=0),

            "RANSACRegressor": RANSACRegressor(random_state=0),

            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(),

            "AdaBoostRegressor": AdaBoostRegressor(random_state=0, n_estimators=100),

            "BaggingRegressor": BaggingRegressor(base_estimator=SVR(), n_estimators=10),

            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=0),

            "VotingRegressor": VotingRegressor([('lr', LinearRegression()), ('rf', RandomForestRegressor(n_estimators=10))]),

            "StackingRegressor": StackingRegressor(estimators=[('lr', RidgeCV()), ('svr', LinearSVR())], final_estimator=RandomForestRegressor(n_estimators=10)),

            "ARDRegression": ARDRegression(),

            "HuberRegressor": HuberRegressor(),

            }
def make_future_dates(last_date, period):

    prediction_dates=pd.date_range(last_date, periods=period+1, freq='B')

    return prediction_dates[1:]



def prepare_data(data2, forecast_out):

    label = np.roll(data2, -forecast_out).reshape((-1))

    X = data2; 

    X_lately = X[-forecast_out:]

    X = X[:-forecast_out] 

    y = label[:-forecast_out] 

    return [X, y, X_lately];



# load data

data = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv',header=0,parse_dates=[0])

data=data[["Date","Price"]].tail(100)

data=data.set_index('Date',drop=True)



# prepare data

forecast_out = 34

x_train, y_train, X_lately = prepare_data(data,forecast_out)



# feature Scaling

stdsc = StandardScaler()

x_train_std = stdsc.fit_transform(x_train)

y_train_std = stdsc.transform(y_train.reshape(-1, 1))

X_lately_std = stdsc.transform(X_lately)



# prediction

future_dates = make_future_dates(data.index[-1],forecast_out)

df_preds=pd.DataFrame({"Date":future_dates})

df_preds=df_preds.set_index('Date',drop=True)

for reg_name, reg in reg_dict.items():

    reg.fit(x_train_std, y_train_std)

    prediction = reg.predict(X_lately_std)

    prediction = stdsc.inverse_transform(prediction.reshape((-1)))

    df_preds[reg_name] = pd.DataFrame({'Price':prediction},index=future_dates)
def disp_all(mode='entire'):

    plt.figure(figsize=(16, 8))

    for col in df_preds.columns:

        plt.plot(df_preds.index[-len(df_preds):], df_preds[col][-len(df_preds):],label=col)



    if mode is 'entire':

        plt.plot(data.index[-100:], data['Price'].tail(100),label="Actual")

        plt.vlines([data.index[-1]], 0, 60, "red", linestyles='dashed')

        plt.text([data.index[-1]], 60, 'Today', backgroundcolor='white', ha='center', va='center')

        plt.vlines([data.index[-1-75]], 0, 60, "red", linestyles='dashed')

        plt.text([data.index[-1-75]], 60, '75 days before', backgroundcolor='white', ha='center', va='center')

        plt.vlines([df_preds.index[-1]], 0, 60, "red", linestyles='dashed')

        plt.text([df_preds.index[-1]], 60, '34 days after', backgroundcolor='white', ha='center', va='center')

    plt.ylim(0, 80)

    plt.title('Predictions ('+mode+')')

    plt.xlabel('Date')

    plt.ylabel('Price')

    plt.legend(loc='best',ncol=2)

    plt.grid(True)

    plt.show()

    

disp_all('entire')

disp_all('zoom')
template = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/sampleSubmission0710_updated.csv',header=0,parse_dates=[0])

template.drop("Price",axis=1,inplace=True)



df2 = data.copy() 

for col in df_preds.columns:

    df2[col] = data['Price'].copy() 

df2 = pd.concat([df2, df_preds])

df2.drop("Price",axis=1,inplace=True)



for col in df2.columns:

    submission = pd.merge(template, df2[col], on='Date', how='left')

    submission.rename(columns={col: 'Price'},inplace=True)

    if submission["Price"].isnull().any():

        submission["Price"].fillna(submission["Price"].mean(),inplace=True)

    submission["Price"] = submission["Price"].round(9)

    submission.to_csv("submission_" + col + ".csv", index=False)