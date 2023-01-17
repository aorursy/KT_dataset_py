# imports
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import style
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.ar_model import AutoReg
from sklearn.neural_network import MLPClassifier

# configs
mpld3.enable_notebook()
warnings.filterwarnings("ignore")

# read csvs & build dataframe
df = pd.read_csv('../input/2-years-restaurant-sale-with-multiple-external-var/sales.csv', ';')
df = df.drop(columns=['Unnamed: 0'])
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')

# input & output
X = df.drop(columns=['DATE', 'SALES'])
y = df.drop(columns=['DATE', 'IS_WEEKEND', 'IS_HOLIDAY', 'IS_FESTIVE_DATE', 'IS_PRE_FESTIVE_DATE', 'IS_AFTER_FESTIVE_DATE', 'IS_PEOPLE_WEEK_PAYMENT', 'IS_LOW_SEASON', 'AMOUNT_OTHER_OPENED_RESTAURANTS', 'WEATHER_PRECIPITATION', 'WEATHER_TEMPERATURE', 'WEATHER_HUMIDITY'])

# standardization & normalization
X = preprocessing.scale(X)

# shuffled and splitted into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# descbring
df.describe()
# plot chart
df.plot(y='SALES', x='DATE', figsize=(13, 4), linestyle='solid', linewidth=1, markersize=2, style="-o")
plt.legend(loc=1)
plt.title('SALE HISTORY')
plt.xlabel('DATE')
plt.ylabel('AMOUNT OF LUNCH SOLD')
plt.show()
# basic predictions 
model = LinearRegression(n_jobs=-1) 
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f'Acurácia do LinearRegression: {round(accuracy*100, 2)}%')

model8 = Ridge(alpha=1.0)
model8.fit(X_train, y_train)
accuracy8 = model8.score(X_test, y_test)
print(f'Acurácia do Ridge: {round(accuracy8*100, 2)}%')

model9 = make_pipeline(PolynomialFeatures(degree=5), LinearRegression(fit_intercept = False))
model9.fit(X_train, y_train)
accuracy9 = model9.score(X_test, y_test)
print(f'Acurácia do PolynomialFeatures: {round(accuracy9*100, 2)}%')