import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import datetime
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
df=pd.read_csv('../input/covidglobalcompetition/covid-global-forcast.csv',parse_dates=["Date"])
df["Province/State"]=df["Province/State"].fillna("")
df=df.sort_values(by=["Date","Country/Region","Province/State"])
df["Location"]=df["Country/Region"] +"/"+ df["Province/State"]
index=pd.MultiIndex.from_frame(df[["Location","Date"]])
df=df.set_index(index,drop=False)
df=df.drop(columns=["Country/Region","Province/State","Lat","Long"])
# Active Case = confirmed - deaths - recovered
df['Active'] = df['# ConfirmedCases'] - df['# Fatalities'] - df['# Recovered_cases']

df["Day"]=df["Date"].dt.day
df["Day of the week"]= df["Date"].dt.weekday
days =["Monday", "Tuesday", "Wednesday", "Thursday", 
                         "Friday", "Saturday", "Sunday"] 
days_dict={x:days[x] for x in range(len(days))}
df["Day of the week"]=df["Day of the week"].map(days_dict)
pandemic_date=datetime.datetime.strptime("11 March 2020","%d %B %Y")
df["Days after/before the pandemic"]=df["Date"] - pandemic_date
df.head(10)

#https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(df["# Fatalities"], lags=10)
plot_pacf(df["# Fatalities"], lags=10)
plot_acf(df["# ConfirmedCases"], lags=10)
plot_pacf(df["# ConfirmedCases"], lags=10)
df["Lag_1_fatalities"]=df.groupby(level=0)["# Fatalities"].shift(1)
df["Lag_1_confirmed_cases"]=df.groupby(level=0)["# ConfirmedCases"].shift(1)
df=df.dropna()
from category_encoders.hashing import HashingEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
y=df["# Fatalities"].values


df=df.drop(columns=["# Fatalities","Date"])


ce_hash=HashingEncoder(cols = ["Location"])
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), ["Day of the week"]),("label",ce_hash,["Location"])])
X=transformer.fit_transform(df)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21)



def RMSLE(predictions,actual_values):
    # root mean squared logarithmic error.
    number_of_predictions=np.shape(predictions)[0]
    predictions=np.log(predictions+1)
    actual_values=np.log(actual_values+1)
    squared_differences=np.power(np.subtract(predictions,actual_values),2)
    total_sum=np.sum(squared_differences)
    avg_squared_diff=total_sum/number_of_predictions
    rmsle=np.sqrt(avg_squared_diff)
    return rmsle
from sklearn.linear_model import Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

models = []
models.append(('LASSO', Lasso()))
models.append(('DF', DecisionTreeRegressor()))
models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees
models.append(('SVR', SVR(gamma='auto'))) # kernel = linear

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    names.append(name)
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    rmsle=RMSLE(predictions,y_test)
    results.append(rmsle)
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()