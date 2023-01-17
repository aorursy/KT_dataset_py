import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score



from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor



from sklearn.preprocessing import StandardScaler
df = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv", index_col="Serial No.")
df.head()
df.info()
fig = plt.figure(figsize=(12,20))



for i in range(len(df.columns)):

    fig.add_subplot(4,2,i+1)

    sns.distplot(df.iloc[:,i], hist=False, rug=True, kde_kws={"bw":0.01}, label="dist")

    plt.xlabel(df.columns[i])



plt.tight_layout()
fig = plt.figure(figsize=(12,20))



for i in range(len(df.columns)):

    fig.add_subplot(4,2,i+1)

    sns.boxplot(y=df.iloc[:,i])

    plt.xlabel(df.columns[i])

    plt.ylabel("Spread of Data")

    

plt.tight_layout()
fig = plt.figure(figsize=(12,20))



for i in range(len(df.columns)):

    fig.add_subplot(4,2,i+1)

    sns.scatterplot(x=df.iloc[:,i], y=df["Chance of Admit "])

    

plt.tight_layout()
sns.heatmap(df.corr() > 0.8, annot=True)
corr = df.corr()

corr["Chance of Admit "].sort_values(ascending=False)
###Model



X = df.drop("Chance of Admit ", axis=1)

y = df["Chance of Admit "]



scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)



X_train = X[:400]

X_test = X[400:]



y_train = y[:400]

y_test = y[400:]
models = [["Linear Regression",LinearRegression()],

         ["KNN",KNeighborsRegressor(n_neighbors=10, n_jobs=8)],

         ["Random Forest", RandomForestRegressor(n_estimators=500, n_jobs=8)],

         ["Decision Tree", DecisionTreeRegressor()],

         ["XGBoost", XGBRegressor(n_jobs=8, n_estimators=1000, learning_rate=0.05)]]
for name, model in models:

    model.fit(X_train,y_train)

    predictions = model.predict(X_test)

    print("{} RMSE: ".format(name), np.sqrt(mean_squared_error(y_test,predictions)), end='\n\n')
linear = LinearRegression()

linear.fit(X_train,y_train)

preds = linear.predict(X_test)



print("Linear Regression RMSE: ", np.sqrt(mean_squared_error(y_test,preds)))