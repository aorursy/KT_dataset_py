import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set_style("whitegrid")
df = pd.read_csv("../input/medium-articles-dataset/medium_data.csv", index_col="id")
df.head()
df.info()
df.drop(columns=["url", "date"], inplace=True, axis=1)
missing = df.isnull().sum()

missing = missing[missing>0]

print(missing)

missing.sort_values(ascending=False)

missing.plot.bar()
df["Has Image"] = df["image"].apply(lambda x: 0 if x is np.nan else 1)

df["Has Subtitle"] = df["subtitle"].apply(lambda x: 0 if x is np.nan else 1)

df["subtitle"] = df["subtitle"].apply(lambda x: "" if x is np.nan else x)

df["Length of Title"] = df["title"].apply(lambda x: len(x))

df["Length of subtitle"] = df["subtitle"].apply(lambda x: len(x))
df.head()
df.drop(["image","title","subtitle"], axis=1, inplace=True)
df.info()
df["responses"] = df["responses"].apply(lambda x: int(x))



### Gives error that we have some rows which have "Read" as it's entry in the response column
df["responses"].replace("Read",0, inplace=True)

df["responses"] = df["responses"].apply(lambda x: int(x))
print(df.publication.nunique())

df.publication.unique()
df = pd.get_dummies(columns=["publication"], prefix="Pub", data=df)
df.head()
df.info()
df.columns
comparable_cols = ["claps","responses","reading_time","Has Subtitle","Has Image","Length of Title","Length of subtitle"]
fig = plt.figure(figsize=(12,6))



for i in range(len(comparable_cols)):

    fig.add_subplot(2,4,i+1)

    sns.distplot(df[comparable_cols[i]].iloc[:], rug=True, hist=False, kde_kws={"bw":0.01})

    

plt.tight_layout()
fig = plt.figure(figsize=(12,6))



for i in range(len(comparable_cols)):

    fig.add_subplot(2,4,i+1)

    sns.boxplot(y=df[comparable_cols[i]])

    

plt.tight_layout()
fig = plt.figure(figsize=(12,6))



for i in range(len(comparable_cols)):

    fig.add_subplot(2,4,i+1)

    sns.scatterplot(y=df["claps"], x=df[comparable_cols[i]])

    

plt.tight_layout()
df.columns
cat_cols = ['Pub_Better Humans','Pub_Better Marketing','Pub_Data Driven Investor','Pub_The Startup','Pub_The Writing Cooperative','Pub_Towards Data Science','Pub_UX Collective']

fig = plt.figure(figsize=(12,6))



for i in range(len(cat_cols)):

    fig.add_subplot(2,4,i+1)

    sns.boxplot(x=df[df[cat_cols[i]] == 1][cat_cols[i]], y=df["claps"])

    

plt.tight_layout()
###Dealing with outliers



df = df.drop(df[df["responses"] > 100].index)

df = df.drop(df[df["reading_time"] > 40].index)

df = df.drop(df[df["Length of Title"] > 200].index)

df = df.drop(df[df["Length of subtitle"] > 200].index)

df = df.drop(df[df["claps"] > 15000].index)
plt.figure(figsize=(10,10))

sns.heatmap(df.corr() > 0.8, annot=True)
corr = df.corr()

corr.claps
###Trying to see how title and subtitle length compare



figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

figure.set_size_inches(20,5)

sns.regplot(x=df["Length of Title"], y=df["claps"], ax=ax1)

sns.regplot(x=df["Length of subtitle"], y=df["claps"], ax=ax2)

sns.regplot(x=df["Length of Title"] + df["Length of subtitle"], y=df["claps"], ax=ax3)



plt.tight_layout()
df["total length"] = df["Length of Title"] + df["Length of subtitle"]
corr = df.corr()

corr["claps"].sort_values(ascending=False)
from sklearn.model_selection import train_test_split



from sklearn.metrics import mean_absolute_error, mean_squared_error



from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
models = [["Linear Regression",LinearRegression()],

         ["KNN",KNeighborsRegressor(n_neighbors=1, n_jobs=-1)],

         ["Decision Tree", DecisionTreeRegressor()],

         ["Random Forest", RandomForestRegressor(n_estimators=100, n_jobs=-1)],

         ["XGBoost", XGBRegressor(n_estimators=500, n_jobs=-1, learning_rate=0.05)]]
X = df.drop(["claps"], axis=1)

y = df.claps



#scaler = StandardScaler().fit(X)

#X = scaler.transform(X)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
for name,model in models:

    model.fit(X_train,y_train)

    predictions = model.predict(X_test)

    print("{} MAE: ".format(name), mean_absolute_error(y_test,predictions))

    print("{} RMSE: ".format(name), np.sqrt(mean_squared_error(y_test,predictions)), end="\n\n")
lin = LinearRegression()

lin.fit(X_train,y_train)

pred = lin.predict(X_test)

print("MAE: ", mean_absolute_error(y_test,pred))
sns.regplot(x=y_test, y=pred)

plt.xlabel("Actual")

plt.ylabel("Predicted")
weight = pd.DataFrame(data={"Feature":df.columns.drop('claps'),"Weights":lin.coef_})

weight.sort_values(ascending=False, by="Weights")