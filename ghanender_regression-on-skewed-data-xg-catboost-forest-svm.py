import pandas as pd
import numpy as np
df=pd.read_csv(r"..\Regression_train.csv")
df.head()
#df = df.drop(['Serial No.'], axis=1)
#df.head()
df.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns


fig = sns.distplot(df['GRE Score'], kde=False)
plt.title("Distribution of GRE Scores")
plt.show()

fig = sns.distplot(df['TOEFL Score'], kde=False)
plt.title("Distribution of TOEFL Scores")
plt.show()

fig = sns.distplot(df['University Rating'], kde=False)
plt.title("Distribution of University Rating")
plt.show()

fig = sns.distplot(df['SOP'], kde=False)
plt.title("Distribution of SOP Ratings")
plt.show()

fig = sns.distplot(df['CGPA'], kde=False)
plt.title("Distribution of CGPA")
plt.show()

plt.show()
fig = sns.regplot(x="GRE Score", y="TOEFL Score", data=df)
plt.title("GRE Score vs TOEFL Score")
plt.show()
fig = sns.regplot(x="GRE Score", y="CGPA", data=df)
plt.title("GRE Score vs CGPA")
plt.show()
fig = sns.regplot(x="GRE Score", y="LOR ", data=df)
plt.title("GRE Score vs LOR")
plt.show()
fig = sns.regplot(x="CGPA", y="LOR ", data=df)
plt.title("CGPA vs LOR")
plt.show()
fig = sns.lmplot(x="CGPA", y="LOR ", data=df, hue="Research")
plt.title("LOR vs CGPA")
plt.show()
corr = df.corr()
fig, ax = plt.subplots(figsize=(8, 8))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.show()
from sklearn.model_selection import train_test_split

X = df.drop(['Chance of Admit '], axis=1)
y = df['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, shuffle=False)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
model=LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, predictions)))
print(df['SOP'].skew())
print(df['GRE Score'].skew())
print(df['University Rating'].skew())
print(df['LOR '].skew())
print(df['CGPA'].skew())
df.columns
# https://www.statisticshowto.com/box-cox-transformation/
from scipy import stats
df['SOP_boxcox'] =stats.boxcox(df['SOP'])[0]
df.head()
pd.Series(df['SOP_boxcox']).skew()
from sklearn.model_selection import train_test_split

X1 = df.drop(['Chance of Admit ','SOP'], axis=1)
y = df['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X1,y,test_size = 0.30, shuffle=False)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, predictions)))
df1=pd.read_csv(r"..car_regression.csv")
df1.head()
df1['distance'].skew()
df1['speed'].skew()
df1['temp_inside'].skew()
df1['temp_outside'].skew()
df1['gas_type'].unique()
from sklearn.preprocessing import OneHotEncoder
df2=pd.get_dummies(df1.gas_type)
df3=pd.concat([df1,df2],axis=1)
df3.head()
from sklearn.model_selection import train_test_split
df3=df3.dropna()
X = df3.drop(['gas_type','consume'], axis=1)
y = df3['consume']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, shuffle=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, predictions)))
df3['boxcox_dist']=stats.boxcox(df3['distance'])[0]
pd.Series(df3['boxcox_dist']).skew()
X3 = df3.drop(['gas_type','consume','distance'], axis=1)
y3 = df3['consume']
X_train, X_test, y_train, y_test = train_test_split(X3,y3,test_size = 0.3, shuffle=False)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, predictions)))
df=pd.read_csv(r"car_price.csv")
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor
from sklearn.metrics import mean_squared_error

models = [['DecisionTree :',DecisionTreeRegressor()],
           ['Linear Regression :', LinearRegression()],
           ['RandomForest :',RandomForestRegressor()],
           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],
           ['SVM :', SVR()],
           ['AdaBoostClassifier :', AdaBoostRegressor()],
           ['GradientBoostingClassifier: ', GradientBoostingRegressor()],
           ['Xgboost: ', XGBRegressor()],
           ['CatBoost: ', CatBoostRegressor(logging_level='Silent')],
           ['Lasso: ', Lasso()],
           ['Ridge: ', Ridge()],
           ['BayesianRidge: ', BayesianRidge()],
           ['ElasticNet: ', ElasticNet()],
           ['HuberRegressor: ', HuberRegressor()]]

print("Results...")


df.head()
df=df.replace("?","")
df.head()
df1=pd.get_dummies(df.make)
df2=pd.concat([df,df1],axis=1)
df2=df2.dropna()
df2.info()
df2['normalized-losses'] = pd.to_numeric(df2['normalized-losses'],errors='coerce')
df2['bore'] = pd.to_numeric(df2['bore'],errors='coerce')
df2['stroke'] = pd.to_numeric(df2['stroke'],errors='coerce')
df2['horsepower'] = pd.to_numeric(df2['horsepower'],errors='coerce')
df2['peak-rpm'] = pd.to_numeric(df2['peak-rpm'],errors='coerce')
df2['price'] = pd.to_numeric(df2['price'],errors='coerce')
df2=df2.dropna()
df2.info()
X = df2.drop(['make','price'], axis=1)
y = df2['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, shuffle=False)
for name,model in models:
    model = model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X_train)
X_pca = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])
for name,model in models:
    model = model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))








