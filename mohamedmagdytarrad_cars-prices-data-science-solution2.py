import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import  SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df=pd.read_csv('../input/dataset-to-estimate-used-car-price/car_pricing.csv',names=headers)
print('The 15 rows of data')
df.head(15)
print('The  last 15 rows of data')
df.tail(15)
# Replace '?' To 'NaN'
df.replace('?',np.nan,inplace=True)
df.head(10)

df.info()
#for column in missing_data.columns.values.tolist():
   # print(column)
   # print(missing_data[column].value_counts())
   # print(' ')
    
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df.dtypes
avg_norm_loss=df[['normalized-losses']].astype(float)
avg_bore=df[['bore']].astype(float)
missing_stroke_values=df[['stroke']].astype(float)
avg_horsepower = df[['horsepower']].astype(float)
avg_peakrpm=df[['peak-rpm']].astype(float)
avg_price=df[['price']].astype(float)
# Creatiing method from SimpleImputer class
impute=SimpleImputer(missing_values=np.nan,strategy='mean')
df[['normalized-losses']]=impute.fit_transform(df[['normalized-losses']])
df[['bore']]=impute.fit_transform(df[['bore']])
df[['stroke']]=impute.fit_transform(df[['stroke']])
df[['horsepower']]=impute.fit_transform(df[['horsepower']])
df[['peak-rpm']]=impute.fit_transform(df[['peak-rpm']])
df[["price"]]=impute.fit_transform(df[["price"]])
impute_2=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
print(df['num-of-doors'].value_counts())
print(df['num-of-doors'].value_counts().idxmax())

df[['num-of-doors']]=impute_2.fit_transform(df[['num-of-doors']])
print('Nums of doors after handeling missing data:', df['num-of-doors'].value_counts())
df.head(15)
df.dtypes
print('The New Data After handeling with missing data',df.info())
# Histogram
#first lets analysis price
sns.distplot(df['price'])
#skewness 
print("Skewness: %f" % df['price'].skew())

# scatter plot

# regplot
sns.regplot(df['normalized-losses'],df['price'])
plt.ylim(0,)
sns.regplot('engine-size', 'price',data=df)
plt.ylim(0,)
sns.regplot('highway-mpg', 'price',data=df)
plt.ylim(0,)

sns.regplot(x='stroke',y='price',data=df)
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
# boxplot
graph = pd.concat([df['price'], df['make']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='make', y="price", data=graph)

#box plot overallqual/saleprice

graph = pd.concat([df['price'], df['fuel-type']], axis=1)
f = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='fuel-type', y="price", data=graph)
# The corr heatmap
corr = df.corr()
f = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True)

k = 10 #number of variables for heatmap
corr = df.corr()
cols = corr.nlargest(k, 'price')['price'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot
sns.set()
cols = ['price','engine-size','curb-weight','horsepower','width','length','wheel-base','bore']
sns.pairplot(df[cols], size = 2.5)
plt.show()
df['fuel-type']=pd.get_dummies(df['fuel-type'])
df['aspiration']=pd.get_dummies(df['aspiration'])
df['make']=pd.get_dummies(df['make'])
df['num-of-doors']=pd.get_dummies(df['num-of-doors'])
df['body-style']=pd.get_dummies(df['body-style'])
df['drive-wheels']=pd.get_dummies(df['drive-wheels'])
df['engine-location']=pd.get_dummies(df['engine-location'])
df['engine-type']=pd.get_dummies(df['engine-type'])
df[ "num-of-cylinders"]=pd.get_dummies(df[ "num-of-cylinders"])
df["fuel-system"]=pd.get_dummies(df[ "fuel-system"])
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
scale=StandardScaler()
X=scale.fit_transform(X)
X=pd.DataFrame(X)
X.head(10)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
Regressor = LinearRegression()
Regressor.fit(X_train, y_train)
y_pred = Regressor.predict(X_test)
linear_accurcy = round(Regressor.score(X_train, y_train) * 100, 2)
linear_accurcy
Randomforest = RandomForestRegressor()
Randomforest.fit(X_train, y_train)
y_pred = Randomforest.predict(X_test)
forest_accurcy = round(Randomforest.score(X_train, y_train) * 100, 2)
forest_accurcy
Tree =DecisionTreeRegressor ()
Tree.fit(X_train, y_train)
y_pred = Tree.predict(X_test)
tree_accurcy = round(Tree.score(X_train, y_train) * 100, 2)
tree_accurcy
ridge = Ridge(alpha=100)
ridge.fit(X_train,y_train)
y_pred=ridge.predict(X_test)
ridge_accurcy=round(ridge.score(X_train, y_train) * 100, 2)
ridge_accurcy
models = pd.DataFrame({
    'Model': [ 'Linear Regression Model','Random forest Model',
              'Descision Tree Model','Ridge model'],
    'Score': [ linear_accurcy,forest_accurcy,
               tree_accurcy ,ridge_accurcy]})

models.sort_values(by='Score', ascending=False)