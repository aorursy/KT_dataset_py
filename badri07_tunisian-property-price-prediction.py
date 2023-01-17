import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/property-prices-in-tunisia/Property Prices in Tunisia.csv')
#See the first rows of the data

df.head()
#Data shape

df.shape
#Statistical description

df.describe()
#Null Values

df.isnull().sum()
#See -1 values

(df==-1).sum()
#Replace -1 with nan

df = df.replace(-1,np.float('nan'))
#Drop duplicate rows

df.drop_duplicates(keep = 'first', inplace = True) 
#Start with the 'type' column

print(df.type.value_counts())
sns.countplot(data = df, x = 'type')
#category column

print(df.category.nunique())

print(df.category.unique())
#A quick barplot

plt.figure(figsize=(20, 4))

sns.countplot(data = df, x = 'category', order=df.category.value_counts().index)
print(df.city.nunique())

print(df.city.unique())
plt.figure(figsize=(10, 6))

sns.countplot(data = df, y = 'city', order = df.city.value_counts().index)
print(df.room_count.nunique())

print(df.room_count.unique())
plt.figure(figsize=(16, 4))

sns.countplot(data = df, x = 'room_count', order = df.room_count.value_counts().index)
print(df.bathroom_count.nunique())

print(df.bathroom_count.unique())
plt.figure(figsize=(10, 4))

sns.countplot(data = df, x = 'bathroom_count', order = df.bathroom_count.value_counts().index)
#Use a density plot

plt.figure(figsize=(10, 6))

sns.distplot(df['size'], hist=True, kde=True, 

              color = 'green',bins = 50,

             kde_kws={'linewidth': 1,'shade': True },

             hist_kws={'edgecolor':'black'})
#Scatter plot price against the size, with the type as a hue

plt.figure(figsize=(10, 7))

sns.scatterplot(data = df , x="size", y="price", hue="type")
#Use a density plot

plt.figure(figsize=(10, 6))

sns.distplot(df['price'], hist=False, kde=True, 

              color = 'blue',bins = 50,

             kde_kws={'linewidth': 1,'shade': True },

             hist_kws={'edgecolor':'black'})
#Scatter plot price against the size, with the type as a hue

plt.figure(figsize=(10, 7))

sns.scatterplot(data = df , x="size", y="log_price", hue="type")
#Use a density plot

plt.figure(figsize=(10, 6))

sns.distplot(df['log_price'], hist=False, kde=True, 

              color = 'blue',bins = 20,

             kde_kws={'linewidth': 1,'shade': True },

             hist_kws={'edgecolor':'black'})
#Require that many non-NA values.

df.dropna(thresh = 9, inplace=True)
#A quick barplot again on the category

plt.figure(figsize=(20, 4))

sns.countplot(data = df, x = 'category', order=df.category.value_counts().index)
dfVendre = df[df.type == 'À Vendre']

dfLouer = df[df.type == 'À Louer']
print(dfVendre.shape)

print(dfLouer.shape)
#Scatter plot price against the size, with the type as a hue

plt.figure(figsize=(10, 7))

sns.scatterplot(data = dfVendre , x="size", y="log_price")
z1 = np.abs(stats.zscore(dfVendre.log_price)) #Calculate Z score for dfVebdre

z2 = np.abs(stats.zscore(dfLouer.log_price))  #Calculate Z score for dfAchat
dfVendre_O  = dfVendre[(z1 < 2.5)]

dfLouer_O  = dfLouer[(z2 < 2.5)]

print('Number of removed rows : ',dfVendre.shape[0]-dfVendre_O.shape[0])

print('Number of removed rows : ',dfLouer.shape[0]-dfLouer_O.shape[0])
#Scatter plot price against the size, with the type as a hue

plt.figure(figsize=(10, 7))

sns.scatterplot(data = dfVendre_O , x="size", y="price", hue="type")
#Scatter plot price against the size, with the type as a hue

plt.figure(figsize=(10, 7))

sns.scatterplot(data = dfLouer_O , x="size", y="price")
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression,SGDRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import time
#Separate categorical values and Numerical Values

Cat_Col = ['category','city','type']

Num_Col = ['room_count','bathroom_count' , 'size']
Pipeline = ColumnTransformer([

    ("num", StandardScaler(), Num_Col),

    ('cat', OrdinalEncoder(),Cat_Col)

])
#Separate Target and Features

Xl = dfLouer_O.drop(['price','log_price','region'],axis = 1)

yl = dfLouer_O.price
#Train_test split

xl_train,xl_test,yl_train,yl_test = train_test_split(Xl,yl,test_size = 0.2,random_state = 42)
#Use the pipeline to transform my features

xl_train = Pipeline.fit_transform(xl_train)

xl_test = Pipeline.transform(xl_test)
#Define my models

names = ["Linear Regression", "SGD Regressor", "Random Forest Regressor"]

Regressors = [LinearRegression(),SGDRegressor(),RandomForestRegressor() ]
for name, Reg in zip(names, Regressors):

  Reg.fit(xl_train, yl_train)

  preds = Reg.predict(xl_test)

  MAE = mean_absolute_error(yl_test,preds)

  R2 = r2_score(yl_test,preds)

  print (name, ' : mean absolute error  :  ', "%.2f" %(MAE), 'R2_Score : ', "%.2f" %(R2))
mymodel = RandomForestRegressor()

mymodel.fit(xl_train,yl_train)
Xl.iloc[[85]]
yl.iloc[85]
mymodel.predict(Pipeline.transform(Xl.iloc[[85]]))
#Concat the two dataframes

df_final = pd.concat([dfLouer_O,dfVendre_O])
df_final.head()
#Features and Target

X = df_final.drop(['price','log_price','region'],axis = 1)

y = df_final.price

#Our Model

model = RandomForestRegressor()
#Train test split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 100)
#Use the pipeline to transform my test features

x_train = Pipeline.fit_transform(x_train)

x_test = Pipeline.transform(x_test)
model.fit(x_train,y_train)
#Generate Predictions

preds = model.predict(x_test)
#Evaluation

result = (mean_absolute_error(y_test,preds))

print(result)
preds_l = model.predict(xl_test)
result = (mean_absolute_error(yl_test,preds_l))

print(result)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import time
#Initialize my gridsearch parameters

Grid_par = [

    {'n_estimators' : [5,10,20,30,50,100] , 'max_features' : [4,5,6]},

    {'bootstrap': [False], 'n_estimators' : [5,10,20,30,50,100] , 'max_features' : [4,5,6]},

    {'n_estimators' : [5,10,20,30,50,100] , 'max_features' : [2,3,4],'max_depth' : [10,20,30]},

    {'bootstrap': [False], 'n_estimators' : [5,10,20,30,50,100] , 'max_features' : [4,5,6],'max_depth' : [10,20,30]}]

    

model = RandomForestRegressor(n_jobs=-1)
GridSearch = GridSearchCV(estimator= model , param_grid=Grid_par, cv = 5,

                         scoring='neg_mean_absolute_error', return_train_score=True)
start = time.time()

GridSearch.fit(xl_train,yl_train)

end = time.time()
print('Time used : ', end - start ,'Second')
results = GridSearch.cv_results_
print('Number of estimators : ' , len(results["params"]))

for mean_score, params in zip(results["mean_test_score"], results["params"]):

    print ((-mean_score),params)
-GridSearch.best_score_
best = GridSearch.best_estimator_
best.fit(xl_train,yl_train)
preds_grid = best.predict(xl_test)
mean_absolute_error(yl_test,preds_grid)
RandSearch = RandomizedSearchCV(estimator=model, param_distributions=Grid_par, cv = 5 ,

                               scoring='neg_mean_absolute_error', return_train_score=True,n_iter=20)
start = time.time()

RandSearch.fit(xl_train,yl_train)

end = time.time()
print('Time used : ', end - start ,'Second')
Randresult = RandSearch.cv_results_
print('Number of estimators : ' , len(Randresult["params"]))

for mean_score, params in zip(Randresult["mean_test_score"], Randresult["params"]):

    print ((-mean_score),params)
best_rand = RandSearch.best_estimator_
best_rand.fit(xl_train,yl_train)
rand_preds = best_rand.predict(xl_test)
mean_absolute_error(yl_test,rand_preds)