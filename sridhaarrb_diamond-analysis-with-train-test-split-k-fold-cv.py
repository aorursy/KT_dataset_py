import pandas as pd
import numpy as np
df = pd.read_csv("../input/diamonds.csv")
df.drop(['Unnamed: 0'],axis=1,inplace=True)
print (df.head())
df.describe()
df.shape
df = df[(df['x']!=0) & (df['y']!=0) & (df['z']!=0)]
df.shape
df['clarity'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
sns.pairplot(df, vars=['carat','depth', 'table','price'])
plt.show()
sns.distplot(df.carat)
plt.show()
sns.countplot(x=df.cut)
plt.show()
sns.countplot(x=df.color)
plt.show()
sns.countplot(x=df.clarity)
plt.show()
#sns.boxplot(x=df.drop(['carat'],axis=1),orient='v')
sns.boxplot(x=df['carat'],orient='v')
plt.show()
diamond_cut = {'Fair':0,
               'Good':1,
               'Very Good':2, 
               'Premium':3,
               'Ideal':4}

diamond_color = {'J':0,
                 'I':1, 
                 'H':2,
                 'G':3,
                 'F':4,
                 'E':5,
                 'D':6}

diamond_clarity = {'I1':0,
                   'SI2':1,
                   'SI1':2,
                   'VS2':3,
                   'VS1':4,
                   'VVS2':5,
                   'VVS1':6,
                   'IF':7}
df['cut'] = df['cut'].map(diamond_cut)
df['color'] = df['color'].map(diamond_color)
df['clarity'] = df['clarity'].map(diamond_clarity)
df.head(20)
df.describe()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
X = df.drop(['price'],axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

models = []
models.append(('LR', LinearRegression()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('SVM', SVR()))
models.append(('RF',RandomForestRegressor()))
# evaluate each model in turn
MSE = []
r2 = []
names = []
score = []
for name, model in models:
    Algo = model.fit(X_train,y_train)
    y_pred = Algo.predict(X_test)
    MSE.append(mean_squared_error(y_test,y_pred))
    r2.append(r2_score(y_test,y_pred))
    names.append(name)


df_TT = pd.DataFrame({'Name':names,'r2_score':r2,'MSE':MSE})
ax = sns.barplot(x="Name", y="r2_score", data=df_TT)
plt.show()
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

models = []
models.append(('LR', LinearRegression()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('SVM', SVR()))
models.append(('RF',RandomForestRegressor()))
# evaluate each model in turn
MSE = []
names = []
r2_score = []
scoring = 'r2'
for name, model in models:
    Algo = model.fit(X_train,y_train)
    r2 =cross_val_score(Algo, X_train, y_train, cv=3, scoring=scoring)
    y_pred = Algo.predict(X_test)
    r2_score.append(np.mean(r2))
    MSE.append(mean_squared_error(y_test,y_pred))
    names.append(name)

df_cv = pd.DataFrame({'Name':names,'r2_score':r2_score,'MSE':MSE})
print (df_cv)
ax = sns.barplot(x="Name", y="r2_score", data=df_cv)
plt.show()
df_TT
df_cv
