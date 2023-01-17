import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import statsmodels.api as st

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')
df.shape
df.info()
df.head()
df.drop('Unnamed: 0',inplace=True,axis=1)
df.isnull().sum()
df.isna().sum()
(df.x == 0).sum()
(df.y == 0).sum()
(df.z == 0).sum()
df[df.x == 0]
df[['x','y','z']]=df[['x','y','z']].replace(0,np.NaN)

df.describe().T
df.plot(kind='box',figsize=(15,10),subplots=True,layout=(3,3))

plt.show()


def outliers(var):

    a = []

    q1 = df[var].quantile(.25)

    q2 = df[var].quantile(.5)

    q3 = df[var].quantile(.75)

    iqr = q3-q1

    ulim = float(q3+(1.5*iqr))

    llim = float(q1-(1.5*iqr))



    for i in df[var]:

        if i > ulim:

            i=np.NaN

        elif i < llim:

            i = np.NaN

        else:

            i=i

        a.append(i)

    return a



for col in df.select_dtypes(exclude='object').columns:

    df[col] = outliers(col)



df.isna().sum()
df.plot(kind='box',figsize=(15,10),subplots=True,layout=(3,3))

plt.show()
sns.boxplot(df['z'])
df.describe()
df.isna().sum()
for i in df.select_dtypes(exclude='object').columns:

    df[i]=df[i].fillna(df[i].mean())

    
df.describe()
df.isna().sum()
df.head()
df_cat = df.select_dtypes(include='object')

df_cat['cut'].value_counts()
df_cat['color'].value_counts()
df_cat['clarity'].value_counts()
le = LabelEncoder()

df_cat = df_cat.apply(le.fit_transform)

df_cat
df = df.drop(df_cat,axis=1)
df = pd.concat([df,df_cat],axis=1)
plt.scatter(df['price'],df['carat'])
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')
X = df.drop('price',axis=1)

y = df['price']
xc = st.add_constant(X)

lm = st.OLS(y,xc).fit()


lm.summary()
vif = [variance_inflation_factor(X.values,col) for col in range(0,X.shape[1])]
pd.DataFrame({'vif':vif,'cols':X.columns})
df.corr()
X = df.drop(['price'],axis=1)

y = df['price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.33,random_state=42)
lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

r2_score(y_test,y_pred)
y_pred

y_test
rr  = RandomForestRegressor()
rr.fit(X_train,y_train)

y_pred = rr.predict(X_test)

r2_score(y_test,y_pred)
rr.get_params
n_estimators = [int(x) for x in np.linspace(10,200,10)]

max_depth = [int(x) for x in np.linspace(10,100,10)]

min_samples_split = [2,3,4,5,10]

min_samples_leaf = [1,2,4,10,15,20]

random_grid = {'n_estimators':n_estimators,'max_depth':max_depth,

               'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}



random_grid

from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator=rf,

                               param_distributions=random_grid,

                               cv = 3)



rf_random.fit(X_train,y_train)
y_pred = rf_random.predict(X_test)

r2_score(y_test,y_pred)
rf_random.best_params_
rf = RandomForestRegressor(n_estimators=178,

                         min_samples_split=5,

                         min_samples_leaf=1,

                         max_depth=50)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

r2_score(y_test,y_pred)