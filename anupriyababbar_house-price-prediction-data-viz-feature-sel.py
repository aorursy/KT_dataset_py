# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd



#def load_data(path, filename, codec='utf-8'):

  #csv_path = os.path.join(path, filename)

  #print(csv_path)

  #return pd.read_csv(csv_path, encoding=codec)



path=('../input/train.csv')
import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.preprocessing import Imputer,LabelEncoder

from scipy.stats import norm, skew

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
#reading of train data

df= pd.read_csv(path)

#checking the contents of table

df.head()

df.columns
fig = plt.figure(2)

ax1 = fig.add_subplot(2, 2, 1)

ax2=fig.add_subplot(2,2,2)

ax3=fig.add_subplot(2,2,3)

ax4=fig.add_subplot(2,2,4)

df.plot.scatter(x='LotFrontage',y='SalePrice',ax=ax1)

df.plot.scatter(x='LotArea',y='SalePrice',ax=ax2)

df.plot.scatter(x='MSSubClass',y='SalePrice',ax=ax3)

df.plot.scatter(x='OverallQual',y='SalePrice',ax=ax4)

plt.show()
#Cheking for nulls in target

df['SalePrice'].isnull().sum()
#assigning target to y

y=df['SalePrice']
sns.set()

cols=list(df.columns)

sns.pairplot(df[cols],size=2.5)

plt.show()
#dropping target from rest of features

X=df.drop('SalePrice',axis=1)
#checking  correlation between the independent features

sns.heatmap(df.corr())
#selecting continous features

X_con=X.select_dtypes(exclude='object')

#X_test_con=df_test.select_dtypes(exclude='object')
X_con_col=list(X_con.columns)
#outlier detection and replacing them with mean

def outlier_detect(df):

    for i in df.describe().columns:

        Q1=df.describe().at['25%',i]

        Q3=df.describe().at['75%',i]

        IQR=Q3 - Q1

        LTV=Q1 - 1.5 * IQR

        UTV=Q3 + 1.5 * IQR

        x=np.array(df[i])

        p=[]

        for j in x:

            if j < LTV or j>UTV:

                p.append(df[i].median())

            else:

                p.append(j)

        df[i]=p

    return df
#calling outlier dectectfunction

X_con=outlier_detect(X_con)
#selecting categorical data

X_cat=X.select_dtypes(include='object')
X_cat_col=list(X_cat.columns)
#checking for nulls in categorical features

X_cat.isnull().sum()
#checking for nulls in continous data

X_con.isnull().sum()
#dropping the features which have maximum values missing

X_cat.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

X_cat.drop('Alley',1,inplace=True)
#again checking for null values----data with no missing values

X_cat.isnull().sum()
#filling missing value of categorical data  with mode

for col in X_cat.columns:

       

    X_cat[col]=X_cat[col].fillna(X_cat[col].mode()[0])
for cols in X_cat.columns:

    sns.set(style="whitegrid")

    ax = sns.barplot(x=cols, y="SalePrice", data=df)

    plt.show()
#again checking for nulls

X_cat.isnull().sum()
#checking nulls in continous features

X_con.isnull().sum()
#replacing NaN with 0 for continous features

for col in X_con.columns:

       

    X_con[col]=X_con[col].replace(to_replace=np.nan,value=0)


for cols in X_con.columns:

    sns.set()

    sns.distplot(X_con[cols])

    plt.show()
#checking for skewness of data and replacing it by sqrt if skewness>1

for feature in X_con.columns:

    if (X_con[feature].skew())>1.0:

        X_con[feature]=np.sqrt(X_con[feature])
X_con.head()
from sklearn import preprocessing
#converting the categorical values into continous values with the hep of label encoding

le = preprocessing.LabelEncoder()
for feat in X_cat:

    le.fit(X_cat[feat])

    X_cat[feat]=le.transform(X_cat[feat])
#merging continous and categorical features

XX=pd.concat([X_con,X_cat],1)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
#splitting of data

X_train,X_test,y_train,y_test=train_test_split(XX,y,test_size=.2,random_state=0)
#creating object of linear model

linear=LinearRegression()
#fitting model on X_train and y_train

linear.fit(X_train,y_train)
#predicting the value of X_test

y_pred=linear.predict(X_test)
from sklearn.metrics import r2_score
lin_score=r2_score(y_test,y_pred)
lin_score
###applying regularization--Lasso

from sklearn import linear_model

lasso = linear_model.Lasso(alpha=0.1)
# Ride---

ridge=linear_model.Ridge(alpha=0.1)
lasso.fit(X_train,y_train)
y_lasso=lasso.predict(X_test)
r2_score(y_test,y_lasso)
ridge.fit(X_train,y_train)

y_ridge=lasso.predict(X_test)

r2_score(y_test,y_ridge)

#apply feature selection

from sklearn.model_selection import cross_validate
from sklearn.feature_selection import chi2



from sklearn.feature_selection import SelectKBest
feat_sel = SelectKBest(score_func=chi2, k=60)

X_train=feat_sel.fit_transform(X_train,y_train)

X_test=feat_sel.transform(X_test)

model=LinearRegression()

model.fit(X_train,y_train)

chi2_score=model.score(X_test,y_test)

print(chi2_score)
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor(random_state=0)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
from sklearn.ensemble import RandomForestRegressor



# Code starts here

rf_reg=RandomForestRegressor(random_state=9)

rf_reg.fit(X_train,y_train)



score_rf=rf_reg.score(X_test,y_test)

print(score_rf)
from sklearn.model_selection import GridSearchCV
parameter_grid={'n_estimators': [20,40,60,80],'max_depth': [8,10],'min_samples_split': [8,10]}
rf_reg=RandomForestRegressor(random_state=9)

rf_reg.fit(X_train,y_train)

grid_search=GridSearchCV(estimator=rf_reg,param_grid=parameter_grid)
grid_search.fit(X_train,y_train)

score_gs=grid_search.score(X_test,y_test)

print(score_gs)
#score_gs