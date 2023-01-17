# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import model_selection

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import neighbors

from sklearn.svm import SVR

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn import preprocessing



from warnings import filterwarnings

filterwarnings('ignore')
df = pd.read_csv("../input/hitters/Hitters.csv")
df.head()
df.shape
df.corr()
df.drop(["League","Division","NewLeague"],axis=1,inplace=True)
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df.head()
df.head()
models = []



models.append(('KNN', KNeighborsRegressor()))

models.append(('SVR', SVR()))

models.append(('CART', DecisionTreeRegressor()))

models.append(('RandomForests', RandomForestRegressor()))

models.append(('GradientBoosting', GradientBoostingRegressor()))

models.append(('XGBoost', XGBRegressor()))

models.append(('Light GBM', LGBMRegressor()))
X = df.drop("Salary",axis=1)

y = df["Salary"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)



for name,model in models:

    mod = model.fit(X_train,y_train)

    y_pred = mod.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(name,rmse)

    print("-------------")
df = pd.read_csv("../input/hitters/Hitters.csv")
df.head()
df["AvgCAtBat"] = df["AtBat"]/df["CAtBat"] 

df["AvgCHits"] = df["Hits"]/df["CHits"] 

df["AvgCHmRun"] = df["HmRun"]/df["CHmRun"] 

df["AvgCruns"] = df["Runs"]/df["CRuns"] 

df["AvgCRBI"] = df["RBI"]/df["CRBI"] 

df["AvgCWalks"] = df["Walks"]/df["CWalks"]
df['Year_lab'] = pd.qcut(df['Years'], 6 ,labels = [1,2,3,4,5,6]) #creating segments for years
df.head()
df.isnull().sum()
df = pd.get_dummies(df,drop_first=True) #one hot encoding
from sklearn.impute import KNNImputer 

imputer = KNNImputer(n_neighbors = 4) 

df_filled = imputer.fit_transform(df)
df = pd.DataFrame(df_filled,columns = df.columns)
df.isnull().sum()
import seaborn as sns

sns.boxplot(x = df["Salary"]);
df[["Salary"]].describe().T
for feature in df:



    Q1 = df[feature].quantile(0.25)

    Q3 = df[feature].quantile(0.75)

    IQR = Q3-Q1

    upper = Q3 + 1.5*IQR

    lower = Q1 - 1.5*IQR



    if df[(df[feature] > upper) | (df[feature] < lower)].any(axis=None):

        print(feature,"yes")

        print(df[(df[feature] > upper) | (df[feature] < lower)].shape[0])

    else:

        print(feature, "no")
Q1 = df.Salary.quantile(0.25)

Q3 = df.Salary.quantile(0.75)

IQR = Q3-Q1

lower = Q1 - 1.5*IQR

upper = Q3 + 1.5*IQR

df.loc[df["Salary"] > upper,"Salary"] = upper
sns.boxplot(x = df["Salary"]);
from sklearn.neighbors import LocalOutlierFactor

lof =LocalOutlierFactor(n_neighbors= 10)

lof.fit_predict(df)
df_scores = lof.negative_outlier_factor_

np.sort(df_scores)[0:30]
threshold = np.sort(df_scores)[7]

threshold
outlier = df_scores > threshold

df = df[outlier]
df.shape
df.head()
df.info()
df.head()
df.isnull().sum()
X = df.drop("Salary",axis=1)

y = df["Salary"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)



for name,model in models:

    mod = model.fit(X_train,y_train)

    y_pred = mod.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(name,rmse)

    print("-------------")