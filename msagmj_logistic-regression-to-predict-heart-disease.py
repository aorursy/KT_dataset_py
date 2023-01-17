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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

sns.set()
data=pd.read_csv("/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv")

data.head(10)
data.describe(include="all")
data.isna().sum()
from statistics import mode 
data["education"].unique()
data["education"]=data["education"].fillna(mode(data["education"]))
data["education"].isna().sum()
data["cigsPerDay"].unique()
data["cigsPerDay"]=data["cigsPerDay"].fillna(data["cigsPerDay"].mean())
data["cigsPerDay"].isna().sum()
data["BPMeds"].unique()
data["BPMeds"]=data["BPMeds"].fillna(mode(data["BPMeds"]))
data["BPMeds"].isna().sum()
data["totChol"].unique()
data["totChol"]=data["totChol"].fillna(data["totChol"].mean())
data["totChol"].isna().sum()
data["glucose"].unique()
data["glucose"]=data["glucose"].fillna(data["glucose"].mean())
data=data.dropna()
data.isna().sum()
data.head(5)
data.describe(include="all")
X_r=data.drop(["TenYearCHD"],axis=1)

y_r=data["TenYearCHD"]
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
bf=SelectKBest(score_func=chi2,k=10)

fit=bf.fit(X_r,y_r)
dfscores=pd.DataFrame(fit.scores_)

dfcolumns=pd.DataFrame(X_r.columns)
featurescores=pd.concat([dfcolumns,dfscores],axis=1)

featurescores.columns=["spec","score"]

featurescores
print(featurescores.nlargest(10,'score'))
X=data[["sysBP","glucose","age","totChol","cigsPerDay","diaBP","prevalentHyp","diabetes","BPMeds","male"]]

y=data["TenYearCHD"]
data["TenYearCHD"].value_counts()
from imblearn.under_sampling import NearMiss
nm= NearMiss()

X_res,y_res=nm.fit_sample(X,y)
X_res.shape,y_res.shape
data.dtypes
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.33, random_state=42)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_test)

print(score)
confusion_matrix = metrics.confusion_matrix(y_test,predictions)

print(confusion_matrix)