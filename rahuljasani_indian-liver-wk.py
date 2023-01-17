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
df = pd.read_csv("/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv")
df.head()
df.info()
import seaborn as sns

from matplotlib import pyplot as plt
sns.countplot(df.Dataset)
sns.countplot(data=df, x='Gender');
g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)

g.map(plt.hist, "Age")

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Disease by Gender and Age');
df.isnull().sum()
df.Albumin_and_Globulin_Ratio.value_counts()
df.Albumin_and_Globulin_Ratio.fillna(1.00,inplace=True)
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df['Gender']=label.fit_transform(df['Gender'])
from sklearn.model_selection import train_test_split

train, test= train_test_split(df,test_size=0.1,random_state=1)

def data_spliting(df):

    x=df.drop(['Dataset'],axis =1)

    y=df.Dataset

    return x,y

x_train,y_train=data_spliting(train)

x_test,y_test=data_spliting(test)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



log_model=LogisticRegression()

log_model.fit(x_train, y_train)

prediction=log_model.predict(x_test)

score= accuracy_score(y_test, prediction)

print(score*100)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

smote = XGBClassifier()

smote.fit(x_train, y_train)



# Predict on test

smote_pred = smote.predict(x_test)

accuracy = accuracy_score(y_test, smote_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
from sklearn.ensemble import RandomForestRegressor

regress = RandomForestRegressor()

regress.fit(x_train , y_train)

reg_train = regress.score(x_train , y_train)

reg_test = regress.score(x_test , y_test)

print(reg_train*100)

print(reg_train*100)