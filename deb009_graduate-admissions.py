# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn import utils

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ap = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

ap.head()
print(ap.dtypes)
ap.info()
#rename the labels



ap.rename(columns={"Serial No.":"Serial_No.","GRE Score":"GRE_Score","TOEFL Score":"TOEFL_Score","University Rating":"University_Rating", "Chance of Admit":"COA"},inplace = True)
ap.head(2)
display(ap.University_Rating.nunique())

ap.University_Rating.unique()

display(ap.Research.nunique())

ap.Research.unique()
sns.heatmap(ap.corr())
Graph = sns.regplot(x="GRE_Score", y="TOEFL_Score", data=ap)

plt.title("GRE Score vs TOEFL Score")

plt.show()



Graph = sns.regplot(x="GRE_Score", y="CGPA", data=ap)

plt.title("GRE Score vs CGPA")

plt.show()
ap.groupby('University_Rating')['GRE_Score'].nunique().plot(kind='bar')

plt.show()
from sklearn.model_selection import train_test_split



X=ap.iloc[:,:-1]

y=ap.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)



#Applying L.R



from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

print("score : ",r2_score(y_test, y_pred))