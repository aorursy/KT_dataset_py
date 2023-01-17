# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import seaborn as sns
import plotly.plotly as py
import plotly.tools as tls

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
#This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables: 
#X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit. 
#X2: Gender (1 = male; 2 = female). 
#X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). 

#X5: Age (year). 
#X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above. 
#X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. 
#X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005. 


payment_data=pd.read_csv("../input/payments.csv")
print (payment_data.head)

payment_data.shape
plt.figure(figsize=(12,4))
sns.countplot(x="AGE",data=payment_data,)
plt.title("AGE graph")
plt.show()
plt.figure(figsize=(12,4))
sns.countplot(x="LIMIT_BAL",hue="MARRIAGE",data=payment_data)
plt.figure(figsize=(12,4))
sns.countplot(x="AGE",hue="MARRIAGE",data=payment_data)
plt.figure(figsize=(12,4))
sns.countplot(x="AGE",hue="SEX",data=payment_data)
payment_data["MARRIAGE"].plot.hist()
payment_data.isnull()
print("Data set length:",len(payment_data))
print("Data set shape:",(payment_data.shape))
X=payment_data.values[:,1:23]
Y=payment_data.values[:,24]
X_train,X_test,y_test,y_train=train_test_split(X,Y,test_size=0.5,random_state=100)
clf_entropy=DecisionTreeClassifier(criterion="entropy",max_depth=3,min_samples_leaf=5)
clf_entropy.fit(X_train,y_train)
y_pred=clf_entropy.predict(X_test)
print (y_pred)
print("Accuracy is",accuracy_score(y_test,y_pred)*100)