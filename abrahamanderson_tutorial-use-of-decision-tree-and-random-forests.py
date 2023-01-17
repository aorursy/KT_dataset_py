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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import cufflinks as cf

cf.go_offline()
df=pd.read_csv("../input/loan_data.csv")

df.head(3)
df.info() 

#There 9578 rows and 14 columns in our dataset
df.describe(include="all") 

#Here we get overall statistical information about the data
plt.figure(figsize=(15,10))

df[df["credit.policy"] == 1]["fico"].hist(color="blue",bins=50,label="Credit Policy = 1",alpha=0.4)

df[df["credit.policy"] == 0]["fico"].hist(color="red",bins=50,label="Credit Policy = 0",alpha=0.4)

plt.legend()

#here we make two different histogram one for those who have credit policy 1 score and the other for those who have 0 score

# we compare their relative fico credit scores
df[df["credit.policy"] == 1]["fico"].iplot(kind="hist",bins=24,colors="blue")

df[df["credit.policy"] == 0]["fico"].iplot(kind="hist",bins=24,colors="orange")

#Here we do the same histogram with iplot library because it is interactive

# this means that when we click somewhere we can get exact score 
plt.figure(figsize=(15,10))

df[df["not.fully.paid"] ==1]["fico"].hist(label="not fully paid = 1",alpha=0.6,color="blue",bins=30)

df[df["not.fully.paid"] ==0]["fico"].hist(label="not fully paid = 0",alpha=0.6,color="red",bins=30)

plt.xlabel("FICO")

plt.title("The FICO credit score of the borrower")

plt.legend()
plt.figure(figsize=(15,10))

sns.countplot(x="purpose",hue="not.fully.paid", data=df, palette="Set1")

plt.title("The Counts of Loans by Purpose")

plt.legend()
sns.jointplot(x="fico",y="int.rate",data=df,color="green",space=0.2)
sns.lmplot(x="fico",y="int.rate",data=df,palette="Set1",hue="credit.policy",col="not.fully.paid")
df.info() # we look again the overall information about the data
cat_feature=["purpose"]

final_data= pd.get_dummies(df,columns=cat_feature,drop_first=True)

final_data.info()
final_data.head()

#Now all of the features in the data has been tranformed into 0 and 1 by adding a new column for each of them
X=final_data.drop("not.fully.paid",axis=1) # All of the columns except from the target column has assigned as the X

y=final_data["not.fully.paid"] # "not.fully.paid" column has been assigned as the target column
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3) # Here we split our data as training and test dataset
from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier()

dtree.fit(X_train,y_train) 
predictions=dtree.predict(X_test)

#df_pred=pd.DataFrame(predictions)

plt.figure(figsize=(15,10))

sns.countplot(predictions,palette="Set1")

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=300)
rfc.fit(X_train,y_train) # We make th new algorithm fit with the training set

rfc_predictions=rfc.predict(X_test) #We make the algorith to predict y test values
print(classification_report(y_test,rfc_predictions))

print(5*"\n")

print(confusion_matrix(y_test,rfc_predictions))