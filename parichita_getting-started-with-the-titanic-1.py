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
train_data=pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data= pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
#Testing for Age

import matplotlib.pyplot as plt

plt.scatter(train_data["Age"],train_data["Survived"])

plt.show()

train_data.boxplot(column=["Age","Survived"],by="Survived")

#Age is not that significant 
#Testing for Fare

import matplotlib.pyplot as plt

plt.scatter(train_data["Fare"],train_data["Survived"])

plt.show()

train_data.boxplot(column=["Fare","Survived"],by="Survived")

#Fare Seems to be significant
#Testing for Embarked

table=pd.crosstab(train_data["Embarked"],train_data["Survived"])

print(table)

from scipy.stats import chi2_contingency

stat, p, dof, expected = chi2_contingency(table)

print(p)

p<0.05

#Dependent
#plt.scatter(train_data["Age"],train_data["Embarked"])

#plt.show()

#train_data.boxplot(column=["Age","Embarked"],by="Embarked")

# As such no relation between them
##Model: Drop all the rows with Nan values in Age and Embarked Column and include the Fare Column



#Data Prep for train data

df=train_data.copy()

df["Embarked"].replace(to_replace= np.nan, value="missing", inplace=True)

df["Age"].replace(to_replace= np.nan, value=-999, inplace=True)



#Data Prep for test data

#features = ["Age","Embarked","Fare","Pclass", "Sex", "SibSp", "Parch"]

df1=test_data.copy()

df1["Age"].fillna(df["Age"].median(),inplace=True)

df1["Fare"].fillna(df["Fare"].median(),inplace=True)





#10::Kaggle: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier



y=df["Survived"]



features = ["Age","Embarked","Fare","Pclass", "Sex", "SibSp", "Parch"]

X=pd.get_dummies(df[features])

X=X.drop("Embarked_missing",axis=1)



#Split the dataset into Training and Testing data

#from sklearn.model_selection import train_test_split

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

X_test=pd.get_dummies(df1[features])



model=RandomForestClassifier(n_estimators=100, max_depth=5,random_state=1)

model.fit(X,y)

predictions=model.predict(X_test)



#Result

#bools1 = predictions==y_test

#print(bools1.sum()) #149

#print(bools1.sum()/len(predictions))

#0.83



output=pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv',index=False)

print("Successfully Saved!")