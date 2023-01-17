# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
%matplotlib inline



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
tests = ("../input/students-performance-in-exams/StudentsPerformance.csv")

test1 = pd.read_csv(tests, sep =",")

testscores = pd.DataFrame(test1) 
testscores.info
type (testscores)
testscores
plt.figure(figsize=(15,2))
plt.xticks(rotation=90)
plt.title("Male To Female Math Testing Scores")


sns.scatterplot(testscores['math score'], testscores['gender'])

plt.show()
plt.figure(figsize=(15,2))
plt.xticks(rotation=90)
plt.title("Reading Male to Female Testing")

sns.scatterplot(testscores['reading score'], testscores['gender'])

plt.show()
plt.figure(figsize=(15,2))
plt.xticks(rotation=90)
plt.title("Writing Male to female testing")

sns.scatterplot(testscores['writing score'], testscores['gender'])

plt.show()
plt.figure(figsize=(10,10))
plt.xticks(rotation=90)
plt.rcParams.update({'font.size':14})
plt.title("Reading Scores for race/ethnicity")
sns.barplot(testscores['race/ethnicity'],testscores['reading score'],data=testscores)
sns.catplot("gender", "math score","race/ethnicity", kind="bar", data=testscores);

sns.catplot("gender", "reading score","race/ethnicity", kind="bar", data=testscores);

sns.catplot("gender", "writing score","race/ethnicity", kind="bar", data=testscores);

plt.figure(figsize=(10,10))
plt.title("Test Prepers Reading Score Avg")

plt.rcParams.update({'font.size':14})
sns.barplot(testscores['test preparation course'],testscores['reading score'],data=testscores)
plt.figure(figsize=(10,10))
plt.title("Test Prepers Math Score Avg")

plt.rcParams.update({'font.size':14})
sns.barplot(testscores['test preparation course'],testscores['math score'],data=testscores)
plt.figure(figsize=(10,10))
plt.title("Test Prepers Writing Score Avg")

plt.rcParams.update({'font.size':14})
sns.barplot(testscores['test preparation course'],testscores['writing score'],data=testscores)
testscores_lunch_free = testscores.loc[testscores['lunch'] == "free/reduced"]
testscores_lunch_free
plt.figure(figsize=(10,10))
plt.title("Test Prepers with low income lunchs Writing Score Avg")

plt.rcParams.update({'font.size':14})
sns.barplot(testscores_lunch_free['test preparation course'],testscores_lunch_free['writing score'],data=testscores_lunch_free)
plt.figure(figsize=(10,10))
plt.title("Test Prepers with low income lunchs Math Score Avg")

plt.rcParams.update({'font.size':14})
sns.barplot(testscores_lunch_free['test preparation course'],testscores_lunch_free['math score'],data=testscores_lunch_free)
plt.figure(figsize=(10,10))
plt.title("Test Prepers with low income lunchs Reading Score Avg")

plt.rcParams.update({'font.size':14})
sns.barplot(testscores_lunch_free['test preparation course'],testscores_lunch_free['reading score'],data=testscores_lunch_free)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit_transform(testscores[['lunch','parental level of education','race/ethnicity','gender','test preparation course',]])
ohe.categories_
#testscores.columns
#y = testscores.math score



#testscores_features = ['test preparation course', 'gender', 'parental level of education', 'race/ethnicity', 'lunch']

#X = testscores[testscores_features]

#testscores.info()
sns.countplot(x="gender", data=testscores)
testscores.loc[:,'gender'].value_counts()
#train, test = train_test_split(testscores, test_size = 100)
#train.shape

#print(testscores)
#testscores
#readingscorelist = testscores['reading score'].tolist()
#
#labelEncoderRating = LabelEncoder()
#labelEncoderRating.fit(readingscorelist)
#labelsRating = labelEncoderRating.transform(readingscorelist)
#testscores['reading score']=pd.Series(labelsRating)
#var=testscores.columns.values.tolist()
#y=testscores['gender']
#x=[i for i in var if i not in ['gender']]
#x=testscores[x]
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)

#from sklearn.preprocessing import MinMaxScaler
#xs=MinMaxScaler()
#x_train[x_train.columns] = xs.fit_transform(x_train[x_train.columns])
#x_test[x_test.columns] = xs.transform(x_test[x_test.columns])

#cy_train=[1 if chance > 0.80 else 0 for chance in y_train]
#cy_train=np.array(cy_train)

#cy_test=[1 if chance > 0.80 else 0 for chance in y_test]
#cy_test=np.array(cy_test)
#from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression()
#lr.fit(x_train, cy_train)
#cy = lr.predict(x_test)
#from sklearn.metrics import confusion_matrix
#import seaborn as sns
# Printing accuracy score & confusion matrix
#from sklearn.metrics import accuracy_score
#print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(cy_test, lr.predict(x_test))))
#print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#from sklearn.metrics import classification_report
#print(classification_report(cy_test, lr.predict(x_test)))
#lr_confm = confusion_matrix(cy, cy_test, [1,0])
#sns.heatmap(lr_confm, annot=True, fmt='.2f',xticklabels = ["Admitted", "Rejected"] , yticklabels = ["Admitted", "Rejected"] )
#plt.ylabel('True class')
#plt.xlabel('Predicted class')
#plt.title('Logistic Regression')
#plt.show()

#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier()
#rf.fit(x_train, cy_train)
#cy = rf.predict(x_test)
# Printing accuracy score & confusion matrix
#print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(cy_test, rf.predict(x_test))))
#print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#print(classification_report(cy_test, rf.predict(x_test)))
#rf_confm = confusion_matrix(cy, cy_test, [1,0])
#sns.heatmap(rf_confm, annot=True, fmt='.2f',xticklabels = ["Admitted", "Rejected"] , yticklabels = ["Admitted", "Rejected"] )
#plt.ylabel('True class')
#plt.xlabel('Predicted class')
#plt.title('Random Forest')
#plt.show()