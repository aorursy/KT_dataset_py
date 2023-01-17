# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing file

data = pd.read_csv('/kaggle/input/titanic/train.csv')

data=data.loc[:,("Survived","Pclass","Sex","Age","SibSp","Parch","Fare")]
data.head(10)
data.count()
data.dropna(inplace=True)

data.count()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

data.Sex=LE.fit_transform(data.Sex)

data.Sex
y= data.iloc[:,0]

x=data.iloc[:,1:]



#here this function is used to randomly split our data

from sklearn.model_selection import train_test_split as split



#Now we are going to keep our data into subdataset

#training=70% and testing=30%

x_train, x_test, y_train, y_test = split(x,y,test_size=.3)



y_train

#type(y_train)

from sklearn.tree import DecisionTreeClassifier

dectree= DecisionTreeClassifier()

#For training features=x_train and target=y_train

dectree.fit(x_train, y_train)  



#prediction would be made on x_test

pred = dectree.predict(x_test)  

from sklearn.metrics import accuracy_score

accuracy_score(y_test,pred)



from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,pred)



from sklearn.metrics import recall_score

recall_score(y_test,pred)



from sklearn.metrics import precision_score

precision_score(y_test,pred)



dectree = DecisionTreeClassifier(criterion='gini')

dectree.fit(x_train, y_train)

pred = dectree.predict(x_test)

print('Criterion=gini', accuracy_score(y_test, pred))

dectree = DecisionTreeClassifier(criterion='entropy')

dectree.fit(x_train, y_train)

pred = dectree.predict(x_test)

print('Criterion=entropy', accuracy_score(y_test, pred))
max_depth = []

acc_gini = []

acc_entropy = []

for i in range(1,30):

 dectree = DecisionTreeClassifier(criterion="gini", max_depth=i)

 dectree.fit(x_train, y_train)

 pred = dectree.predict(x_test)

 acc_gini.append(accuracy_score(y_test, pred))

 ####

 dectree = DecisionTreeClassifier(criterion="entropy", max_depth=i)

 dectree.fit(x_train, y_train)

 pred = dectree.predict(x_test)

 acc_entropy.append(accuracy_score(y_test, pred))

 ####

 max_depth.append(i)

d = pd.DataFrame({"acc_gini":pd.Series(acc_gini), 

 "acc_entropy":pd.Series(acc_entropy),

 "max_depth":pd.Series(max_depth)})

# visualizing changes in parameters

import matplotlib.pyplot as plt

plt.plot("max_depth","acc_gini", data=d, label="gini")

plt.plot("max_depth","acc_entropy", data=d, label="entropy")

plt.xlabel("max_depth")

plt.ylabel("accuracy")

plt.legend()
#First with level 7

dectree = DecisionTreeClassifier(criterion="entropy", max_depth=7)

dectree.fit(x_train, y_train)

pred = dectree.predict(x_test)

print('Accuracy with level 7 and criterion as entropy = ',accuracy_score(y_test, pred))



#First with level 8

dectree2 = DecisionTreeClassifier(criterion="entropy", max_depth=8)

dectree2.fit(x_train, y_train)

pred = dectree2.predict(x_test)

print('Accuracy with level 8 and criterion as entropy = ',accuracy_score(y_test, pred))
print(dict(zip(data.columns,dectree.feature_importances_)))

x_axis =data.columns

# Sort feature importances in descending order

feature_imp = dectree.feature_importances_

indices = np.argsort(feature_imp)[::-1]

# Rearrange feature names so they match the sorted feature importances

features_names = data.columns

names = [features_names[i] for i in indices]
import matplotlib.pyplot as plt

# Create plot

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])



# Create plot title

plt.title("Feature Importance")

ax.bar(names,feature_imp)

plt.show()



data2 = pd.read_csv('/kaggle/input/titanic/test.csv')



data2=data.loc[:,("Pclass","Sex","Age","SibSp","Parch","Fare")]

data2
prediction = dectree.predict(data2)

type(prediction)

print(len(prediction))
df = pd.DataFrame({ 'prediction': prediction })

df


df.to_csv('results.csv', index = False)





from IPython.display import FileLink

FileLink(r'results.csv')