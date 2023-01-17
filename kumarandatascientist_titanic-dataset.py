#importing numpy and panda
import pandas as pd
import numpy as np

#load data
data = pd.read_csv('../input/train.csv')

#head
data.head()





#target column
columns_target = ['Survived']

#train columns
columns_train = ['PassengerId','Age','Pclass','Sex','Fare']

#separate the data
x = data[columns_train]
y = data[columns_target]
# cleaning data // checking
x['Sex'].isnull().sum()
x['Pclass'].isnull().sum()
x['Fare'].isnull().sum()
x['Age'].isnull().sum()
x['Age'] = x['Age'].fillna(x['Age'].median())
x['Age'].isnull().sum()
#sklearn we cannot pass String values as categorial variable
#so coverting male = 0 and female = 1
d = {'male':0,'female':1}
x['Sex'] = x['Sex'].apply(lambda x:d[x])
x['Sex'].head()
#lets see the final dataset
x
x.head()
# split the data to train  a test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
#now splitted data
# support vector machine
from sklearn import svm

clf = svm.LinearSVC()

print (clf)
# train the model
clf.fit(x_train,y_train)
#predict the model
print(clf.predict(x_test[0:1]))
print(clf.predict(x_test[0:10]))
print(clf.predict(x_test))
# check the accuracy
print(clf.score(x_test,y_test))
import numpy as np
Survived = np.array(clf.predict(x_test))
data1 = x_test['PassengerId']
PassengerId = np.array(data1)

#len(Survived)
PassengerId
import pandas as pd
dataset = pd.DataFrame({'PassengerId': PassengerId, 'Survived': list(Survived)}, columns=['PassengerId', 'Survived'])
dataset
dataset.to_csv("../input/gender_submission.csv", index=False)
