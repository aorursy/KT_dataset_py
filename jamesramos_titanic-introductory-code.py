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
#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Reading the dataset
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")

#Printing the first two elements of "train"
train.head(2)
#Determining female survivorship
women = train.loc[train.Sex == "female","Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
#Determining male survivorship
men = train.loc[train.Sex == "male","Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
objects = ("Men", "Women")
x_pos = np.arange(len(objects))
performance = [rate_men, rate_women]

plt.bar(x_pos, performance)
plt.xticks(x_pos, objects)
plt.ylabel("Percentage of Survivors")
plt.title("Survivorship Based on Sex")

plt.show()
chart = pd.crosstab(train.Pclass, train.Survived, margins=True)

chart
#Code to split data into x (independent) and y (dependent) variables

y_train = train['Survived']
features = ['Sex','Pclass']
x_train = pd.get_dummies(train[features])
x_test = pd.get_dummies(test[features])
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})
output.to_csv('decision_tree.csv', index=False)
print("Your submission was successfully saved!")
from sklearn.naive_bayes import GaussianNB
nbayes = GaussianNB()
nbayes.fit(x_train, y_train)
y_pred = nbayes.predict(x_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})
output.to_csv('naive_bayes.csv', index=False)
print("Your submission was successfully saved!")
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

convnn = Sequential()
convnn.add(Conv2D(64, kernel_size=3, activation= "relu", input_shape=(28,28,1)))
convnn.add(Conv2D(32, kernel_size=3, activation= "relu"))
convnn.add(Flatten())
convnn.add(Dense(10, activation= "softmax"))

convnn.compile(optimizer = "adam", loss = "mean_squared_error")

convnn.fit(x_train, y_train)
y_pred = convnn.predict(x_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})
output.to_csv('naive_bayes.csv', index=False)
print("Your submission was successfully saved!")