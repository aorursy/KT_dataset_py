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
path1 = '../input/titanic/train.csv'

path2 = '../input/titanic/test.csv'
data1 = pd.read_csv(path1)

data2 = pd.read_csv(path2)

women = data1.loc[data1.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men =data1.loc[data1.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
data1
data1['Age'].fillna(data1['Age'].mean(), inplace = True)

data2['Age'].fillna(data1['Age'].mean(), inplace = True)

data2['Fare'].fillna(data2['Fare'].mean(), inplace = True)
data1.describe()
# comparison of the gender of people survived

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 5)

sns.countplot(data1[data1['Survived']== 1]['Sex'], palette = 'pink')

plt.title('the gendre of people survived', fontsize = 20)

plt.show()
# comparison of the gender of people not survived

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 5)

sns.countplot(data1[data1['Survived']== 0]['Sex'], palette = 'pink')

plt.title('the gendre of people not survived', fontsize = 20)

plt.show()
# plotting a pie chart to represent share of dead people in Titanic with gender



labels = ['Male', 'Female']

sizes = data1[data1['Survived']== 0]['Sex'].value_counts()

colors = plt.cm.copper(np.linspace(0, 1, 5))



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(sizes, labels = labels, colors = colors,  shadow = True)

plt.title('share of dead people in Titanic with gender', fontsize = 20)

plt.legend()

plt.show()
# plotting a pie chart to represent the share of Embarked survived people



size = data1[data1['Survived']== 0]['Embarked'].value_counts()

colors = plt.cm.Wistia(np.linspace(0, 1, 5))

labels = ['S', 'C', 'Q']

plt.pie(size,  colors = colors, labels = labels, shadow = True, startangle = 90)

plt.title('the share of Embarked survived peoples', fontsize = 25)

plt.legend()

plt.show()
# different Embarked acquired by the Passangers 



plt.figure(figsize = (18, 8))

plt.style.use('fivethirtyeight')

ax = sns.countplot('Embarked', data = data1, palette = 'bone')

ax.set_xlabel(xlabel = 'Different Embarked acquired by the Passangers ', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of Passangers', fontsize = 16)

plt.show()
# Comparing the Passangers' Ages



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (15, 5)

sns.distplot(data1['Age'], color = 'blue')

plt.xlabel('Age Range for Passangers', fontsize = 16)

plt.ylabel('Count of the Passangers', fontsize = 16)

plt.title('Distribution of Ages of Passangers', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
# Comparing the Survived Passangers' Ages



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (15, 5)

sns.distplot(data1[data1['Survived']==1]['Age'])

plt.xlabel('Age Range forSurvived Passangers', fontsize = 16)

plt.ylabel('Count of the Passangers', fontsize = 16)

plt.title('Distribution of Ages of Survived Passangers', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
#  Pclass of Survived Passangers



plt.figure(figsize = (10, 8))

ax = sns.countplot(x = 'Pclass', data = data1[data1['Survived']==1], palette = 'pastel')

ax.set_title(label = 'Count of Survived Passangers on Basis of their Pclass', fontsize = 20)

ax.set_xlabel(xlabel = 'Number Class', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
# SibSp of Survived Passangers



plt.figure(figsize = (13, 8))

ax = sns.countplot(x = 'SibSp', data = data1[data1['Survived']==1], palette = 'dark')

ax.set_title(label = 'Count of Survived Passangers on Basis of SipSp', fontsize = 20)

ax.set_xlabel(xlabel = 'SibSp ', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()

# Parch of Survived Passangers



plt.figure(figsize = (13, 8))

ax = sns.countplot(x = 'Parch', data = data1[data1['Survived']==1], palette = 'dark')

ax.set_title(label = 'Count of Survived Passangers on Basis of Parch', fontsize = 20)

ax.set_xlabel(xlabel = 'Parch ', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()

# To show Different Fares of Survived passengers



plt.figure(figsize = (20, 5))

sns.distplot(data1[data1['Survived']==1]['Fare'], color = 'pink')

plt.title(' Different Fares of Survived passengers', fontsize = 20)

plt.xlabel('Fare associated with Survived Passangers', fontsize = 16)

plt.ylabel('count of Survived Passangers', fontsize = 16)

plt.show()
# To show that there are Passangers having same age

# Histogram: number of Passangers's age



sns.set(style = "dark", palette = "colorblind", color_codes = True)

x = data1['Age']

plt.figure(figsize = (15,8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'g')

ax.set_xlabel(xlabel = "Passanger's age", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of Passangers', fontsize = 16)

ax.set_title(label = 'Histogram of passanger age', fontsize = 20)

plt.show()
# plotting a correlation heatmap



plt.rcParams['figure.figsize'] = (30, 20)

sns.heatmap(data1.corr(), annot = True)



plt.title('Histogram of the Dataset', fontsize = 30)

plt.show()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data1['Sex'] = le.fit_transform(data1['Sex'])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data2['Sex'] = le.fit_transform(data2['Sex'])

features = ['Pclass','Sex',  'SibSp','Parch']
X_train = data1[features]

X_test = data2[features]

y_train = data1['Survived']

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

predictions
y_test = pd.read_csv('../input/titanic/gender_submission.csv')

from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)

y_test=y_test['Survived'].values
cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
output = pd.DataFrame({'PassengerId': data2.PassengerId ,'Survived': predictions})



filename = 'Titanic Predictions.csv'



output.to_csv(filename,index=False)



print('Saved file: ' + filename)