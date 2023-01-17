# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df
train_df.isnull().sum()
train_df = train_df.drop(['Cabin'], axis = 1)

train_df = train_df.dropna()



test_df = test_df.drop(['Cabin'], axis = 1)

test_df = test_df.dropna()
print('Train Dataframe:')

print(train_df.isnull().sum())

print('\nTest Dataframe:')

print(test_df.isnull().sum())
train_df.Survived.value_counts().plot(kind = 'bar', alpha = 0.5)

plt.title('Distribution of Survival (Survived = 1)')

plt.show()
plt.scatter(train_df.Survived, train_df.Age, alpha = 0.2)

plt.ylabel('Age')

plt.title('Survived by Age (Survived = 1)')

plt.show()
train_df.Pclass.value_counts().plot(kind = 'barh', alpha = 0.5)

plt.title('Class Distribution')

plt.show()
plt.figure(figsize = (15, 5))



for class_val in np.unique(train_df.Pclass):

    train_df.Age[train_df.Pclass == class_val].plot(kind = 'kde')

    

plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(('1st Class', '2nd Class', '3rd Class'))

plt.show()
train_df.Embarked.value_counts().plot(kind = 'bar', alpha = 0.5)

plt.title('Passengers per boarding location')

plt.show()
males = train_df.Survived[train_df.Sex == 'male'].value_counts()

females = train_df.Survived[train_df.Sex == 'female'].value_counts()



fig = plt.figure(figsize = (18, 6))



ax1 = fig.add_subplot(121)

males.plot(kind = 'bar', color = 'blue', label = 'Male')

females.plot(kind = 'bar', color = '#FA2379',label = 'Female')

ax1.set_xlim(-1, 2)

plt.title("Survival w/ respect to Gender (raw value counts)")

plt.legend(loc = 'best')



ax2 = fig.add_subplot(122)

(males/float(males.sum())).plot(kind = 'bar', color = 'blue', label = 'Male')

(females/float(females.sum())).plot(kind = 'bar', color = '#FA2379',label = 'Female')

ax2.set_xlim(-1,2)

plt.title("Survival w/ respect to Gender (proportional value counts)")

plt.legend(loc = 'best')



plt.show()
fig = plt.figure(figsize = (18,4))



female_upperclass = train_df.Survived[train_df.Sex == 'female'][train_df.Pclass != 3].value_counts().sort_index()

male_upperclass = train_df.Survived[train_df.Sex == 'male'][train_df.Pclass != 3].value_counts().sort_index()

female_lowerclass = train_df.Survived[train_df.Sex == 'female'][train_df.Pclass == 3].value_counts().sort_index()

male_lowerclass = train_df.Survived[train_df.Sex == 'male'][train_df.Pclass == 3].value_counts().sort_index()



gender_and_classes = {'Female upperclass': female_upperclass,'Female lowerclass': female_lowerclass,

                      'Male upperclass': male_upperclass,'Male lowerclass': male_lowerclass}

i = 0

colors = ['#FA2479','pink','steelblue','lightblue']



fig.suptitle('Survival w/ respect to Gender and Class')

for k, v in gender_and_classes.items():

    ax = fig.add_subplot(1,4,i+1)

    v.plot(kind = 'bar', label = k, color = colors[i], alpha = 0.65)

    ax.set_xticklabels(['Died','Survived'], rotation = 45)

    ax.set_xlim(-1, len(v))

    ax.set_ylim(0, max(v) + 50)

    plt.legend(loc='best')

    i += 1

    

plt.show()
X = np.array(train_df.iloc[:, [2,4,5,9]])

Y = np.array(train_df.Survived)



X_test = np.array(test_df.iloc[:, [1,3,4,8]])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



lbl_enc = LabelEncoder()

X[:,0] = lbl_enc.fit_transform(X[:,0])

X[:,1] = lbl_enc.fit_transform(X[:,1])



X_test[:,0] = lbl_enc.fit_transform(X_test[:,0])

X_test[:,1] = lbl_enc.fit_transform(X_test[:,1])



ohe = OneHotEncoder(categorical_features = [0])

X = pd.DataFrame(ohe.fit_transform(X).toarray())

X = X.drop([0], axis = 1)



X_test = pd.DataFrame(ohe.fit_transform(X_test).toarray())

X_test = X_test.drop([0], axis = 1)
from sklearn.preprocessing import StandardScaler



std_sclr = StandardScaler()

X = std_sclr.fit_transform(X)

X_test = std_sclr.fit_transform(X_test)
from sklearn.model_selection import StratifiedShuffleSplit



strat_split = StratifiedShuffleSplit(test_size = 0.2, random_state = 0)

for train_index, val_index in strat_split.split(X, Y):

    X_train, X_val = X[train_index], X[val_index]

    Y_train, Y_val = Y[train_index], Y[val_index]
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(random_state = 0)

lr.fit(X_train, Y_train)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(random_state = 0)

rf.fit(X_train, Y_train)
from sklearn.svm import SVC



svc = SVC(random_state = 0)

svc.fit(X_train, Y_train)
preds = {'Logistic Regression':lr.predict(X_val), 'Random Forest':rf.predict(X_val), 'SVC':svc.predict(X_val)}
from sklearn.metrics import confusion_matrix, accuracy_score



for k,v in preds.items():

    cm = confusion_matrix(Y_val, v)

    acc = accuracy_score(Y_val, v)

    print('Confusion Matrix for {}:\n{}'.format(k, cm))

    print('Accuracy for {}:{}'.format(k,acc * 100))