# Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

%matplotlib inline
# Let's import the data and start exploring it

data = pd.read_csv('../input/mushrooms.csv')

data.head()
data.info()
data.describe()
labelEncoder = preprocessing.LabelEncoder()

for col in data.columns:

    data[col] = labelEncoder.fit_transform(data[col])

    

# Train Test Split

X = data.drop('class', axis=1)

y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
keys = []

scores = []

models = {'Logistic Regression': LogisticRegression(), 'Decision Tree': DecisionTreeClassifier(),

          'Random Forest': RandomForestClassifier(n_estimators=30), 

          'K-Nearest Neighbors':KNeighborsClassifier(n_neighbors=1),

            'Linear SVM':SVC(kernel='rbf', gamma=.10, C=1.0)}



for k,v in models.items():

    mod = v

    mod.fit(X_train, y_train)

    pred = mod.predict(X_test)

    print('Results for: ' + str(k) + '\n')

    print(confusion_matrix(y_test, pred))

    print(classification_report(y_test, pred))

    acc = accuracy_score(y_test, pred)

    print(acc)

    print('\n' + '\n')

    keys.append(k)

    scores.append(acc)

    table = pd.DataFrame({'model':keys, 'accuracy score':scores})



print(table)
# Re-training the Random Forest

rfc = RandomForestClassifier(n_estimators = 30)

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)



importances = rfc.feature_importances_

plot = sns.barplot(x=X.columns, y=importances)



for item in plot.get_xticklabels():

    item.set_rotation(90)
sns.countplot(x = 'odor', data = data, hue='class', palette='coolwarm')

plt.show()