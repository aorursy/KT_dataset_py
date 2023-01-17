import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv("../input/ANAD_Normalized.csv")
df.head()
df.info()
#how many categories we have?

df['Emotion '].unique()
plt.figure(figsize = (10, 8))

sns.countplot(df['Emotion '])

plt.show()
df['Emotion '].value_counts()
df.isnull().sum().sum() #no missing values
df.drop('Type', 1, inplace = True) #not needed
#split into features and labels sets

X = df.drop(['name', 'Emotion '], axis = 1) #features

y = df['Emotion '] #labels
X.info() #we have only numerical features, so we are ready for training
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)
from sklearn.linear_model import LogisticRegression



m1 = LogisticRegression()

m1.fit(X_train, y_train)

pred1 = m1.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred1)) #not bad, but not very good for surprised emotion
labels = ['angry', 'happy', 'surprised']

cm1 = pd.DataFrame(confusion_matrix(y_test, pred1), index = labels, columns = labels)
plt.figure(figsize = (10, 8))

sns.heatmap(cm1, annot = True, cbar = False, fmt = 'g')

plt.ylabel('Actual values')

plt.xlabel('Predicted values')

plt.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
grid = {'n_estimators': [10, 50, 100, 300]}



m2 = GridSearchCV(RandomForestClassifier(), grid)

m2.fit(X_train, y_train)
m2.best_params_  #I got n_estimators = 50
pred2 = m2.predict(X_test)

print(classification_report(y_test, pred2)) #much better, but recall is still low
cm2 = pd.DataFrame(confusion_matrix(y_test, pred2), index = labels, columns = labels)



plt.figure(figsize = (10, 8))

sns.heatmap(cm2, annot = True, cbar = False, fmt = 'g')

plt.ylabel('Actual values')

plt.xlabel('Predicted values')

plt.show()
from sklearn.ensemble import GradientBoostingClassifier
#m3 = GradientBoostingClassifier(learning_rate = 0.5, max_depth = 3, n_estimators = 100)

#m3.fit(X_train, y_train)
grid = {

    'learning_rate': [0.03, 0.1, 0.5], 

    'n_estimators': [100, 300], 

    'max_depth': [1, 3, 9]

}



m3 = GridSearchCV(GradientBoostingClassifier(), grid, verbose = 2)

m3.fit(X_train, y_train) 
m3.best_params_
pred3 = m3.predict(X_test)



print(classification_report(y_test, pred3))
cm3 = pd.DataFrame(confusion_matrix(y_test, pred3), index = labels, columns = labels)



plt.figure(figsize = (10, 8))

sns.heatmap(cm3, annot = True, cbar = False, fmt = 'g')

plt.ylabel('Actual values')

plt.xlabel('Predicted values')

plt.show()
from sklearn.svm import SVC
grid = {

    'C': [1, 5, 50],

    'gamma': [0.05, 0.1, 0.5, 1, 5]

}



m5 = GridSearchCV(SVC(), grid)

m5.fit(X_train, y_train)
m5.best_params_ #I got C = 50, gamma = 0.05
pred5 = m5.predict(X_test)



print(classification_report(y_test, pred5))
cm5 = pd.DataFrame(confusion_matrix(y_test, pred5), index = labels, columns = labels)



plt.figure(figsize = (10, 8))

sns.heatmap(cm5, annot = True, cbar = False, fmt = 'g')

plt.ylabel('Actual values')

plt.xlabel('Predicted values')

plt.show()