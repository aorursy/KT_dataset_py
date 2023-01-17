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
covid_data = pd.read_csv('../input/symptoms-and-covid-presence/Covid Dataset.csv')
print(np.shape(covid_data))

covid_data.head()
covid_data.describe().T
#Preprocessing the data

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

covid_data = covid_data.apply(LabelEncoder().fit_transform)
covid_data.head()
# Extract X and Y from the dataset

X_total = covid_data.iloc[:, 0:20].values

y_total = covid_data.iloc[:,20].values
#SPLIT THE DATA INTO TRAIN AND TEST DATA

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size = 0.3, random_state = 0)
#LOGISTIC REGRESSION



from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score





# fit data to LR model

lr = LogisticRegression()

lr.fit(X_train, y_train)





# Importance of X-features

importance = lr.coef_[0]

plt.figure(figsize=(20,10))





#PLOT THE IMPORTANCE GRAPH

plt.title('LOGISTIC REGRESSION FEATURE IMPORTANCE', color ='blue')

plt.xlabel('FEATURES', color ='blue')

plt.ylabel('LR - IMPORTANCE', color ='blue')

sns.barplot([i for i in range(len(importance))], importance)

plt.show()





#PREDICT ON THE X-TEST VALUES

y_pred = lr.predict(X_test)





#CHECK THE ACCURACY OF THE NB MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#DECISION TREE CLASSIFIER(DT)

from sklearn.tree import DecisionTreeClassifier



#TRAIN THE DATA WITH DT MODEL

dtree = DecisionTreeClassifier(max_depth=4, random_state=42)

dtree.fit(X_train, y_train)





# Importance of X-features

importance = dtree.feature_importances_

plt.figure(figsize=(20,10))





#PLOT THE IMPORTANCE GRAPH 

plt.title('DECISION TREE CLASSIFIER FEATURE IMPORTANCE', color ='blue')

plt.xlabel('FEATURES', color ='blue')

plt.ylabel('DT - IMPORTANCE', color ='blue')

sns.barplot([i for i in range(len(importance))], importance)

plt.show()





#PREDICT ON THE X-TEST VALUES

y_pred = dtree.predict(X_test)





#CHECK THE ACCURACY OF THE DT MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#KNeighbors Classifier(KNN)

from sklearn.neighbors import KNeighborsClassifier





# to calculate importance 

from sklearn.inspection import permutation_importance





#TRAIN THE DATA WITH DT MODEL

knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')

knn.fit(X_train, y_train)





# Importance of X-features

results = permutation_importance(knn, X_train, y_train, scoring='neg_mean_squared_error')

importance = results.importances_mean

plt.figure(figsize=(20,10))





#PLOT THE IMPORTANCE GRAPH

plt.title('KNeighbors Classifier FEATURE IMPORTANCE', color ='blue')

plt.xlabel('FEATURES', color ='blue')

plt.ylabel('KNN - IMPORTANCE', color ='blue')

sns.barplot([i for i in range(len(importance))], importance)

plt.show()





#PREDICT ON THE X-TEST VALUES

y_pred = knn.predict(X_test)





#CHECK THE ACCURACY OF THE DT MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#Gradient Boosting Classifier (GB)

from sklearn.ensemble import GradientBoostingClassifier



#TRAIN THE DATA WITH GB MODEL

gb = model = GradientBoostingClassifier()

gb.fit(X_train, y_train)





# Importance of X-features

importance = gb.feature_importances_

plt.figure(figsize=(20,10))





#PLOT THE IMPORTANCE GRAPH

plt.title('Gradient Boosting Classifier FEATURE IMPORTANCE', color ='blue')

plt.xlabel('FEATURES', color ='blue')

plt.ylabel('GB - IMPORTANCE', color ='blue')

sns.barplot([i for i in range(len(importance))], importance)

plt.show()





#PREDICT ON THE X-TEST VALUES

y_pred = gb.predict(X_test)





#CHECK THE ACCURACY OF THE DT MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#Gaussian Naive Bais(NB)

from sklearn.naive_bayes import GaussianNB





#TRAIN THE DATA WITH NB MODEL

nb = GaussianNB()

nb.fit(X_train, y_train)





# Importance of X-features

results = permutation_importance(nb, X_train, y_train, scoring='neg_mean_squared_error')

importance = results.importances_mean

plt.figure(figsize=(20,10))





#PLOT THE IMPORTANCE GRAPH

plt.title('Gaussian Naive Bais FEATURE IMPORTANCE', color ='blue')

plt.xlabel('FEATURES', color ='blue')

plt.ylabel('NB - IMPORTANCE', color ='blue')

sns.barplot([i for i in range(len(importance))], importance)

plt.show()





#PREDICT ON THE X-TEST VALUES

y_pred = nb.predict(X_test)





#CHECK THE ACCURACY OF THE NB MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#Support Vector Machines(SVM)

from sklearn import svm





#TRAIN THE DATA WITH SVM MODEL

svc = svm.SVC()

svc.fit(X_train, y_train)





# Importance of X-features

results = permutation_importance(svc, X_train, y_train, scoring='neg_mean_squared_error')

importance = results.importances_mean

plt.figure(figsize=(20,10))





#PLOT THE IMPORTANCE GRAPH

plt.title('Support Vector Machines FEATURE IMPORTANCE', color ='blue')

plt.xlabel('FEATURES', color ='blue')

plt.ylabel('SVM - IMPORTANCE', color ='blue')

sns.barplot([i for i in range(len(importance))], importance)

plt.show()





#PREDICT ON THE X-TEST VALUES

y_pred = svc.predict(X_test)





#CHECK THE ACCURACY OF THE SVM MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#Random Forrest Classifier(RF)

from sklearn.ensemble import RandomForestClassifier





#TRAIN THE DATA WITH RF MODEL

rf = RandomForestClassifier(n_estimators=10, random_state =0)

rf.fit(X_train, y_train)





# Importance of X-features

importance = rf.feature_importances_

plt.figure(figsize=(20,10))







#PLOT THE IMPORTANCE GRAPH

plt.title('Random Forrest Classifier FEATURE IMPORTANCE', color ='blue')

plt.xlabel('FEATURES', color ='blue')

plt.ylabel('RF - IMPORTANCE', color ='blue')

sns.barplot([i for i in range(len(importance))], importance)

plt.show()





#PREDICT ON THE X-TEST VALUES

y_pred = rf.predict(X_test)





#CHECK THE ACCURACY OF THE RF MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))