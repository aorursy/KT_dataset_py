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
dataframe = pd.read_csv("/kaggle/input/gender-classification/Transformed Data Set - Sheet1.csv")

dataframe.info()
dataframe.rename(columns={'Favorite Color' :'FavoriteColor', 'Favorite Music Genre':'FavoriteMusicGenre', 

                          'Favorite Beverage':'FavoriteBeverage', 'Favorite Soft Drink':'FavoriteSoftDrink'}, inplace=True)
'''

#Method 1 when we have all the columns as type object

from sklearn.preprocessing import LabelEncoder

dataframe.apply(LabelEncoder().fit_transform)

'''


#Fetch features of type Object

objFeatures = dataframe.select_dtypes(include="object").columns



#Iterate a loop for features of type object

from sklearn import preprocessing

le = preprocessing.LabelEncoder()



for feat in objFeatures:

    dataframe[feat] = le.fit_transform(dataframe[feat].astype(str))

    

dataframe.info()

import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot('FavoriteColor', hue='Gender', data=dataframe, palette='Blues')
dataframe.info()
X = dataframe.drop(['Gender'], axis = 1)

y = dataframe.Gender
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix



def clf_scores(clf, y_predicted):

    # Accuracy

    acc_train = clf.score(X_train, y_train)*100

    acc_test = clf.score(X_test, y_test)*100

    

    roc = roc_auc_score(y_test, y_predicted)*100 

    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()

    cm = confusion_matrix(y_test, y_predicted)

    correct = tp + tn

    incorrect = fp + fn

    

    return acc_train, acc_test, roc, correct, incorrect, cm

#1. Logistic regression



from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression()

clf_lr.fit(X_train, y_train)



Y_pred_lr = clf_lr.predict(X_test)

print(clf_scores(clf_lr, Y_pred_lr))
#2. KNN



from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors=3)

clf_knn.fit(X_train, y_train)



Y_pred_knn = clf_knn.predict(X_test)

print(clf_scores(clf_knn, Y_pred_knn))
#3. Naive Bayes



from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()

clf_gnb.fit(X_train, y_train)



Y_pred_gnb = clf_gnb.predict(X_test)

print(clf_scores(clf_gnb, Y_pred_gnb))
#4. SVM 

from sklearn.svm import SVC



clf_svm = SVC()

clf_svm.fit(X_train, y_train)



Y_pred_svm = clf_svm.predict(X_test)

print(clf_scores(clf_svm, Y_pred_svm))
#5. Decision tree

from sklearn.tree import DecisionTreeClassifier



clf_dt = DecisionTreeClassifier(random_state=0)

clf_dt.fit(X_train, y_train)

clf_dt.fit(X_train, y_train)



Y_pred_dt = clf_dt.predict(X_test)

print(clf_scores(clf_dt, Y_pred_dt))
#6. Radom forest classifier



from sklearn.ensemble import RandomForestClassifier



clf_rfc = RandomForestClassifier(max_depth=10, random_state=42)

clf_rfc.fit(X_train, y_train)



Y_pred_rfc = clf_rfc.predict(X_test)

print(clf_scores(clf_rfc, Y_pred_rfc))
#7. Gradient boosting classifier



from sklearn.ensemble import GradientBoostingClassifier



clf_gbc = GradientBoostingClassifier(random_state=42)

clf_gbc.fit(X_train, y_train)



Y_pred_gbc = clf_gbc.predict(X_test)

print(clf_scores(clf_gbc, Y_pred_gbc))