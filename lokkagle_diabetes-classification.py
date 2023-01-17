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
# loading packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# load data

data = pd.read_csv('/kaggle/input/kaggle_diabetes.csv')

data.head()
data.isna().sum()
data.info()
data.describe().T
# checking target class weightage

sns.countplot(x = 'Outcome', data = data)

plt.show()
# pairplot for data distribution and outcome variable importance

sns.pairplot(data, hue= 'Outcome')

plt.show()
# for data distributions

data.hist(figsize= (15,15))

plt.show()
# To check the skewness in each attribute in the data

data.skew(axis = 0)
# checking for zero variables

data.isin([0]).sum()
data[data.Pregnancies == 0]
data.isin([np.NaN]).any()
# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN

df_copy = data.copy(deep=True)

df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

df_copy.isnull().sum()
# Replacing NaN value by mean, median depending upon distribution



attributes = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']



for i in attributes:

    if i in ['Glucose','BloodPressure']:

        df_copy[i].fillna(df_copy[i].mean(), inplace = True)

    else :

        df_copy[i].fillna(df_copy[i].median(), inplace = True)

        

# check for NaN values 

df_copy.isna().sum()
df_copy.hist(figsize=(12,12))

plt.show()
X = df_copy.iloc[:, :-1].values

y = df_copy.iloc[:, -1].values
# splitting the data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 125)
# scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Using GridSearchCV to find the best algorithm for this problem

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



# Creating a function to calculate best model for this problem

def find_best_model(X, y):

    models = {

        'logistic_regression': {

            'model': LogisticRegression(solver='lbfgs', multi_class='auto'),

            'parameters': {

                'C': [1,5,10]

               }

        },

        

        'decision_tree': {

            'model': DecisionTreeClassifier(splitter='best'),

            'parameters': {

                'criterion': ['gini', 'entropy'],

                'max_depth': [5,10]

            }

        },

        

        'random_forest': {

            'model': RandomForestClassifier(criterion='gini'),

            'parameters': {

                'n_estimators': [10,15,20,50,100,200]

            }

        },

        

        'svm': {

            'model': SVC(gamma='auto'),

            'parameters': {

                'C': [1,10,20],

                'kernel': ['rbf','linear']

            }

        }



    }

    

    scores = [] 

    cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)

        

    for model_name, model_params in models.items():

        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = cv_shuffle, return_train_score=False)

        gs.fit(X, y)

        scores.append({

            'model': model_name,

            'best_parameters': gs.best_params_,

            'score': gs.best_score_

        })

        

    return pd.DataFrame(scores, columns=['model','best_parameters','score'])



find_best_model(X_train, y_train)
# checking for cross validations (k-fold cross validation)

from sklearn.model_selection import cross_val_score

for i in [5,10]:

    CV_score = cross_val_score(estimator= RandomForestClassifier(n_estimators = 100, random_state = 0), X = X_train,y = y_train, cv = i )

    print('CV score: {} for cv = {}'.format(CV_score, i))
# from our observations randomforest is best fit

classifier = RandomForestClassifier(n_estimators= 100, random_state= 0, n_jobs= -1)

classifier.fit(X_train, y_train)
# PCA is used for dimentionality reduction technique

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X = pca.fit_transform(X)
# to check for variable importance

explained_variance = pca.explained_variance_ratio_

explained_variance
# plotting scatter plot for 2d data

plt.scatter(x = X[:, 0], y= X[:, 1])

plt.xlabel('first priciple component')

plt.ylabel('second principle component')

plt.show()
# check for model scores as part of model evalution

print("training score: {} and test score: {} ".format(classifier.score(X_train, y_train), classifier.score(X_test, y_test)))
# model predictions

y_pred = classifier.predict(X_test)
# model metrics for classification problem

from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

cm = confusion_matrix(y_test, y_pred)

cm
# classification report

print(classification_report(y_test, y_pred))
# accuracy score

print(accuracy_score(y_test, y_pred))
# confusion metrics parameters for auc-roc curve

from sklearn.metrics import roc_auc_score, roc_curve

fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
# checking for auc-roc curve

roc_score = roc_auc_score(y_test, y_pred)

roc_score
# roc curve

plt.figure()

plt.plot(fpr, tpr, label = 'Logistic Regression (sensitivity = %0.3f)' %roc_score)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Postive Rate')

plt.legend(loc = 'lower right')

plt.show()