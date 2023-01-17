# Importing libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Supress warnings

import warnings

warnings.filterwarnings("ignore")



# Classification

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier 







# Regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet, LogisticRegression

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 











# Modelling Helpers :

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score, ShuffleSplit, cross_validate



# Preprocessing :

from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder



# Metrics :

# Regression

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 

# Classification

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report



print("Setup complete...")
asd = pd.read_csv("../input/Toddler Autism dataset July 2018.csv")

print("Dataset loaded...")
asd.head()
asd.describe()
asd.columns
asd.drop(['Case_No', 'Who completed the test'], axis = 1, inplace = True)

asd.columns
asd.dtypes
corr = asd.corr()

plt.figure(figsize = (15,15))

sns.heatmap(data = corr, annot = True, square = True, cbar = True)
plt.figure(figsize = (16,8))

sns.countplot(x = 'Ethnicity', data = asd)
asd.columns
asd.drop('Qchat-10-Score', axis = 1, inplace = True)
le = LabelEncoder()

columns = ['Ethnicity', 'Family_mem_with_ASD', 'Class/ASD Traits ', 'Sex', 'Jaundice']

for col in columns:

    asd[col] = le.fit_transform(asd[col])

asd.dtypes
X = asd.drop(['Class/ASD Traits '], axis = 1)

Y = asd['Class/ASD Traits ']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 7)
models = []

models.append(('Logistic Regression:', LogisticRegression()))

models.append(('Decision Tree      :', DecisionTreeClassifier()))

models.append(('Naive Bayes        :', GaussianNB()))

models.append(('SVM                :', SVC()))

models.append(('Random Forest      :', RandomForestRegressor()))





for name, model in models:

    model.fit(x_train, y_train)

    pred = model.predict(x_test).astype(int)

    print(name, accuracy_score(y_test, pred))