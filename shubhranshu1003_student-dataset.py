import xgboost
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

data = pd.read_excel('../input/research_student (1).xlsx')

data.head()
data.info()
train, test = train_test_split(data, test_size=0.3)
train.head(20)
test.head()
data.Gender.replace(['Male', 'Female'], \
                     [0,1], inplace = True)
data.Gender
test.Gender.replace(['Male', 'Female'], \
                     [0,1], inplace = True)
test.Gender

data.columns
data = data.fillna(0)

#Removing null values
data.head()

data = data.drop(['Board[10th]', 'Board[12th]', 'Category'], axis =1)

data.head()

data.index
data.rename({'GPA 1': 'GPA_1','GPA 2': 'GPA_2', 'GPA 3': 'GPA_3', 'GPA 4': 'GPA_4', 'GPA 5': 'GPA_5', 'GPA 6': 'GPA_6'}, axis = 1)
data.columns
data.columns = ['Branch', 'Marks10', 'Marks12', 'Gender', 'GPA_1', 'Rank',
       'NormalizedRank', 'CGPA', 'CurrentBack', 'EverBack', 'GPA_2',
       'GPA_3', 'GPA_4', 'GPA_5', 'GPA_6', 'OlympiadsQualified',
       'TechnicalProjects', 'TechQuiz', 'EnggCoaching',
       'NTSEScholarships', 'MiscellanyTechEvents']
data.head()
data.info
#Removing outliners

gpalist = ['GPA_1', 'GPA_2', 'GPA_3', 'GPA_4', 'GPA_5', 'GPA_6']
global data
for i in gpalist:
    for j in data[i]:
        if j > 10.0:
            data = data.drop(data[data[i]==j].index)
data.describe
#Scaling

scale_list = ['Rank', 'NormalizedRank']

sc = data[scale_list]
sc.head()
scaler = StandardScaler()
sc = scaler.fit_transform(sc)
data[scale_list] = sc
data[scale_list].head()
plt.scatter(data.Branch, data.GPA_1)
train, test = train_test_split(data, test_size=0.3)
train.shape
test.shape
train.info
train.describe()
result_y = train['CGPA']
result_x = train.drop('CGPA', axis =1)
X_train, X_test, y_train, y_test = train_test_split(result_x, result_y ,test_size=0.3)
X_train.shape
logreg=LinearRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
y_test
y_pred
print(metrics.mean_squared_error(y_test, y_pred))
xgb = xgboost.XGBRegressor(n_estimators=25000, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)
xgb.fit(X_train,y_train)
predictions = xgb.predict(X_test)
print(metrics.mean_squared_error(y_test, predictions))

