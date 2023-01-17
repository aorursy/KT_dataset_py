import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings  

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_set = pd.read_csv('/kaggle/input/learn-together/train.csv.zip',index_col='Id')

test_set = pd.read_csv('/kaggle/input/learn-together/test.csv.zip',index_col='Id')
train_set.shape
test_set.shape
train_set.info()
test_set.info()
train_set.head()
train_set.describe().transpose()
train_set['Cover_Type'].value_counts()
train_set.iloc[:,list(range(10))+[-1]].corr()
wilderness_area = train_set.loc[:,['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']].sum()

wilderness_area.index = ['Rawah', 'Neota', 'Comanche Peak', 'Cache la Poudre']



figure1 = plt.figure(figsize=(10,5))

axes1 = figure1.add_axes([0,0,1,1])

axes1.set_title('Wilderness areas count')

axes1.set_xlabel('Wilderness areas')

axes1.set_ylabel('Count')

sns.barplot(x = wilderness_area.index, y = wilderness_area.values, palette='rainbow')

figure1.tight_layout()
soil_types = train_set.iloc[:,14:-1].sum()

figure2 = plt.figure(figsize=(10,5))

axes2 = figure2.add_axes([0,0,1,1])

axes2.set_title('Soil types count')

axes2.set_xlabel('Soil types')

plt.xticks(rotation= 90)

axes2.set_ylabel('Count')

sns.barplot(x = soil_types.index, y = soil_types.values, palette='BrBG')

figure2.tight_layout()
figure3 = plt.figure(figsize=(10,10))

axes3 = figure3.add_axes([0,0,1,1])

axes3.set_title('Correlations')

sns.heatmap(train_set.iloc[:,list(range(10))+[-1]].corr(),cmap='coolwarm',annot=True)
sns.pairplot(train_set[['Hillshade_3pm', 'Hillshade_9am','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Aspect']])

plt.show()
from sklearn.model_selection import StratifiedShuffleSplit

X = train_set.iloc[:,:-1]

y = train_set["Cover_Type"]



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

for train_index, val_index in split.split(X, y):

    X_train = X.iloc[train_index]

    y_train = y.iloc[train_index]

    X_val = X.iloc[val_index]

    y_val = y.iloc[val_index]
#from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler





column_pipeline = ColumnTransformer([("scaler", StandardScaler(), list(X.columns)[:10])],

                                    remainder = 'passthrough')

X_train_prepared = column_pipeline.fit_transform(X_train)

X_val_prepared = column_pipeline.transform(X_val)

X_test = column_pipeline.transform(test_set)

y_val.shape
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV



param_grid_LR = [{'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0]}]

LR = LogisticRegression(random_state = 10)

grid_search_LR = GridSearchCV(LR, param_grid_LR, cv=5)

grid_search_LR.fit(X_train_prepared, y_train)

print(grid_search_LR.best_score_)

import pickle

from sklearn.svm import SVC

param_grid_SVC = [{'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0], 'kernel':['rbf']},

                 {'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0], 'kernel':['poly'], 'degree': [2,3,4]}]

SV = SVC(random_state = 10, probability = True)

grid_search_SV = GridSearchCV(SV, param_grid_SVC, cv=3)

grid_search_SV.fit(X_train_prepared, y_train)



with open("SVC_model", 'wb') as file:  

    pickle.dump(grid_search_SV.best_estimator_, file)

print(grid_search_SV.best_score_)
from sklearn.neighbors import KNeighborsClassifier

KN = KNeighborsClassifier()

param_grid_KN = [{'n_neighbors': [4,5,6,7], 'leaf_size':[20,30,40]}]

grid_search_KN = GridSearchCV(KN, param_grid_KN, cv=3)

grid_search_KN.fit(X_train_prepared, y_train)

print(grid_search_KN.best_score_)
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(random_state = 10)

param_grid_DT = [{'max_depth':[2,4,8,16]}, {'min_samples_split':[2,3,4,5,6]}, {'min_samples_leaf':[1,2,3,4]}]

grid_search_DT = GridSearchCV(DT, param_grid_DT, cv=3)

grid_search_DT.fit(X_train_prepared, y_train)

print(grid_search_DT.best_score_)
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 10)



param_grid_RF = [{'n_estimators': [100, 200, 300], 'max_depth':[2,4,8,16]},

                 {'n_estimators': [100, 200, 300],'min_samples_split':[2,3,4,5,6]},

                 {'n_estimators': [100, 200, 300],'min_samples_leaf':[1,2,3,4]}]

grid_search_RF = GridSearchCV(RF, param_grid_RF, cv=3)

grid_search_RF.fit(X_train_prepared, y_train)

print(grid_search_RF.best_score_)
from sklearn.ensemble import ExtraTreesClassifier

ET = ExtraTreesClassifier()



param_grid_ET = [{'n_estimators': [100, 200, 300], 'max_depth':[2,4,8,16]},

                 {'n_estimators': [100, 200, 300],'min_samples_split':[2,3,4,5,6]},

                 {'n_estimators': [100, 200, 300],'min_samples_leaf':[1,2,3,4]}]

grid_search_ET = GridSearchCV(ET, param_grid_ET, cv=3)

grid_search_ET.fit(X_train_prepared, y_train)

print(grid_search_ET.best_score_)
from sklearn.ensemble import VotingClassifier

VC = VotingClassifier(estimators = [('svc', grid_search_SV.best_estimator_), 

                                    ('rf', grid_search_RF.best_estimator_), 

                                    ('et', grid_search_ET.best_estimator_)], 

                      voting = 'soft')

VC.fit(X_train_prepared, y_train)

print(accuracy_score(y_val, VC.predict(X_val_prepared)))



with open("voting_model", 'wb') as file:  

    pickle.dump(VC, file)
predictions = VC.predict(X_test)

submission = pd.DataFrame({'Id':test_set.index,'Cover_Type':predictions}, 

                          columns=['Id', 'Cover_Type'])

submission.to_csv("submission.csv", index=False)