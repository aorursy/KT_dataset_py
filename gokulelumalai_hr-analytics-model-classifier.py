# import libraries for data exploration

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# %config InlineBackend.figure_Format='retina'

import warnings

warnings.filterwarnings('ignore')
# let's load the dataset

hr = pd.read_csv(r'train_LZdllcl (1).csv')

bckup = hr.copy()

hr.head()
hr.info() # basic descr
hr = hr.drop('employee_id', axis=1)
hr['is_promoted'].value_counts()
sns.countplot(hr['is_promoted'])
hr.isnull().sum()
hr['education'].value_counts()
hr['previous_year_rating'].value_counts()
labels = hr['is_promoted'].copy()

hr = hr.drop('is_promoted', axis=1)
# let's impute the missing values with mode and median value for now.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
num_pipline = Pipeline([  # create pipelines for feature transformations

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())

])
cat_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('encoder', OneHotEncoder())

])
num_attribs = list(hr.select_dtypes(include=np.number))

cat_attribs = list(hr.select_dtypes(exclude=np.number))
full_pipeline = ColumnTransformer([

    ('num_attribs', num_pipline, num_attribs),

    ('cat_attribs', cat_pipeline, cat_attribs)

])
hr_prepared = full_pipeline.fit_transform(hr)
hr_prepared.shape
# hr_prepared = pd.DataFrame(hr_prepared, columns=list(hr))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(hr_prepared, labels, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report, f1_score # choosing confusion matrix and F1 score
log_reg = LogisticRegression()



log_reg.fit(X_train, y_train)
predicted = log_reg.predict(X_test)
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))
f1_score(y_test, predicted)
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X, y = sm.fit_resample(hr_prepared, labels)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
new_model = LogisticRegression()

new_predict = new_model.fit(X_train, y_train).predict(X_test)
print(confusion_matrix(y_test, new_predict))
print(classification_report(y_test, new_predict))
f1_score(y_test, new_predict)
hr.isnull().sum()
cat_hr = bckup.copy()



education_mode = cat_hr['education'].mode()[0]

pyr_median = cat_hr['previous_year_rating'].median()



cat_hr['education'].fillna(education_mode, inplace=True)

cat_hr['previous_year_rating'].fillna(pyr_median, inplace=True)
# cat_hr = bckup.dropna(how='any')
cat_hr.isnull().sum().sum()
cat_hr.drop('employee_id', axis=1, inplace=True)

# cat_hr.head()
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTENC
sm = SMOTENC(categorical_features=[0, 1, 2, 3, 4]) # categorical feature column index are given as input
X = cat_hr.drop('is_promoted', axis=1)

y = cat_hr['is_promoted'].copy()



X, y = sm.fit_resample(X, y)
cat_hr.columns
X = pd.DataFrame(X, columns=['department', 'region', 'education', 'gender', 'recruitment_channel',

       'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service',

       'KPIs_met >80%', 'awards_won?', 'avg_training_score'])
np.unique(y, return_counts=True)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
new_cbr = CatBoostClassifier(verbose=400, eval_metric='F1')
new_cbr.fit(X_train, y_train,cat_features=[0,1,2,3,4],eval_set=(X_test, y_test),plot=True) #index of cat. features are mentioned
test_data = pd.read_csv(r'test_2umaH9m.csv') 

# test_data.head()
# test_data_prepared = full_pipeline.fit_transform(test_data)
test_data.isnull().sum()
test_data['education'].fillna(test_data['education'].mode()[0], inplace=True)

test_data['previous_year_rating'].fillna(test_data['previous_year_rating'].median(), inplace=True)
final = test_data.drop('employee_id', axis=1)
cbr_predicted = new_cbr.predict(final)



cbr_predicted = pd.DataFrame(cbr_predicted, columns=['is_promoted'])



df = pd.concat([test_data['employee_id'], cbr_predicted], axis=1)
df.to_csv(r'C:\Users\gokul\Downloads\results.csv')