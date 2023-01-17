# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head(15)
df = df.fillna(0)
print(df)
x = df.drop(columns=['status'])
print(x)
y = df['status']
print(y)
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()  
x= x.apply(label_encoder.fit_transform)
print(x)
y= label_encoder.fit_transform(y)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=109)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
#separating numerical and categorical col
numerical_col = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
categorical_col = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
#Creating Pipeline to Missing Data 

#inpute numerical missing data with median
numerical_transformer = make_pipeline(SimpleImputer(strategy='median'),
                                      StandardScaler())

#inpute categorical data with the most frequent value of the feature and make one hot encoding
categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                        OneHotEncoder(handle_unknown='ignore'))

preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_col),
                                               ('cat', categorical_transformer, categorical_col)])

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier())])
#Using GradientBoostingClassifier with GridSearchCV to get better parameters

param_grid = {'model__learning_rate':[0.001, 0.01, 0.1], 
              'model__n_estimators':[100, 150, 200, 300, 350, 400]}

#param_grid = {'model__learning_rate':[0.1], 
#              'model__n_estimators':[150]}

#use recall score
grid = GridSearchCV(clf, param_grid, cv=2, scoring='accuracy', n_jobs=-1)
grid.fit(x_train, y_train)
grid.best_params_
from sklearn.metrics import classification_report,confusion_matrix
predictions = grid.predict(x_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x="degree_t", data=df, hue='specialisation')
plt.title("Candidate degree vs  Placement")
plt.xlabel("courses")
plt.ylabel("Number of candidate")
plt.show()
df.plot.scatter(x='salary', y='mba_p',title='Candidate Performance')
df.drop(['status'], axis=1).plot.line(title='Candidate Performance')
df['salary'].plot.hist()
df['status'].value_counts().sort_index().plot.bar()