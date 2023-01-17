import pandas as pd

import numpy as np
Path = '../input/pima-indians-diabetes-database/diabetes.csv'

train_df = pd.read_csv(Path)
train_df.head()
train_df.info()
train_df.describe()
#only iterates through feature columns leaving the label (Outcome)

#features = train_df.drop('Outcome', axis=1)

#for cols in features:

#    train_df[cols] = train_df[cols].replace(0,train_df[cols].mean())

train_df['Glucose'] = train_df['Glucose'].replace(0, train_df['Glucose'].mean())

train_df['BloodPressure'] = train_df['BloodPressure'].replace(0, train_df['BloodPressure'].mean())

train_df['SkinThickness'] = train_df['SkinThickness'].replace(0, train_df['SkinThickness'].mean())

train_df['Insulin'] = train_df['Insulin'].replace(0, train_df['Insulin'].mean())

train_df['BMI'] = train_df['BMI'].replace(0, train_df['BMI'].mean())
train_df.describe()
import seaborn as sb

import matplotlib.pyplot as plt
#Correlation matrix

# help from: https://www.kaggle.com/zacharywyman/titanic-top-10-beginner-tutorial/data?select=train.csv
corr_matrix = train_df.corr()

fig, ax = plt.subplots(figsize = (10,8))

sb.heatmap(corr_matrix, annot = True, fmt='.2g', vmin = -1,

            vmax = 1, center = 0, cmap = 'coolwarm')
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(train_df.drop('Outcome', axis=1), train_df['Outcome'], 

                                                  test_size = 0.2)
print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)
model = RandomForestClassifier(n_estimators=250, max_features='auto',

                              max_depth = 6, criterion='entropy')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy score of Random Forest: ", (accuracy_score(y_pred, y_test))*100)