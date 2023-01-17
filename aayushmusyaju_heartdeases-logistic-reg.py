import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
heart = pd.read_csv('/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv')
heart.head()
sns.heatmap(heart.isnull())
sns.heatmap(heart.corr(), cmap='coolwarm')
sns.heatmap(heart[['diabetes', 'glucose']].corr(), )
diab = heart[heart['diabetes'] == 1]['glucose'].mean()
no_diab = heart[heart['diabetes'] == 0]['glucose'].mean()
def fill(cols):
    if pd.isnull(cols[1]):
        if cols[0] == 1:
            return diab
        else:
            return no_diab
    else:
        return cols[1]
            
heart['glucose'] = heart[['diabetes', 'glucose']].apply(fill, axis=1)
sns.heatmap(heart.isnull())
heart.dropna(axis=0, inplace=True)
sns.heatmap(heart.isnull())
logmodel= LogisticRegression()
X = heart.drop('TenYearCHD', axis=1)
y = heart['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=102)

X_train_scale = preprocessing.scale(X_train)
X_test_scale = preprocessing.scale(X_test)
logmodel.fit(X_train_scale, y_train)

predictions = logmodel.predict(X_test_scale)
coefficient = pd.DataFrame(np.transpose(logmodel.coef_), X_train.columns, columns=['Coefficient'])
print('classification Report:')
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))