import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/diabetes.csv')
df.describe()
df[df['Insulin'] == 0].shape
# calculate the average value of blood pressure
df['BloodPressure'].sum() / (df.shape[0] - df[df['BloodPressure']==0].shape[0])
# fill in the missing value with the average value
df['BloodPressure'] = df['BloodPressure'].replace(0, 72.4)
# calculate the average value of Glucose
df['Glucose'].sum() / (df.shape[0] - df[df['Glucose']==0].shape[0])
df['Glucose'] = df['Glucose'].replace(0, 121.7)
# calculate the average BMI
df['BMI'].sum() / (df.shape[0] - df[df['BMI']==0].shape[0])
df['BMI'] = df['BMI'].replace(0, 32.46)
mask = (df['Insulin'] != 0)
data = df[mask]
data.describe()
age_class = []
for _, row in data.iterrows():
    age = row['Age']
    if age < 30:
        age_class.append(2)
    elif age < 40:
        age_class.append(3)
    elif age < 50:
        age_class.append(4)
    elif age < 60:
        age_class.append(5)
    elif age < 70:
        age_class.append(6)
    else:
        age_class.append(7)
        
data['age_class'] = age_class      
data.head()
data['age_class'] = data['age_class'].astype('category')
data['Pregnancies'] = data['Pregnancies'].astype('category')
data = data.drop(['Age'], axis=1)
data.info()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
numeric_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
categorical_features = ['Pregnancies', 'age_class']
scaled = StandardScaler().fit_transform(data[numeric_features])
X_numeric = pd.DataFrame(data=scaled, columns=numeric_features, index=data.index)
X_categorical = pd.get_dummies(data[categorical_features], drop_first=True)
X = pd.concat([X_categorical, X_numeric], axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
y.value_counts()
y_test.value_counts()
logreg = LogisticRegression()
params = {'C': np.logspace(-5, 2, 8)}
cv = GridSearchCV(logreg, param_grid=params, cv=2, scoring='roc_auc')
cv.fit(X_train, y_train)
print('test score:{}'.format(cv.score(X_test, y_test)))
print('best score:{} | best parameter: {}'.format(cv.best_score_, cv.best_params_))
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
