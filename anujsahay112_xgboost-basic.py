



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head(10)
df.columns
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age']]

y = df['Outcome']
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=7)
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("The accuracy of simple XGBoost is :",round(accuracy*100,2),"%")
from xgboost import plot_tree

from xgboost import plot_importance

plot_tree(model, rankdir = 'LR')
plot_importance(model)