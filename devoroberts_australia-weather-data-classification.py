import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv('../input/weatherAUS.csv')
data.head()
data[data.isnull().any(axis=1)]
del data['Date']
del data['Evaporation']
del data['Sunshine']
data.head()
del data['Location']
before_rows = data.shape[0]
data = data.dropna()
after_rows = data.shape[0]
before_rows
after_rows
before_rows - after_rows
clean_data = data.copy()
clean_data['RainTomorrow'] = clean_data['RainTomorrow'].map({'No':0, 'Yes':1})
clean_data['RainToday'] = clean_data['RainToday'].map({'No':0, 'Yes':1})
clean_data.head(10)
features = ['WindSpeed9am', 'Humidity9am', 'Pressure9am', 
            'Cloud9am', 'Temp9am', 'RainToday']
X = clean_data[features].copy()
y = clean_data['RainTomorrow'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)
y_train.describe()
rain_classifier = DecisionTreeClassifier(max_leaf_nodes=8, random_state=0)
rain_classifier.fit(X_train, y_train)
predictions = rain_classifier.predict(X_test)
predictions[:10]
y_test[:10]
accuracy_score(y_true = y_test, y_pred = predictions)


