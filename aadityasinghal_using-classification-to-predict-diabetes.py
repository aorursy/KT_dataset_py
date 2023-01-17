import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
dataset = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
dataset.head()
dataset.isnull().sum()
sns.heatmap(dataset.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
dataset.info()
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values
imputer = SimpleImputer(missing_values=0, strategy='mean')
imputer.fit(X[:, 1:6])
X[:, 1:6] = imputer.transform(X[:, 1:6])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred_test = classifier.predict(X_test)
print('Confusion Matrix :')
print(confusion_matrix(y_test, y_pred_test)) 
print('Accuracy Score :',accuracy_score(y_test, y_pred_test))
print('Report : ')
print(classification_report(y_test, y_pred_test))