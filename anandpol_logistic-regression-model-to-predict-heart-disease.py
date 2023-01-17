import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X)
X = imputer.transform(X)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
from sklearn.metrics import accuracy_score
print('Accuracy of my model on testing set :' ,accuracy_score(y_test,y_pred))