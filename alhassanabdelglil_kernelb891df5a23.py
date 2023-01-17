import numpy as np
import pandas as pd
dataset = pd.read_csv('../input/Dataset_for_Classification.csv')
dataset.head()
dataset.describe()
dataset.isnull().sum()
y = dataset.loc[:,['Attrition']]
y.describe()
dataset.drop(['Attrition'],inplace=True,axis = 1)
X = dataset.iloc[:,:]
X.head()
#X_cat = X.loc[:,['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime',]]
X = pd.get_dummies(X,drop_first=True)
X.head()
y = pd.get_dummies(y,drop_first=True)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
#sc_y = StandardScaler()
#y = sc_y.fit_transform(y)
print(pd.DataFrame(y))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state = 0)
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 3, random_state = 0)
regressor.fit(X_train, y_train)
# Predicting a new result
y_pred = regressor.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
y=pd.DataFrame(y_pred)
y.to_csv('out.csv',index=False,header=False)

print('done')
