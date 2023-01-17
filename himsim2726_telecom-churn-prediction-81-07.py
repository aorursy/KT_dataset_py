!pip install sweetviz
import numpy as np
import pandas as pd
import sweetviz as sv
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.head()
data.shape
data.isnull().sum()  # Checking the number values missing values in each feature 
sns.heatmap(data.isnull(),cmap = 'magma' )
eda = sv.analyze([data, "data"],target_feat='Churn')
eda.show_html('Report.html')
data.drop('customerID', axis = 1, inplace = True)  
from sklearn.preprocessing import LabelEncoder                 # Converting categorical churn column into numerical
le = LabelEncoder()
data['Churn'] = le.fit_transform(data.Churn)
df = pd.get_dummies(data)                           
X = df.drop('Churn', axis = 1)                # Defining X and y variables
y = df['Churn'] 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
lm = LogisticRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

lm_accuracy = round(lm.score(X_test, y_test) * 100, 2)
print('Test Accuracy: ', lm_accuracy)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

knn_accuracy = round(knn.score(X_test, y_test) * 100, 2)
print('Test Accuracy: ', knn_accuracy)
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

svc_accuracy = round(svc.score(X_test, y_test) * 100, 2)
print('Test Accuracy: ', svc_accuracy)
dt = DecisionTreeClassifier(max_depth = 5)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

dt_accuracy = round(dt.score(X_test, y_test) * 100, 2)
print('Test Accuracy: ', dt_accuracy)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rf_accuracy = round(rf.score(X_test, y_test) * 100, 2)
print('Test Accuracy: ', rf_accuracy)
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

xgb_accuracy = round(xgb.score(X_test, y_test) * 100, 2)
print('Test Accuracy: ', xgb_accuracy)