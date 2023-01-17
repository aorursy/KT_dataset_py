import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.head()
data.shape
sns.heatmap(data.isnull(),cmap = 'magma' )
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data.diagnosis) 
data.shape
data.drop('Unnamed: 32', axis = 1, inplace = True)
cor = data.corr()
sns.heatmap(cor, annot = False, center= 0)
plt.show()
from scipy.stats import zscore
z_scores = zscore(data)

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_data = data[filtered_entries]
df = pd.DataFrame(new_data)
df
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
X = df.iloc[:,:]
calculate_vif(X)
drop_cols = ['perimeter_mean','area_mean','perimeter_se','area_se','perimeter_worst','area_worst','id']
df.drop(drop_cols, axis = 1, inplace = True)
df.head()
X = df.drop('diagnosis', axis = 1)                # Defining X and y variables
y = df['diagnosis']
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