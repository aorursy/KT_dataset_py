# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from pandas.plotting import scatter_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
df.info()
df.describe()
sns.pairplot(data=df)
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
sns.distplot(df['CGPA'], kde=False, color='r')
sns.distplot(df['GRE Score'], kde=False, color='r')
sns.distplot(df['TOEFL Score'], kde=False, color='r')
df.columns
#cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
plt.figure(figsize=(10,10))
sns.scatterplot(x = 'CGPA', y = 'Chance of Admit ', data=df, hue = 'University Rating', size = 'University Rating')
fig = sns.regplot(x="GRE Score", y="CGPA", data=df)
plt.title("GRE Score vs CGPA")
plt.show()
sns.regplot(x='GRE Score', y = 'Chance of Admit ', data=df)
sns.lmplot(x='GRE Score', y = 'Chance of Admit ', hue="Research",data=df)
sns.regplot(x='LOR ', y = 'Chance of Admit ', data=df)
sns.lmplot(x='CGPA', y = 'Chance of Admit ', data=df, hue="SOP")
X =  df.drop(['Chance of Admit '], axis=1)
y = df['Chance of Admit ']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False, random_state = 1)

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
linreg = LinearRegression()
linreg.fit(X_train, y_train)
linpreds = linreg.predict(X_test)
print (np.sqrt(mean_squared_error(y_test, linpreds)))
k_range = range(1, 15)
rmse = []
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores = np.sqrt(mean_squared_error(y_test, knn.predict(X_test)))
    print(k, scores)
    rmse.append(scores)
min(rmse)
# plot to see clearly
plt.plot(k_range, rmse)
plt.xlabel('Value of K for KNN')
plt.ylabel('RMSE')
plt.show()
knn =  KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)
score = np.sqrt(mean_squared_error(y_test, knn.predict(X_test)))
print (score)
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
print (grid_search.best_params_)

best_grid = grid_search.best_estimator_

model = RandomForestRegressor(bootstrap= True, max_depth= 100, max_features= 3, min_samples_leaf= 4, min_samples_split=12)
model.fit(X_train, y_train)
print(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
feature_names = X.columns
importance_frame = pd.DataFrame()
importance_frame['Features'] = X.columns
importance_frame['Importance'] = model.feature_importances_
importance_frame = importance_frame.sort_values(by=['Importance'], ascending=False)
importance_frame
