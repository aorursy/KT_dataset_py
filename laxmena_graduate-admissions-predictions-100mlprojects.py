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
data = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')

data2 = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')



dataset = pd.concat([data, data2])



dataset.sample(5)
dataset.drop(columns=['Serial No.'], axis=1, inplace=True)



X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values
print(dataset.columns)
dataset.describe()
dataset.isnull().sum()
import seaborn as sns

sns.pairplot(dataset)

plt.show()
corr = dataset.corr()

fig, ax = plt.subplots(figsize=(8, 8))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

dropSelf = np.zeros_like(corr)

dropSelf[np.triu_indices_from(dropSelf)] = True

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)

plt.title('Graduate Admissions - Features Correlations\n#100MLProjects #laxmena')

plt.show()

fig.savefig('Correlation.png')
import seaborn as sns



sns.distplot(dataset.iloc[:,0].values)
# GRE Scores

gre_median = np.median(dataset.iloc[:,0].values)

gre_mean = np.mean(dataset.iloc[:,0].values)



print("GRE Scores Summary")

print("GRE Median: ", gre_median)

print("GRE Mean: ", gre_mean)
import seaborn as sns



sns.distplot(dataset.iloc[:,1].values)
# TOEFL Scores

toefl_median = np.median(dataset.iloc[:,1].values)

toefl_mean = np.mean(dataset.iloc[:,1].values)



print("TOEFL Scores Summary")

print("TOEFL Median: ", toefl_median)

print("TOEFL Mean: ", toefl_mean)
import seaborn as sns



sns.distplot(dataset.iloc[:,-3].values)
# GRE Scores

cgpa_median = np.median(dataset.iloc[:,-3].values)

cgpa_mean = np.mean(dataset.iloc[:,-3].values)



print("CGPA Scores Summary")

print("CGPA Median: ", cgpa_median)

print("CGPA Mean: ", cgpa_mean)


threshold = 0.75

plt.scatter(X[:,0][y>threshold], y[y>threshold], color='green', label='>' + str(threshold*100)+' Chance of Admit')

plt.scatter(X[:,0][y<=threshold], y[y<=threshold], color='red', label='<' + str(threshold*100)+' Chance of Admit')

plt.xlabel('GRE Score')

plt.ylabel('Chance of Admit')

plt.title('GRE Score vs Chance of Admit')

plt.legend()

plt.show()
threshold = 0.8

plt.scatter(X[:,1][y>threshold], y[y>threshold], color='green', label='>' + str(threshold*100)+' Chance of Admit')

plt.scatter(X[:,1][y<=threshold], y[y<=threshold], color='red', label='<' + str(threshold*100)+' Chance of Admit')

plt.xlabel('TOEFL Score')

plt.ylabel('Chance of Admit')

plt.title('TOEFL Score vs Chance of Admit')

plt.legend()

plt.show()
plt.scatter(X[:,0], X[:,1], color='blue')

plt.xlabel('GRE Score')

plt.ylabel('TOEFL Score')

plt.title('GRE Score vs TOEFL Score')

plt.show()
threshold = 0.8

plt.scatter(X[:,-2][y>threshold], y[y>threshold], color='green', label='>' + str(threshold*100)+' Chance of Admit')

plt.scatter(X[:,-2][y<=threshold], y[y<=threshold], color='red', label='<' + str(threshold*100)+' Chance of Admit')

plt.xlabel('CGPA')

plt.ylabel('Chance of Admit')

plt.title('CGPA vs Chance of Admit')

plt.legend()

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



X_train, X_test, y_train, y_test = train_test_split(X, y)



def feature_scaler(X):

  sc = StandardScaler()

  X[:,:-1] = sc.fit_transform(X[:,:-1])  

  return X



X_train = feature_scaler(X_train)

X_test = feature_scaler(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression



linear_regressor = LinearRegression()

linear_regressor = linear_regressor.fit(X_train, y_train)
y_pred_lr = linear_regressor.predict(X_test)



linear_regressor_score = round(linear_regressor.score(X_test, y_test)*100, 2)

linear_regressor_mas = mean_absolute_error(y_test,y_pred_lr)

linear_regressor_rmse = np.sqrt(mean_squared_error(y_test,y_pred_lr))



print('Accuracy Score: ',linear_regressor_score,'%')

print('Mean Absolute Error: ', linear_regressor_mas)

print('RMSE: ', linear_regressor_rmse)
print('Intercept: \n', linear_regressor.intercept_)

print('Coefficients: \n', linear_regressor.coef_)
from sklearn.tree import DecisionTreeRegressor



decision_tree_regressor = DecisionTreeRegressor()

decision_tree_regressor = decision_tree_regressor.fit(X_train, y_train)
y_pred_dt = decision_tree_regressor.predict(X_test)



decision_tree_score = round(decision_tree_regressor.score(X_test, y_test)*100, 2)

decision_tree_mas = mean_absolute_error(y_test,y_pred_dt)

decision_tree_rmse = np.sqrt(mean_squared_error(y_test,y_pred_dt))



print('Accuracy: ', decision_tree_score,'%')

print('Mean Absolute Error: ', decision_tree_mas)

print('RMSE: ', decision_tree_rmse)
### Random Forest Regression Model
from sklearn.ensemble import RandomForestRegressor



random_forest_regressor = RandomForestRegressor()

random_forest_regressor = random_forest_regressor.fit(X_train, y_train)
y_pred_rf = random_forest_regressor.predict(X_test)



random_forest_score = round(random_forest_regressor.score(X_test, y_test)*100, 2)

random_forest_mas = mean_absolute_error(y_test,y_pred_rf)

random_forest_rmse = np.sqrt(mean_squared_error(y_test,y_pred_rf))



print('Accuracy: ', random_forest_score,'%')

print('Mean Absolute Error: ', random_forest_mas)

print('RMSE: ', random_forest_rmse)
from sklearn.svm import SVR



svr_regressor = SVR(kernel='linear')

svr_regressor = svr_regressor.fit(X_train, y_train)
y_pred_svr = svr_regressor.predict(X_test)



svr_score = round(svr_regressor.score(X_test, y_test)*100, 2)

svr_mas = mean_absolute_error(y_test,y_pred_svr)

svr_rmse = np.sqrt(mean_squared_error(y_test,y_pred_svr))



print('Accuracy: ', svr_score,'%')

print('Mean Absolute Error: ', svr_mas)

print('RMSE: ', svr_rmse)
df = pd.DataFrame({'Regression Model': ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression', 'SVR Model'],

                  'Accuracy Score': [linear_regressor_score, decision_tree_score, random_forest_score, svr_score],

                   'Mean Absolute Error': [linear_regressor_mas, decision_tree_mas, random_forest_mas, svr_mas],

                   'Root Mean Squared Error': [linear_regressor_rmse, decision_tree_rmse, random_forest_rmse, svr_rmse]},

                  columns= ['Regression Model', 'Accuracy Score', 'Mean Absolute Error', 'Root Mean Squared Error'])

print(df.to_markdown())
importance_frame = pd.DataFrame()

importance_frame['Importance'] = random_forest_regressor.feature_importances_

importance_frame['Features'] = dataset.columns[:-1]
plt.barh([1,2,3,4,5,6,7], importance_frame['Importance'], align='center', alpha=0.5)

plt.yticks([1,2,3,4,5,6,7], importance_frame['Features'])

plt.xlabel('Importance')

plt.title('Feature Importances')

plt.show()