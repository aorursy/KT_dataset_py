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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
filename = '/kaggle/input/insurance-premium-prediction/insurance.csv'

df = pd.read_csv(filename)

df.head()
df.rename(columns = {'expenses':'charges'}, inplace = True) 
df.head()
df.shape
df.info()
df.describe()
corr = df.corr()

corr
df.isnull().sum()
plt.figure(figsize=(8,6))

sns.heatmap(corr, annot = True)
fig, axes = plt.subplots(ncols = 3, figsize = (15,6), squeeze=True)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

df.plot(kind='scatter', x='age', y='charges', ax=axes[0])

df.plot(kind='scatter', x='children', y='charges', ax=axes[1])

df.plot(kind='scatter', x='bmi', y='charges', ax=axes[2])
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15,10))



df.plot(kind='hist', y='age', ax=axes[0][0], color = 'blue')

df.plot(kind='hist', y='bmi', ax=axes[0][1], color = 'orange', bins = 54)

df.plot(kind='hist', y='children', ax=axes[1][0], color = 'red', bins = 6)

df.plot(kind='hist', y='charges', ax=axes[1][1], color = 'green', bins = 80)
palette=['#EB5050','#3EA2FF']

fig, axes = plt.subplots(ncols = 3, figsize = (15,6), squeeze=True)

sns.scatterplot(x='bmi', y='charges', ax=axes[0], data=df,hue='sex', palette=palette)

sns.scatterplot(x='bmi', y='charges', ax=axes[1], data=df,hue='smoker', palette=palette)

sns.scatterplot(x='bmi', y='charges', ax=axes[2], data=df,hue='region')
fig, axes = plt.subplots(ncols=3, figsize = (15,6))

df['sex'].value_counts().plot(kind='bar', color = 'orange', ax=axes[0],title="Sex", legend = 'sex') 

df['region'].value_counts().plot(kind='bar', color = 'green', ax=axes[1],title="Region", legend = 'region')

df['smoker'].value_counts().plot(kind='bar', ax=axes[2],title="Smoker", legend = 'smoker')

palette=['#EB5050','#3EA2FF']

sns.catplot(x='sex', y='charges', kind='violin', palette=palette, data=df)
palette=['#EB5050','#2DFFAB'] 

sns.catplot(x='sex', y='charges', kind='violin', hue='smoker', palette=palette, data=df)
from scipy import stats

from scipy.stats import norm

fig =plt.figure(figsize=(18,6))

plt.subplot(1,2,1)

sns.distplot(df['charges'], fit=norm)

(mu,sigma)= norm.fit(df['charges'])

plt.legend(['For Normal dist. mean: {:.2f} | std: {:.2f}'.format(mu,sigma)])

plt.ylabel('Frequency')

plt.title('Distribution of Charges')

palette=['#EB5050','#2DFFAB'] 

sns.set(style="ticks")

sns.pairplot(data=df, hue='smoker', palette=palette)
df.head()
df.drop(["region"], axis=1, inplace=True) 

df.head()
# Changing binary categories to 1s and 0s

df['sex'] = df['sex'].map(lambda s :1  if s == 'female' else 0)

df['smoker'] = df['smoker'].map(lambda s :1  if s == 'yes' else 0)



df.head()
X = df.drop(['charges'], axis = 1)

y = df.charges

print('Shape of X: ', X.shape)

print('Shape of y: ', y.shape)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

lr = LinearRegression().fit(X_train, y_train)



y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



print(lr.score(X_test, y_test))
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

results
# Normalize the data

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
pd.DataFrame(X_train).head()
pd.DataFrame(y_train).head()
def model_summary(model, model_name, cvn=20): # Default value for cvn = 20

    print(model_name)

    y_pred_model_train = model.predict(X_train)

    y_pred_model_test = model.predict(X_test)

    accuracy_model_train = r2_score(y_train, y_pred_model_train)

    print("Training Accuracy: ", accuracy_model_train)

    accuracy_model_test = r2_score(y_test, y_pred_model_test)

    print("Testing Accuracy: ", accuracy_model_test)

    RMSE_model_train = sqrt(mean_squared_error(y_train, y_pred_model_train))

    print("RMSE for Training Data: ", RMSE_model_train)

    RMSE_model_test = sqrt(mean_squared_error(y_test, y_pred_model_test))

    print("RMSE for Testing Data: ", RMSE_model_test)

#     if model == polynomial_reg:

#         polynomial_features = PolynomialFeatures(degree=3)

#         y_pred_cv_PR = cross_val_predict(model, polynomial_features.fit_transform(X), y, cv=20)

#     else:

    y_pred_cv_model = cross_val_predict(model, X, y, cv=cvn)

    accuracy_cv_model = r2_score(y, y_pred_cv_model)

    print("Accuracy for", cvn,"- Fold Cross Predicted: ", accuracy_cv_model)
from math import sqrt 

from sklearn.model_selection import cross_val_predict  

from sklearn.metrics import r2_score, mean_squared_error  
from sklearn.linear_model import LinearRegression  



multiple_linear_reg = LinearRegression(fit_intercept=False)  

multiple_linear_reg.fit(X_train, y_train)  

model_summary(multiple_linear_reg, "Multiple_linear_Regression")
from sklearn.svm import SVR  



support_vector_reg = SVR(gamma="auto", kernel="linear", C=1000)  

support_vector_reg.fit(X_train, y_train)  

model_summary(support_vector_reg, "Support_Vector_Regressor")
from sklearn.preprocessing import PolynomialFeatures



polynomial_features = PolynomialFeatures(degree=3)  

x_train_poly = polynomial_features.fit_transform(X_train)  

x_test_poly = polynomial_features.fit_transform(X_test) 



polynomial_reg = LinearRegression(fit_intercept=False)  

polynomial_reg.fit(x_train_poly, y_train)  

print("PolynomialFeatures")

y_pred_PR_train = polynomial_reg.predict(x_train_poly)

y_pred_PR_test = polynomial_reg.predict(x_test_poly)

accuracy_PR_train = r2_score(y_train, y_pred_PR_train)

print("Training Accuracy: ", accuracy_PR_train)

accuracy_PR_test = r2_score(y_test, y_pred_PR_test)

print("Testing Accuracy: ", accuracy_PR_test)

RMSE_PR_train = sqrt(mean_squared_error(y_train, y_pred_PR_train))

print("RMSE for Training Data: ", RMSE_PR_train)

RMSE_PR_test = sqrt(mean_squared_error(y_test, y_pred_PR_test))

print("RMSE for Testing Data: ", RMSE_PR_test)

y_pred_cv_PR = cross_val_predict(polynomial_reg, polynomial_features.fit_transform(X), y, cv=20)

accuracy_cv_PR = r2_score(y, y_pred_cv_PR)

print("Accuracy for 20-Fold Cross Predicted: ", accuracy_cv_PR)
from sklearn.tree import DecisionTreeRegressor



decision_tree_reg = DecisionTreeRegressor(max_depth=5, random_state=13)  

decision_tree_reg.fit(X_train, y_train) 

model_summary(decision_tree_reg, "Decision_Tree_Regression")
import sklearn

from sklearn import tree

fig, ax = plt.subplots(figsize=(10, 10))

sklearn.tree.plot_tree(decision_tree_reg)
from sklearn.ensemble import RandomForestRegressor  



random_forest_reg = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=13)  

random_forest_reg.fit(X_train, y_train) 

model_summary(random_forest_reg, "Random_Forest_Regression")