# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
data = pd.read_csv("/kaggle/input/concrete-compressive-strength-data-set/compresive_strength_concrete.csv")
len(data)
data.head()
req_col_names = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",

                 "CoarseAggregate", "FineAggregare", "Age", "CC_Strength"]

curr_col_names = list(data.columns)



mapper = {}

for i, name in enumerate(curr_col_names):

    mapper[name] = req_col_names[i]



data = data.rename(columns=mapper)
data.head()
data.isna().sum()
sns.pairplot(data)

plt.show()
corr = data.corr()



sns.heatmap(corr, annot=True, cmap='Blues')

b, t = plt.ylim()

plt.ylim(b+0.5, t-0.5)

plt.title("Feature Correlation Heatmap")

plt.show()
ax = sns.distplot(data.CC_Strength)

ax.set_title("Compressive Strength Distribution")
fig, ax = plt.subplots(figsize=(10,7))

sns.scatterplot(y="CC_Strength", x="Cement", hue="Water", size="Age", data=data, ax=ax, sizes=(20, 200))

ax.set_title("CC Strength vs (Cement, Age, Water)")

ax.legend(loc="upper left", bbox_to_anchor=(1,1))

plt.show()
X = data.iloc[:,:-1]         # Features - All columns but last

y = data.iloc[:,-1]          # Target - Last Column
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Importing models

from sklearn.linear_model import LinearRegression, Lasso, Ridge



# Linear Regression

lr = LinearRegression()

# Lasso Regression

lasso = Lasso()

# Ridge Regression

ridge = Ridge()



# Fitting models on Training data 

lr.fit(X_train, y_train)

lasso.fit(X_train, y_train)

ridge.fit(X_train, y_train)



# Making predictions on Test data

y_pred_lr = lr.predict(X_test)

y_pred_lasso = lasso.predict(X_test)

y_pred_ridge = ridge.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")

print("""LinearRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(

            np.sqrt(mean_squared_error(y_test, y_pred_lr)),mean_squared_error(y_test, y_pred_lr),

            mean_absolute_error(y_test, y_pred_lr), r2_score(y_test, y_pred_lr)))

print("""LassoRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(

            np.sqrt(mean_squared_error(y_test, y_pred_lasso)),mean_squared_error(y_test, y_pred_lasso),

            mean_absolute_error(y_test, y_pred_lasso), r2_score(y_test, y_pred_lasso)))

print("""RidgeRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(

            np.sqrt(mean_squared_error(y_test, y_pred_ridge)),mean_squared_error(y_test, y_pred_ridge),

            mean_absolute_error(y_test, y_pred_ridge), r2_score(y_test, y_pred_ridge)))
coeff_lr = lr.coef_

coeff_lasso = lasso.coef_

coeff_ridge = ridge.coef_



labels = req_col_names[:-1]



x = np.arange(len(labels)) 

width = 0.3



fig, ax = plt.subplots(figsize=(10,6))

rects1 = ax.bar(x - 2*(width/2), coeff_lr, width, label='LR')

rects2 = ax.bar(x, coeff_lasso, width, label='Lasso')

rects3 = ax.bar(x + 2*(width/2), coeff_ridge, width, label='Ridge')



ax.set_ylabel('Coefficient')

ax.set_xlabel('Features')

ax.set_title('Feature Coefficients')

ax.set_xticks(x)

ax.set_xticklabels(labels, rotation=45)

ax.legend()



def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

autolabel(rects1)

autolabel(rects2)

autolabel(rects3)



fig.tight_layout()

plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,4))



ax1.scatter(y_pred_lr, y_test, s=20)

ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

ax1.set_ylabel("True")

ax1.set_xlabel("Predicted")

ax1.set_title("Linear Regression")



ax2.scatter(y_pred_lasso, y_test, s=20)

ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

ax2.set_ylabel("True")

ax2.set_xlabel("Predicted")

ax2.set_title("Lasso Regression")



ax3.scatter(y_pred_ridge, y_test, s=20)

ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

ax3.set_ylabel("True")

ax3.set_xlabel("Predicted")

ax3.set_title("Ridge Regression")



fig.suptitle("True vs Predicted")

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
from sklearn.tree import DecisionTreeRegressor



dtr = DecisionTreeRegressor()



dtr.fit(X_train, y_train)



y_pred_dtr = dtr.predict(X_test)



print("Model\t\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")

print("""Decision Tree Regressor \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(

            np.sqrt(mean_squared_error(y_test, y_pred_dtr)),mean_squared_error(y_test, y_pred_dtr),

            mean_absolute_error(y_test, y_pred_dtr), r2_score(y_test, y_pred_dtr)))



plt.scatter(y_test, y_pred_dtr)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

plt.xlabel("Predicted")

plt.ylabel("True")

plt.title("Decision Tree Regressor")

plt.show()
from sklearn.ensemble import RandomForestRegressor



rfr = RandomForestRegressor(n_estimators=100)



rfr.fit(X_train, y_train)



y_pred_rfr = rfr.predict(X_test)



print("Model\t\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")

print("""Random Forest Regressor \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(

            np.sqrt(mean_squared_error(y_test, y_pred_rfr)),mean_squared_error(y_test, y_pred_rfr),

            mean_absolute_error(y_test, y_pred_rfr), r2_score(y_test, y_pred_rfr)))



plt.scatter(y_test, y_pred_rfr)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

plt.xlabel("Predicted")

plt.ylabel("True")

plt.title("Decision Tree Regressor")

plt.show()
from sklearn.neural_network import MLPRegressor



mlp = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000)



mlp.fit(X_train, y_train)



y_pred_mlp = rfr.predict(X_test)



print("Model\t\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")

print("""Multi Layer Perceptron \t\t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(

            np.sqrt(mean_squared_error(y_test, y_pred_mlp)),mean_squared_error(y_test, y_pred_mlp),

            mean_absolute_error(y_test, y_pred_mlp), r2_score(y_test, y_pred_mlp)))



plt.scatter(y_test, y_pred_mlp)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

plt.xlabel("Predicted")

plt.ylabel("True")

plt.title("Decision Tree Regressor")

plt.show()
models = [lr, lasso, ridge, dtr, rfr, mlp]

names = ["Linear Regression", "Lasso Regression", "Ridge Regression", 

         "Decision Tree Regressor", "Random Forest Regressor", "Multi Layer Perceptron"]

rmses = []



for model in models:

    rmses.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))



x = np.arange(len(names)) 

width = 0.3



fig, ax = plt.subplots(figsize=(10,7))

rects = ax.bar(x, rmses, width)

ax.set_ylabel('RMSE')

ax.set_xlabel('Models')

ax.set_title('RMSE with Different Algorithms')

ax.set_xticks(x)

ax.set_xticklabels(names, rotation=45)

autolabel(rects)

fig.tight_layout()

plt.show()