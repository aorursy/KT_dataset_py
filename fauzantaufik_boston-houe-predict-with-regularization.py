import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import sklearn as sk
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error



data_latih = pd.read_csv('../input/boston_train.csv', sep=',', index_col = ['ID'])
target = data_latih['medv']
print("Search For Missing Values")
print(data_latih.isnull().sum())
data_latih = data_latih.drop(['medv'], axis=1)
original_data = data_latih
print(data_latih.head())

print(data_latih.describe())
print(data_latih.shape)
plt.figure(figsize=(7,7))
sns.heatmap(data_latih.corr(), square=True)
plt.show()
print(original_data[['tax', 'rad']].head(8))
rata2=target.mean()
col = []
for n in target[:] :
    if n>=rata2 :
        col.append('green')
    else :
        col.append('red')

col = pd.Series(col)
pd.plotting.scatter_matrix(original_data, c = col, figsize=(30,30), marker='.')
plt.show()
## Linier Regression
linreg = LinearRegression(normalize=True)
linreg.fit(original_data, target)
r2 = linreg.score(original_data, target)

#importing Test Data
data_test = pd.read_csv('../input/boston_train.csv', sep=',', index_col = ['ID'])
test_target = data_test['medv']
data_test=data_test.drop(['medv'], axis=1)

y_pred=linreg.predict(data_test)
mse = mean_squared_error(test_target, y_pred)
rmse=np.sqrt(mse)
print(r2)
print(rmse)
#Lasso Regression
alphas=np.linspace(0.001, 1, 100)
lass = Lasso(normalize=True)
lasso_scores = []
lasso_scores_std = []
rmse=[]
for alpha in alphas:

    # Specify the alpha value to use: ridge.alpha
    lass.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    lasso_cv_scores = cross_val_score(lass, original_data, target, cv=5)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    lasso_scores.append(np.mean(lasso_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    lasso_scores_std.append(np.std(lasso_cv_scores))
    

#Display Plot Function
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alphas, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alphas[0], alphas[-1]])
    ax.set_xscale('log')
    plt.show()
    
# Display the plot
display_plot(lasso_scores, lasso_scores_std)


model = Lasso(alpha=0.001, normalize=True)
model.fit(original_data, target)
r2=model.score(original_data, target)
test_pred=model.predict(data_test)
rmse=np.sqrt(mean_squared_error(test_target, test_pred))
coefs = model.coef_
intercept = model.intercept_
print("Intercept : {}".format(intercept))
print("RMSE : {}".format(rmse))
print("R2 : {}".format(r2))

plt.figure(figsize=(7,5))
plt.plot(range(len(original_data.columns)), coefs)
plt.xticks(range(len(original_data.columns)), original_data.columns, rotation=60)
plt.margins(0.02)
plt.show()
