import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
import plotly.express as px
import plotly.figure_factory as ff
import xgboost as xgb

sns.set()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

%matplotlib inline
dataset = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
dataset.sample(5)
dataset.describe().T
# Dropping Serial No. as it doesn't give any additional data.

dataset = dataset.drop('Serial No.', axis=1)
dataset.columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research','Chance of admission']
print(pd.isnull(dataset).sum())
sns.distplot(dataset['GRE Score'], bins = 10, color = 'orange', label = 'KDE')
plt.legend()
plt.gcf().set_size_inches(12, 5)
sns.boxplot(x=dataset['GRE Score'], color = 'orange')
plt.title('GRE boxplot', fontsize = 20)
plt.show()
sns.distplot(dataset['TOEFL Score'], bins = 10, color = 'blue', label = 'KDE')
plt.legend()
plt.gcf().set_size_inches(12, 5)
sns.boxplot(x=dataset['TOEFL Score'], color = 'blue')
plt.title('TOEFL boxplot', fontsize = 20)
plt.show()
hist_data = [dataset['SOP'], dataset['LOR'], dataset['CGPA']]
group_labels = [ 'SOP','LOR','CGPA']

fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.5, 0.5,0.1], colors = ['#D4323E', 'turquoise','#177415'])
fig.update_layout(title_text='SOP, LOR and CGPA')
fig.show()
sns.boxplot(x=dataset['SOP'], color = 'pink')
plt.title('SOP boxplot', fontsize = 20)
plt.show()
sns.boxplot(x=dataset['CGPA'], color = 'green')
plt.title('CGPA boxplot', fontsize = 20)
plt.show()
sns.boxplot(x=dataset['LOR'], color = 'turquoise')
plt.title('LOR boxplot', fontsize = 20)
plt.show()
LOR = dataset['LOR'] # Variable data
LOR_Q1 = LOR.quantile(0.25) # Q1 inf limit
LOR_Q3 = LOR.quantile(0.75) # Q3 sup limit
LOR_IQR = LOR_Q3 - LOR_Q1 # IQR
LOR_lowerend = LOR_Q1 - (1.5 * LOR_IQR) # q1 - 1.5 * q1
LOR_upperend = LOR_Q3 + (1.5 * LOR_IQR) # q3 + 1.5 * q3

LOR_outliers = LOR[(LOR < LOR_lowerend) | (LOR > LOR_upperend)] # Outlier index
LOR_outliers
sns.distplot(dataset['Chance of admission'], bins = 10, color = 'red', label = 'KDE')
plt.legend()
plt.gcf().set_size_inches(12, 5)
sns.boxplot(x=dataset['Chance of admission'], color = 'red')
plt.title('Chance of admission', fontsize = 20)
plt.show()
Adm = dataset['Chance of admission'] # Variable data
Adm_Q1 = Adm.quantile(0.25) # Q1 inf limit
Adm_Q3 = Adm.quantile(0.75) # Q3 sup limit
Adm_IQR = Adm_Q3 - Adm_Q1 # IQR
Adm_lowerend = Adm_Q1 - (1.5 * Adm_IQR) # q1 - 1.5 * q1
Adm_upperend = Adm_Q3 + (1.5 * Adm_IQR) # q3 + 1.5 * q3

Adm_outliers = Adm[(Adm < Adm_lowerend) | (Adm > Adm_upperend)] # Outlier index
Adm_outliers
Research = pd.DataFrame(dataset['Research'].value_counts()).reset_index()
Research.columns = ['Research','Total']
fig = px.pie(Research, values = 'Total', names = 'Research', title='Research', hole=.4, color = 'Research',width=800, height=400)
fig.show()
sns.heatmap(dataset.corr(), annot = True, linewidths=.5, cmap= 'YlGnBu')
plt.title('Correlations', fontsize = 20)
plt.gcf().set_size_inches(12, 7)
plt.show()
correlaciones = sns.pairplot(dataset, corner = True)
X = dataset.iloc[:,0:7].values 

y = dataset.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # Set Splitting

# K fold definition

kfold = KFold(n_splits = 10)
def regression_model(model):
    """
    Will fit the regression model passed and will return the regressor object and the score
    """
    regressor = model
    regressor.fit(X_train, y_train)
    score = regressor.score(X_test, y_test) # R2
    cv_results = cross_val_score(model, X_train, y_train, cv = kfold)
    cv_results = cv_results.mean()
    return regressor, score, cv_results
model_performance = pd.DataFrame(columns = ["Model", "Score","10 - fold cross val"])

models_to_evaluate = [xgb.XGBRegressor(),
                      LinearRegression(), Ridge(), SVR(kernel = 'linear'), RandomForestRegressor()]

for model in models_to_evaluate:
    regressor, score, cv_results = regression_model(model)
    model_performance = model_performance.append({"Model": model, "Score": score, "10 - fold cross val":cv_results},
                                                 ignore_index=True)

model_performance
ridge_parameters = {'alpha': [1,0.1,0.01,0.001,0.0001,0]}

ridge_Regressor = Ridge()
grid = GridSearchCV(ridge_Regressor, ridge_parameters , cv = 10)
grid.fit(X_train,y_train)
print('The parameters combination that would give best accuracy is : ')
print(grid.best_params_)
print('The best accuracy achieved after parameter tuning via grid search is : ', grid.best_score_)
XGB_parameters = {'n_estimators': [50, 100, 200],
        'subsample': [ 0.6, 0.8, 1.0],
        'max_depth': [1,2,3,4],
        'learning_rate': [0.1,0.2, 0.3, 0.4, 0.5]}

XGB_Regressor = xgb.XGBRegressor()
grid = GridSearchCV(XGB_Regressor, XGB_parameters , cv = 10)
grid.fit(X_train,y_train)
print('The parameters combination that would give best accuracy is : ')
print(grid.best_params_)
print('The best accuracy achieved after parameter tuning via grid search is : ', grid.best_score_)
regressor = Ridge(alpha=1)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.scatter(y_test, y_pred, color = 'red')
plt.xlabel('True admission chance',fontsize = 15)
plt.ylabel('Predicted admission chance', fontsize = 15)
plt.title('Ridge regressor', fontsize = 20)
plt.grid(True)
features = {'GRE Score': [300, 295, 314], 'TOEFL Score': [105, 94, 101],
                          'University Rating': [3, 3, 4], 'SOP':[3.4, 4.2, 3.1],
                          'LOR':[3.3, 2.5, 2.9], 'CGPA':[7.12, 7.99, 9],
                            'Research':[0, 1, 1]}
featuresDF = pd.DataFrame.from_dict(features)
featuresDF
Admission = regressor.predict(featuresDF)

Admission_DF = pd.DataFrame({'Admission':Admission})
joined = featuresDF.join(Admission_DF)
joined