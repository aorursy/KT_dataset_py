import os
import pandas as pd
print(os.listdir("../input"))
df = pd.read_csv('../input/winequality-red.csv')
df.head()
df.describe() # -> no missing data
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df.hist(bins=10, figsize=(20,20))
plt.show()
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr_matrix,annot=True,linewidths=.5,ax=ax)
plt.show()
corr_matrix['quality'].sort_values(ascending=False)
df_copy = df.copy()
df.drop(['free sulfur dioxide', 'residual sugar', 'pH', 'chlorides', 'fixed acidity'], axis=1, inplace=True)
df.head()
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X = df.drop('quality', axis=1)
X = scaler.fit_transform(X)
new_df = pd.DataFrame(X, columns=df.drop('quality',axis=1).columns)
new_df['quality'] = df['quality'] 
new_df.head()
plt.scatter(new_df['volatile acidity'], new_df['quality'])
plt.show()
from scipy import stats

# alcohol
slope_alcohol, intercept_alcohol, r_value, p_value, std_err = stats.linregress(new_df['alcohol'], new_df['quality'])
# volatile acidity
slope_volatile, intercept_volatile, r_value, p_value, std_err = stats.linregress(new_df['volatile acidity'], new_df['quality'])
# citric acid
slope_citric, intercept_citric, r_value, p_value, std_err = stats.linregress(new_df['citric acid'], new_df['quality'])
# total sulfur dioxide
slope_sulfur, intercept_sulfur, r_value, p_value, std_err = stats.linregress(new_df['total sulfur dioxide'], new_df['quality'])
# density
slope_density, intercept_density, r_value, p_value, std_err = stats.linregress(new_df['density'], new_df['quality'])
# sulphates
slope_sulphates, intercept_sulphates, r_value, p_value, std_err = stats.linregress(new_df['sulphates'], new_df['quality'])
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

alcohol = go.Scatter(x=new_df['alcohol'], 
                     y=slope_alcohol*new_df['alcohol']+intercept_alcohol,
                     name='alcohol')
volatile = go.Scatter(x=new_df['volatile acidity'], 
                      y=slope_volatile*new_df['volatile acidity']+intercept_volatile,
                     name='volatile acidity')
citric = go.Scatter(x=new_df['citric acid'], 
                      y=slope_citric*new_df['citric acid']+intercept_citric,
                     name='citric acid')
sulfur = go.Scatter(x=new_df['total sulfur dioxide'], 
                      y=slope_sulfur*new_df['total sulfur dioxide']+intercept_sulfur,
                     name='total sulfur dioxide')
density = go.Scatter(x=new_df['density'], 
                      y=slope_density*new_df['density']+intercept_density,
                     name='density')
sulphates = go.Scatter(x=new_df['sulphates'], 
                      y=slope_sulphates*new_df['sulphates']+intercept_sulphates,
                     name='sulphates')


layout = dict(title = 'Relationshop between features and quality',
              yaxis = dict(
                  title='quality',
                  zeroline=False,
                  gridwidth=2
              ),
              xaxis = dict(zeroline = False)
             )
data = [alcohol, volatile, citric, sulfur, density, sulphates]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
from sklearn.preprocessing import LabelEncoder
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
new_df['quality'] = pd.cut(new_df['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
new_df['quality'] = label_quality.fit_transform(new_df['quality'])
new_df['quality'].value_counts()
new_df.head()
from sklearn.model_selection import train_test_split

X = new_df.drop('quality', axis=1)
y = new_df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
print(classification_report(y_test, log_reg_pred))
from sklearn.model_selection import GridSearchCV

param_log_reg = {
    'tol':[1e-4, 1e-3],
    'C':[0.5, 1, 2],
    'solver':['newton-cg','lbfgs','liblinear','sag','saga']
}
grid_log_reg = GridSearchCV(log_reg, param_log_reg, scoring='accuracy')
grid_log_reg.fit(X_train, y_train)
grid_log_reg.best_params_
grid_log_reg_pred = grid_log_reg.predict(X_test)
print(classification_report(y_test, grid_log_reg_pred))
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log') # the ‘log’ loss gives logistic regression
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)
print(classification_report(y_test, sgd_pred))
from sklearn.ensemble import GradientBoostingClassifier

gradient = GradientBoostingClassifier()
gradient.fit(X_train, y_train)
gradient_pred = gradient.predict(X_test)
print(classification_report(y_test, gradient_pred))
param_gradient = {
    'learning_rate':[0.05,0.1,0.2],
    'n_estimators':[50,100,200,500],
    'max_depth':[1,3,5]
}
grid_gradient = GridSearchCV(gradient, param_gradient, scoring='accuracy')
grid_gradient.fit(X_train, y_train)
grid_gradient.best_params_
gradient_grid_pred = grid_gradient.predict(X_test)
print(classification_report(y_test, gradient_grid_pred))
from sklearn.model_selection import cross_val_score
grad_cross = cross_val_score(grid_gradient.best_estimator_, X_train, y_train, scoring='accuracy',n_jobs=-1,
                            cv=10)
grad_cross.mean()
