pwd
import warnings

warnings.filterwarnings('ignore')

import os

import pandas as pd

from pandas import DataFrame

import pylab as pl

import numpy as np

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt

%matplotlib inline
Fuel_cons=pd.read_csv("../input/petrol_consumption.csv") #Importing Data
Fuel_cons.head()
print(Fuel_cons.shape)

Fuel_cons.info()
pd.options.display.float_format = '{:.4f}'.format

data_summary=Fuel_cons.describe()

data_summary.T
for k, v in Fuel_cons.items():

    q1 = v.quantile(0.25)

    q3 = v.quantile(0.75)

    irq = q3 - q1

    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]

    perc = np.shape(v_col)[0] * 100.0 / np.shape(Fuel_cons)[0]

    print("Column %s outliers = %.2f%%" % (k, perc))
plt.figure(figsize=(12,5))

Fuel_cons.boxplot(patch_artist=True,vert=False)
my_corr=Fuel_cons.corr()

my_corr
plt.figure(figsize=(12,5))

sns.heatmap(my_corr,linewidth=0.5)

plt.show()
pearson_coef, p_value = stats.pearsonr(Fuel_cons['Petrol_tax'], Fuel_cons['Petrol_Consumption'])

print("The Pearson Correlation Coefficient of Petrol_tax is", pearson_coef, " with a P-value of P =", p_value)  

sns.regplot(x="Petrol_tax", y="Petrol_Consumption", data=Fuel_cons)

plt.ylim(0,)
pearson_coef, p_value = stats.pearsonr(Fuel_cons['Average_income'], Fuel_cons['Petrol_Consumption'])

print("The Pearson Correlation Coefficient of Petrol_tax is", pearson_coef, " with a P-value of P =", p_value)  

sns.regplot(x="Average_income", y="Petrol_Consumption", data=Fuel_cons)

plt.ylim(0,)
pearson_coef, p_value = stats.pearsonr(Fuel_cons['Paved_Highways'], Fuel_cons['Petrol_Consumption'])

print("The Pearson Correlation Coefficient of Petrol_tax is", pearson_coef, " with a P-value of P =", p_value)  

sns.regplot(x="Paved_Highways", y="Petrol_Consumption", data=Fuel_cons)

plt.ylim(0,)
pearson_coef, p_value = stats.pearsonr(Fuel_cons['Population_Driver_licence(%)'], Fuel_cons['Petrol_Consumption'])

print("The Pearson Correlation Coefficient of Petrol_tax is", pearson_coef, " with a P-value of P =", p_value)  

sns.regplot(x="Population_Driver_licence(%)", y="Petrol_Consumption", data=Fuel_cons)

plt.ylim(0,)
plt.figure(figsize=(12,5))

sns.distplot(Fuel_cons['Petrol_tax'])
plt.figure(figsize=(12,5))

sns.distplot(Fuel_cons['Paved_Highways'])
plt.figure(figsize=(12,5))

sns.distplot(Fuel_cons['Average_income'])
plt.figure(figsize=(12,5))

sns.distplot(Fuel_cons['Petrol_Consumption'])
plt.figure(figsize=(12,5))

sns.distplot(Fuel_cons['Population_Driver_licence(%)'])
a = sns.FacetGrid(Fuel_cons, hue = 'Petrol_Consumption', aspect=4 )

a.map(sns.kdeplot, 'Petrol_tax', shade= True )

a.set(xlim=(0 ,Fuel_cons['Petrol_tax'].max()))

a.add_legend()
axes = sns.factorplot('Petrol_tax','Paved_Highways',data=Fuel_cons, aspect = 2.5, )
predictor_var= Fuel_cons[['Petrol_tax','Average_income','Paved_Highways','Population_Driver_licence(%)']] #all columns except the last one

target_var= Fuel_cons['Petrol_Consumption'] #only the last column
predictor_var.shape
target_var.shape
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(predictor_var,target_var, test_size=0.30, random_state=123)
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=4,max_features=4)
tree.fit(X_train, Y_train)
predictions = tree.predict(X_test)
df=pd.DataFrame({'Actual':Y_test, 'Predicted':predictions})

df.head(5)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test,predictions))

print('Mean Squared Error:', metrics.mean_squared_error(Y_test,predictions))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test,predictions)))

print('r2_score:', metrics.r2_score(Y_test,predictions))
tree.feature_importances_

pd.Series(tree.feature_importances_,index=predictor_var.columns).sort_values(ascending=False)
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree, filled=True, rounded=True, feature_names=predictor_var.columns, out_file=None)
import graphviz
graphviz.Source(dot_data)
from sklearn.model_selection import GridSearchCV
param_grid = [{"max_depth":[3,4,5, None], "max_features":[3,4,5,6,7]}]
gs = GridSearchCV(estimator=DecisionTreeRegressor(random_state=123),param_grid = param_grid,cv=10)
gs.fit(X_train, Y_train)
gs.cv_results_['params']
gs.cv_results_['rank_test_score']
gs.best_estimator_
tree = DecisionTreeRegressor(max_depth=3,max_features=4)
tree.fit(X_train, Y_train)
predictions = tree.predict(X_test)
df=pd.DataFrame({'Actual':Y_test, 'Predicted':predictions})

df.head(5) #Check the top 5 predictions and actual values.
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test,predictions))

print('Mean Squared Error:', metrics.mean_squared_error(Y_test,predictions))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test,predictions)))

print('r2_score:', metrics.r2_score(Y_test,predictions))
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree, filled=True, rounded=True, feature_names=predictor_var.columns, out_file=None)
import graphviz
graphviz.Source(dot_data)
DT_Regressor=[['Max_Depth',4,3],['Max_Feature',4,4],['Mean Abs. Error',106.73,96.6],['Mean Square Error',18466.34,15143.73],['Root Mean Square',135.89,123.05],['r2_Score',0.20,0.344]]

Result_Summary2= pd.DataFrame(DT_Regressor, columns = ['Parameters','Without Grid Search','With Grid Search'])

Result_Summary2