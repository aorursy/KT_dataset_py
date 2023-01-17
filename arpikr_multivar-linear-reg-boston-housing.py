import os
import pandas as pd
from pandas import DataFrame
import pylab as pl
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline
boston=pd.read_csv("../input/boston.csv") #Importing Data
pd.options.display.float_format = '{:.2f}'.format
print(boston.shape)     #Checking rows and Column
print(boston.head())    #Checking the data firts 5 Columns 
boston.info()  #Checking for Nulls and Datatypes of each Predictors
pd.options.display.float_format = '{:.4f}'.format
boston.describe()
plt.figure(figsize=(15,10))
boston.boxplot(patch_artist=True,vert=False)
for k, v in boston.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(boston)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))
pd.options.display.float_format = '{:.4f}'.format
my_corr=boston.corr()
my_corr
plt.figure(figsize=(15,10))
sns.heatmap(my_corr,linewidth=0.5)
plt.show()
pearson_coef, p_value = stats.pearsonr(boston['CRIM'], boston['MV'])
print("The Pearson Correlation Coefficient of CRIM is", pearson_coef, " with a P-value of P =", p_value)  
sns.regplot(x="CRIM", y="MV", data=boston)
plt.ylim(0,)
sns.distplot(boston['CRIM'])       # We can see that CRIM data is Positevely Skewed and also has an outliers
pearson_coef, p_value = stats.pearsonr(boston['INDUS'], boston['MV'])
print("The Pearson Correlation Coefficient of INDUS is", pearson_coef, " with a P-value of P =", p_value)  
sns.regplot(x="INDUS", y="MV", data=boston)
plt.ylim(0,)
sns.distplot(boston['INDUS'])                  # this predictor shows Multimodal Distribution  
pearson_coef, p_value = stats.pearsonr(boston['NOX'], boston['MV'])
print("The Pearson Correlation Coefficient of NOX is", pearson_coef, " with a P-value of P =", p_value) 
sns.regplot(x="NOX", y="MV", data=boston)
plt.ylim(0,)
sns.distplot(boston['NOX'])     
pearson_coef, p_value = stats.pearsonr(boston['RM'], boston['MV'])
print("The Pearson Correlation Coefficient of RM is", pearson_coef, " with a P-value of P =", p_value) 
sns.regplot(x="RM", y="MV", data=boston)
plt.ylim(0,)
sns.distplot(boston['RM'])          # This distribution seems Normally distributed 
pearson_coef, p_value = stats.pearsonr(boston['AGE'], boston['MV'])
print("The Pearson Correlation Coefficient of AGE is", pearson_coef, " with a P-value of P =", p_value) 
sns.regplot(x="AGE", y="MV", data=boston)
plt.ylim(0,)
sns.distplot(boston['AGE'])
pearson_coef, p_value = stats.pearsonr(boston['DIS'], boston['MV'])
print("The Pearson Correlation Coefficient of DIS is", pearson_coef, " with a P-value of P =", p_value) 
sns.regplot(x="DIS", y="MV", data=boston)
plt.ylim(0,)
sns.distplot(boston['DIS'])
pearson_coef, p_value = stats.pearsonr(boston['TAX'], boston['MV'])
print("The Pearson Correlation Coefficient of TAX is", pearson_coef, " with a P-value of P =", p_value)  
sns.regplot(x="TAX", y="MV", data=boston)
plt.ylim(0,)
sns.distplot(boston['TAX'])    # Multimodal Distribution
pearson_coef, p_value = stats.pearsonr(boston['PT'], boston['MV'])
print("The Pearson Correlation Coefficient of PT is", pearson_coef, " with a P-value of P =", p_value) 
sns.regplot(x="PT", y="MV", data=boston)
plt.ylim(0,)
sns.distplot(boston['PT'])
pearson_coef, p_value = stats.pearsonr(boston['B'], boston['MV'])
print("The Pearson Correlation Coefficient of B is", pearson_coef, " with a P-value of P =", p_value)  
sns.regplot(x="B", y="MV", data=boston)
plt.ylim(0,)
sns.distplot(boston['B'])
sns.distplot(boston['MV'])     # Distribution of Target Variable 
Corr_result=[['CRIM',-0.3883046116575091,1.1739862423663313e-19], ['INDUS',-0.4837251712814335,4.900242319351878e-31], ['NOX',-0.4273207763683765,7.06503408465202e-24], ['RM',0.695359937127267,2.4872456897496148e-74], ['AGE',-0.37695456714288655,1.5699814570835983e-18], ['DIS',0.24992873873512159,1.206610952424503e-08], ['TAX',-0.46853593528654547,5.637730675534297e-29], ['PT',-0.5077867038116088,1.609499278902899e-34], ['B',0.3334608226834166,1.3181119682130765e-14]]
Pearson_Pvalue = pd.DataFrame(Corr_result, columns = ['Predictors', 'pearson_Correlation','P-value'])
pd.options.display.float_format = '{:.5f}'.format
Pearson_Pvalue 
#Variance Inflation Factor

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
X = add_constant(boston)
vifs = [vif(X.values, i) for i in range(len(X.columns))]
pd.Series(data=vifs, index=X.columns).sort_values(ascending=False)
boston.columns
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(boston.drop("MV", axis=1), boston['MV'], test_size = 0.2,\
                                                    random_state=112)
from sklearn.linear_model import LinearRegression
my_model = LinearRegression(normalize=True)  #Create an object of LinearRegression class.
my_model.fit(X_train, Y_train)               #Fitting the linear regression model to our training set.
predictions = my_model.predict(X_test)       #Make predictions on the test set
pd.DataFrame({'actual value': Y_test, 'predictions':predictions}).sample(5)   #Compare a sample of 5 actual Y values from test set and corresponding predicted values 
my_model.score(X_test, Y_test)           #Check the  R2  value
my_model.coef_
my_model.intercept_
from sklearn import metrics
print('MAE',metrics.mean_absolute_error(Y_test,predictions))
print('MSE',metrics.mean_squared_error(Y_test,predictions))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))
plt.scatter(Y_test,predictions)
plt.xlabel('Y_test')
plt.ylabel('Predicted_Y')
import statsmodels.api as sm         #Import statsmodels API
from sklearn.model_selection import train_test_split  #Divide the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(boston.drop("MV", axis=1), boston['MV'], test_size = 0.2,\
                                                    random_state=112)
X_train = sm.add_constant(X_train)   #Add the constant term to the training data
my_model = sm.OLS(Y_train, X_train)  #Fit the OLS model
result = my_model.fit()
print(result.summary()) 
from sklearn.metrics import r2_score
predictions = result.predict(sm.add_constant(X_test))
r2_score(Y_test, predictions)
print('MAE',metrics.mean_absolute_error(Y_test,predictions))
print('MSE',metrics.mean_squared_error(Y_test,predictions))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))
plt.scatter(Y_test,predictions)
plt.xlabel('Y_test')
plt.ylabel('Predicted_Y')
X_train = X_train.drop("TAX", axis=1)
updated_model_result = sm.OLS(Y_train, X_train).fit()
print(updated_model_result.summary())
from sklearn.metrics import r2_score
X_test = X_test.drop("TAX", axis=1)
predictions = updated_model_result.predict(sm.add_constant(X_test))
r2_score(Y_test, predictions)
print('MAE',metrics.mean_absolute_error(Y_test,predictions))
print('MSE',metrics.mean_squared_error(Y_test,predictions))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))
plt.scatter(Y_test,predictions)
plt.xlabel('Y_test')
plt.ylabel('Predicted_Y')
X_train = X_train.drop("INDUS", axis=1)
updated_model_result1 = sm.OLS(Y_train, X_train).fit()
print(updated_model_result1.summary())
X_test = X_test.drop("INDUS", axis=1)
predictions = updated_model_result1.predict(sm.add_constant(X_test))
r2_score(Y_test, predictions)
print('MAE',metrics.mean_absolute_error(Y_test,predictions))
print('MSE',metrics.mean_squared_error(Y_test,predictions))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))
# Visual inspection of Measured and Predicted
fig, ax = plt.subplots()
ax.scatter(Y_test, predictions)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('measured')
ax.set_ylabel('predicted')
plt.show()
sns.residplot(predictions,Y_test)    # Residual Plot
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
X = add_constant(boston)
vifs = [vif(X.values, i) for i in range(len(X.columns))]
pd.Series(data=vifs, index=X.columns).sort_values(ascending=False)
Model_Iterations=[['Trial 1(a)',0.7256164320858531,3.527748598537018,25.057161469613312, 5.00571288325782], ['Trial 1(b)',0.7256164320858531,3.5277485985370145,25.05716146961326, 5.005712883257814], ['Trial 2',0.7259104836332606,3.508617903749012,25.030308195709896,5.003029901540655], ['Trial 3',0.734199332192343,3.462822477847541,24.27335682897541,4.926799856801107]]
Scores = pd.DataFrame(Model_Iterations, columns = ['Iterations','R-square_Score','MAE', 'MSE','RMSE '])
pd.options.display.float_format = '{:.5f}'.format
Scores
sns.distplot(boston['CRIM'])
boston['log_CRIM'] = np.log(boston['CRIM'])

#boston.hist('log_DIS',figsize=(8,5))
#plt.title('MV vs log(CRIM)')
#plt.ylabel('MV')
#plt.xlabel("log(CRIM)")
#sns.distplot(boston['log_CRIM'])
sns.jointplot(x="log_CRIM", y="MV", data=boston, kind="reg");
boston.columns
import statsmodels.formula.api as smf
r_style_model = smf.ols('MV~INDUS+NOX+RM+AGE+DIS+TAX+PT+B+log_CRIM', data=boston)
result = r_style_model.fit()
print(result.summary())
print('MAE',metrics.mean_absolute_error(Y_test,predictions))
print('MSE',metrics.mean_squared_error(Y_test,predictions))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))
plt.scatter(Y_test,predictions)
plt.xlabel('Y_test')
plt.ylabel('Predicted_Y')
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
X = add_constant(boston)
vifs = [vif(X.values, i) for i in range(len(X.columns))]
pd.Series(data=vifs, index=X.columns).sort_values(ascending=False)
boston.drop(['TAX'],axis=1,inplace=True)
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
X = add_constant(boston)
vifs = [vif(X.values, i) for i in range(len(X.columns))]
pd.Series(data=vifs, index=X.columns).sort_values(ascending=False)
boston.drop(['INDUS'],axis=1,inplace=True)
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
X = add_constant(boston)
vifs = [vif(X.values, i) for i in range(len(X.columns))]
pd.Series(data=vifs, index=X.columns).sort_values(ascending=False)