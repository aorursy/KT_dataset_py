import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv(r"../input/advertising-data/Advertising.csv",index_col=0,header=0)

data.head()
print(data.dtypes,"\n")

print(data.shape,"\n")
#Assumption 1 - No outliers in the data

#can be checked using box plot

f,axes=plt.subplots(2,2,figsize=(7,7))

sns.boxplot(y="TV",data=data,ax=axes[0,0],color="orange")

sns.boxplot(y="Radio",data=data,ax=axes[0,1],color="pink")

sns.boxplot(y="Newspaper",data=data,ax=axes[1,0],color="grey")

sns.boxplot(y="Sales",data=data,ax=axes[1,1],color="brown")
#Assumption 2 (Assumption of Linearity):

#Every independent variable should have a linear relationship with dependent variable



sns.pairplot(data,x_vars=["TV","Radio","Newspaper"],y_vars="Sales",kind="reg")
#Assumption 3:

#The dependent variable should follow an approx. normal distribution

#can be checked using a distplot



sns.distplot(data.Sales,bins=10,color="green",hist=True)
#Assumption 4-There should no multi collinearity in the data

#Can be checked using correlation and VIF(Variance inflation factor)

#Relationship between the independent variables should not exsit



ind = data[["TV","Radio","Newspaper"]]



corr_df = ind.corr(method = "pearson")

print(corr_df,"\n")



sns.heatmap(corr_df,vmax =1.0,vmin=-1.0,annot=True)

#VIF - 1/(1-R^2)

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif



vif_df = pd.DataFrame()

vif_df["features"] = ind.columns

vif_df["VIF Factor"] = [vif(ind.values, i) for i in range(ind.shape[1])]

vif_df.round(2)
data.info()

#No Null's in the data
data.describe()
data.boxplot(column="Newspaper")
data.hist(bins = 20,figsize=(10,7))
#Transforming data based on Skewness



from scipy.stats import skew

data_num_skew = data.apply(lambda x: skew(x.dropna()))

data_num_skewed = data_num_skew[(data_num_skew > .75) | (data_num_skew < -.75)]



print(data_num_skew,"\n")

print(data_num_skewed)



import numpy as np

data[data_num_skewed.index] = np.log1p(data[data_num_skewed.index])

data.hist(bins=20)
df = data.copy()

df.shape
#Creating x and Y:

X=data[["TV","Radio","Newspaper"]]

Y=data["Sales"]

print(X.head(),"\n")

print(Y.head())
from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.2,random_state=10)
print(X_train.shape,"\n")

print(Y_train.shape,"\n")

print(X_test.shape,"\n")

print(Y_test.shape)
from sklearn.linear_model import LinearRegression



lm = LinearRegression()



lm.fit(X_train,Y_train)



print(lm.intercept_,"\n")

print(list(zip(X.columns,lm.coef_)))
#Prediction

Y_pred = lm.predict(X_test)

print(Y_pred)
#EVALUATING model manually, just to cross check sice the data is very small

#by comparing the actual Y values and predicted Y values



new_df = pd.DataFrame()

new_df =X_test



new_df["Actual Sales"] =Y_test

new_df["Predicted Sales"] = Y_pred

new_df
from sklearn.metrics import r2_score,mean_squared_error



import numpy as np



r2 = r2_score(Y_test,Y_pred)

print("R^2 = ",r2,"\n")



adjusted_r_squared = 1 - (1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1) #formula for adjusted R2

print("Adjusted R^2 = ",adjusted_r_squared,"\n")



rmse = np.sqrt(mean_squared_error(Y_test,Y_pred))#sqrt(SME) = RMSE

print("RMSE = ",rmse,"\n")



#Checking whether RMSE is a good value or not



per=str(round(((rmse/max(Y_test))*100),2)) + "%"

print("Y_min = ",min(Y_test),",  Y_max = ",max(Y_test), ", Percentage of Errors :", per)
new_df = pd.DataFrame()

new_df = X_train



new_df["Sales"] = Y_train

new_df.shape
import statsmodels.formula.api as sm



lm_model = sm.ols(formula="Sales ~ TV + Radio + Newspaper",data=new_df).fit()



print(lm_model.params,"\n")

print(lm_model.summary())



#Since p-value of Newspaper > 0.05, we can drop it
Y_pred_new = lm_model.predict(X_test)



from sklearn.metrics import r2_score,mean_squared_error

import numpy as np



r2 = r2_score(Y_test,Y_pred_new)

print("R^2 :",r2,"\n")#good value of R2(towards 1)



rmse = np.sqrt(mean_squared_error(Y_test,Y_pred_new))#sqrt(SME) => RMSE

print("RMSE :", rmse,"\n")



adjusted_r_squared = 1 - (1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)#formula for adjusted R2

print("Adjusted R^2 :",adjusted_r_squared)
import statsmodels.formula.api as sm



lm_model = sm.ols(formula="Sales ~ TV + Radio",data=new_df).fit()



print(lm_model.params,"\n")

print(lm_model.summary())
Y_pred_new = lm_model.predict(X_test)



from sklearn.metrics import r2_score,mean_squared_error

import numpy as np



r2 = r2_score(Y_test,Y_pred_new)

print("R^2 :",r2,"\n")#good value of R2(towards 1)



rmse = np.sqrt(mean_squared_error(Y_test,Y_pred_new))#sqrt(SME) => RMSE

print("RMSE :", rmse,"\n")



adjusted_r_squared = 1 - (1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)#formula for adjusted R2

print("Adjusted R^2 :",adjusted_r_squared)
#Assumption 5 : No Auto Correlation in the data

#Can be checked by Durbin Watson Test



#Threshold - [0,4]

#close to 2 - no auto correlation

#close to 0 - positive auto correlation

#close to 4 - negative auto correlation



#From the above Summary DWT = 2.1 (No Auto Correlation)
#Assumption 6 :Errors should be random 

#Can be checked using Residuals(Errors) Vs. Fitted values(actual) plot



plot_lm_1 = plt.figure(1)

plot_lm_1.set_figheight(8)

plot_lm_1.set_figwidth(12)



model_fitted_y = lm_model.fittedvalues



plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'Sales', data=new_df, lowess=True)



plot_lm_1.axes[0].set_title('Residuals vs Fitted')

plot_lm_1.axes[0].set_xlabel('Fitted values')

plot_lm_1.axes[0].set_ylabel('Residuals')
#Assumption 7: Errors should follow an approximate normal distribution

#can be checked using the normal Quantile-Quantile plot (normal qq plot)





res = lm_model.resid

import statsmodels.api as stm

import scipy.stats as stats

fig = stm.qqplot(res, fit=True, line='45')

plt.title('Normal Q-Q')

plt.xlabel('Theoretical Quantiles')

plt.ylabel('Standardized Residuals')

plt.show() 
#Assumption 8 : Errors should follow a constant variance (Homoskedasticity)

#Can be checked using the scale location plot



# normalized residuals

model_norm_residuals = lm_model.get_influence().resid_studentized_internal

# absolute squared normalized residuals

model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))



plot_lm_3 = plt.figure(3)

plot_lm_3.set_figheight(8)

plot_lm_3.set_figwidth(12)

plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)

sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, lowess=True)





plot_lm_3.axes[0].set_title('Scale-Location')

plot_lm_3.axes[0].set_xlabel('Fitted values')

plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$')
X=df[["TV","Radio","Newspaper"]]

Y=df["Sales"]

print(X.head(),"\n")

print(Y.head())



#Newspaper variable is considered just to understand Ridge Model
from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.2,random_state =10)
print(X_train.shape,"\n")

print(X_test.shape,"\n")

print(Y_train.shape,"\n")

print(Y_test.shape,"\n")
from sklearn.linear_model import Ridge



lm = Ridge()

lm.fit(X_train,Y_train)



print(lm.intercept_,"\n")

print(lm.coef_)

from sklearn.linear_model import Lasso



lm = Lasso()

lm.fit(X_train,Y_train)



print(lm.intercept_,"\n")

print(lm.coef_)

test_sales = X_test.copy()



test_sales["Sales"] = Y_pred

test_sales["Predicted_Sales"] = round(test_sales.Sales,2)

test_sales.drop("Sales",axis=1,inplace=True)

test_sales.head()