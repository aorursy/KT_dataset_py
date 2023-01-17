import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Output variable = mpg(miles per gallon)
# Input data files are available in the "../input/" directory.
auto = pd.read_csv("../input/Auto.csv")
auto_df = pd.DataFrame(auto)
print(auto_df.describe())

#there are some missing value in mpg and horsepower dilling them ith their mean
auto_df['mpg'].fillna(auto_df.mpg.mean(),inplace=True)
auto_df['horsepower'].fillna(auto_df.horsepower.mean(),inplace=True)
#pairplot 
sns.pairplot(data=auto_df,x_vars=['displacement','horsepower','weight','acceleration'],y_vars=['mpg'])

#displacement,horsepower and weight seems to follow polynomial 
# Assuming mpg > 40 as outliers 
auto_ = auto_df[auto_df['mpg'] <= 40]

#modellling

#out of all possible models for the input as displacement,horsepower,weight.  I found that mpg is highly 
# dependent on horsepower

#considering mpg and horsepower to be linearly dependent 
model_l = smf.ols(formula='mpg~horsepower',data = auto_).fit()
print(model_l.summary())

# we could see that the adjusted R square is .584 which could be a ok model.
auto_['predicted_linear_mpg'] = model_l.predict(auto_['horsepower'])

# RSE and error
SSD = np.sum((auto_['mpg']-auto_['predicted_linear_mpg'])**2)
RSE = np.sqrt((1/(len(auto_['mpg'])-1-1))*SSD)
print("[+] RSE = ",RSE)
error = RSE/np.mean(auto_['mpg'])
print("[+] Error = ",error*100)
# 20.18 % error, improvising model
#from the scatter plot it seems that horsepower is following some polynomial function

#quadratic
model_q = smf.ols(formula='mpg~horsepower+np.power(horsepower,2)',data=auto_).fit()
print(model_q.rsquared)

auto_['predicted_quadratic_mpg'] = model_q.predict(auto_['horsepower'])
# RSE and error
SSD_q = np.sum((auto_['mpg']-auto_['predicted_quadratic_mpg'])**2)
RSE_q = np.sqrt((1/(len(auto_['mpg'])-1-1))*SSD_q)
print("[+] RSE = ",RSE_q)
error_q = RSE_q/np.mean(auto_['mpg'])
print("[+] Error = ",error_q*100)

#error and RSE decreased, adjusted R square increased => model improved

#improvising model with higher power polynomial through scikit-learn
X = auto_['horsepower']
Y = auto_['mpg']
poly = PolynomialFeatures(degree=5)
X_ = poly.fit_transform(X[:,np.newaxis])
lr = LinearRegression()
lr.fit(X_,Y)

#rsquare
print(lr.score(X_,Y))

auto_['predicted_poly_mpg'] = lr.predict(X_)
SSD_p = np.sum((auto_['mpg']-auto_['predicted_poly_mpg'])**2)
RSE_p = np.sqrt((1/(len(auto_['mpg'])-1-1))*SSD_p)
print("[+] RSE = ",RSE_p)
error_p = RSE_p/np.mean(auto_['mpg'])
print("[+] Error = ",error_p*100)
# adjusted R is improved a bit so the model is bit improved

print(auto_.head())
#plotting regression line and linear line

_ = sns.regplot(auto_['horsepower'],auto_['mpg'])
plt.plot(auto_['horsepower'],auto_['predicted_poly_mpg'],'r.',linestyle='none')