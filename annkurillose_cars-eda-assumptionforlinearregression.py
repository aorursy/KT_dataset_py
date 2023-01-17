import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

sns.set_style('darkgrid')
df = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')

df.head()
df.head()
df.shape
df.info()
df.isnull().sum()
a=['Car_Name','Fuel_Type','Seller_Type','Transmission']
# Showing how many unique values are there in categorical columns

for i in a:

    print(i ,len(df[i].unique()))

    
# Since there are 98 unique different items in the column Car_Names we will drop that column.

df=df.drop('Car_Name',axis=1)
# Percentage of each category in Fuel_Type,Seller_Type,Transmission

fig, axes=plt.subplots(1,3,figsize=(15,10))

df['Fuel_Type'].value_counts().plot(kind='pie',autopct='%.3f%%',ax=axes[0],textprops={'fontsize': 13})

df['Seller_Type'].value_counts().plot(kind='pie',autopct='%.3f%%',ax=axes[1],textprops={'fontsize': 13})

df['Transmission'].value_counts().plot(kind='pie',autopct='%.3f%%',ax=axes[2],textprops={'fontsize': 13})

plt.show()
# Number of previous owners

sns.countplot(df['Owner'],palette='husl')

plt.show()
plt.figure(figsize=(8,6))

sns.heatmap(df.corr(),annot=True)

plt.show()
# Distribution of Selling_Price, Year, Present_Price, Kms_Driven

fig, axes=plt.subplots(2,2,figsize=(15,5))

sns.distplot(df['Selling_Price'],ax=axes[0,0])

sns.distplot(df['Year'],ax=axes[0,1])

sns.distplot(df['Present_Price'],ax=axes[1,0])

sns.distplot(df['Kms_Driven'],ax=axes[1,1])



plt.show()
# Plotting 'Year', 'Present_Price', 'Kms_Driven', 'Owner' against the traget variable selling price to find their relation

sns.pairplot(df,x_vars=['Year', 'Present_Price', 'Kms_Driven', 'Owner'],y_vars=['Selling_Price'],height=4)

plt.show()
# To find further inference 

sns.lmplot(x='Present_Price',y='Selling_Price',data=df, fit_reg=False,col='Transmission',hue='Fuel_Type',height=4,aspect=1.5)

plt.show()
# Count of outliers in each numerical column

a=['Year','Selling_Price','Present_Price','Kms_Driven','Owner']



for i in a:

    q1 = df[i].quantile(0.25)

    q3 = df[i].quantile(0.75)

    iqr = q3-q1



    UL = q3 + (1.5 * iqr)

    LL = q1 - (1.5 * iqr)

    print(i,df[(df[i]>UL) | (df[i]<LL)].count()[i])

    #print(cars[(cars[i]>UL) | (cars[i]<LL)][i])

# Outliers of Selling price

q1 = df['Selling_Price'].quantile(0.25)

q3 = df['Selling_Price'].quantile(0.75)

iqr = q3-q1

UL = q3 + (1.5 * iqr)

LL = q1 - (1.5 * iqr)

df[(df['Selling_Price']>UL) | (df['Selling_Price']<LL)].sort_index()

# Outliers of Present Price

q1 = df['Present_Price'].quantile(0.25)

q3 = df['Present_Price'].quantile(0.75)

iqr = q3-q1

UL = q3 + (1.5 * iqr)

LL = q1 - (1.5 * iqr)

df[(df['Present_Price']>UL) | (df['Present_Price']<LL)].sort_index()

# Visual representation of outliers

fig, axes=plt.subplots(2,2,figsize=(15,8))

sns.boxplot('Selling_Price',data=df,ax=axes[0,0])

sns.boxplot('Year',data=df,ax=axes[0,1])

sns.boxplot('Present_Price',data=df,ax=axes[1,0])

sns.boxplot('Kms_Driven',data=df,ax=axes[1,1])

plt.show()

df=pd.get_dummies(df,drop_first=True)
df.info()
X=df.drop('Selling_Price',axis=1)

y=df['Selling_Price']

X.head()
# Standardizing the data by taking mean to 0 and standard deviation to 1.

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_std = ss.fit_transform(X)

X_std=pd.DataFrame(X_std, columns=X.columns)
import statsmodels.api as sm

Xc=sm.add_constant(X_std)

ols=sm.OLS(y,Xc)

model=ols.fit()

model.summary()
y_pred=model.predict(Xc)
# Plotting predicted values vs actual values

plt.scatter(y_pred,y)

plt.plot(y_pred,y_pred,'r')

plt.xlabel('y predicted')

plt.ylabel('y actual')

plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=pd.DataFrame()

vif['VIF']=[variance_inflation_factor(X_std.values,i) for i in range(X_std.shape[1])]

vif['feature']=X_std.columns

vif.sort_values('VIF',ascending=False)
X_vif=X_std.copy()

while vif['VIF'].max()>10:

    a=vif[vif['VIF']==vif['VIF'].max()].iloc[0,1]

    X_vif=X_vif.drop(a,axis=1)

    

    vif=pd.DataFrame()

    vif['VIF']=[variance_inflation_factor(X_vif.values,i) for i in range(X_vif.shape[1])]

    vif['feature']=X_vif.columns

vif



# We can see that one columns have been removed which brings all the vif values below 10.
sns.regplot(y,model.predict(),line_kws={'color':'red'})

plt.show()
from statsmodels.stats.diagnostic import linear_rainbow

linear_rainbow(res=model,frac=0.5) 

# Since pvalue > 0.05 we conclude that the data is  linear.
from scipy.stats import norm

sns.distplot(model.resid,fit=norm)

norm.fit(model.resid)

plt.show()
import scipy.stats as stats

stats.shapiro(model.resid)

# p value = 0 < 0.05 hence we reject null hypothesis (ie.It is normally distributed) Which means that it is not normally distributed.
#### QQ plot ( quantile quantile plot)



import scipy.stats as stats

stats.probplot(model.resid,plot=plt)

plt.show()



# here only the extreme values are going from normality.
import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(model.resid, lags=40 , alpha=0.05)

acf.show()

sns.residplot(model.predict(Xc),model.resid,lowess =True, line_kws ={'color':'red'} )

plt.xlabel('Predicted Values')

plt.ylabel('Residuals')

plt.show()

# from the graph it is hetro ( since there is high varience in the output )

# we can further check using 
import statsmodels.stats.api as sms

from statsmodels.compat import lzip

name=['F-stat','p=value']

test=sms.het_goldfeldquandt(y=model.resid,x=Xc)

lzip(name, test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)



from sklearn.metrics import r2_score, mean_squared_error

print('R^2 on the test data', r2_score(y_test, y_pred))

from sklearn.linear_model import Lasso, Ridge
lasso=Lasso(alpha=0.01)

lasso.fit(X_train,y_train)
pd.DataFrame(lasso.coef_,index=X_train.columns,columns=['coefs'])
y_pred = lasso.predict(X_test)

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
ridge=Ridge(alpha=0.01)

ridge.fit(X_train,y_train)
pd.DataFrame(ridge.coef_,index=X_train.columns,columns=['coefs'])
y_pred = ridge.predict(X_test)

r2_score(y_test,y_pred)
#### There is no much difference in the score. This is because lasso and ridge needs many columns to make a difference in the prediction. Hence we try using all the columns

cars = pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")

cars=pd.get_dummies(cars,drop_first=True)
X=cars.drop('Selling_Price',axis=1)

y=cars['Selling_Price']
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_std = ss.fit_transform(X)

X_std=pd.DataFrame(X_std, columns=X.columns)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.3,random_state=0)
from sklearn.linear_model import Lasso, Ridge

from sklearn.metrics import r2_score





lasso=Lasso(alpha=0.01)

lasso.fit(X_train,y_train)



pd.DataFrame(lasso.coef_,index=X_train.columns,columns=['coefs'])



y_pred = lasso.predict(X_test)

r2_score(y_test,y_pred)
ridge=Ridge(alpha=0.01)

ridge.fit(X_train,y_train)



pd.DataFrame(ridge.coef_,index=X_train.columns,columns=['coefs'])



y_pred = ridge.predict(X_test)

r2_score(y_test,y_pred)
# In lasso the scores have improved to 0.88 as compared to basic model with score 0.85
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor
models = {'Lasso': Lasso(alpha=0.01),

          'Ridge':Ridge(alpha=0.01),

          'RandomForest' : RandomForestRegressor(),

          'DecisionTree' : DecisionTreeRegressor(),

          'GradientBoosting' : GradientBoostingRegressor(),

          'AdaBoost' : AdaBoostRegressor()}





def Different_model_scores(models):

    model_scores = {}    

    for name, model in models.items():        

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        model_scores[name]=r2_score(y_test,y_pred)

    return model_scores

model_scores = Different_model_scores(models)

model_scores