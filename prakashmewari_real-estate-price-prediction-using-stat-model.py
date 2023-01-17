import pandas as pd 

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt 
df = pd.read_csv("../input/realestate.csv")
df.head()
df.columns=df.columns.str.lower()
df.head()
df.corr()
sns.pairplot(df)
df.describe()
df.columns
plt.figure(1)

plt.subplot(121)

sns.distplot(df['unit_area']);

plt.subplot(122)

df['unit_area'].plot.box(figsize=(16,5))

plt.show()
# univariate analyis : ---

plt.figure(1)

plt.subplot(121)

sns.distplot(df['houseage']);

plt.subplot(122)

df['houseage'].plot.box(figsize=(16,5))

plt.show()
plt.figure(1)

plt.subplot(121)

sns.distplot((df['distance']));

plt.subplot(122)

(df['distance']).plot.box(figsize=(16,5))

plt.show()
df.columns
plt.figure(1)

plt.subplot(121)

sns.distplot(df['stores']);

plt.subplot(122)

df['stores'].plot.box(figsize=(16,5))

plt.show()
# checking na 

df.isna().sum()
df['transactiondate']=df['transactiondate'].astype(int)
sns.catplot(x = 'transactiondate' , y = 'unit_area' , data =df)
sns.lineplot(x ='transactiondate' , y ='unit_area' , data =df)
df['transactiondate'].value_counts()
for col in df.columns:

    print("--------****-------")

    print (df[col].value_counts().head())

    
df.head()
df['distance'].describe()
df['distance'].plot.box()
df['distance']=df['distance'].clip(23,1453)
df['distance'].plot.box()
df['distance'].plot.hist()
df['houseage']=df['houseage'].astype(int)
df.head()
df['distance']=df['distance'].astype(int)
# collect x and y

df.columns

X = df[['houseage', 'distance', 'stores']]

y = df['unit_area']
df.columns
import statsmodels.formula.api as smf # for regression model
ml1 = smf.ols('unit_area~transactiondate+houseage+distance+stores',data=df).fit() # regression model

ml1.summary()
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
df_new=df.drop(df.index[270],axis=0)
df_new
ml2 = smf.ols('unit_area~transactiondate+houseage+distance+stores',data=df_new).fit() # regression model
ml2.summary()
sm.graphics.influence_plot(ml2,)
rsquared=smf.ols('unit_area~transactiondate+houseage+distance+stores',data=df_new).fit().rsquared
vif_sp = 1/(1-rsquared)

vif_sp  # vif value
rsquared2=smf.ols('unit_area~transactiondate+houseage+distance+stores',data=df).fit().rsquared
vif_sp = 1/(1-rsquared2)

vif_sp  # vif value
# Added varible plot 

sm.graphics.plot_partregress_grid(ml2)
# predict the data 

pred=ml2.predict(df_new)
# checking linearity

# Observed values VS Fitted values

plt.scatter(df_new.unit_area,pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
#Residuals VS Fitted Values 

plt.scatter(pred,ml2.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

########    Normality plot for residuals ######

# histogram

plt.hist(ml2.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 

import pylab          

import scipy.stats as st



# Checking Residuals are normally distributed

st.probplot(ml2.resid_pearson, dist="norm", plot=pylab)



############ Homoscedasticity #######



# Residuals VS Fitted Values 

plt.scatter(pred,ml2.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

### Splitting the data into train and test data 



from sklearn.model_selection import train_test_split

df_train,df_test  = train_test_split(df_new,test_size = 0.2) # 20% size
df_train.shape,df_test.shape
model_train = smf.ols('unit_area~transactiondate+houseage+distance+stores',data=df_train).fit()
model_train
# train_data prediction

train_pred = model_train.predict(df_train)
train_pred.head()
# trian residual values 

train_resid  = train_pred - df_train.unit_area
train_resid.plot.hist()
# RMSE value for train data 

train_rmse = np.sqrt(np.mean(train_resid*train_resid))

train_rmse
# prediction on test data set 

test_pred = model_train.predict(df_test)
# test residual values 

test_resid  = test_pred - df_test.unit_area
# RMSE value for test data 

test_rmse = np.sqrt(np.mean(test_resid*test_resid))
test_rmse,train_rmse