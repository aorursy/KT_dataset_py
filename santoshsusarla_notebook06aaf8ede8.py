# supress warnings 

from warnings import filterwarnings

filterwarnings('ignore')



# 'Os' module provides functions for interacting with the operating system 

import os



# 'Pandas' is used for data manipulation and analysis

import pandas as pd 



# 'Numpy' is used for mathematical operations on large, multi-dimensional arrays and matrices

import numpy as np



# 'Matplotlib' is a data visualization library for 2D and 3D plots, built on numpy

import matplotlib.pyplot as plt

%matplotlib inline



# 'Seaborn' is based on matplotlib; used for plotting statistical graphics

import seaborn as sns



# 'Scikit-learn' (sklearn) emphasizes various regression, classification and clustering algorithms

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import ElasticNet



# 'Statsmodels' is used to build and analyze various statistical models

import statsmodels

import statsmodels.api as sm

import statsmodels.stats.api as sms

from statsmodels.tools.eval_measures import rmse

from statsmodels.compat import lzip

from statsmodels.graphics.gofplots import ProbPlot



# 'SciPy' is used to perform scientific computations

from scipy.stats import f_oneway

from scipy.stats import jarque_bera

from scipy import stats
data = pd.read_csv('../input/seoulbikedatacsv/SeoulBikeData (1).csv',engine='python')
data.head(4)
data.info()
data.shape
data[(data['Rented Bike Count']==0)].head()
index=data.index[data['Functioning Day']=='No']  # fiding the poisition if non-functioning days
data=data.drop(index)
data=data.reset_index(drop=True)
data['Functioning Day'].value_counts()                # checking the count of functioning day
data['Holiday'].value_counts()                   # checking the count of Holiday
data.head(1)
data.corr() # checking the correlation
num_data=data.select_dtypes(include=np.number)

cat_data=data.select_dtypes(exclude=np.number)
num_data.head(1)
cat_data.head(1)
hr=data.groupby(['Seasons','Hour'])['Rented Bike Count'].mean()
sr=data.groupby(['Seasons'])['Rented Bike Count'].mean()

sr
hr=pd.DataFrame([hr])

hr.head()


hr1=hr.T
hr1.head()
data['Rented Bike Count'].plot(kind='kde')
num_data.head(1)


num_data.plot(kind='kde',subplots=True,layout=(4,3),sharex=False,figsize=(15,10))

plt.show()
sns.countplot(cat_data['Seasons'])
sns.countplot(cat_data['Holiday'])
plt.hist(data['Rented Bike Count'],bins=10)
sns.distplot(data['Rented Bike Count'])
plt.boxplot(data['Rented Bike Count'],vert=False)
sns.violinplot(data['Rented Bike Count'])
plt.figure(figsize=(15, 8))           # plotting a heatmap to see correlation

corr=num_data.corr()

sns.heatmap(corr,annot=True,fmt='0.2f',vmax=1,vmin=-1)
plt.figure(figsize=(15, 8))        # checking the number of bikes plotted every hou

sns.boxplot(data['Hour'],data['Rented Bike Count']).set_title('Rented Bike Count distribution across hour')
plt.figure(figsize=(15, 8))         #Represntation of the rented bike count based on seasons

sns.boxplot(data['Seasons'],data['Rented Bike Count'],order=['Autumn','Spring','Summer','Winter']).set_title('Rented Bike Count distribution across Seasons')
plt.figure(figsize=(15, 8))

sns.boxplot(data['Holiday'],data['Rented Bike Count']).set_title('Rented Bike Count distribution across Holidays')
plt.figure(figsize=(15, 8))

sns.boxplot(x=data['Hour'],y=data['Rented Bike Count'],hue=data['Seasons'],hue_order=['Autumn','Spring','Summer','Winter'])
plt.figure(figsize=(15, 8))

sns.countplot(data['Wind speed (m/s)']) 

plt.xticks(rotation=90)
from scipy.stats import f_oneway

f_oneway(data['Rented Bike Count'][data['Holiday'] == 'Holiday'], 

             data['Rented Bike Count'][data['Holiday'] == 'No Holiday'])
tab=pd.crosstab(data['Holiday'],data['Seasons'])/data['Seasons'].value_counts()*100
(tab.T).plot(kind='bar',stacked=True,figsize=(10,5))

plt.legend(framealpha=0.2,bbox_to_anchor=(1,0,0.2,0.5))
cat_data.head(1)
cat_data=cat_data.drop('Date',1)          # dropping date as it is insignificant 
data1=pd.concat([num_data,cat_data],1)
data1.head(1)


plt.figure(figsize=(15, 8))

plt.title('Box Plot for Numerical Variable')

data1.boxplot()
plt.figure(figsize=(15, 8))

plt.title('Box Plot for Numerical Variable by dropping Rented Bike Count and Visibilty')

data1.drop(['Rented Bike Count','Visibility (10m)'],1).boxplot()
plt.figure(figsize=(15,8))

plt.title('Box Plot for Rented Bike Count ')

sns.boxplot(data1['Rented Bike Count'])
plt.figure(figsize=(15,8))

plt.title('Box Plot for visibilty ')

sns.boxplot(data['Visibility (10m)'])
#Capping to remove outliers

Q1 = data1.quantile(0.25)

Q3 = data1.quantile(0.75)

# calculate of interquartile range 

IQR = Q3 - Q1

data1=data1[~((data1<(Q1-1.5*IQR))|(data1>(Q3+1.5*IQR))).any(axis=1)]
data1.shape
plt.figure(figsize=(15,8))

data1.boxplot(column=[  'Temperature(�C)', 'Humidity(%)',

       'Wind speed (m/s)',

       'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)'])

plt.xticks(rotation=90)
plt.figure(figsize=(15,8))

plt.title('Box Plot for Rented Bike Count after capping')

sns.boxplot(data1['Rented Bike Count'])
data1=data1.drop(['Rainfall(mm)', 'Snowfall (cm)'],1)
plt.figure(figsize=(15,8))

corr=data1.corr()

corr

sns.heatmap(corr,annot=True,fmt='0.2f',vmax=1,vmin=-1)

data1.corr()
data1.head(1)
data1.shape

num_data1=data1.select_dtypes(include=np.number)

cat_data1=data1.select_dtypes(exclude=np.number)
cat_data1=pd.get_dummies(cat_data1,columns=['Holiday','Seasons'],drop_first=True)  #using one-hot encoding technique
cat_data1=pd.get_dummies(cat_data1,columns=['Functioning Day'])
cat_data1.head()
num_data1.head(1)
data2=pd.concat([num_data1,cat_data1],1)        #combining both categorical and numerical variables

data2.head()
datav1=data2.copy()

datav1.head()
out=pd.DataFrame(np.power(datav1['Rented Bike Count'],0.3))
datav1['pow_rbc']=out

datav1['pow_rbc'].skew()           # data is slightly left skewed
data2=pd.concat([num_data1,cat_data1],1)        #combining both categorical and numerical variables

data2.head()
datav1=data2.copy()

datav1.head()
out=pd.DataFrame(np.power(datav1['Rented Bike Count'],0.3))
datav1['pow_rbc']=out

datav1['pow_rbc'].skew()           # data is slightly left skewed
data_ols1=datav1.copy()
from sklearn.preprocessing import StandardScaler       

sc=StandardScaler()

num_two_col_sc=sc.fit_transform(num_data1.drop('Rented Bike Count',1))

num_two_col_sc=pd.DataFrame(num_two_col_sc,columns=num_data1.iloc[:,1:].columns)

num_two_col_sc.head(2)
num_sc=num_two_col_sc.reset_index(drop=True)
cat_data1=cat_data1.reset_index(drop=True)
final_data=pd.concat([num_two_col_sc,cat_data1],axis=1)

final_data.head(2)
final_data.shape
##Base model-1 using ols packag
#Load the independent and dependent features

inp=final_data

out=num_data1['Rented Bike Count']
#Load required libraries

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_squared_error
#Split the dataset into train and test 

xtrain,xtest,ytrain,ytest= train_test_split(inp,out,test_size=.3,random_state=48)
#check the dimensions

print('Dimensions of x_train:',xtrain.shape)

print('Dimensions of y_train:',ytrain.shape)

print('Dimensions of x_test:',xtest.shape)

print('Dimensions of y_test:',ytest.shape)
##Base model-2 using sklearn package
score_card = pd.DataFrame(columns=['Model_Name', 'R-Squared', 'Adj. R-Squared', 'RMSE'])



linreg_full_base_model_using_ols = pd.Series({

                     'Model_Name': "LRM_using _OLS",

                     'RMSE':ols_rmse,

                     'R-Squared':ols_r2 ,

                     'Adj. R-Squared': ols_adj_r2    

                   })



score_card = score_card.append(linreg_full_base_model_using_ols , ignore_index=True)



#call scorecard

score_card
Lr=LinearRegression()

Lr.fit(xtrain,ytrain) 

ypred=Lr.predict(xtest)

print('predicted values:',ypred,'\n')

intercept=Lr.intercept_

print('Intercept:',intercept,'\n')

Coeff=Lr.coef_

print('Coefficents :',Coeff)
#find Performance

from sklearn.metrics import r2_score,mean_squared_error

from statsmodels.tools.eval_measures import rmse



#rmse

lr_rmse=rmse(ytest,ypred)

#r2_score

lr_r2=r2_score(ytest,ypred)

#adj_r2_score

lr_adj_r2=1-(((1-lr_r2)*(len(xtest)-1))/(len(xtest)-len(xtest.columns)-1))



print('rmse:',lr_rmse,'\n','r2_score:',lr_r2,'\n','adj_r2:',lr_adj_r2)
#model-3 output transformation model
# Build a model with the output transformation





#As we know that dependent variable is not normal distributed so transform it

import scipy.stats as stats

box_out,lam=stats.boxcox(num_data1['Rented Bike Count'])

box_out=pd.DataFrame(box_out)

print('skewness of price:',box_out.skew()[0])    #equal to zero so normal distributed now



#split the  input and output variables 

xtrain,xtest,ytrain,ytest= train_test_split(inp,box_out,test_size=.3,random_state=48)





#build a model using Sklearn package

Lr=LinearRegression()

Lr.fit(xtrain,ytrain) 

ypred=Lr.predict(xtest)

print('predicted values:',ypred,'\n')

intercept=Lr.intercept_

print('Intercept:',intercept,'\n')

Coeff=Lr.coef_

print('Coefficents :',Coeff)

#check
#find performance

#rmse

lr_box_cox_rmse=rmse(ytest,ypred)

#r2_score

lr_box_cox_r2=r2_score(ytest,ypred)

#adj_r2_score

lr_box_cox_adj_r2=1-(((1-lr_box_cox_r2)*(len(xtest)-1))/(len(xtest)-len(xtest.columns)-1))



print('rmse:',lr_box_cox_rmse,'\n','r2_score:',lr_box_cox_r2,'\n','adj_r2:',lr_box_cox_adj_r2)
linreg_full_model_with_transformed_Rented_bike_Count = pd.Series({

                     'Model_Name': "LRM_full_with_transformed_Rented_bike_Count",

                     'RMSE':lr_box_cox_rmse,

                     'R-Squared':lr_box_cox_r2 ,

                     'Adj. R-Squared': lr_box_cox_adj_r2     

                   })



score_card = score_card.append(linreg_full_model_with_transformed_Rented_bike_Count, ignore_index=True)



#call scorecard

score_card
 #Assumptions
#a) Multicolinearity
#import library

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif=pd.DataFrame()

vif['VIF']=[variance_inflation_factor(xtrain.values,i) for i in range(xtrain.shape[1])]          #dataframe to np array

vif['features']=xtrain.columns

vif.sort_values('VIF',ascending=False)
inpv1=xtrain.drop(['Dew point temperature(�C)','Holiday_No Holiday'],1)

vif=pd.DataFrame()

vif['VIF']=[variance_inflation_factor(inpv1.values,i) for i in range(inpv1.shape[1])]          

vif['features']=inpv1.columns

vif.sort_values('VIF',ascending=False)
#model=4 significant features with transformed output# check
#inp=inpv1             have only signficant features   (these two are train set)

#out=ytrain           transformed output
#Build a model 

import statsmodels.api as sm



inpc1=sm.add_constant(inpv1)

ols=sm.OLS(ytrain,inpc1)

linear_model_using_ols=ols.fit() 

print(linear_model_using_ols.summary())
#Autocollinearity
#Statistical test to check the linearity

from statsmodels.stats.diagnostic import linear_rainbow 

print('% of Linearity:',linear_rainbow(res=linear_model_using_ols,frac=0.5)[1]*100)
#Normality
#plot the distribution of residual

sns.distplot(linear_model_using_ols.resid,color='orange')

plt.show()
#Check skewness

print('Skewness of residual is :',linear_model_using_ols.resid.skew())
#Using qqplot

from statsmodels.graphics.gofplots import qqplot

qqplot(linear_model_using_ols.resid,line='r')   #r=regression line

#check
#Find the Homascadasticity

sns.residplot(linear_model_using_ols.predict(),linear_model_using_ols.resid)

plt.show()
#statsitic test for homascadsticity

from statsmodels.stats.api import het_goldfeldquandt



#H0: model is homascadsticity

print('Pvalue is :',het_goldfeldquandt(linear_model_using_ols.resid,linear_model_using_ols.model.exog)[1])
#Feature selection
inpv1.head(2)
inpv_1=inpv1.copy()
#import library

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
#a) Backward elimination method
lr=LinearRegression()

lr_back=sfs(estimator=lr,k_features='best',forward=False) #k_features='best' gives significant features among all the features

sfs_back=lr_back.fit(inpv_1,ytrain)

back_feat=sfs_back.k_feature_names_

back_feat=list(back_feat)

print('These are the significant features for best model by considering backward_elimin :','\n',back_feat)
#Check score

print('Score of backward elimination :',sfs_back.k_score_)
 #Forward selection method
lr=LinearRegression()

lr_forw=sfs(estimator=lr,k_features='best',forward=True)   

sfs_forw=lr_forw.fit(inpv_1,ytrain)

forw_feat=sfs_forw.k_feature_names_

forw_feat=list(forw_feat)

print('These are the significant features for best model by considering forward_selec :','\n',forw_feat)
#Check score

print('Score of forward selection :',sfs_forw.k_score_)
#RFE-Recursive feature elimination
lr=LinearRegression()

from sklearn.feature_selection import RFECV       #its gives possible features so thats why we dont need to write n_features_to_select

rfe_mod=RFECV(estimator=lr)

rfe_feat=rfe_mod.fit(inpv_1,ytrain)



rank=pd.DataFrame()

rank['Rank']=rfe_feat.ranking_

rank['Feature']=inpv_1.columns



r_feat=rank[rank['Rank']==1]
#print features

rfe_featu=r_feat['Feature']

print('These are the significant features for best model by considering RFE :','\n',rfe_featu)
#Check score

print('Score of RFECV method :',np.mean(rfe_feat.grid_scores_))
#model-5 model with best features using backward elimination

import statsmodels.api as sm



inpc2=sm.add_constant(inpv_1[back_feat])

ols=sm.OLS(ytrain,inpc2)

linear_model_using_ols_best_feat=ols.fit() 

print(linear_model_using_ols_best_feat.summary())
#model-6 model with best features using RFE method
#Build a model

inpc3=sm.add_constant(inpv_1[rfe_featu])

ols=sm.OLS(ytrain,inpc3)

linear_model_using_ols_best_feat_RFE=ols.fit() 

print(linear_model_using_ols_best_feat_RFE.summary())
sign_feat=list(inpv_1[rfe_featu].columns)  #bcoz we need to consider only best features obtained from RFE method





from statsmodels.api import add_constant

xtest_with_constant=add_constant(xtest[sign_feat],has_constant='add')

ypred_ols=linear_model_using_ols_best_feat_RFE.predict(xtest_with_constant)

ypred_ols.shape
#find Performance

from sklearn.metrics import r2_score,mean_squared_error

from statsmodels.tools.eval_measures import rmse



#rmse

ols_rmse=np.sqrt(mean_squared_error(ytest,ypred_ols))

#r2_score

ols_r2=r2_score(ytest,ypred_ols)

#adj_r2_score

ols_adj_r2=1-(((1-ols_r2)*(len(xtest[sign_feat])-1))/(len(xtest[sign_feat])-len(xtest[sign_feat].columns)-1))



print('rmse:',ols_rmse,'\n','r2_score:',ols_r2,'\n','adj_r2:',ols_adj_r2)
linreg_model_with_signifi_feat_using_RFECV_method = pd.Series({

                     'Model_Name': "LRM_with_signif_feat_using_RFECV",

                     'RMSE':ols_rmse,

                     'R-Squared':ols_r2 ,

                     'Adj. R-Squared': ols_adj_r2     

                   })



score_card = score_card.append(linreg_model_with_signifi_feat_using_RFECV_method , ignore_index=True)



#call scorecard

score_card
#Interaction effect (Joint effect)
# Interaction effect with backward elimination
inpv_1_back=inpv_1[back_feat]

inpv_1_back.head(2)
#concatenate ytrain and inpv_1_back[back_feat]

inpv_1_back=pd.concat([inpv_1_back,ytrain],axis=1)

inpv_1_back.rename({0: 'Rented Bike Count'}, axis=1, inplace=True)

inpv_1_back.head(2)

#check
#bring to one scale form

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

inpv_1_back['Rented Bike Count']=sc.fit_transform(inpv_1_back['Rented Bike Count'].values.reshape(-1,1))

inpv_1_back.head(2)
inpv_2_back=inpv_1_back.copy()
inpv_2_back['Hour*Temp']=inpv_2_back['Temperature(�C)']*inpv_2_back['Hour']
inpv_2_back.head(2)
inpv_2_back_d=inpv_2_back.drop(['Rented Bike Count'],1)
inpv_2_back_d.head()
#Build a model 

import statsmodels.api as sm



inpc4=sm.add_constant(inpv_2_back_d)

ols=sm.OLS(ytrain,inpc4)

linear_model_using_ols_best_feat_backward_with_interaction=ols.fit() 

print(linear_model_using_ols_best_feat_backward_with_interaction.summary())
#same we need for test also#check

xtest[back_feat].head(2)
xtest_backward=pd.concat([xtest[back_feat],ytest],axis=1)

xtest_backward.rename({0: 'Rented Bike Count'}, axis=1, inplace=True)

xtest_backward.head(2)

#bring to one scale form

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

xtest_backward['Rented Bike Count']=sc.fit_transform(xtest_backward['Rented Bike Count'].values.reshape(-1,1))

xtest_backward.head(2)
xtest_backward_1=xtest_backward.copy()
xtest_backward_1['Hour*Temp']=xtest_backward['Temperature(�C)']*xtest_backward['Hour']
xtest_backward_1.head(2)
xtest_backward_2=xtest_backward_1.drop('Rented Bike Count',1)
#find Performance

from sklearn.metrics import r2_score,mean_squared_error

from statsmodels.tools.eval_measures import rmse



#rmse

ols_rmse=np.sqrt(mean_squared_error(ytest,ypred_ols))

#r2_score

ols_r2=r2_score(ytest,ypred_ols)

#adj_r2_score

ols_adj_r2=1-(((1-ols_r2)*(len(xtest_backward)-1))/(len(xtest_backward)-len(xtest_backward.columns)-1))



print('rmse:',ols_rmse,'\n','r2_score:',ols_r2,'\n','adj_r2:',ols_adj_r2)
linreg_model_with_signifi_feat_backwrd_interaction = pd.Series({

                     'Model_Name': "LRM_with_signif_feat_Backwrd_Intrctn",

                     'RMSE':ols_rmse,

                     'R-Squared':ols_r2 ,

                     'Adj. R-Squared': ols_adj_r2     

                   })



score_card = score_card.append(linreg_model_with_signifi_feat_backwrd_interaction , ignore_index=True)



#call scorecard

score_card
#Interaction effect with RFECV method
inpv_1_rfe=inpv_1[rfe_featu]

inpv_1_rfe.head(2)
#concatenate ytrain and inpv_1_rfe[back_feat]

inpv_1_rfe=pd.concat([inpv_1_rfe,ytrain],axis=1)

inpv_1_rfe.rename({0: 'Rented Bike Count'}, axis=1, inplace=True)

inpv_1_rfe.head(2)
#bring to one scale form

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

inpv_1_rfe['Rented Bike Count']=sc.fit_transform(inpv_1_rfe['Rented Bike Count'].values.reshape(-1,1))

inpv_1_rfe.head(2)
inpv_2_rfe=inpv_1_rfe.copy()
inpv_2_rfe['Hour*Temp']=inpv_2_rfe['Temperature(�C)']*inpv_2_rfe['Hour']
inpv_2_rfe=inpv_2_rfe.drop('Rented Bike Count',1)

inpv_2_rfe.head(2)
#model-8 interaction effect of price difference with rfe method
#Build a model 

import statsmodels.api as sm



inpc5=sm.add_constant(inpv_2_rfe)

ols=sm.OLS(ytrain,inpc5)

linear_model_using_ols_best_feat_RFECV_with_interaction=ols.fit() 

print(linear_model_using_ols_best_feat_RFECV_with_interaction.summary())
#same we need for test also

xtest[rfe_featu].head(2)
#concatenate

xtest_rfe=pd.concat([xtest[rfe_featu],ytest],axis=1)

xtest_rfe.rename({0: 'Rented Bike Count'}, axis=1, inplace=True)

xtest_rfe.head(2)
#bring to one scale form

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

xtest_rfe['Rented Bike Count']=sc.fit_transform(xtest_rfe['Rented Bike Count'].values.reshape(-1,1))

xtest_rfe.head(2)
xtest_rfe_1=xtest_rfe.copy()

xtest_rfe_1['Hour*Temperature']=xtest_rfe['Hour']*xtest_rfe['Temperature(�C)']

xtest_rfe_1=xtest_rfe_1.drop('Rented Bike Count',1)

xtest_rfe_1.head(2)
from statsmodels.api import add_constant

xtest_with_constant=add_constant(xtest_rfe_1)

ypred_ols=linear_model_using_ols_best_feat_RFECV_with_interaction.predict(xtest_with_constant)

ypred_ols.shape
#find Performance

from sklearn.metrics import r2_score,mean_squared_error

from statsmodels.tools.eval_measures import rmse



#rmse

ols_rmse=np.sqrt(mean_squared_error(ytest,ypred_ols))

#r2_score

ols_r2=r2_score(ytest,ypred_ols)

#adj_r2_score

ols_adj_r2=1-(((1-ols_r2)*(len(xtest_backward)-1))/(len(xtest_backward)-len(xtest_backward.columns)-1))



print('rmse:',ols_rmse,'\n','r2_score:',ols_r2,'\n','adj_r2:',ols_adj_r2)
linreg_model_with_signifi_feat_RFECV_interaction = pd.Series({

                     'Model_Name': "LRM_with_signif_feat_RFECV_intrctn",

                     'RMSE':ols_rmse,

                     'R-Squared':ols_r2 ,

                     'Adj. R-Squared': ols_adj_r2     

                   })



score_card = score_card.append(linreg_model_with_signifi_feat_RFECV_interaction , ignore_index=True)



#call scorecard

score_card
#Optimisation
#Cross Validation
inp=final_data

out=box_out
#import library

from sklearn.model_selection import cross_val_score,KFold



kf=KFold(n_splits=5,shuffle=True,random_state=48)





#use r2 score

from sklearn.linear_model import LinearRegression     #split the dataset into 5 groups

lr=LinearRegression()

res=cross_val_score(lr,inp,out,cv=kf,scoring='r2')

print('R2 values for 5sets of dataset :',res ) 
#bias error

be=1-np.mean(res)



#variance error

ve=np.std(res)



print('Bias error interms of R2:',be)

print('Varaince error interms of R2:',ve)      #std(res)=0 model if free of overfitting 
# Dataset is having overfitting problem... to overcome this we can use regualarization 
 #Regularization with Hyperparameter
#import hyperparameter library

from sklearn.model_selection import GridSearchCV 

#import regularization library

from sklearn.linear_model import Lasso,Ridge,ElasticNet
#Lasso Regularization
loss=Lasso()     #hyper paramater of lasso is alpha

param={'alpha':[0.000001,0.00001,0.0001,0.001,0.01,0.1,1,2,5,8,10,50,60,70,90,100]}



grid=GridSearchCV(loss,param_grid=param,cv=5,scoring='r2') 



hyp_rid=grid.fit(inpv_2_rfe,ytrain)



print('Best hyperparameter is :',hyp_rid.best_params_ )

print('Score :',hyp_rid.best_score_ )
#Ridge Regularization
rid=Ridge()     #hyper paramater of ridge is alpha

param={'alpha':[0.000001,0.00001,0.0001,0.001,0.01,0.1,1,2,5,8,10,50,60,70,90,100]}



grid=GridSearchCV(rid,param_grid=param,cv=5,scoring='r2')



hyp_rid=grid.fit(inpv_2_rfe,ytrain)



print('Best hyperparameter is :',hyp_rid.best_params_ )

print('Score :',hyp_rid.best_score_ )

#ElasticNet Regularization
enet=ElasticNet()     #hyper paramater of elastic  is alpha and l1_ratio

param={'alpha':[0.000001,0.00001,0.0001,0.001,0.01,0.1,1,2,5,8,10,50,60,70,90,100],'l1_ratio':[0.1,.2,0.3,0.4,.5,0.6,0.7,0.8,0.9]}



grid=GridSearchCV(enet,param_grid=param,cv=5,scoring='r2')



hyp_rid=grid.fit(inpv_2_rfe,ytrain)



print('Best hyperparameter is :',hyp_rid.best_params_ )

print('Score :',hyp_rid.best_score_ )
#model-9  building a model using Ridge regularization
rid=Ridge(alpha=8)                 

rid.fit(inpv_2_rfe,ytrain)



ypred_train=rid.predict(inpv_2_rfe)

ypred_test=rid.predict(xtest_rfe)



rmse_train=np.sqrt(mean_squared_error(ytrain,ypred_train))

rmse_test=np.sqrt(mean_squared_error(ytest,ypred_test))



r2_train=r2_score(ytrain,ypred_train)

r2_test=r2_score(ytest,ypred_test)



adj_r2_train=1-(((1-r2_train)*(len(inpv_2_rfe)-1))/(len(inpv_2_rfe)-len(inpv_2_rfe.columns)-1))

adj_r2_test=1-(((1-r2_test)*(len(xtest_rfe)-1))/(len(xtest_rfe)-len(xtest_rfe.columns)-1))



print('rmse_train:',rmse_train,'\n','rmse_test:',rmse_test,'\n','r2_score_train:',r2_train,'\n','r2_score_test:',r2_test,'\n','adj_r2_train:',adj_r2_train,'\n','adj_r2_test:',adj_r2_test)
linreg_model_with_signifi_feat_RFECV_interaction_ridge = pd.Series({

                     'Model_Name': "LRM_with_signif_feat_RFECV_interctn_Ridge",

                     'RMSE':rmse_test,

                     'R-Squared':r2_test ,

                     'Adj. R-Squared': adj_r2_test     

                   })



score_card = score_card.append(linreg_model_with_signifi_feat_RFECV_interaction_ridge , ignore_index=True)



#call scorecard

score_card
#Stochastic Gradient Descent
#import library

from sklearn.linear_model import SGDRegressor
sgd=SGDRegressor()

sgd.fit(inpv_2_rfe,ytrain)





ypred_train=sgd.predict(inpv_2_rfe)

ypred_test=sgd.predict(xtest_rfe)



rmse_train=np.sqrt(mean_squared_error(ytrain,ypred_train))

rmse_test=np.sqrt(mean_squared_error(ytest,ypred_test))



r2_train=r2_score(ytrain,ypred_train)

r2_test=r2_score(ytest,ypred_test)



adj_r2_train=1-(((1-r2_train)*(len(inpv_2_rfe)-1))/(len(inpv_2_rfe)-len(inpv_2_rfe.columns)-1))

adj_r2_test=1-(((1-r2_test)*(len(xtest_rfe)-1))/(len(xtest_rfe)-len(xtest_rfe.columns)-1))



print('rmse_train:',rmse_train,'\n','rmse_test:',rmse_test,'\n','r2_score_train:',r2_train,'\n','r2_score_test:',r2_test,'\n','adj_r2_train:',adj_r2_train,'\n','adj_r2_test:',adj_r2_test)
linreg_model_with_signifi_feat_RFECV_interaction_sgd = pd.Series({

                     'Model_Name': "LRM_with_signif_feat_RFECV_intrctn_SGD",

                     'RMSE':rmse_test,

                     'R-Squared':r2_test ,

                     'Adj. R-Squared': adj_r2_test     

                   })



score_card = score_card.append(linreg_model_with_signifi_feat_RFECV_interaction_sgd  , ignore_index=True)



#call scorecard

score_card
#SVM Algorithm
from sklearn import svm

model=svm.SVR()

model.fit(inpv_2_rfe,ytrain)





ypred_train=model.predict(inpv_2_rfe)

ypred_test=model.predict(xtest_rfe)



rmse_train=np.sqrt(mean_squared_error(ytrain,ypred_train))

rmse_test=np.sqrt(mean_squared_error(ytest,ypred_test))



r2_train=r2_score(ytrain,ypred_train)

r2_test=r2_score(ytest,ypred_test)



adj_r2_train=1-(((1-r2_train)*(len(inpv_2_rfe)-1))/(len(inpv_2_rfe)-len(inpv_2_rfe.columns)-1))

adj_r2_test=1-(((1-r2_test)*(len(xtest_rfe)-1))/(len(xtest_rfe)-len(xtest_rfe.columns)-1))



print('rmse_train:',rmse_train,'\n','rmse_test:',rmse_test,'\n','r2_score_train:',r2_train,'\n','r2_score_test:',r2_test,'\n','adj_r2_train:',adj_r2_train,'\n','adj_r2_test:',adj_r2_test)





linreg_model_with_signifi_feat_RFECV_interaction_SVM = pd.Series({

                     'Model_Name': "LRM_with_signif_feat_RFECV_intrctn_SVM",

                     'RMSE':rmse_test,

                     'R-Squared':r2_test ,

                     'Adj. R-Squared': adj_r2_test     

                   })



score_card = score_card.append(linreg_model_with_signifi_feat_RFECV_interaction_SVM  , ignore_index=True)



#call scorecard

score_card
#Automatic Relevance Determination Regression (ARD) Algorithm
from sklearn.linear_model import ARDRegression
model=ARDRegression()

model.fit(inpv_2_rfe,ytrain)
ypred_train=model.predict(inpv_2_rfe)

ypred_test=model.predict(xtest_rfe)



rmse_train=np.sqrt(mean_squared_error(ytrain,ypred_train))

rmse_test=np.sqrt(mean_squared_error(ytest,ypred_test))



r2_train=r2_score(ytrain,ypred_train)

r2_test=r2_score(ytest,ypred_test)



adj_r2_train=1-(((1-r2_train)*(len(inpv_2_rfe)-1))/(len(inpv_2_rfe)-len(inpv_2_rfe.columns)-1))

adj_r2_test=1-(((1-r2_test)*(len(xtest_rfe)-1))/(len(xtest_rfe)-len(xtest_rfe.columns)-1))



print('rmse_train:',rmse_train,'\n','rmse_test:',rmse_test,'\n','r2_score_train:',r2_train,'\n','r2_score_test:',r2_test,'\n','adj_r2_train:',adj_r2_train,'\n','adj_r2_test:',adj_r2_test)
linreg_model_with_signifi_feat_RFECV_interaction_ARD = pd.Series({

                     'Model_Name': "LRM_with_signif_feat_RFECV_intrctn_ARD",

                     'RMSE':rmse_test,

                     'R-Squared':r2_test ,

                     'Adj. R-Squared': adj_r2_test     

                   })



score_card = score_card.append(linreg_model_with_signifi_feat_RFECV_interaction_ARD  , ignore_index=True)



#call scorecard

score_card
#Bayesian Ridge Regression Algorithm
from sklearn.linear_model import BayesianRidge
model=BayesianRidge()

model.fit(inpv_2_rfe,ytrain)



ypred_train=model.predict(inpv_2_rfe)

ypred_test=model.predict(xtest_rfe)



rmse_train=np.sqrt(mean_squared_error(ytrain,ypred_train))

rmse_test=np.sqrt(mean_squared_error(ytest,ypred_test))



r2_train=r2_score(ytrain,ypred_train)

r2_test=r2_score(ytest,ypred_test)



adj_r2_train=1-(((1-r2_train)*(len(inpv_2_rfe)-1))/(len(inpv_2_rfe)-len(inpv_2_rfe.columns)-1))

adj_r2_test=1-(((1-r2_test)*(len(xtest_rfe)-1))/(len(xtest_rfe)-len(xtest_rfe.columns)-1))



print('rmse_train:',rmse_train,'\n','rmse_test:',rmse_test,'\n','r2_score_train:',r2_train,'\n','r2_score_test:',r2_test,'\n','adj_r2_train:',adj_r2_train,'\n','adj_r2_test:',adj_r2_test)
linreg_model_with_signifi_feat_RFECV_interaction_BR = pd.Series({

                     'Model_Name': "LRM_with_signif_feat_RFECV_intrctn_BR",

                     'RMSE':rmse_test,

                     'R-Squared':r2_test ,

                     'Adj. R-Squared': adj_r2_test     

                   })



score_card = score_card.append(linreg_model_with_signifi_feat_RFECV_interaction_BR  , ignore_index=True)



#call scorecard

score_card
#Passive Aggressive Algorithms
from sklearn.linear_model import PassiveAggressiveRegressor
model=PassiveAggressiveRegressor()

model.fit(inpv_2_rfe,ytrain)



ypred_train=model.predict(inpv_2_rfe)

ypred_test=model.predict(xtest_rfe)



rmse_train=np.sqrt(mean_squared_error(ytrain,ypred_train))

rmse_test=np.sqrt(mean_squared_error(ytest,ypred_test))



r2_train=r2_score(ytrain,ypred_train)

r2_test=r2_score(ytest,ypred_test)



adj_r2_train=1-(((1-r2_train)*(len(inpv_2_rfe)-1))/(len(inpv_2_rfe)-len(inpv_2_rfe.columns)-1))

adj_r2_test=1-(((1-r2_test)*(len(xtest_rfe)-1))/(len(xtest_rfe)-len(xtest_rfe.columns)-1))



print('rmse_train:',rmse_train,'\n','rmse_test:',rmse_test,'\n','r2_score_train:',r2_train,'\n','r2_score_test:',r2_test,'\n','adj_r2_train:',adj_r2_train,'\n','adj_r2_test:',adj_r2_test)
linreg_model_with_signifi_feat_RFECV_interaction_PAR = pd.Series({

                     'Model_Name': "LRM_with_signif_feat_RFECV_intrctn_PAR",

                     'RMSE':rmse_test,

                     'R-Squared':r2_test ,

                     'Adj. R-Squared': adj_r2_test     

                   })



score_card = score_card.append(linreg_model_with_signifi_feat_RFECV_interaction_PAR  , ignore_index=True)



#call scorecard

score_card
#Robust Multivariate Regression Algorithms(TheilSenRegressor
from sklearn.linear_model import TheilSenRegressor
model=TheilSenRegressor()

model.fit(inpv_2_rfe,ytrain)



ypred_train=model.predict(inpv_2_rfe)

ypred_test=model.predict(xtest_rfe)



rmse_train=np.sqrt(mean_squared_error(ytrain,ypred_train))

rmse_test=np.sqrt(mean_squared_error(ytest,ypred_test))



r2_train=r2_score(ytrain,ypred_train)

r2_test=r2_score(ytest,ypred_test)



adj_r2_train=1-(((1-r2_train)*(len(inpv_2_rfe)-1))/(len(inpv_2_rfe)-len(inpv_2_rfe.columns)-1))

adj_r2_test=1-(((1-r2_test)*(len(xtest_rfe)-1))/(len(xtest_rfe)-len(xtest_rfe.columns)-1))



print('rmse_train:',rmse_train,'\n','rmse_test:',rmse_test,'\n','r2_score_train:',r2_train,'\n','r2_score_test:',r2_test,'\n','adj_r2_train:',adj_r2_train,'\n','adj_r2_test:',adj_r2_test)
linreg_model_with_signifi_feat_RFECV_interaction_TSR = pd.Series({

                     'Model_Name': "LRM_with_signif_feat_RFECV_intrctn_TSR",

                     'RMSE':rmse_test,

                     'R-Squared':r2_test ,

                     'Adj. R-Squared': adj_r2_test     

                   })



score_card = score_card.append(linreg_model_with_signifi_feat_RFECV_interaction_TSR  , ignore_index=True)



#call scorecard

score_card
#Random Fprest 
from sklearn.ensemble import RandomForestRegressor
inpv_2_rfe
regressor=RandomForestRegressor(n_estimators=10,random_state=2)

regressor.fit(inpv_2_rfe,ytrain)

ypred_train=regressor.predict(inpv_2_rfe)

ypred_test=regressor.predict(xtest_rfe)



rmse_train=np.sqrt(mean_squared_error(ytrain,ypred_train))

rmse_test=np.sqrt(mean_squared_error(ytest,ypred_test))



r2_train=r2_score(ytrain,ypred_train)

r2_test=r2_score(ytest,ypred_test)



adj_r2_train=1-(((1-r2_train)*(len(inpv_2_rfe)-1))/(len(inpv_2_rfe)-len(inpv_2_rfe.columns)-1))

adj_r2_test=1-(((1-r2_test)*(len(xtest_rfe)-1))/(len(xtest_rfe)-len(xtest_rfe.columns)-1))



print('rmse_train:',rmse_train,'\n','rmse_test:',rmse_test,'\n','r2_score_train:',r2_train,'\n','r2_score_test:',r2_test,'\n','adj_r2_train:',adj_r2_train,'\n','adj_r2_test:',adj_r2_test)
linreg_model_with_RandomForestRegressor = pd.Series({

                     'Model_Name': "LRM_with_RandomForestRegressor",

                     'RMSE':rmse_test,

                     'R-Squared':r2_test ,

                     'Adj. R-Squared': adj_r2_test     

                   })



score_card = score_card.append(linreg_model_with_RandomForestRegressor  , ignore_index=True)



#call scorecard

score_card
#Selecting best model
score_card['RMSE']=score_card['RMSE'].astype('float')
score_card.plot(secondary_y=['R-Squared','Adj. R-Squared'])



# display just the plot

plt.axvline(x=15, color='black',label='BEST model')

plt.title('Selecting best model',fontsize=20)

plt.show()
xtrain.head(1)
linreg_model_with_RandomForestRegressor_2 = pd.Series({

                     'Model_Name': "LRM_with_RandomForestRegressor_2",

                     'RMSE':rmse_test,

                     'R-Squared':r2_test ,

                     'Adj. R-Squared': adj_r2_test     

                   })



score_card = score_card.append(linreg_model_with_RandomForestRegressor_2  , ignore_index=True)



#call scorecard

score_card
score_card.plot(secondary_y=['R-Squared','Adj. R-Squared'])



# display just the plot

plt.axvline(x=16, color='black',label='BEST model')

plt.title('Selecting best model',fontsize=20)

plt.show()