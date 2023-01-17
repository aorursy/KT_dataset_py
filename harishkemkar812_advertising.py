

import os

print(os.listdir("../input/"))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





## for creating  train and test data 



from sklearn.model_selection import train_test_split

## For creating model  



from sklearn.linear_model import LinearRegression 



#E:\Harish\DataScience\Machine learning\Demo Datasets\Lesson 4



adv_df = pd.read_csv('../input/Advertising.csv',delimiter =  ',' ,engine = 'python' )  

#adv_df.info()



## Features TV,radio,newspapaer , 

## Label -  sales 

## 200 rows nota single null value



## plotting data 

#adv_df['sales'].plot()

#adv_df['TV'].plot()

#adv_df['newspaper'].plot()



#adv_df.plot()

## Histogram for sales numbers 

plt.hist(adv_df['sales'],color = 'r')
## Histogram for TV numbers 

plt.hist(adv_df['TV'],color = 'b')
## Histogram for Radio numbers 

plt.hist(adv_df['radio'],color = 'g')
## Scatter plot between Tv aand sales , to see if there seems any linear relationship between TV and Sales 



plt.close('All')

plt.scatter(adv_df['TV'],adv_df['sales'])

plt.xlabel('TV')

plt.ylabel('Sales')

plt.title('TV and sales ')

## Scatter plot between Sales and newspaper , to see if there is any linear relationship between sales and newspaper



plt.close('All')



plt.scatter(adv_df['newspaper'],adv_df['sales'])

plt.xlabel('newspaper')

plt.ylabel('Sales')

plt.title('newspapaer and sales ')



## does not lok much linera dependent 

## Scatter plot between Sales and radio , to see if there is any linear relationship between sales and newspaper



plt.close('All')



plt.scatter(adv_df['radio'],adv_df['sales'])

plt.xlabel('radio')

plt.ylabel('Sales')

plt.title('radio and sales ')



## find correlation between variables  

adv_corr =  adv_df.corr()

print(adv_corr)



sns.heatmap(data = adv_corr, square = True , cmap = 'bwr')
## This means sales is highly corerelated with  TV and then with radio and not much with newspapaer

## Creating models   first ceating moel with three variable and then with two variables 

## will compare there scores then 





print(adv_df)


adv_df.columns =['Seq','TV','radio','newspaper','sales']



adv_df.head()

adv_df.drop(['Seq'],axis = 1)

## Dividing Dataset into train and test data 



# diabet_dataset =  pd.concat([df,df_target],axis = 1)



X_features =  pd.concat([adv_df['TV'],adv_df['radio'],adv_df['newspaper']],axis = 1)



Y_target = pd.DataFrame(adv_df['sales']) 



#print(X_features)

#print(Y_target)



## X_train,X_test,Y_train,Y_test =  train_test_split(X_features,Y_target,train_size = 0.3,test_size = 0.7) 

X_train,X_test,Y_train,Y_test = train_test_split(X_features,Y_target,train_size = 0.3,test_size = 0.7)





## Details of train and test Datasets 



#X_train.info() ## 60 records

#X_test.info() ## 140records

## Creating model with Three variables 





linereg = LinearRegression()

linereg.fit(X_train,Y_train)
linereg.intercept_
linereg.coef_
## Model output 



Y_model = linereg.predict(X_test)



print(Y_model)
## now we will evaluate the model



from sklearn.metrics import mean_squared_error 

from sklearn.metrics import mean_absolute_error



# Mean Squarred of this model 



msr1 =  mean_squared_error(Y_model,Y_test)

np.sqrt(msr1)



## 1.807228924192492

## Now creating model with two variables and dropping newspaper 



X_features_2 =  pd.concat([adv_df['TV'],adv_df['radio']],axis = 1)



Y_target_2 = pd.DataFrame(adv_df['sales']) 



X_train2,X_test2,Y_train2,Y_test2 = train_test_split(X_features_2,Y_target_2,train_size = 0.3,test_size = 0.7)



linereg2 = LinearRegression()

linereg2.fit(X_train2,Y_train2)
Y_model2 = linereg2.predict(X_test2)
msr2 =  mean_squared_error(Y_model2,Y_test2)
np.sqrt(msr2)



# 1.6776033398016748
## Creating models with different number of features and comparing those models 

data = pd.read_csv('../input/Advertising.csv',delimiter =  ',' ,engine = 'python' )  

#adv_df.info()



data.head()



data.drop([0])
data.head()

data.columns = ['Seq','TV','radio','newspaper','sales']
data.head()

data = pd.DataFrame(data.drop(['Seq'],axis = 1))

data.head()
## Visualise relation ship  tv , radio and newspaper and sales using scatter plot  

fig,axs = plt.subplots(1,3,sharey = True)

#fig,axs = plt.subplots(1,3)

data.plot(kind = 'scatter' , x= 'TV' ,y='sales',ax = axs[0] )

data.plot(kind = 'scatter' ,x = 'radio',y ='sales',ax = axs[1])

data.plot(kind = 'scatter',x = 'newspaper', y = 'sales',ax = axs[2])

          
## Creating model with only one depenendet variable 

feature_cals = ['TV']

label_cols  =  ['sales']

X =  data[feature_cals]

#print(X)

Y = data[label_cols]

#print(Y)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()



lm.fit(X,Y)
lm.intercept_
lm.coef_
## Now checking p value and others stats for this model

import statsmodels.formula.api as smf

## Applying ols on TV and sales 



lm =  smf.ols(formula = 'sales ~ TV' , data = data).fit()
## This gives you summary of the model 



lm.summary()



## below are the poinits to be noted 





## Anova section 

#R Squared = 0.612

# Adj r Squared =  0.610

# Fstatic (relevant for the model ) --  312.1

# pvalue of the model - rob(F-Statistics)  --  1.47e-42 , since this is less than  5 percent , 

#wwwill reject our null hypothesis , whcih was there isno relationship between TV and sales numbers







## Analysis of variable section 



# T statistics of the variable TV is 17.668

## and P > Tsatistics is 0.000 , whcih means we can reject our null hypothesisby 5 oerxent confidence interval 

## which was , that coefficient for TV is zero, 



## also if we see the 95% confidence interval we can see that for TV the 

## confidence isbetween 0.042 and 0.053 whcih does not include zero

## hence we can say that the coeficient for TV will be non zero and 



## from sklearn import model_selection

from sklearn.model_selection import train_test_split  



#print(data)

## Creating model with 3 variables

feature_cals = ['TV','radio','newspaper']

X = data[feature_cals]

X_train,Y_test,X_test,Y_test = train_test_split(X,Y,train_size = 0.3,test_size=0.7)
#X_test.info() ## 140 records

#X_train.info() # 60 
lm3  =  LinearRegression()

lm3.fit(X,Y)
## Import statsmodel.formula.api as smf

lm_model_3 =  smf.ols(formula = 'sales ~ TV + radio + newspaper',data = data).fit()



lm_model_2 =  smf.ols(formula = 'sales ~ TV + radio',data = data).fit()



lm =  smf.ols(formula = 'sales ~ TV' , data = data).fit()
lm.summary()
lm_model_2.summary()
lm_model_3.summary()
## Hence looking at all the three model we can say that model with two variables is the best 


