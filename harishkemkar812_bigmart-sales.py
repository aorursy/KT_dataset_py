import numpy as np 

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression  

from sklearn.metrics import r2_score,mean_squared_error



from math import sqrt



import seaborn as sns
## Below is to check which is your current working directory while using kAggle and to extract files into your dataset



import os

print(os.listdir("../input/bmdataset"))
## Importing Train data Set

##pd.read_csv('E:/Harish/DataScience/Python_eBook/Projects_for_Submission/Project4_Movielens/movies.dat', delimiter = '::',engine='python')

#E:/Harish/DataScience/Machine learning/Demo Datasets/Lesson 4/bigmart_train.csv



train =  pd.read_csv('../input/bmdataset/bigmart_train.csv',delimiter = ',' ,engine = 'python')
train.head()
train.shape
train.isnull().sum()



##  here we can see Item_weight and Out let Size have alot of null values 
train['Item_Fat_Content'].unique()



##we can see only two type of values for Fat content column but there are multiple Strings used to depict  same values

## Like Low Fat,LF and low fat --  all represent same value of low fat
train['Outlet_Establishment_Year'].unique().max()



train['Outlet_Establishment_Year'].unique().min()

## We can se outlet establishment year varies from  1985 to  2009 , based on these we can find out the age of an outlet 
## Calculating Outlet Age 





train['Outlet_Age'] =  2019 - train['Outlet_Establishment_Year']



print(train['Outlet_Age'])





## Adding the outlet age column to the DatSet

train.head()
## Checking unique values in outlet_size

train['Outlet_Size'].unique()
train.describe()



train.info()
train['Item_Fat_Content'].value_counts()
train['Outlet_Size'].mode()
train['Outlet_Size'].isnull().value_counts()



## there are 2410 records where outlet size is null 

## mode is the most frequent number in the list

## here in our case mode is "Medium"
train['Outlet_Size'] = train['Outlet_Size'].fillna("Medium")
train['Outlet_Size'].isnull().value_counts()



## Now we can see no null value in OUtlet_Size , and null have been replaced by "Medium"
## Checking Data fiels Item_weight 

train['Item_Weight'].isnull().value_counts()



## So we can seee here 7060 rows are not null and 1463 rows are null

## So we will replace item weight with mean weights 

train['Item_Weight'].mean()
train['Item_Weight'] =  train['Item_Weight'].fillna(train['Item_Weight'].mean())
train['Item_Weight'].isnull().any()



## Now we can see there is no null value in  'Item_Weight' column
### Checking Item_visibility column





train['Item_Visibility'].hist(bins =20)

## Also to see outliers in  ['Item_Visibility'] column we can use box plot



sns.boxplot(x = train['Item_Visibility'])





## Hence we can see outliers lie between 0.20 and 0.30 and maximum data is between  0.0 to 1.0
Q1 =  train['Item_Visibility'].quantile(0.25)

Q3 =  train['Item_Visibility'].quantile(0.75)



#print(Q1)
## now calculating IQR (inter Quartile range )



IQR =  Q3 - Q1



print(IQR)
## NoW removing Outliers 



#boston_df_out = boston_df_o1[~((boston_df_o1 < (Q1 - 1.5 * IQR)) |(boston_df_o1 > (Q3 + 1.5 * IQR))).any(axis=1)]

#print(boston_df_o1 < (Q1 - 1.5 * IQR)) |(boston_df_o1 > (Q3 + 1.5 * IQR))



filt_train =  train.query('(@Q1 - 1.5*@IQR) <= Item_Visibility <=(@Q3 + 1.5*@IQR)')
filt_train
## Now checking if outliers are removed from the filt_train data sets 



filt_train.shape



### (8379, 13)
train.shape



###(8523, 13) , hence we can see outliers records have been removed from the data sets 



## Replacing new filtered data frame with train dataframe  





train =  filt_train
train.shape

train.info()

## Checking Item Visibility column 



#train['Item_Visibility'].isnull().any()

## no null values in this  Column



## Modifying Item_visibility into categorical variable



#train['Item_Visibility'].value_counts()

#pd.cut(np.array([1, 7, 5, 4, 6, 3]),3, labels=["bad", "medium", "good"])



#newdf = pd.cut(train['Item_Visibility'],3,labels = ['Low Viz','Viz','High Viz'])

#train['Item_Visibility_bins_hk']  =  pd.cut(train['Item_Visibility'],3,labels = ['Low Viz','Viz','High Viz'])
train['Item_Visibility_bins']  =  pd.cut(train['Item_Visibility'],[0.000,0.005,0.13,0.2],labels = ['Low Viz','Viz','High Viz'])
#train['Item_Visibility_bins_hk'].value_counts()
train['Item_Visibility_bins'].isnull().value_counts()



## We can see there are null values after chnaging the continuous variable into categorical variables 

## We need to replace these null values with lowvisibility variable 
train['Item_Visibility_bins'] = train['Item_Visibility_bins'].fillna("Low Viz")
## Now checking if we still have any null values left in the Item_visibility column



train['Item_Visibility_bins'].isnull().any()



##so no null values in thiscolumn
train['Item_Fat_Content'] =  train['Item_Fat_Content'].replace(['low fat','LF'],'Low Fat')



train['Item_Fat_Content'] =  train['Item_Fat_Content'].replace(['reg'],'Regular')

train['Item_Fat_Content'].value_counts()



## so now we can see this column contains only two types of categorical variables -- Low Fat  and Regular
## lets start encoding all Categorical variables into  numericals using label encoder 

## From sklear.preprocessing import LabelEncoder



le = LabelEncoder()



train['Item_Fat_Content'].value_counts()

train['Item_Visibility_bins'].value_counts()

train['Outlet_Location_Type'].value_counts()



train['Outlet_Location_Type'].isnull().value_counts()
train['Outlet_Location_Type'].value_counts()



## Value counts this is already a categorical variable 
## Now using label encoding 



train['Item_Fat_Content'] =  le.fit_transform(train['Item_Fat_Content'])

train['Item_Visibility_bins'] =  le.fit_transform(train['Item_Visibility_bins'])

train['Outlet_Location_Type'] =  le.fit_transform(train['Outlet_Location_Type'])

train.info()
#train['Item_Fat_Content'].value_counts()

#train['Item_Visibility_bins'].value_counts()

#train['Outlet_Location_Type'].value_counts()



## so we have converted these three variables into categorical variables , 

## These includes both numerical and categorical values

##



## Creating OUtlet_type 





dummy = pd.get_dummies(train['Outlet_Type'])



dummy.head()

## Now merging bot train and Test Data  , beow is the new train data 





train = pd.concat([train,dummy],axis = 1)
train.isnull().any()
## Now taking only relenatvariables for model creation 



## Dropping irrelevant columns  





train = train.drop(['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size','Outlet_Type'],axis =1)
train.columns
## Finding Correlation in between variables in between indipendent variables 

train_corr =  train.corr()
## PLotting Heat map for correlation  

sns.heatmap(data = train_corr,square =  True,cmap = 'bwr' )
X = train.drop('Item_Outlet_Sales',axis =1)

Y = pd.DataFrame(train.Item_Outlet_Sales)
X.columns

Y.columns



X.info()


##Splitting Available data into train aand test data



##from sklearn import model_selection -- This is no longer used , insted below is used 

from sklearn.model_selection import train_test_split



Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.3)



Xtrain.info()
### Creating  Regression model witn test Data 

## Model Number 1 , linear regression 

lin = LinearRegression()



lin.fit(Xtrain,Ytrain)

predictions =  lin.predict(Xtest)
# Now getting RMSE for linear regression model 



sqrt(mean_squared_error(Ytest,predictions))



lin.score(Xtest,Ytest)
## ridge model  



from sklearn.linear_model import Ridge 



ridgereg = Ridge(alpha = 0.001,normalize = True)



ridgereg.fit(Xtrain,Ytrain)

pred_rig = ridgereg.predict(Xtest)





sqrt(mean_squared_error(Ytest,pred_rig))



ridgereg.score(Xtest,Ytest)

### Lasso model  

from sklearn.linear_model import Lasso



lassoreg = Lasso(alpha = 0.001,normalize = True)

lassoreg.fit(Xtrain,Ytrain)



pred_lasso = lassoreg.predict(Xtest) 



sqrt(mean_squared_error(Ytest,pred_lasso))



lassoreg.score(Xtest,Ytest)
from sklearn.linear_model import ElasticNet





elasreg = ElasticNet(alpha = 0.001,normalize = True)

elasreg.fit(Xtrain,Ytrain)



pred_elasreg = elasreg.predict(Xtest) 



sqrt(mean_squared_error(Ytest,pred_elasreg))



lassoreg.score(Xtest,Ytest)

### hence all three models are giving same values of model Score and RMSE 