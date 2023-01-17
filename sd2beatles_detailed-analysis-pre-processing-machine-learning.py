import numpy as np

import pandas as pd

import plotly.figure_factory as ff

import category_encoders as ce

from sklearn.preprocessing import LabelEncoder

import copy

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn import svm

from sklearn import metrics

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

import statsmodels.api as sm
train=pd.read_csv('../input/big-mart-sales-prediction/Train.csv')

train.head()
def data_modified(df,pred=None):

    obs=df.shape[0] # return the number of rows

    types=df.dtypes #return the type of data

    counts=df.apply(lambda x:x.count()) #store the number of not null values for eac column

    nulls=df.apply(lambda x:x.isnull().sum())#store the total number of nulls for each column

    distincts=df.apply(lambda x:x.unique().shape[0])#sotre the unique memeber of each column

    missing_ratio=round(df.isnull().sum()/obs*100,2) 

    skewness=round(df.skew(),2)

    kurtosis=round(df.kurt(),2)

    if pred is None:

        cols=['types','counts','nulls','distincts','missing_ratio','skewness','kurtosis']

        result=pd.concat([types,counts,nulls,distincts,missing_ratio,skewness,kurtosis],axis=1)

    else:

        corr=round(df.corr()[pred],2) #computing correlation between each column and SalePrice

        corr_name='corr '+pred

        result=pd.concat([types,counts,nulls,distincts,missing_ratio,skewness,kurtosis,corr],axis=1)

        cols=['types','counts','nulls','distincts','missing_ratio','skewness','kurtosis',corr_name]

    result.columns=cols

    result=result.sort_values(by=corr_name,ascending=False)

    result=result.reset_index()

    return result

    
data_modified(train,pred='Item_Outlet_Sales')
object_column_number=[i for i,j in enumerate(train.dtypes) if j=='object']

object_columns=train.iloc[:,object_column_number]

#drop item_idtentifier which does not help in our predicting model

object_columns=object_columns.drop(['Item_Identifier','Outlet_Identifier'],axis=1)

object_columns_labels=object_columns.columns

table=[[object_columns_labels[i],list(object_columns.iloc[:,i].unique())] for i in range(5)]

table.insert(0,['columns','members'])
#prepare a table containing columns and their members

result=ff.create_table(table)

result.layout.annotations[5].font.size=10

result.layout.annotations[11].font.size=10

result.layout.update(width=1700)

result.show()

data=train.copy()

data.pivot_table('Item_Outlet_Sales',index='Outlet_Size',columns='Outlet_Location_Type',aggfunc='count',margins=True)
data.pivot_table('Item_Outlet_Sales',index='Outlet_Type',columns='Outlet_Location_Type',aggfunc='count',margins=True)
data.groupby(['Outlet_Location_Type','Outlet_Type'])['Outlet_Size'].value_counts()
display(data.pivot_table(['Item_Outlet_Sales','Item_MRP'],index=[data.Outlet_Type,data.Outlet_Location_Type],aggfunc='mean'))

#to find out the unique member of business belong to Grocery Store currently operating in tier 1

data[(data.Outlet_Location_Type=='Tier 1')&(data.Outlet_Type=='Grocery Store')].loc[:,'Outlet_Size'].unique()
index=data[(data.Outlet_Size.isnull())&(data.Outlet_Type=='Grocery Store')].loc[:,'Outlet_Size'].index

#assign 'small'

data.loc[index,'Outlet_Size']='Small'
index=data[data.Outlet_Size.isnull()].loc[:,'Outlet_Size'].index

data.loc[index,'Outlet_Size']='Small'

data.isnull().any()

base_data=data.drop(['Item_Identifier','Outlet_Identifier',],axis=1)

base_columns=base_data.columns
#Chosee every row with Item_Weight having some value 

base_data=base_data[base_data.Item_Weight.isnull()==False]

predict_value=base_data.Item_Weight.mean()

#Generate random indicies

random_rows=np.random.choice(base_data.index,np.int(base_data.index.shape[0]*0.25))

#Store up the true value of Item_Weight

test_value=base_data.loc[random_rows,'Item_Weight']

error=test_value.map(lambda x:np.abs(x-predict_value))

print('Average Baseline Error:{0} degrees'.format(round(np.mean(error),2)))


data.Item_Fat_Content=data.Item_Fat_Content.map(lambda x: 'low fat' if x=='Low Fat'and 'LF'and 'low fat' else 'regular')



# mapping Item_Fat_Content to either 1 or 2

ce_ord=ce.OrdinalEncoder()

data.Item_Fat_Content=ce_ord.fit_transform(data.Item_Fat_Content)



#maping Item_Type to ordinal category from 1 to 16

ce_ord=ce.OrdinalEncoder()

data.Item_Type=ce_ord.fit_transform(data.Item_Type)



#Outlet location



data.Outlet_Location_Type=data.Outlet_Location_Type.map(lambda x:x[-1]).astype(int)



#Outlet Type

ce_ord=ce.OrdinalEncoder()

data.Outlet_Type=ce_ord.fit_transform(data.Outlet_Type)



#Outlet Size

data.Outlet_Size=data.Outlet_Size.map(lambda x: 0 if x=='Small' else 1 if x=='Medium' else  2 if x=='High' else x )







data_null_free=data[data.Item_Weight.isnull()==False]

data_null=data[data.Item_Weight.isnull()==True]

X=data_null_free.drop(['Item_Identifier','Item_Weight','Outlet_Identifier'],axis=1)

y=data_null_free['Item_Weight']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

#Instantiate model with 2000 decision trees 

rf=RandomForestRegressor(n_estimators=2000,random_state=42,min_samples_split=3)

#train the model on training data

rf.fit(X_train,y_train)

predictions=rf.predict(X_test)

errors=abs(predictions-y_test)

print('Mean Absoulte Error:{0} degrees'.format(round(np.mean(errors),2)))
#Calculate mean absoulte the percentage error

error_percent=100*errors/y_test

accuracy=100-np.mean(error_percent)

print("Accuracy: {0}%".format(round(accuracy,2)))

importance=list(rf.feature_importances_)

features=X.columns

features_importances=[(features,round(importances*100,2)) for features,importances in zip(features,importance)]

features_importances=sorted(features_importances,key=lambda x:x[1],reverse=True)

[print('Variable: {:30} Importance: {}'.format(*pair)) for pair in features_importances];
sns.set()

fig,ax=plt.subplots(figsize=(11,5))

plt.bar(features,importance,alpha=0.7,color='coral')

ax.set_xticklabels(features,rotation=45)

ax.set_xlabel('features',fontsize=15)

ax.set_ylabel('importance(percentage)',fontsize=15)

ax.set_title('Features Importances',fontsize=20)
data_null=data_null.drop(['Item_Identifier','Item_Weight','Outlet_Identifier'],axis=1)

#store the indicies with missing values

data_null_index=data_null.index

#predict the values with our random forest model

predict=rf.predict(data_null)

#assign them to the null values 

data.iloc[data_null_index,1]=predict
# double check wether we have removed all the missing values in every feature

data.isnull().any()
# Assigin the finding results to original data(train)

train.Outlet_Size=data.Outlet_Size

train.Outlet_Size=train.Outlet_Size.map(lambda x: 'small' if x==0 else 'medium' if x==1 else 'high')

train.Outlet_Size=train.Outlet_Size.astype('category')

train.Item_Weight=data.Item_Weight
#Finding out the correlation with Item_Outlet_Sales

result=data.corr()['Item_Outlet_Sales']

print('Correlation with Item_Outlet_Sales')

print('-'*100)

#Sort the results in a descedning order

result=result.sort_values(ascending=False)

display(result)

result_columns=result.index

data=data.loc[:,result_columns]
sns.set()

fig,axes=plt.subplots(figsize=(15,10))

sns.scatterplot(x=data.Item_MRP,y=data.iloc[:,0],ax=axes,hue=data.Outlet_Size,palette='Spectral')
fig,ax=plt.subplots(3,1,figsize=(15,20))

weight_vis=data.Item_Weight*data.Item_Visibility

weight_by_vis=data.Item_Weight/data.Item_Visibility

sns.scatterplot(x=data.Item_Visibility,y=data.iloc[:,0],ax=ax[0],hue=data.Outlet_Location_Type,palette='Spectral')

ax[0].set_title('Correaltion with Sales: {0}'.format(data.Item_Outlet_Sales.corr(data.Item_Visibility)),fontsize=14)

sns.scatterplot(x=weight_vis,y=data.Item_Outlet_Sales,hue=data.Outlet_Size,palette='Spectral',ax=ax[1])

ax[1].set_title('Correaltion with Sales: {0}'.format(data.Item_Outlet_Sales.corr(weight_vis)),fontsize=14)

sns.scatterplot(x=weight_by_vis,y=data.iloc[:,0],hue=data.Outlet_Size,palette='Spectral',ax=ax[2])
#remove the outliers

data=data[data.Item_Weight*data.Item_Visibility<4.7]

data=data[data.Item_Weight/data.Item_Visibility<2500]

data=data[data.Item_Outlet_Sales<12000]

fig,ax=plt.subplots(3,1,figsize=(15,20))

weight_fat=data.Item_Weight*data.Item_Fat_Content

weight_by_fat=data.Item_Weight/data.Item_Fat_Content

sns.scatterplot(x=data.Item_Weight,y=data.iloc[:,0],ax=ax[0],hue=data.Outlet_Location_Type,palette='Spectral')

ax[0].set_title('Correlation between Item_Weight and Sale revenue',fontsize=20)

ax[0].text(x=4,y=11000,s='Correaltion:{0}'.format(data.Item_Outlet_Sales.corr(data.Item_Weight)),fontsize=14)

sns.scatterplot(x=weight_fat,y=data.iloc[:,0],ax=ax[1],hue=data.Outlet_Location_Type,palette='Spectral')

ax[1].set_title('Correlation between Weight_fat and Sale revenue',fontsize=20)

ax[1].text(x=4,y=11000,s='Correaltion:{0}'.format(data.Item_Outlet_Sales.corr(weight_fat)),fontsize=14)

data.Item_Outlet_Sales.corr(weight_vis)

sns.scatterplot(x=weight_by_fat,y=data.iloc[:,0],ax=ax[2],hue=data.Outlet_Location_Type,palette='Spectral')

ax[2].set_title('Correlation between Weight_by_fat and Sale revenue',fontsize=20)

ax[2].text(x=2.5,y=11000,s='Correaltion:{0}'.format(data.Item_Outlet_Sales.corr(weight_by_fat)),fontsize=14)

fig,axes=plt.subplots(2,2,figsize=(15,12))

sns.boxplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',ax=axes[0,0],data=data)

sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',ax=axes[0,1],data=data)

sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',ax=axes[1,0],data=data)

sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',ax=axes[1,1],data=data)
data=data.drop(['Item_Weight','Item_Visibility'],axis=1)

y=data.Item_Outlet_Sales

X=data.iloc[:,1:]

X.iloc[:,1:6]=X.iloc[:,1:6].astype('category')

# Preparing the data sets

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#Train the model 

sl=LinearRegression()

sl.fit(X_train,y_train)
#predict the taraget variable based on X_test

predict_sl=sl.predict(X_test)

#Calculate Mean Squared Error

mse=np.mean((predict_sl-y_test)**2)

#Score

sl_score=np.sqrt(mse)

print('Score of Simple regrssion model : {0}'.format(sl_score))
r=Ridge(alpha=0.5,solver='cholesky')

r.fit(X_train,y_train)

predict_r=r.predict(X_test)

mse=np.mean((predict_r-y_test)**2)

r_score=np.sqrt(mse)

r_score

print('Score of Rigid Regression : {0}'.format(sl_score))
l=Lasso(alpha=0.01)

l.fit(X_train,y_train)

predict_r=r.predict(X_test)

mse=np.mean((predict_r-y_test)**2)

l_score=np.sqrt(mse)

l_score

print('Score of Lasso : {0}'.format(l_score))
en=ElasticNet(alpha=0.01)

en.fit(X_train,y_train)

predict_r=en.predict(X_test)

mse=np.mean((predict_r-y_test)**2)

l_score=np.sqrt(mse)

l_score

print('Score of Elastic Net: {0}'.format(l_score))
svm=SVR(epsilon=15,kernel='linear')

svm.fit(X_train,y_train)

predict_r=svm.predict(X_test)

mse=np.mean((predict_r-y_test)**2)

l_score=np.sqrt(mse)

l_score

print('Score of Support Vector machine: {0}'.format(l_score))
dtr=DecisionTreeRegressor()

dtr.fit(X_train,y_train)

predict_r=dtr.predict(X_test)

mse=np.mean((predict_r-y_test)**2)

l_score=np.sqrt(mse)

l_score
y=train.Item_Outlet_Sales

X=train.iloc[:,[2,4,5,7,8,9,10]]

columns=['Item_MRP',

         'Item_Fat_Content',

         'Item_Type',

         'Outlet_Size',

         'Outlet_Location_Type',

         'Outlet_Type',

        'Outlet_Establishment_Year']

#rearrange the columns

X=pd.DataFrame(X,columns=columns)

#All the object varibles are converted into categories one

X.iloc[:,1:]=X.iloc[:,1:7].astype('category')

X=pd.get_dummies(X)

#Create a OLS model

model=sm.OLS(y,X)

results=model.fit()
results.summary()