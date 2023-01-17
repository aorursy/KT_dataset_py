from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
import pandas as pd
import numpy as np
import os
import warnings
import copy
!ls ../input/Melbourne_housing_FULL.csv
content =  pd.read_csv("../input/Melbourne_housing_FULL.csv")
# to take a glance at the dataset
object_columns = content.select_dtypes(['object']).columns.tolist()
print(object_columns)
#print( content['Type'] )
# to convert the object type into category type
for column in object_columns:
    try:
        content[column] = content[column].astype( 'category' )
    except:
        pass


# to process the dataframe type
content['Date'] = pd.to_datetime( content['Date'] )
content[ 'Day' ] = [ each - content['Date'].min() for each in content['Date'] ]
print( content.columns )
content[ 'Age' ] = 2018 - content['YearBuilt']
# to look at the mean/cnt and other agg types
cnt = content.sort_values( 'Date',ascending=False ).dropna().groupby(['Date']).count()
mean = content.sort_values( 'Date',ascending=False ).dropna().groupby(['Date']).mean()
cnt.drop( columns=[ 'Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
        'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount', 'Day' ],inplace=True )
cnt.columns = ['Count']

mean.drop( columns= [ 'Distance', 'Postcode', 'Bedroom2', 'Car',
        'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude',
       'Propertycount', 'Age'],inplace=True)
desctiption = mean.join( cnt )

print( desctiption )
# to find out the NAN value
import matplotlib.pyplot as plt
null_value_cnt = content.isnull().sum()
percentage = null_value_cnt / content.__len__() * 100
percentage.plot( kind='bar',width=0.5,color = 'red',label='before' )
plt.ylim( -100.0,100.0 )
#to transform the data in TYPE AND REGION
content = content.drop( content[content['Price'].isnull()].index ,axis=0 )

content['Type']=content['Type'].map( {'h':'House','t':'Townhouse','u':'Unit'} )
content['Regionname'] = content['Regionname'].map({'Northern Metropolitan':'N Metro',
                                            'Western Metropolitan':'W Metro',
                                            'Southern Metropolitan':'S Metro',
                                            'Eastern Metropolitan':'E Metro',
                                            'South-Eastern Metropolitan':'SE Metro',
                                            'Northern Victoria':'N Vic',
                                            'Eastern Victoria':'E Vic',
                                            'Western Victoria':'W Vic'})
print(content.head())

# find out the outlier by visualizing the boxplot
content['Landsize'] = content['Landsize'].fillna( 0 )
print( content.columns )



from sklearn.impute import SimpleImputer as Imputer

index=[ ]
data =[ ]
for each in content[ 'Type' ].unique( ):
    print(each)

    conv = Imputer(missing_values=0.0, strategy='mean')
    raw = content[(content['Type'] == each) ]['Landsize']
    indics = raw.index.tolist()
    raw = np.reshape( raw.values,(-1,1) )
    convertion = conv.fit_transform(raw).reshape(-1)
    index.extend(indics)
    data.extend(convertion)
    #print( len( index ),'---->', len( data )  )

temp = pd.Series( data,name='Landsize',index=index )
content.update( temp )
print( 'landsize has been converted\n' )


content['Bathroom'] = content['Bathroom'].fillna( 1.0 )
index = content[ content['Bathroom']==0 ].index.tolist()
data = [ 1 for _ in index ]
temp = pd.Series( data,name='Bathroom',index=index )
content.update( temp )


index=[ ]
data =[ ]
for each in content[ 'Type' ].unique( ):
    print( each )
    for i in content['Suburb'].unique():
        conv = Imputer(missing_values=np.NaN, strategy='most_frequent')
        raw = content[(content['Type'] == each) & (content['Suburb'] == i)]['Age']
        
        empty = raw.isnull().sum()
        indics = raw.index.tolist()
        raw = np.reshape(raw.values, newshape=(-1, 1))
        if len(indics)>empty:
            convertion = conv.fit_transform(raw).reshape(-1)
        else:
            convertion=[ ]
            indics = [  ]
        index.extend(indics)
        data.extend(convertion)

print( 'age has been converted\n' )
temp = pd.Series( data,name='Age',index=index )
content.update( temp )

# to remove outlier
content = content.drop( content[ (content['Landsize'] > 1100.0) ].index,axis=0 )
content = content.drop( content[ (content['Landsize'] < 50.0) ].index,axis=0 )
content = content.drop( content[ content['Bathroom']>6].index,axis=0 )
content = content.drop( content[ (content['Age']>200) ].index,axis=0 )
content = content.drop( content[ (content['Age']<0) ].index,axis=0 )
content = content.drop( content[ content['Age'].isnull() ].index,axis=0 )

null_value_cnt = content.isnull().sum()
percentage = -null_value_cnt / content.__len__() * 100
percentage.plot( kind='bar',width=0.5,color = 'blue',label='after' )
plt.title( 'the percentage of Null values ' )
plt.legend()
plt.show()
# to plot the relationship between features and the price

import seaborn as sns

sns.set_style( 'darkgrid' )
fig,axes = plt.subplots(2,2,figsize=[20,20])
sns.distplot( content['Price'],ax=axes[0,0] )
axes[0,0].set_xlabel( 'Price' )
axes[0,0].set_ylabel( 'the distribution' )
axes[0,0].set_title( 'the distribution of price' )

sns.boxplot(x='Regionname',y='Price',data=content ,ax=axes[0,1] )
axes[0,1].set_xlabel( 'Regionname' )
axes[0,1].set_ylabel( 'Price' )
axes[0,1].set_title( 'Price vs regionname' )


axes[1,0].scatter( x='Bathroom',y='Price',data=content,edgecolor='b' )
axes[1,0].set_xlabel( 'Bathroom' )
axes[1,0].set_ylabel( 'Price' )
axes[1,0].set_title( 'Price vs Bathroom' )

sns.boxplot( x='Type',y='Price',data=content ,ax=axes[1,1] )
axes[1,1].set_xlabel( 'Type' )
axes[1,1].set_ylabel( 'Price' )
axes[1,1].set_title( 'Price vs Type' )
plt.title( 'Melbourne House Price ' )
plt.show()

# prepare the dataset and label for training models

content.drop( columns=[ 'Bedroom2', 'Car','BuildingArea','YearBuilt','Lattitude','Longtitude','Day', 'Method', 'SellerG',
       'Date', 'CouncilArea','Suburb', 'Address','Postcode'] ,inplace=True)

content['Type']=content['Type'].map(  {'House':1,'Townhouse':2,'Unit':3}  )

key = content['Regionname'].unique( )
value = list( range( 1,key.__len__()+1 ) )
region_dict = dict( zip( key,value ) )

content['Regionname']=content['Regionname'].map( region_dict )

print( content.isnull().any() )


content = content.dropna()


print( content.describe() )
dataset = content.drop( columns=['Price'] )
final_column = dataset.columns
label = content['Price']

# to implement the regression models

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

dataset = MinMaxScaler().fit_transform(dataset)
train_data,test_data,train_label,test_label=train_test_split(dataset,label,test_size=0.3,random_state=40)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

reg1 = LinearRegression()
reg2 = DecisionTreeRegressor()
reg3 = ExtraTreesRegressor()
reg4 = XGBRegressor()
reg5 = SVR()

reg1.fit( train_data,train_label )
reg2.fit( train_data,train_label )
reg3.fit( train_data,train_label )
reg4.fit( train_data,train_label )
reg5.fit( train_data,train_label )

label1 = reg1.predict( test_data )
label2 = reg2.predict( test_data )
label3 = reg3.predict( test_data )
label4 = reg4.predict( test_data )
label5 = reg5.predict( test_data )

# compare the loss of different models

from sklearn.metrics import mean_squared_error
print( 'the loss of LinearRegression is ',mean_squared_error(test_label,label1) )
print( 'the loss of DecisionTreeRegressor is ',mean_squared_error(test_label,label2) )
print( 'the loss of ExtraTreesRegressor is ',mean_squared_error(test_label,label3) )
print( 'the loss of XGBRegressor is ',mean_squared_error(test_label,label4) )
print( 'the loss of SVR is ',mean_squared_error(test_label,label5) )

print( '==='*10 )
# to compare the r^2 value of different regression models
# to chech the percentage of explained samples

from sklearn.metrics import r2_score
print( 'the r2 of LinearRegression is ',r2_score(test_label,label1) )
print( 'the r2 of DecisionTreeRegressor is ',r2_score(test_label,label2) )
print( 'the r2 of ExtraTreesRegressor is ',r2_score(test_label,label3) )
print( 'the r2 of XGBRegressor is ',r2_score(test_label,label4) )
print( 'the r2 of SVR is ',r2_score(test_label,label5) )

print( '++'*10 )
#to figure out the coef_ or the feature_importance

feature_data = [ reg1.coef_,reg2.feature_importances_,reg3.feature_importances_,reg4.feature_importances_]
evaluation = pd.DataFrame( feature_data,index=['LinearRegression','DecisionTreeRegressor','ExtraTreesRegressor','XGBRegressor'],columns=final_column )
print( evaluation )

print( 'aparently, ExtraTreeRegressor performs the best in all the linear models with the loss presenting the least' )
print('---'*10)
print( 'from the best model we could know that the history of a poverty plays the most significant role, followed by the type of the accommodation.\nPlus, the landsize and bathroom are both considered much by the buyers' )