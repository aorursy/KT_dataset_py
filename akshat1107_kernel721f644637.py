import numpy as np

import pandas as pd



#inporting datasets

train_data=pd.read_csv('train.csv')

test_data=pd.read_csv('test.csv')





#info about data

print(train_data.shape)

print(test_data.shape)

print(train_data.info())

print(test_data.info())

print(train_data.isna().sum())

print(test_data.isna().sum())



for i in train_data.columns:

    print(i,"\t\t\t\t\t",train_data[i].isna().sum(),train_data[i].dtype)



for i in test_data.columns:    

    print(i,"\t\t\t\t\t",test_data[i].isna().sum(),test_data[i].dtype)

#drop columns

drop_columns = ['Id','Alley','PoolQC','Fence','MiscFeature']

train_data=train_data.drop(drop_columns,axis=1)

test_data=test_data.drop(drop_columns,axis=1)



#finding null columns and object type columns

object_columns=[]

train_na=[]

test_na=[]

object_position=[]

c=0



for i in test_data.columns:

    

    if train_data[i].isna().sum()>0:

        print(train_data[i].isna().sum())

        train_na.append(i)

    if test_data[i].isna().sum()>0:

        test_na.append(i)

    if train_data[i].dtype=='object':

        object_columns.append(i)

        object_position.append(c)

    c+=1

print(train_na)

print(test_na)

print(object_columns)

print(object_position)

train_numeric=list(set(train_na)-set(object_columns))

print(train_numeric)

test_numeric=list(set(test_na)-set(object_columns))

print(test_numeric)



#filling numeric data

"""

train_data['LotFrontage']=train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())

train_data['MasVnrArea']=train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].mean())

train_data['GarageYrBlt']=train_data['GarageYrBlt'].fillna(train_data['GarageYrBlt'].mean())



test_data['LotFrontage']=test_data['LotFrontage'].fillna(test_data['LotFrontage'].mean())

test_data['MasVnrArea']=test_data['MasVnrArea'].fillna(test_data['MasVnrArea'].mean())

test_data['GarageYrBlt']=test_data['GarageYrBlt'].fillna(test_data['GarageYrBlt'].mean())

test_data['GarageCars']=test_data['GarageCars'].fillna(test_data['GarageCars'].mean())

test_data['BsmtFinSF2']=test_data['BsmtFinSF2'].fillna(test_data['BsmtFinSF2'].mean())

test_data['BsmtHalfBath']=test_data['BsmtHalfBath'].fillna(test_data['BsmtHalfBath'].mean())

test_data['BsmtFinSF1']=test_data['BsmtFinSF1'].fillna(test_data['BsmtFinSF1'].mean())

test_data['GarageArea']=test_data['GarageArea'].fillna(test_data['GarageArea'].mean())

test_data['BsmtFullBath']=test_data['BsmtFullBath'].fillna(test_data['BsmtFullBath'].mean())

test_data['BsmtUnfSF']=test_data['BsmtUnfSF'].fillna(test_data['BsmtUnfSF'].mean())

test_data['TotalBsmtSF']=test_data['TotalBsmtSF'].fillna(test_data['TotalBsmtSF'].mean())

"""

#Do it like this below

for i in train_numeric:

    train_data[i]=train_data[i].fillna(train_data[i].mean())



for i in test_numeric:

    test_data[i]=test_data[i].fillna(test_data[i].mean())



#filling object data

for i in train_na:

    print(train_data[i].value_counts())

    train_data[i]=train_data[i].fillna(train_data[i].mode()[0])

for i in test_na:

    test_data[i]=test_data[i].fillna(test_data[i].mode()[0])

    

    



print(train_data.shape)

print(test_data.shape)



#combining train and test data

combine=pd.concat([train_data,test_data])

print(combine.shape)



object_combine=[]

c=0

for i in combine.columns:

    if i in object_columns:

        object_combine.append(c)

    c+=1        





#converting to categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

cat=LabelEncoder()

for i in object_columns:

    

 #   print(X_combine[:,i])

    combine[i]=cat.fit_transform(combine[i])

  #  print(X_combine[:,i])

ohe=OneHotEncoder(categorical_features=object_combine)

#droppping the last column

combine=combine.drop(['SalePrice'],axis=1)

X_combine=combine.values

X_combine=ohe.fit_transform(X_combine).toarray()



print(pd.DataFrame(X_combine).isna().sum().sum())

print(pd.DataFrame(X_combine).shape)



#dividing into test and train data

x_train=X_combine[0:1460,:]

print(pd.DataFrame(x_train).shape)

x_test=X_combine[1460:,:]

print(pd.DataFrame(x_test).shape)



y_train=train_data['SalePrice'].values





from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

test_data_size=0.33

seed=7                    #Always use same rows

Xtt,Xt,Ytt,Yt=train_test_split(x_train,y_train,test_size=test_data_size,random_state=seed)



model=LinearRegression()

model.fit(Xtt,Ytt)

result=model.score(Xt,Yt)

print("Accuracy= %f %%"%(result*100))



from sklearn.ensemble import RandomForestRegressor

reg=RandomForestRegressor(n_estimators=279,random_state=0)

reg.fit(Xtt,Ytt)

result=reg.score(Xt,Yt)





print("Accuracy= %f %%"%(result*100))







from sklearn.tree import DecisionTreeRegressor

reg=DecisionTreeRegressor(random_state=0)

reg.fit(Xtt,Ytt)

result=reg.score(Xt,Yt)





print("Accuracy= %f %%"%(result*100))







y_pred=reg.predict(x_test)

ydf=pd.DataFrame(y_pred)

sample_data=pd.read_csv('sample_submission.csv')

datasets=pd.concat([sample_data['Id'],ydf],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('sample_submission.csc',index=False)






