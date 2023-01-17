import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings

from sklearn.preprocessing import StandardScaler

from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

%matplotlib inline



warnings.filterwarnings('ignore')
traindata=pd.read_csv('../input/train.csv')#reading the data

testdata=pd.read_csv('../input/test.csv')

print('Number of rows and columns of the training set: ',traindata.shape)

print('Number of rows and columns of the test set: ',testdata.shape)
numeric_cols=traindata.select_dtypes(include=[np.number]).columns#select only numerical

nominal_cols=traindata.select_dtypes(exclude=[np.number]).columns#select only non numerical

print(numeric_cols.shape[0],'numeric columns: ',numeric_cols)

print(nominal_cols.shape[0],'nominal columns: ',nominal_cols)
fig, ax = plt.subplots()

ax.hist(traindata['SalePrice'],40)

ax.set_xlabel('SalePrice')

plt.show()
fig, ax = plt.subplots()

ax.hist(np.log(traindata['SalePrice']),40)

ax.set_xlabel('log(SalePrice)')

plt.show()
fig, ax = plt.subplots()

ax.scatter(traindata['SalePrice'],traindata['GrLivArea'])

ax.set_xlabel('SalePrice')

ax.set_ylabel('GrLivArea')

plt.show()
traindata=traindata[traindata['GrLivArea']<4000] #drop outliers in train

traindata=traindata.drop('Id',axis=1) #remove the column from train

test_ids = testdata['Id'] #save id column from test

testdata=testdata.drop('Id',axis=1) #remove the column from test

numeric_cols=numeric_cols[numeric_cols!='Id'] #remove the column name from this list as well



data=pd.concat([traindata,testdata],axis=0,ignore_index=True) #concatenate training and test set for future transformations

print(data['SalePrice'].head())#Don't worry about the SalePrice variable that is not in test. 

print(data['SalePrice'].tail()) #It's filled with NAs

data['SalePrice'] = np.log(data['SalePrice']) #apply the logarithm to SalePrice
correlation=traindata[numeric_cols].corr() #obtain the correlation matrix

sns.set()

fig, ax = plt.subplots(figsize=(16,8))

sns.heatmap(correlation,ax=ax)

plt.show() #draw the correlation matrix
aux=(abs(correlation)-np.identity(correlation.shape[0])).max() #maximum correlation of each variable

selected_feats=aux[aux>0.5].index#take only variables whose maximum correlation is strong.

sns.set()

fig, ax = plt.subplots(figsize=(16,8))

sns.heatmap(correlation.loc[selected_feats,selected_feats], annot=True,fmt='.2f',ax=ax)

plt.show()
data=data.drop(['GarageArea','1stFlrSF','GarageYrBlt'],axis=1) #remove columns

numeric_cols=numeric_cols[numeric_cols!='GarageArea'] #remove them from our list too

numeric_cols=numeric_cols[numeric_cols!='1stFlrSF']

numeric_cols=numeric_cols[numeric_cols!='GarageYrBlt']



correlation=traindata[numeric_cols].corr() #calculate again the correlation matrix (without the removed columns)

aux=abs(correlation['SalePrice']).sort_values(ascending=False) #sort variables by their correlation with SalePrice

selected_feats=aux[0:19].index #Take the best 19. Why 19? because.

sns.set()

fig, ax = plt.subplots(figsize=(16,8))

sns.heatmap(correlation.loc[selected_feats,selected_feats], annot=True,fmt='.2f',ax=ax)

plt.show()
selected_feats=selected_feats[1:] # don't take SalePrice



fig, axes = plt.subplots(nrows=6,ncols=3,figsize=(16,32),sharey=True)

axes=axes.flatten()

for i in range(len(axes)):

    axes[i].scatter(traindata[selected_feats[i]],traindata['SalePrice'])

    axes[i].set_xlabel(selected_feats[i])

    axes[i].set_ylabel('SalePrice')

plt.show()
selected_nominal_feats= np.random.choice(nominal_cols,18,replace=False)

fig, axes = plt.subplots(nrows=6,ncols=3,figsize=(16,32),sharey=True)

axes=axes.flatten()

for i in range(len(axes)):

    sns.set()

    sns.stripplot(x=selected_nominal_feats[i], y='SalePrice', data=traindata,ax=axes[i],jitter=True)

    axes[i].set_xlabel(selected_nominal_feats[i])

    axes[i].xaxis.set_tick_params(rotation=60)

plt.subplots_adjust(hspace = 0.5)

plt.show()
missing_values=data.isnull().sum() #obtain the number of missing values by column

numeric_missing=missing_values[numeric_cols] #separate the numeric variables

numeric_missing['SalePrice']=0 #We don't want to detect SalePrice's NAs because they are not real, they just belong to test set

numeric_missing=numeric_missing[numeric_missing>0] #we only want to see variables with 1 or more missings

numeric_missing_df= pd.DataFrame()

numeric_missing_df[['absolute','relative']]= pd.concat([numeric_missing,numeric_missing/data.shape[0]],axis=1)

numeric_missing_df #table of missing numeric values 
#exactly the same for nominal variables

nominal_missing=missing_values[nominal_cols] 

nominal_missing=nominal_missing[nominal_missing>0]

nominal_missing_df= pd.DataFrame()

nominal_missing_df[['absolute','relative']]= pd.concat([nominal_missing,nominal_missing/data.shape[0]],axis=1)

nominal_missing_df
data=data.drop(['Alley','PoolQC','MiscFeature'],axis=1) #remove columns

nominal_cols=nominal_cols[nominal_cols!='Alley']

nominal_cols=nominal_cols[nominal_cols!='PoolQC']

nominal_cols=nominal_cols[nominal_cols!='MiscFeature']
pos_params=data['LotFrontage'].describe()#get position parameters of the variable

pos_params=[pos_params['25%'],pos_params['50%'],pos_params['75%']]

chosen_values=np.random.choice(pos_params,numeric_missing['LotFrontage'],p=[0.25,0.5,0.25]) #randomly choose between 1sQ, median and 3rdQ

data.loc[data['LotFrontage'].isnull(),'LotFrontage']=chosen_values #fill missings

    

for fillvar in numeric_missing.index:

    data[fillvar]=data[fillvar].fillna(data[fillvar].median()) #fill with median



numeric_missing=data[numeric_cols].isnull().sum() 

numeric_missing['SalePrice']=0

print('Remaining numeric missing values: ',numeric_missing.sum())

data=data.fillna('NA') #fill nominal ones with 'NA' label

print('Remaining nominal missing values: ',data[nominal_cols].isnull().sum().sum())
skewness=data.skew(axis=0,numeric_only=True) #the skewness of the numeric variables (Salesprice is not included)

posskewness = skewness[skewness > 0.5] #We take only the positively skewed vriables

posskewed_features = posskewness.index #The names of that variables        

data[posskewed_features] = np.log1p(data[posskewed_features]) #we apply the log(x+1) to each variable x

print('Corrected features: ',posskewed_features)
data=pd.concat([data[numeric_cols],pd.get_dummies(data[nominal_cols])],axis=1)

print('Number of rows and columns: ',data.shape)
data[selected_feats+'2']=np.power(data[selected_feats],2) #create new variables powering to 2

data[selected_feats+'3']=np.power(data[selected_feats],3) #create new variables powering to 3

numeric_cols=np.hstack([numeric_cols,selected_feats+'2',selected_feats+'3']) #add the new features to our list

print('Number of rows and columns: ',data.shape)
traindata=data.iloc[:traindata.shape[0],:] 

testdata=data.iloc[traindata.shape[0]:,:]

testdata=testdata.drop('SalePrice',axis=1) #We drop the unknown variable in the test. It was just filled with NAs



train, test= train_test_split(traindata, test_size = 0.25, random_state = 0)



stdSc = StandardScaler()



numeric_cols=numeric_cols[numeric_cols!='SalePrice'] #We don't want to scale SalePrice

train.loc[:, numeric_cols] = stdSc.fit_transform(train.loc[:, numeric_cols])#scaling tranformation

test.loc[:, numeric_cols] = stdSc.transform(test.loc[:, numeric_cols])



traindata.loc[:, numeric_cols] = stdSc.fit_transform(traindata.loc[:, numeric_cols])

testdata.loc[:, numeric_cols] = stdSc.transform(testdata.loc[:, numeric_cols])



X_train=train.drop('SalePrice',axis=1)#Separate the target variable form the rest

y_train=train['SalePrice']

X_test=test.drop('SalePrice',axis=1)

y_test=test['SalePrice']



X_traindata=traindata.drop('SalePrice',axis=1)

y_traindata=traindata['SalePrice']
def rmse_cv(model,X, y):# function that calculates the root mean squared error of the test set through cross validation.

    rmse= np.sqrt(-cross_val_score(model, X, y, scoring = scorer, cv = 10))

    return(rmse)



scorer = make_scorer(mean_squared_error, greater_is_better = False)

param_keridge={'alpha':[0.05, 0.1, 0.3, 0.6, 1, 1.5, 3, 5, 10, 15, 30, 50, 80],

               'kernel':['linear','poly','rbf'],

               'degree':[2,3],

               'coef0':[0.5,1,1.5,2,2.5,3,3.5]} #parameters for our GridSearchCV to explore

regressor=GridSearchCV(KernelRidge(), param_keridge).fit(X_train,y_train).best_estimator_ #obtain the regressor with the best hyperparameters comnination

print('Best estimator found: ',regressor)

print('Root mean square error on the test partition :',rmse_cv(regressor,X_train,y_train).mean()) #show the estimate of the test error



regressor.fit(X_traindata,y_traindata) #train our regressor with all the train data



result=pd.DataFrame()

result['Id']=test_ids

result['SalePrice']=np.exp(regressor.predict(testdata)) #make the prediction and transform it back by aplying exp

print('The description of the submision:\n',result.describe())

result.to_csv('submision.csv',index=False)