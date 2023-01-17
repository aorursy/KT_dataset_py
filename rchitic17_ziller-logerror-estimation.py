import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor

color= sns.color_palette()

%matplotlib inline
import os
print(os.listdir("../input"))
train_df=pd.read_csv("../input/train_2016_v2.csv")
train_df.shape
train_df.head()
train_df.info()
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
plt.xlabel('index',fontsize=12)
plt.ylabel('logerror',fontsize=12)
plt.show()           
plt.figure(figsize=(12,8))
sns.distplot(train_df.logerror.values,bins=500,kde=False)
plt.xlabel('logerror',fontsize=12)
plt.xlim([-0.4,0.4])
plt.show()
train_df['transactiondate']=pd.to_datetime(train_df['transactiondate'])
train_df['transaction_month']=train_df['transactiondate'].dt.month

month_count=train_df['transaction_month'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(month_count.index, month_count.values)
plt.xlabel('Month of transaction',fontsize=12)
plt.ylabel('Number of Occurrences',fontsize=12)
plt.show()
(train_df['parcelid'].value_counts()).value_counts()
prop_df=pd.read_csv("../input/properties_2016.csv")
prop_df.shape
prop_df.head()
prop_df.info()
missing_df=prop_df.isnull().sum(axis=0).reset_index()
missing_df.columns=['column_name', 'missing_count']
missing_df=missing_df.sort_values(by='missing_count')
missing_df
ind= np.arange(missing_df.shape[0])
fig, ax =plt.subplots(figsize=(12,18))
rects=ax.barh(ind, missing_df.missing_count.values)
ax.set_yticklabels(missing_df.column_name.values,rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()
train_df=pd.merge(train_df, prop_df, on='parcelid', how='left')
train_df.head()
train_df.info()
plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.longitude.values, y=train_df.latitude.values, size=10)
plt.ylabel('Latitude',fontsize=12)
plt.xlabel('Longitude',fontsize=12)
plt.show()
#plot heatmap to find correlations between variables
plt.figure(figsize = (12,8))
sns.heatmap(data=train_df.corr())
plt.show()
plt.gcf().clear()
#missing_df=train_df.isnull().sum(axis=0).reset_index()
#missing_df.columns=['column_name','missing_count']
#missing_df['missing_ratio']=(missing_df['missing_count'])/train_df.shape[0]
#missing_df.ix[missing_df['missing_ratio']>0.5]
#reduntant features
dropcols = ['finishedsquarefeet12','finishedsquarefeet13', 'finishedsquarefeet15','finishedsquarefeet6','finishedsquarefeet50']
#identical to bathroomcnt
dropcols.append('calculatedbathnbr')
dropcols.append('fullbathcnt')
#if there is no data for hot tub, most likely there is no hot tub
index = train_df.hashottuborspa.isnull()
train_df.loc[index,'hashottuborspa'] = "None"
#too many missing values
dropcols.append('pooltypeid10')
#if pooltype id is null, most likely the pool doesnt exist
index = train_df.pooltypeid2.isnull()
train_df.loc[index,'pooltypeid2'] = 0

index = train_df.pooltypeid7.isnull()
train_df.loc[index,'pooltypeid7'] = 0

index = train_df.poolcnt.isnull()
train_df.loc[index,'poolcnt'] = 0
#for poolsizesum, fill in with median only for properties that have a pool
poolsizesum_median = train_df.loc[train_df['poolcnt'] > 0, 'poolsizesum'].median()
train_df.loc[(train_df['poolcnt'] > 0) & (train_df['poolsizesum'].isnull()), 'poolsizesum'] = poolsizesum_median
#If property doesn't have a pool then poolsizesum is 0 
train_df.loc[(train_df['poolcnt'] == 0), 'poolsizesum'] = 0
#fireplaceflag and fireplacecnt
train_df['fireplaceflag']= "No"
train_df.loc[train_df['fireplacecnt']>0,'fireplaceflag']= "Yes"

index = train_df.fireplacecnt.isnull()
train_df.loc[index,'fireplacecnt'] = 0

#Tax deliquency flag - if it is null, most likely it doesn't exist
index = train_df.taxdelinquencyflag.isnull()
train_df.loc[index,'taxdelinquencyflag'] = "None"
#garagecarcnt and garagetotalsqft
#if garage count is null, most likely there are no garages
index = train_df.garagecarcnt.isnull()
train_df.loc[index,'garagecarcnt'] = 0
#if no garage, garage size is 0
#Likewise no garage means the size is 0 by default
index = train_df.garagetotalsqft.isnull()
train_df.loc[index,'garagetotalsqft'] = 0

#fill in missing values with most reasonable/common value
#airconditioningtypeid
train_df['airconditioningtypeid'].value_counts()
index = train_df.airconditioningtypeid.isnull()
train_df.loc[index,'airconditioningtypeid'] = 1

#heatingorsystemtypeid
print(train_df['heatingorsystemtypeid'].value_counts())
index = train_df.heatingorsystemtypeid.isnull()
train_df.loc[index,'heatingorsystemtypeid'] = 2

#threequarterbathnbr
print(train_df['threequarterbathnbr'].value_counts())
index = train_df.threequarterbathnbr.isnull()
train_df.loc[index,'threequarterbathnbr'] = 1
#drop variables with more than 97% of values missing
missingvalues_prop = (train_df.isnull().sum()/len(train_df)).reset_index()
missingvalues_prop.columns = ['field','proportion']
missingvalues_prop = missingvalues_prop.sort_values(by = 'proportion', ascending = False)
print(missingvalues_prop)
missingvaluescols = missingvalues_prop[missingvalues_prop['proportion'] > 0.97].field.tolist()
dropcols = dropcols + missingvaluescols
train_df = train_df.drop(dropcols, axis=1)
def fillna_knn( df, base, target, fraction = 1, threshold = 10, n_neighbors = 5 ):
    assert isinstance( base , list ) or isinstance( base , np.ndarray ) and isinstance( target, str ) 
    whole = [ target ] + base
    
    miss = df[target].isnull()
    notmiss = ~miss 
    nummiss = miss.sum()
    
    enc = OneHotEncoder()
    X_target = df.loc[ notmiss, whole ].sample( frac = fraction )
    
    enc.fit( X_target[ target ].unique().reshape( (-1,1) ) )
    
    Y = enc.transform( X_target[ target ].values.reshape((-1,1)) ).toarray()
    X = X_target[ base  ]
    
    print( 'fitting' )
    n_neighbors = n_neighbors
    clf = neighbors.KNeighborsClassifier( n_neighbors, weights = 'uniform' )
    clf.fit( X, Y )
    
    print( 'the shape of active features: ' ,enc.active_features_.shape )
    
    print( 'predicting' )
    Z = clf.predict(df.loc[miss, base])
    
    numunperdicted = Z[:,0].sum()
    if numunperdicted / nummiss *100 < threshold :
        print( 'writing result to df' )    
        df.loc[ miss, target ]  = np.dot( Z , enc.active_features_ )
        print( 'num of unperdictable data: ', numunperdicted )
        return enc
    else:
        print( 'out of threshold: {}% > {}%'.format( numunperdicted / nummiss *100 , threshold ) )

#function to deal with variables that are actually string/categories
def zoningcode2int( df, target ):
    storenull = df[ target ].isnull()
    enc = LabelEncoder( )
    df[ target ] = df[ target ].astype( str )

    print('fit and transform')
    df[ target ]= enc.fit_transform( df[ target ].values )
    print( 'num of categories: ', enc.classes_.shape  )
    df.loc[ storenull, target ] = np.nan
    print('recover the nan value')
    return enc

#fill in features that depend on location  
fillna_knn( df = train_df,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'buildingqualitytypeid', fraction = 0.15, n_neighbors = 1 )


zoningcode2int( df = train_df,
                            target = 'propertycountylandusecode' )
fillna_knn( df = train_df,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'propertycountylandusecode', fraction = 0.15, n_neighbors = 1 )

zoningcode2int( df = train_df,
                            target = 'propertyzoningdesc' )

fillna_knn( df = train_df,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'propertyzoningdesc', fraction = 0.15, n_neighbors = 1 )

#regionidcity, regionidneighborhood & regionidzip - assume it is the same as the nereast property. 
#As mentioned above, this is ok if there's a property very nearby to the one with missing values (I leave it up to the reader to check if this is the case!)
fillna_knn( df = train_df,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'regionidcity', fraction = 0.15, n_neighbors = 1 )

fillna_knn( df = train_df,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'regionidneighborhood', fraction = 0.15, n_neighbors = 1 )

fillna_knn( df = train_df,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'regionidzip', fraction = 0.15, n_neighbors = 1 )

#unitcnt - the number of structures the unit is built into. Assume it is the same as the nearest properties. If the property with missing values is in a block of flats or in a terrace street then this is probably ok - but again I leave it up to the reader to check if this is the case!
fillna_knn( df = train_df,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'unitcnt', fraction = 0.15, n_neighbors = 1 )

#yearbuilt - assume it is the same as the nearest property. This assumes properties all near to each other were built around the same time
fillna_knn( df = train_df,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'yearbuilt', fraction = 0.15, n_neighbors = 1 )

#lot size square feet - not sure what to do about this one. Lets use nearest neighbours. Assume it has same lot size as property closest to it
fillna_knn( df = train_df,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'lotsizesquarefeet', fraction = 0.15, n_neighbors = 1 )

fillna_knn( df = train_df,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'numberofstories', fraction = 0.15, n_neighbors = 1 )
#finishedfloor1squarefeet
plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.finishedfloor1squarefeet.values, y=train_df.calculatedfinishedsquarefeet.values)
plt.ylabel('calculatedfinishedsquarefeet', fontsize=12)
plt.xlabel('finishedfloor1squarefeet', fontsize=12)
plt.title("finishedfloor1squarefeet Vs calculatedfinishedsquarefeet", fontsize=15)
plt.show()

#There are some properties where finishedfloor1squarefeet and calculatedfinishedsquarefeetare are both exactly the same - probably because its a studio flat of some sort so that the area on the first floor is equivalent to the total area, lets see how many there are
#For now assume if the number of stories is 1 then the finishedfloor1squarefeet is the same as calculatedfinishedsquarefeet
train_df.loc[(train_df['finishedfloor1squarefeet'].isnull()) & (train_df['numberofstories']==1),'finishedfloor1squarefeet'] = train_df.loc[(train_df['finishedfloor1squarefeet'].isnull()) & (train_df['numberofstories']==1),'calculatedfinishedsquarefeet']

#I also discovered that there seems to be two properties that have finishedfloor1squarefeet greater than calculated finishedsquarefeet. Notice also that they have big logerrors aswell - my guess is that the Zillow House price model found it difficult to predict these points due to the fact that they probably had potentially 'incorrect' data input values?
#Discussion point - should we be removing these points or leave them in as they are or 'fix' them? I think it really depends on whether the test data has similar points which may be wrong as we'll want to predict big log errors for these incorrect points aswell I guess...
#For now just remove them.
print(train_df.loc[train_df['calculatedfinishedsquarefeet']<train_df['finishedfloor1squarefeet']])
droprows = train_df.loc[train_df['calculatedfinishedsquarefeet']<train_df['finishedfloor1squarefeet']].index
train_df = train_df.drop(droprows)
print(train_df.isnull().sum())

#taxvaluedollarcnt & landtaxvaluedollarcnt - set it equal to the tax amount (most correlated value). Single story property so assume they are all the same
train_df.loc[train_df.taxvaluedollarcnt.isnull(),'taxvaluedollarcnt'] = train_df.loc[train_df.taxvaluedollarcnt.isnull(),'taxamount']
train_df.loc[train_df.landtaxvaluedollarcnt.isnull(),'landtaxvaluedollarcnt'] = train_df.loc[train_df.landtaxvaluedollarcnt.isnull(),'taxamount']

#structure tax value dollar - fill this in using its most correlated variable
x =  train_df.corr()
print(x.structuretaxvaluedollarcnt.sort_values(ascending = False))

#taxvaluedollarcnt is most correlated variable, let's see how they are related 
plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.structuretaxvaluedollarcnt.values, y=train_df.taxvaluedollarcnt.values)
plt.ylabel('taxvaluedollarcnt', fontsize=12)
plt.xlabel('structuretaxvaluedollarcnt', fontsize=12)
plt.title("structuretaxvaluedollarcnt Vs taxvaluedollarcnt", fontsize=15)
plt.show()

#Lets look at the distribution of taxvaluedollar cnt where structuretaxvaluedollarcnt is missing just to make sure we are predicting missing values in the body of the taxvaluedollarcnt distribution
print(train_df.loc[train_df['structuretaxvaluedollarcnt'].isnull(),'taxvaluedollarcnt'].describe())
print(train_df['taxvaluedollarcnt'].describe())

#Slightly amend the k nearest neighbour function so it works on regression
def fillna_knn_reg( df, base, target, n_neighbors = 5 ):
    cols = base + [target]
    X_train = df[cols]
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train[base].values.reshape(-1, 1))
    rescaledX = scaler.transform(X_train[base].values.reshape(-1, 1))

    X_train = rescaledX[df[target].notnull()]
    Y_train = df.loc[df[target].notnull(),target].values.reshape(-1, 1)

    knn = KNeighborsRegressor(n_neighbors, n_jobs = -1)    
    # fitting the model
    knn.fit(X_train, Y_train)
    # predict the response
    X_test = rescaledX[df[target].isnull()]
    pred = knn.predict(X_test)
    df.loc[train_df[target].isnull(),target] = pred
    return

#fill in structuretaxvaluedollarcnt using taxvaluedollarcnt as per the above
fillna_knn_reg(df = train_df, base = ['taxvaluedollarcnt'], target = 'structuretaxvaluedollarcnt')

#Do the same thing for tax amount, as taxvaluedollarcnt is its most correlated variable
fillna_knn_reg(df = train_df, base = ['taxvaluedollarcnt'], target = 'taxamount')
print(train_df.isnull().sum())
#fill in total square feet based on number of bedrooms (assumed to be proportional)
fillna_knn_reg(df = train_df, base = ['bedroomcnt'], target = 'calculatedfinishedsquarefeet')
#fill in unit count based on total square feet (assumed to be proprtional)
fillna_knn_reg(df = train_df, base = ['calculatedfinishedsquarefeet'], target = 'unitcnt')

train_df=train_df.drop(['numberofstories','finishedfloor1squarefeet'],axis=1)
train_df['censustractandblock']=train_df['censustractandblock'].fillna(train_df['censustractandblock'].mean())
print(train_df.isnull().sum())
train_y=train_df['logerror'].values
cat_cols=["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
train_x=train_df_new.drop(['parcelid', 'logerror', 'transactiondate','airconditioningtypeid']+cat_cols,axis=1)
train_x=train_x.groupby(train_x.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))
feat_names=train_df.columns.values

from sklearn import ensemble
model=ensemble.ExtraTreesRegressor(n_estimators=25,max_depth=30,max_features=0.3,n_jobs=-1,random_state=0)
model.fit(train_x,train_y)

#plot variable importances
importances=model.feature_importances_
#std=np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
#indices=np.argsort(importances)[::-1][:20]
indices=range(len(model.feature_importances_))
plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(indices,importances[indices],color='b')
plt.xticks(indices,feat_names[indices],rotation='vertical')
plt.xlim([-1,len(indices)])
plt.show()
parameters = {'n_estimators':[5,10,15],'n_jobs':[-1],'oob_score':[False]}  # this can be extended
model = ensemble.RandomForestRegressor()
from sklearn import model_selection
grid = model_selection.GridSearchCV(model,param_grid=parameters,scoring='neg_mean_absolute_error',cv=3)  
grid.fit(train_x,train_y)
cv_results = pd.DataFrame(grid.cv_results_)
print(cv_results[["param_n_estimators","mean_test_score","std_test_score"]])

feat_imps = grid.best_estimator_.feature_importances_
fi = pd.DataFrame.from_dict({'feat':train_x.columns,'imp':feat_imps})
fi.set_index('feat',inplace=True,drop=True)
fi = fi.sort_values('imp',ascending=False)
fi.head(20).plot.bar()
samp = pd.read_csv(r"../input/sample_submission.csv")  #The parcelid's for the testset
#impute all numerical missing values in the properties dataframe
prop_df_new = prop_df.groupby(prop_df.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))
test = samp.loc[:,['ParcelId']].merge(prop_df_new,how='left',left_on='ParcelId',right_on='parcelid')
test_x = test.drop(['ParcelId','parcelid']+cat_cols,axis=1,inplace=False)
#,'airconditioningtypeid','architecturalstyletypeid','assessmentyear','basementsqft','bathroomcnt','bedroomcnt','buildingclasstypeid'
test_x=test_x.groupby(test_x.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))

test_y = grid.predict(test_x)
test_y = pd.DataFrame(test_y)
test_y[1] = test_y[0]
test_y[2] = test_y[0]
test_y[3] = test_y[0]
test_y[4] = test_y[0]
test_y[5] = test_y[0]  #For simplicity make identical predictions for all months
test_y.columns = ["201610","201611","201612","201710","201711","201712"]
submission = test_y.copy()
submission["parcelid"] = samp["ParcelId"].copy()
cols = ["parcelid","201610","201611","201612","201710","201711","201712"]
submission = submission[cols]
filename = "Prediction_" + str(submission.columns[0]) + re.sub("[^0-9]", "",str(datetime.datetime.now())) + '.csv'
print(filename)
submission.to_csv(filename,index=False)
y=train_df.logerror
x=train_df.drop(['parcelid','logerror','transactiondate',],axis=1)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=148)

# Consider the following models
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

gb = XGBRegressor(n_jobs=1, random_state=148)
ls = Lasso(alpha=1e-6, normalize=True)
el = ElasticNet(alpha=1e-6, normalize=True)
rf = RandomForestRegressor(random_state=148)

base_learners = [
    ('ls', ls), ('el', el), ('rf', rf), ('gb', gb)
]
P = np.zeros((xtest.shape[0], len(base_learners)))
P = pd.DataFrame(P, columns=[e for e, _ in base_learners])

from sklearn.metrics import mean_absolute_error

for est_name, est in base_learners:
    est.fit(xtrain, ytrain)
    p = est.predict(xtest)
    P.loc[:, est_name] = p
    print("%3s : %.4f" % (est_name, mean_absolute_error(ytest, p)))
xtrain.head()
train_df.loc[train_df['taxdelinquencyflag']=='None','taxdelinquencyflag']=0
train_df.loc[train_df['taxdelinquencyflag']=='Y','taxdelinquencyflag']=1
train_df.loc[train_df['hashottuborspa']=='None','hashottuborspa']=0
train_df.loc[train_df['hashottuborspa']=='True','hashottuborspa']=1
train_df.loc[train_df['fireplaceflag']=='No','fireplaceflag']=0
train_df.loc[train_df['fireplaceflag']=='Yes','fireplaceflag']=1
xtrain.info()
