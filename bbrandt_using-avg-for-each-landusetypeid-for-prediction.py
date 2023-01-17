import numpy as np
import pandas as pd
import gc


##### READ IN RAW DATA
def read_data():
    # see https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options
    # for the dtype option and why it is important
    print( "Reading data from disk ...")
    dtype={'parcelid': np.int32, 'logerror': np.float32, 'transactiondate': str, 'transaction_month': np.int32, 'airconditioningtypeid': np.float32, 'architecturalstyletypeid': np.float32, 'basementsqft': np.float32, 'bathroomcnt': np.float32, 'bedroomcnt': np.float32,  'buildingclasstypeid': np.float32, 'buildingqualitytypeid': np.float32,  'calculatedbathnbr': np.float32, 'decktypeid': np.float32, 'finishedfloor1squarefeet': np.float32,  'calculatedfinishedsquarefeet': np.float32, 'finishedsquarefeet12': np.float32,  'finishedsquarefeet13': np.float32, 'finishedsquarefeet15': np.float32,  'finishedsquarefeet50': np.float32, 'finishedsquarefeet6': np.float32, 'fips': np.float32,  'fireplacecnt': np.float32, 'fullbathcnt': np.float32, 'garagecarcnt': np.float32,  'garagetotalsqft': np.float32, 'hashottuborspa': object, 'heatingorsystemtypeid': np.float32,  'latitude': np.float32, 'longitude': np.float32, 'lotsizesquarefeet': np.float32, 'poolcnt': np.float32,  'poolsizesum': np.float32, 'pooltypeid10': np.float32, 'pooltypeid2': np.float32, 'pooltypeid7': np.float32,  'propertycountylandusecode': object, 'propertylandusetypeid': np.float32,  'propertyzoningdesc': object, 'rawcensustractandblock': np.float32, 'regionidcity': np.float32,  'regionidcounty': np.float32, 'regionidneighborhood': np.float32, 'regionidzip': np.float32,  'roomcnt': np.float32, 'storytypeid': np.float32, 'threequarterbathnbr': np.float32,  'typeconstructiontypeid': np.float32, 'unitcnt': np.float32, 'yardbuildingsqft17': np.float32,  'yardbuildingsqft26': np.float32, 'yearbuilt': np.float32, 'numberofstories': np.float32,  'fireplaceflag': object, 'structuretaxvaluedollarcnt': np.float32,  'taxvaluedollarcnt': np.float32, 'assessmentyear': np.float32, 'landtaxvaluedollarcnt': np.float32,  'taxamount': np.float32, 'taxdelinquencyflag': object, 'taxdelinquencyyear': np.float32,  'censustractandblock': np.float32}
    train_full = pd.read_csv("../input/train_2016_v2.csv",dtype=dtype,parse_dates=['transactiondate'])
    prop_full = pd.read_csv('../input/properties_2016.csv',dtype=dtype)
    sample_full = pd.read_csv('../input/sample_submission.csv')
    sample_full['parcelid'] = sample_full['ParcelId'] #rename so that parcelid spelling is the same
    print("shape of full training data:",train_full.shape)
    print("shape of full property data:",prop_full.shape)
    print("shape of full sample data:",sample_full.shape)
    print("\nFinished reading data from disk ...")
    return prop_full,train_full,sample_full

prop_full,train_full,sample_full=read_data()
landusetypeIDs=np.sort(prop_full['propertylandusetypeid'].unique())
print("Mean log error for entire data set:")
print(sum(train_full['logerror'])/len(train_full['logerror']))
#calculate and store the mean logerror for every propertylandusetypeid
meandict={}

for typeId in landusetypeIDs:
    print("propertylandusetypeid =", typeId)
    props=prop_full.loc[prop_full['propertylandusetypeid'] == typeId]
    propst=props.loc[props['parcelid'].isin(train_full.parcelid)]
    print("    number of properties with that typeID in (test,train) data =",len(props.index),len(propst.index))
    meandict[typeId]=0.0114572195563 #see description at the beginning
    if(len(propst.index)>1000): #we want more than 1000 properties in the training data for the respective landusetypeid so that the bias is hopefully statistically significant. Change this to whatever value you feel comfortable with 
        tprops=train_full.loc[train_full['parcelid'].isin(propst.parcelid)]    
        meandict[typeId]=sum(tprops['logerror'])/len(tprops.index)
        print("    average abs logerror = ",sum(abs(tprops['logerror']))/len(tprops.index))
        print("    average logerror = ",sum(tprops['logerror'])/len(tprops.index))
train_full=train_full.drop_duplicates(subset='parcelid',keep='first') #need to drop duplicates and triplicates

df_sample_full = sample_full.merge(prop_full, how='left', on='parcelid')
df_sample_full = df_sample_full.merge(train_full, how='left', on='parcelid')
df_sample_full = df_sample_full[['parcelid','propertylandusetypeid','logerror']] 
del prop_full;gc.collect();
del train_full;gc.collect();
del sample_full;gc.collect();
df_sample_full.head(n=20)
# predict the logerror based on the mean for the respective propertylandusetypeid
def myfillna2(logerr,propid):
    # almost all properties have a propertylandusetypeid set. However some very few don't 
    if np.isnan(propid):
        return 0.0114572195563 #see description at the beginning
    else:
        return meandict[propid]

pred = df_sample_full.apply(lambda x: myfillna2(x.logerror,x.propertylandusetypeid), axis=1)   

pred.head(n=20)
##### WRITE THE RESULTS

print( "\nPreparing results for write ..." )
y_pred=[]

for i,predict in enumerate(pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': df_sample_full['parcelid'],
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
from datetime import datetime

print( "\nWriting results to disk ..." )
output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ..." )