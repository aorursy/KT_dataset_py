from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import operator
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import mean_absolute_error
import seaborn as sns

### Data from Kaggle
properties = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv("../input/train_2016_v2.csv")


print(properties.shape)
print(train.shape)
properties = properties.rename(columns = {'calculatedfinishedsquarefeet':'cal_fin_sqft'})
properties = properties.rename(columns = {'structuretaxvaluedollarcnt':'struc_TVDC'})
properties = properties.rename(columns = {'taxvaluedollarcnt':'TVDC'})
properties = properties.rename(columns = {'landtaxvaluedollarcnt':'land_TVDC'})
properties = properties.rename(columns = {'finishedsquarefeet15':'fin_sqft15'})
properties = properties.rename(columns = {'finishedsquarefeet12':'fin_sqft12'})
properties = properties.rename(columns = {'finishedsquarefeet6':'fin_sqft6'})
plt.figure(figsize=(12,8))
plt.hist(train.logerror.values, bins=100)
plt.title('error metric distribution')
plt.xlabel('logerror', fontsize=12)
plt.style.use('ggplot')
plt.show()
plt.figure(figsize=(12,12))
plt.scatter(y=properties.iloc[0:100000].latitude.values, x=properties.iloc[0:100000].longitude.values)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('Greater Los Angeles Metro Area')
plt.show()



#life of property
properties['N-life'] = 2018 - properties['yearbuilt']

#life of property
properties['N-LivingAreaError'] = properties['cal_fin_sqft']/properties['fin_sqft12']

#proportion of living area
properties['N-LivingAreaProp'] = properties['cal_fin_sqft']/properties['lotsizesquarefeet']
properties['N-LivingAreaProp2'] = properties['fin_sqft12']/properties['fin_sqft15']

#Amout of extra space
properties['N-ExtraSpace'] = properties['lotsizesquarefeet'] - properties['cal_fin_sqft'] 
properties['N-ExtraSpace-2'] = properties['fin_sqft15'] - properties['fin_sqft12'] 

#Total number of rooms
properties['N-TotalRooms'] = properties['bathroomcnt']*properties['bedroomcnt']

#Average room size
properties['N-AvRoomSize'] = properties['cal_fin_sqft']/properties['roomcnt'] 

# Number of Extra rooms
properties['N-ExtraRooms'] = properties['roomcnt'] - properties['N-TotalRooms'] 

#Ratio of the built structure value to land area
properties['N-ValueProp'] = properties['struc_TVDC']/properties['land_TVDC']

#Does property have a garage, pool or hot tub and AC?
properties['N-GarPoolAC'] = ((properties['garagecarcnt']>0) & (properties['pooltypeid10']>0) & (properties['airconditioningtypeid']!=5))*1 

properties["N-location"] = properties["latitude"] + properties["longitude"]
properties["N-location-2"] = properties["latitude"]*properties["longitude"]
properties["N-location-2round"] = properties["N-location-2"].round(-4)

properties["N-latitude-round"] = properties["latitude"].round(-4)
properties["N-longitude-round"] = properties["longitude"].round(-4)


gbycol = 'rawcensustractandblock'
col1 = 'taxamount'
properties['gby_single_'+gbycol+col1] = properties[gbycol].map(properties.groupby([gbycol])[col1].median().to_dict())


c = 'cal_fin_sqft'
properties[c+'gby_median']= properties.loc[~properties['rawcensustractandblock'].isnull(),['rawcensustractandblock',c]].groupby(['rawcensustractandblock']).transform(lambda x: x/(x.median()))

#Ratio of the built structure value to land area
properties['N-ValueRatio'] = properties['TVDC']/properties['taxamount']

col2 = 'bathroomcnt'
properties['gby_single_'+gbycol+col2] = properties[gbycol].map(properties.groupby([gbycol])[col2].median().to_dict())




properties.columns
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))


# xgboost params
xgb_params = {
    'eta': 0.1,
    'max_depth': 12,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda':0.8,
    'alpha': 0.4, 
    'base_score': y_mean,
    'silent': 1,
    'colsample_bylevel':1.0,
    'min_child_weight':4.0,
    'colsample_bytree':0.5
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

# cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   nfold=5,
                   num_boost_round=3900,
                   early_stopping_rounds=50,
                   verbose_eval=10, 
                   show_stdv=False
                  )
num_boost_rounds = len(cv_result)
print("Number of boosting rounds {}".format(num_boost_rounds))
# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
pred = model.predict(dtest)


output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': pred, '201611': pred, '201612': pred,
        '201710': pred, '201711': pred, '201712': pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
from datetime import datetime
filename = 'zillow_teja_programming_sample.csv'
#output.to_csv(filename.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
import operator
sorted_x = sorted(model.get_fscore().items(), key=operator.itemgetter(1))
sorted_df = pd.DataFrame(sorted_x)
sorted_df.columns = ['feature_name','feature_imp']
sorted_df.tail(10)
plt.figure(figsize = (15,5))
plt.plot(sorted_df.loc[:,'feature_name'][-10:],sorted_df.loc[:,'feature_imp'][-10:])
plt.style.use('ggplot')
plt.ylabel('Feature Importance', fontsize=12)
plt.xlabel('Feature Name', fontsize=12)
plt.title('Feature Importance of different variables')
plt.tight_layout()
plt.xticks(rotation = 45)
plt.show()
imp_cols = sorted_df.loc[:,'feature_name'][-10:].values
imp_cols = np.append(imp_cols,'logerror')


corrmat = train_df[imp_cols].corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()

