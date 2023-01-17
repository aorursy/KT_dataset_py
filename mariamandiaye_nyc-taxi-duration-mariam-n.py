





import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

%matplotlib inline



import os







#train = os.path.join('..', 'input', 'train.csv')

#test = os.path.join('..', 'input', 'train.csv')



#df_train = pd.read_csv(train )

#df_test  = pd.read_csv(test)



df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.head()
df_train.describe()
df_train.hist(bins=60, figsize=(30,15))

plt.show()
df_train.loc[df_train['trip_duration'] < 5000, 'trip_duration'].hist();



plt.title('trip_duration')

plt.show()
plt.subplots(figsize=(18,6))

plt.title("Visualisation des valeurs aberrantes ")

df_train.boxplot();
df_train.isna().sum()





sh =df_train.shape[0]



df_train.head()
 

#print('il ya ', df_train[df_train['distances']<=0.01].shape[0], 'voyage moins de 1min')

#df_train = df_train[df_train['distances']>0.01]

#print('il ya ', df_train[df_train['distances']<=0.01].shape[0], 'voyage moins de 1min')


print('il ya', df_train[df_train['trip_duration']<=1*60].shape[0], "voyage moins de 1min")

df_train = df_train[df_train['trip_duration']>1*60]

print('il ya', df_train[df_train['trip_duration']<=1*60].shape[0], 'voyage moins de 1min')


#df_train[df_train['distances']/(df_train['trip_duration'])>=200/3600].shape[0]

#df_train = df_train[df_train['distances']/(df_train['trip_duration'])<200/3600]

#print('There is', df_train[df_train['distances']/(df_train['trip_duration'])>=200/3600].shape[0], 'se déplace à une vitesse moyenne supérieure à 200 km / h après filtrage')


#print('', df_train[df_train['trip_duration']>=3*3600].shape[0], )

#df_train = df_train[df_train['trip_duration']<3*3600]

#print('', df_train[df_train['trip_duration']>=3*3600].shape[0]


#print('il ya ', df_train[df_train['distances']/(df_train['trip_duration'])<=1/3600].shape[0]

#df_train = df_train[df_train['distances']/(df_train['trip_duration'])>1/3600]

#print('il ya ', df_train[df_train['distances']/(df_train['trip_duration'])<=1/3600].shape[0]
print('nous avons filtrer','{:.3}'.format((sh-df_train.shape[0])/sh*100) , '' )
def Ftr_sel(df_in):

    CATEG = [ 'store_and_fwd_flag' ]

    NUME = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'zone', 'distances', 'pickup_Month', 'pickup_Hour', 'pickup_Weekend', 'passenger_count', 'vendor_id' ]

    categ = CATEG

    nume = NUME



    X=df_in.loc[:, categ + nume]



    for cat in categ:

        X[cat] = X[cat].astype('category').cat.codes



    return X
X_train = Ftr_sel(df_train)

target = 'trip_duration'

y_train = df_train.loc[:, target]

print(X_train.shape, y_train.shape)

y_train = np.log1p( y_train )

X_train.head()
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_squared_log_error as MSLE

import xgboost
X_train_sample, X_validation, y_train_sample, y_validation = train_test_split(X_train, y_train, test_size=.2, random_state=42 )

print(X_train_sample.shape, y_train_sample.shape , X_validation.shape, y_validation.shape)

X_train_sample.head(5)
#min_samples_leaf = {  1: 0.14335025261894946,  2: 0.13981831370645642,   3: 0.13852060557356807,  4: 0.1374604137021863, 5: 0.13701190316428685, 6: 0.13719592541154788,  7: 0.13647552678899308,   8: 0.13668619429239404,  9: 0.13678934918189598, 10: 0.13720206662667936, 15: 0.1378838545097919,   20: 0.13858468007164235,  25: 0.1397767624826059, 30: 0.14040835836429333, 35: 0.14162848146663448,  40: 0.14219905657487034,  45: 0.14265841548835242, 50: 0.14374664124817566, 100: 0.14924626267746,   150: 0.15302159678464494, 200: 0.15600849362124466, 250: 0.1578977545855252,  300: 0.16053779676581148, }

#plt.plot(min_samples_leaf.keys(), min_samples_leaf.values());

#plt.title('min_samples_leaf optimization');

#plt.legend(" with hyperparameters: RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features=0.4, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=9, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1, oob_score=False, random_state=50, verbose=0, warm_start=False)") #plt.legend(" with features: pickup_latitude	pickup_longitude	dropoff_latitude	dropoff_longitude	zone	distances	pickup_Month	pickup_Hour	pickup_Weekend	passenger_count ")

#plt.xlabel('min_samples_leaf');

#plt.ylabel('MSLE score');

#min(min_samples_leaf, key=min_samples_leaf.get)


#params = { 'booster':'gbtree', 'verbosity':1, 'max_depth':15, 'subsample': 1, 'lamda':0, 'max_delta_step':3, 'objective':'reg:linear', 'learning_rate':0.08, 'colsample_bytree':0.9, 'colsample_bylevel':0.9}

#data_train = xgboost.DMatrix(X_train_sample,y_train_sample)

#model = xgboost.train(params, data_train, num_boost_round=200)
#real = list(np.expm1(y_validation))

#predicted = list(np.expm1(model.predict(xgboost.DMatrix(X_validation))))

#print('\nMean Square Log Error score:', MSLE(real, predicted))


#rf = RandomForestRegressor( n_estimators=100, min_samples_leaf=1, max_depth=None, max_features=.4, oob_score=False, bootstrap=True, n_jobs=-1 )



params = { 'booster':'gbtree', 'verbosity':1, 'max_depth':15, 'subsample': 1, 'lamda':0, 'max_delta_step':3, 'objective':'reg:linear', 'learning_rate':0.08, 'colsample_bytree':0.9, 'colsample_bylevel':0.9}

data_train = xgboost.DMatrix(X_train,y_train)

bost = xgboost.train(params, data_train, num_boost_round=200)


#rf.fit( X_train, y_train );

#rf.fr
#rf1_scores=-cross_val_score( rf, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error' )

#rf1_scores, np.mean(rf1_scores)
X_test = Ftr_sel(df_test)

X_test.head()
#y_test_predict = model.predict(X_test)

#y_test_predict = np.expm1(y_test_predict)

#y_test_predict[:5]
y_test_predict = np.expm1(bost.predict(xgboost.DMatrix(X_test)))

y_test_predict[:5]
my_submission = pd.DataFrame(df_test.loc[:, 'id'])

my_submission['trip_duration'] = y_test_predict

print(my_submission.shape)

my_submission.head(5)
my_submission.to_csv("submite.csv", index=False)