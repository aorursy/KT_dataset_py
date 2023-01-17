import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import Geohash



import time



import xgboost as xgb



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



from sklearn.metrics import mean_squared_error





import seaborn as sns
df = pd.read_csv('../input/training.csv')
df = df.sample(10000)
def round3(x):

    return round(float(x)*10000)/10000
df['lat'] = df.apply(lambda x: round3(Geohash.decode_exactly(x['geohash6'])[0]), axis=1)



df['lng'] = df.apply(lambda x: round3(Geohash.decode_exactly(x['geohash6'])[1]), axis=1)
df['hour'] = df.apply(lambda x: float(x['timestamp'].split(':')[0]), axis=1)



df['minute'] = df.apply(lambda x: float(x['timestamp'].split(':')[1]), axis= 1)
df['dow'] =  df.apply(lambda x: x['day']%7, axis =1)
df.info()
df.corr()['demand']
sns.lineplot(data=df, x='hour', y='demand')
sns.lineplot(data=df, x='dow', y='demand')
sns.lineplot(data=df, x='lat', y='demand')
sns.lineplot(data=df, x='lng', y='demand')
sns.scatterplot(data=df, x='lng', y='lat', hue='demand')
sns.lineplot(data=df, x='minute', y='demand')
selectedColumn = ['lat','lng','hour','dow']
clf = RandomForestRegressor(max_depth=25,  n_estimators=240)
dfTrain, dfTest = train_test_split(df,test_size=0.2)
clf.fit(X=dfTrain[selectedColumn],y=dfTrain['demand'])
dfTest['predict'] = clf.predict(X=dfTest[selectedColumn])
mean_squared_error(dfTest['demand'], dfTest['predict'])
xgb_reg = xgb.XGBRegressor(learning_rate=0.01,max_depth=25,n_estimators=240, tree_method='hist')
xgb_reg.fit(X=dfTrain[selectedColumn],y=dfTrain['demand'])
dfTest['predict_xgb'] = xgb_reg.predict(data=dfTest[selectedColumn])
mean_squared_error(dfTest['demand'], dfTest['predict_xgb'])
xgb.plot_importance(xgb_reg)
xgb.plot_importance(xgb_reg, importance_type='cover')
df = pd.read_csv('../input/training.csv')
df['lat'] = df.apply(lambda x: round3(Geohash.decode_exactly(x['geohash6'])[0]), axis=1)



df['lng'] = df.apply(lambda x: round3(Geohash.decode_exactly(x['geohash6'])[1]), axis=1)
df['hour'] = df.apply(lambda x: float(x['timestamp'].split(':')[0]), axis=1)



df['minute'] = df.apply(lambda x: float(x['timestamp'].split(':')[1]), axis= 1)
df['dow'] =  df.apply(lambda x: x['day']%7, axis =1)
dfTrain, dfTest = train_test_split(df,test_size=0.2)
xgb_reg = xgb.XGBRegressor(learning_rate=0.01,max_depth=25,n_estimators=240, tree_method='gpu_hist')
starttime = time.time()

xgb_reg.fit(X=dfTrain[selectedColumn],y=dfTrain['demand'])

print('Training Time (s): ', time.time() - starttime)
dfTest['predict_xgb'] = xgb_reg.predict(data=dfTest[selectedColumn])
mean_squared_error(dfTest['demand'], dfTest['predict_xgb'])
xgb_reg.save_model('xgbReg.model')
def predictDemand(savedModel, geohash6='qp03wc', day=100, timestamp='00:00'):

    

    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



    import Geohash

    

    def round3(x):

        return round(float(x)*10000)/10000

    

    

    lat = round3(Geohash.decode_exactly(geohash6)[0])

    lng = round3(Geohash.decode_exactly(geohash6)[1])

    

    hour = float(timestamp.split(':')[0])

    

    dow = day%7

    

    dataX = pd.DataFrame({'lat': [lat], 'lng': [lng], 'hour': [hour], 'dow': [dow] })

    

    output = savedModel.predict(dataX)

    

    return output[0]

                 

                 
import xgboost as xgb



# load model

savedModel = xgb.XGBRegressor()

savedModel.load_model('xgbReg.model')



predictDemand(savedModel,geohash6='qp09sy', day=39, timestamp='3:0' )