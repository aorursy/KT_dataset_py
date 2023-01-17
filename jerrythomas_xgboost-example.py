import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from patsy import dmatrices
from subprocess import check_output
from sklearn import metrics
import xgboost as xgb
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.columns.values
test.columns.values
#merge train and test data
def merge(train, test):
   train_h = list(train.columns.values)
   test_h = list(test.columns.values)
   #print test_h
   #print train_h
   all_h = set(train_h + test_h)
   #print all_h
   add_to_test = list(all_h - set(test_h))
   add_to_train = list(all_h - set(train_h))
   for col in add_to_test:
      test[col] = np.nan
   for col in add_to_train:
      train[col] = np.nan
   test = test.reindex_axis(sorted(test.columns), axis=1)
   train = train.reindex_axis(sorted(train.columns), axis=1)
   print (test.shape)
   print (train.shape)
   merged = pd.concat([test, train], axis=0)
   print (merged.shape)
   return merged
#time engineering
def timeEngineering(data):
   data['Dates'] = pd.to_datetime(data['Dates'])
   data['year'] = data['Dates'].dt.year
   data['month'] = data['Dates'].dt.month
   data['week'] = data['Dates'].dt.week
   data['day'] = data['Dates'].dt.day
   data['hour'] = data['Dates'].dt.hour
   return data
#convert textual features to numerical
def featureToInteger(data, feature):
   notnan = data[feature].dropna(how='all')
   unique = notnan.unique()
   #print unique
   data[feature+'_1'] = np.nan
   for cls in unique:
      cls_ind = np.where(unique==cls)[0][0]
      data[feature+'_1'][data[feature]==cls] = cls_ind
   return data


#extract address features
def addressFeatures(data):
   feature = 'Address'
   index = 0
   feat_extr = {'addr_num':0,
                'addr_block':0,
                'addr1':'',
                'addr1_type':'',
                'addr2':'', 
                'addr2_type':''}
   entries = []
   for row in zip(data[feature]):
     string = row[0]
     string = string.upper()
     #print string
     if '/' in string:
         #print 'two addresses'
         #print string
         feat_extr['addr_num'] = 2 #two addresses
         string = string.split('/')
         feat_extr['addr2'] = string[0].lstrip().rstrip()
         tmp = string[0].split()
         feat_extr['addr2_type'] = tmp[-1]
         string = string[1]
         #print string
     else:
         #print 'only one address'
         feat_extr['addr_num'] = 1 #only one address  
         feat_extr['addr2'] = 'none'
         feat_extr['addr2_type'] = 'none'
     if 'BLOCK OF' in string:
         #print 'it is a block'
         #print string
         string = string.replace('BLOCK OF', '')
         string = string.replace('  ', ' ')
         feat_extr['addr_block'] = 1 # is a block
     else: 
         #print 'it is not a block'
         feat_extr['addr_block'] = 0 #not a block
     feat_extr['addr1'] = string.lstrip().rstrip()
     tmp = string.split()
     try:
        feat_extr['addr1_type'] = tmp[-1]
     except IndexError:
        feat_extr['addr1_type'] = 'none'
     #print feat_extr
     entry = feat_extr
     #print (entry)
     entries.append(entry)
     index = index + 1
     if index%500000 == 0:
        print ('processed %d rows'%index)
   df = pd.DataFrame.from_records(entries)
   print ('extracted address features:', df.shape)
   return df


#make a class label
def makeClass(data):
   categories = data['Category'].dropna(how='all')
   classes = categories.unique()
   #print classes
   data['class'] = np.nan
   for cls in classes:
      cls_ind = np.where(classes==cls)[0][0]
      data['class'][data['Category']==cls] = cls_ind
   #print (classes)
   df_classes = pd.DataFrame(classes)
   #df_classes.to_csv('categories.csv')
   return classes, data
def extractFeatures(train, test):
   data = merge(train, test)
   df_addr = addressFeatures(data)
   data = timeEngineering(data)
   data = data.reset_index()
   #data = pd.read_csv('merged.csv')
   #df_addr = pd.read_csv('addr_features.csv')
   data = pd.concat([data, df_addr], axis=1)
   #data = data.drop('index')
   data = data[list(set(data.columns)-set(['index']))]
   print (data.head())
   #features to convert: textual to numerical
   #features = ['Resolution', 'Address', 'DayOfWeek', 'Descript', 'PdDistrict', 'addr1', 'addr1_type', 'addr2', 'addr2_type']
   features = ['Resolution', 'DayOfWeek', 'PdDistrict', 'addr1_type', 'addr2_type']
   for feature in features:
      print (feature)
      data = featureToInteger(data, feature)
   #data.to_csv('data.csv', index=False)
   classes, data = makeClass(data)
   #drop columns not used in prediction
   features_drop = ['Dates', 'Address', 'DayOfWeek', 'Descript', 'PdDistrict', 'Resolution', 'addr1', 'addr2', 'addr1_type', 'addr2_type']
   features_left = list(set(data.columns)-set(features_drop))
   #print (features_left)
   data = data[features_left]
   return classes, data
def dataForML(data):
   features_left_x = list(set(data.columns)-set(['class','Category']))
   #data['addr1_type_1'] = data['addr1_type_1'].astype(int)
   #data['addr2_type_1'] = data['addr2_type_1'].astype(int)
   data_train = data[~data['class'].isnull()]
   data_test = data[data['class'].isnull()]
   data_train_x = data_train[features_left_x]
   data_train_y = data_train['class']
   data_test_x = data_test[features_left_x]
   data_test_y = data_test['class']

   data = dict(x_train=data_train_x,
               x_test=data_test_x,
               y_train=data_train_y,
               y_test=data_test_y)
   return data

def predict(data, classes):
  print (list(data['x_train'].columns))
  xg_train = xgb.DMatrix(data['x_train'], label=data['y_train'])
  xg_test = xgb.DMatrix(data['x_test'], label=data['y_test'])
  print ('matrices created')
  # setup parameters for xgboost
  param = {}
  # use softprob multi-class classification
  param['objective'] = 'multi:softprob'
  param['eta'] = 0.1
  #param['eta'] = 0.8
  param['max_depth'] = 6
  param['silent'] = 1
  param['nthread'] = 4
  param['num_class'] = len(classes)
  #param['max_delta_step'] = 1
  num_round = 15
  bst = xgb.train(param, xg_train, num_round)
  print ('model built')
  # get prediction
  pred = bst.predict(xg_test)
  print ('prediction done')
  #format the prediction
  df_pred = pd.DataFrame(pred)
  df_pred.columns = classes
  df_pred = df_pred.reindex_axis(sorted(df_pred.columns), axis=1)
  df_pred = pd.concat([data['x_test']['Id'], df_pred],axis=1)
  df_pred['Id'] = df_pred['Id'].astype(int)
  return df_pred

classes, features = extractFeatures(train, test)
data = dataForML(features)
df_pred = predict(data, classes)
print ('prediction completed')
df_pred.to_csv('submission.csv', index=False,  float_format='%.6f')