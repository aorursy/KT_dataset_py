import os

print((os.listdir('../input/')))



import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import numpy as np

from sklearn.feature_selection import SelectFromModel

import xgboost as xgb

from sklearn.preprocessing import StandardScaler
#import keras

#from keras.layers import Dense

#from keras.models import Sequential

#from keras import regularizers

#from keras.utils import to_categorical

#from sklearn.preprocessing import StandardScaler

#model = Sequential()
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

X_train, X_dev, y_train, y_dev = train_test_split( df_train.loc[:, 'V1':'V16'], df_train.loc[:, 'Class'], test_size=0.15, random_state=42)
test_index=df_test['Unnamed: 0'] #copying test index for later
X_train['V6'] = StandardScaler().fit_transform(X_train['V6'].values.reshape(-1, 1))



X_dev['V6'] = StandardScaler().fit_transform(X_dev['V6'].values.reshape(-1, 1))



df_test['V6'] = StandardScaler().fit_transform(df_test['V6'].values.reshape(-1, 1))



features = ['V2','V3', 'V4', 'V7','V8','V16']

for feature in features:

    for column in range(pd.get_dummies(X_train[feature]).columns.size):

        X_train[feature + '_' + str(column + 1)] = pd.get_dummies(X_train[feature])[column]

        

for feature in features:

    for column in range(pd.get_dummies(X_dev[feature]).columns.size):

        X_dev[feature + '_' + str(column + 1)] = pd.get_dummies(X_dev[feature])[column]



for feature in features:

    for column in range(pd.get_dummies(df_test[feature]).columns.size):

        df_test[feature + '_' + str(column + 1)] = pd.get_dummies(df_test[feature])[column]





X_train.drop(columns=features, axis = 1, inplace = True)

X_dev.drop(columns=features, axis = 1, inplace = True)

df_test.drop(columns=features, axis = 1, inplace = True)



df_test = df_test.loc[:, 'V1':'V16_4']

train_X = df_train.loc[:, 'V1':'V16']

train_y = df_train.loc[:, 'Class']

model = xgb.XGBClassifier(learning_rate = 0.12, max_depth = 2, min_child_weight = 1, subsample = 0.8, colsample_bytree = 1, n_estimators = 300)

model.fit(X_train, y_train)
#from sklearn.model_selection import GridSearchCV



#xgb_model = xgb.XGBClassifier()

#optimization_dict = {

#                      'learning_rate': [0.10, 0.12, 0.04, 0.06, 0.08, 0.14, 0.16],

#                        'max_depth': [1, 2,4,6, 8, 10],

#                     'n_estimators': [50,100,200, 300, 400]}



#model = GridSearchCV(xgb_model, optimization_dict, 

#                     scoring='accuracy', verbose=1)



#model.fit(X_train,y_train)

#print(model.best_score_)

#print(model.best_params_)
#df_test = df_test.loc[:, 'V1':'V16']

pred = model.predict_proba(df_test)

pred
pred_dev = model.predict(X_dev) 

pred_dev = pd.DataFrame(pred_dev)

score = roc_auc_score(y_dev, pred_dev)

score
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:, 1])

result.to_csv('output.csv', index=False)

result
#import keras

#from keras.layers import Dense

#from keras.models import Sequential

#from keras import regularizers

#from keras.utils import to_categorical

#from sklearn.preprocessing import StandardScaler

#model = Sequential()

#from sklearn.feature_selection import SelectKBest

#from sklearn.feature_selection import chi2

#bestfeatures = SelectKBest(score_func=chi2, k='all')

#fit = bestfeatures.fit(df_train.loc[:, 'V1':'V16'].abs() ,df_train.loc[:, 'Class'])

#dfscores = pd.DataFrame(fit.scores_)

#dfcolumns = pd.DataFrame(df_train.loc[:, 'V1':'V16'].columns)

#concat two dataframes for better visualization 

#featureScores = pd.concat([dfcolumns,dfscores],axis=1)

#featureScores.columns = ['Specs','Score']  #naming the dataframe columns

#print(featureScores.nlargest(10,'Score'))
#X_train['V13'] = StandardScaler().fit_transform(X_train['V13'].values.reshape(-1, 1))

#X_train['V15'] = StandardScaler().fit_transform(X_train['V15'].values.reshape(-1, 1))

#X_train['V6'] = StandardScaler().fit_transform(X_train['V6'].values.reshape(-1, 1))

#X_train['V12'] = StandardScaler().fit_transform(X_train['V12'].values.reshape(-1, 1))

#X_train['V14'] = StandardScaler().fit_transform(X_train['V14'].values.reshape(-1, 1))



#X_dev['V15'] = StandardScaler().fit_transform(X_dev['V15'].values.reshape(-1, 1))

#X_dev['V13'] = StandardScaler().fit_transform(X_dev['V13'].values.reshape(-1, 1))

#X_dev['V6'] = StandardScaler().fit_transform(X_dev['V6'].values.reshape(-1, 1))

#X_dev['V12'] = StandardScaler().fit_transform(X_dev['V12'].values.reshape(-1, 1))

#X_dev['V14'] = StandardScaler().fit_transform(X_dev['V14'].values.reshape(-1, 1))





#X_train['V3_1'] = pd.get_dummies(X_train['V3'])[0]

#X_train['V3_2'] = pd.get_dummies(X_train['V3'])[1]

#X_train['V3_3'] = pd.get_dummies(X_train['V3'])[2]



#X_train['V7_1'] = pd.get_dummies(X_train['V7'])[0]

#X_train['V7_2'] = pd.get_dummies(X_train['V7'])[1]



#X_train['V8_1'] = pd.get_dummies(X_train['V8'])[0]

#X_train['V8_2'] = pd.get_dummies(X_train['V8'])[1]



#X_train['V9_1'] = pd.get_dummies(X_train['V9'])[0]

#X_train['V9_2'] = pd.get_dummies(X_train['V9'])[1]

#X_train['V9_3'] = pd.get_dummies(X_train['V9'])[2]



#X_dev['V3_1'] = pd.get_dummies(X_dev['V3'])[0]

#X_dev['V3_2'] = pd.get_dummies(X_dev['V3'])[1]

#X_dev['V3_3'] = pd.get_dummies(X_dev['V3'])[2]



#X_dev['V7_1'] = pd.get_dummies(X_dev['V7'])[0]

#X_dev['V7_2'] = pd.get_dummies(X_dev['V7'])[1]



#X_dev['V8_1'] = pd.get_dummies(X_dev['V8'])[0]

#X_dev['V8_2'] = pd.get_dummies(X_dev['V8'])[1]



#X_dev['V9_1'] = pd.get_dummies(X_dev['V9'])[0]

#X_dev['V9_2'] = pd.get_dummies(X_dev['V9'])[1]

#X_dev['V9_3'] = pd.get_dummies(X_dev['V9'])[2]



#X_train.drop(columns=['V3', 'V7', 'V8', 'V9'],axis = 1, inplace = True)

#X_dev.drop(columns=['V3', 'V7', 'V8', 'V9'], axis = 1, inplace = True)
#lam = 0.1

#model.add(Dense(X_train.columns.size, activation='relu', input_dim=X_train.columns.size))

#model.add(Dense(24, activation='relu'))

#model.add(Dense(32, activation='relu'))

#model.add(Dense(36, activation='relu'))

#model.add(Dense(40, activation='relu'))

#model.add(Dense(34, activation='relu'))

#model.add(Dense(30, activation='relu'))

#model.add(Dense(20, activation='relu'))

#model.add(Dense(10, activation='relu'))

#model.add(Dense(4, activation='relu'))

#model.add(Dense(1, activation='sigmoid'))



#model.compile(optimizer='adam', loss='binary_crossentropy')

#model.fit(X_train, y_train, epochs=70, batch_size=256)

#pred = model.predict(X_dev)
#pred_dev = pd.DataFrame(pred[:])

#score = roc_auc_score(y_dev, pred_dev)

#df_test['V1'] = StandardScaler().fit_transform(X_dev['V1'].values.reshape(-1, 1))

#df_test['V13'] = StandardScaler().fit_transform(df_test['V13'].values.reshape(-1, 1))

#df_test['V15'] = StandardScaler().fit_transform(df_test['V15'].values.reshape(-1, 1))



#df_test['V6'] = StandardScaler().fit_transform(df_test['V6'].values.reshape(-1, 1))

#df_test['V12'] = StandardScaler().fit_transform(df_test['V12'].values.reshape(-1, 1))

#df_test['V14'] = StandardScaler().fit_transform(df_test['V14'].values.reshape(-1, 1))



#df_test['V3_1'] = pd.get_dummies(df_test['V3'])[0]

#df_test['V3_2'] = pd.get_dummies(df_test['V3'])[1]

#df_test['V3_3'] = pd.get_dummies(df_test['V3'])[2]



#df_test['V7_1'] = pd.get_dummies(df_test['V7'])[0]

#df_test['V7_1'] = pd.get_dummies(df_test['V7'])[0]

#df_test['V7_2'] = pd.get_dummies(df_test['V7'])[1]

#df_test['V7_2'] = pd.get_dummies(df_test['V7'])[1]



#df_test['V8_1'] = pd.get_dummies(df_test['V8'])[0]

#df_test['V8_2'] = pd.get_dummies(df_test['V8'])[1]



#df_test['V9_1'] = pd.get_dummies(df_test['V9'])[0]

#df_test['V9_2'] = pd.get_dummies(df_test['V9'])[1]

#df_test['V9_3'] = pd.get_dummies(df_test['V9'])[2]



#df_test.drop(columns=['V3', 'V7', 'V8', 'V9'], axis = 1, inplace = True)



#df_test = df_test.loc[:, 'V1':'V9_3']

#pred = model.predict(df_test)



#result=pd.DataFrame()

#result['Id'] = test_index

#result['PredictedValue'] = pd.DataFrame(pred)

#result['PredictedValue'] = ['%.4f' % elem for elem in result['PredictedValue']]

#result.to_csv('output.csv', index=False)

#result