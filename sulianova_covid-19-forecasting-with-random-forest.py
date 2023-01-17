# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv", index_col = 'Id')

test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv", index_col = 'ForecastId')

pd.set_option('display.max_columns', 150)

pd.set_option('display.max_rows', 150)

train_df.head()
train_df.info()
test_df.head()
train_df.shape, test_df.shape
y_train_cc = np.array(train_df['ConfirmedCases'].astype(int))

y_train_ft = np.array(train_df['Fatalities'].astype(int))

cols = ['ConfirmedCases', 'Fatalities']



full_df = pd.concat([train_df.drop(cols, axis=1), test_df])

index_split = train_df.shape[0]

full_df = pd.get_dummies(full_df, columns=full_df.columns)



x_train = full_df[:index_split]

x_test= full_df[index_split:]

#x_train.shape, x_test.shape, y_train_cc.shape, y_train_ft.shape
#Regular Random Forest Regressor

rf = RandomForestRegressor(n_estimators=100, n_jobs= -1, min_samples_leaf=3, random_state=17)



rf.fit(x_train,y_train_cc)
y_pred_cc = rf.predict(x_test)

y_pred_cc = y_pred_cc.astype(int)

y_pred_cc[y_pred_cc <0]=0
rf_f = RandomForestRegressor(n_estimators=100, n_jobs= -1, min_samples_leaf=3, random_state=17)



rf_f.fit(x_train,y_train_ft)
y_pred_ft = rf_f.predict(x_test)

y_pred_ft = y_pred_ft.astype(int)

y_pred_ft[y_pred_ft <0]=0

predicted_df_rf = pd.DataFrame([y_pred_cc, y_pred_ft], index = ['ConfirmedCases','Fatalities'], columns= np.arange(1, y_pred_cc.shape[0] + 1)).T

predicted_df_rf.to_csv('submission_rf.csv', index_label = "ForecastId")
from sklearn.preprocessing import LabelEncoder



cols = ['ConfirmedCases', 'Fatalities']

index_split = train_df.shape[0]



full_df = pd.concat([train_df.drop(cols, axis=1), test_df])

full_df.Date = pd.to_datetime(full_df.Date)

full_df.Date = full_df.Date.astype('int64')

#full_df['Date'] = full_df['Date'].apply(pd.to_datetime)

#full_df['day_of_week'] = full_df['Date'].apply(lambda ts: ts.weekday()).astype('int')

#full_df['month'] = full_df['Date'].apply(lambda ts: ts.month)

#full_df['day'] = full_df['Date'].apply(lambda ts: ts.day)

#full_df.drop(['Date', 'Province_State'],axis=1, inplace= True )

full_df.drop(['Province_State'],axis=1, inplace= True )



le = LabelEncoder()

def CustomLabelEncoder(df):

    for c in df.columns:

        if df.dtypes[c] == object:

            le.fit(df[c].astype(str))

            df[c] = le.transform(df[c].astype(str))

    return df



full_df_encoded = CustomLabelEncoder(full_df)



train_encoded = full_df[:index_split]

test_encoded= full_df[index_split:]
#from sklearn.ensemble import RandomForestClassifier



rf_params = {'max_features':  [1, 2, 3], 'min_samples_leaf': [5, 10, 15, 20], 'max_depth': [8, 10, 20, 30]}

rf = RandomForestRegressor(n_estimators=100, random_state=17, n_jobs= -1)

#gcv = GridSearchCV(rf, rf_params, n_jobs=-1, cv=5, verbose=1)

#gcv.fit(train_encoded,y_train_cc)
#gcv.best_params_ #for RF Classifier {'max_depth': 8, 'max_features': 1, 'min_samples_leaf': 15}

                 #RF Regressor 
rf = RandomForestRegressor(max_depth = 8, min_samples_leaf=20, random_state=17, n_estimators=100, n_jobs= -1)



rf.fit(train_encoded,y_train_cc)
y_pred_cc = rf.predict(test_encoded)

y_pred_cc = y_pred_cc.astype(int)

y_pred_cc[y_pred_cc <0]=0
y_train_ft.shape, train_encoded.shape, test_encoded.shape
#gcv.fit(train_encoded,y_train_ft)
#gcv.best_params_ #RF Classifier {'max_depth': 8, 'max_features': 1, 'min_samples_leaf': 20}

                 #RF Regressor 
rf = RandomForestRegressor(max_depth = 8, min_samples_leaf=20, random_state=17, n_estimators=100, n_jobs= -1)



rf.fit(train_encoded,y_train_ft)
y_pred_ft = rf.predict(test_encoded)

y_pred_ft = y_pred_ft.astype(int)

y_pred_ft[y_pred_ft <0] = 0
from sklearn.ensemble import RandomForestClassifier



rfcla = RandomForestClassifier(n_estimators=100, max_samples=0.8,

                        random_state=1)

# We train model

rfcla.fit(train_encoded, y_train_cc)
predictions = rfcla.predict(test_encoded)
rfcla.fit(train_encoded,y_train_ft)
predictions1 = rfcla.predict(test_encoded)
submission = pd.DataFrame({'ForecastId': test_df.index,'ConfirmedCases':predictions,'Fatalities':predictions1})

filename = 'submission.csv'



submission.to_csv(filename,index=False)
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_squared_log_error, make_scorer

from sklearn.model_selection import cross_val_score



def RMSLError(y_test, predictions):

    return np.sqrt(mean_squared_log_error(y_test, predictions))

    

rmsle_score = make_scorer(RMSLError, greater_is_better=False)

time_split = TimeSeriesSplit(n_splits=10)



cv_scores = cross_val_score(rfcla, train_encoded, y_train_cc, cv=time_split,  scoring=rmsle_score, n_jobs=2)

cv_scores * (-1)
np.array([2.79932766, 2.00873436, 4.07550427, 1.90746791, 2.11054789,

       1.32860041, 2.52991252, 2.78177521, 1.13328816, 1.39190609]).mean()
cv_scores = cross_val_score(rf, train_encoded, y_train_cc, cv=time_split,  scoring=rmsle_score, n_jobs=2)

[cv_scores * (-1)].mean()
np.array([2.56756208, 2.02996862, 6.17504173, 1.62328543, 2.08334431, \

 2.64283866, 1.9898001 , 3.19179292, 1.26391612, 1.83892938]).mean()