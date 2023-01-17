import pandas as pd 
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt 
from currency_converter import CurrencyConverter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gender_guesser.detector as gender


# read in train and test data
data = pd.read_csv('train_set.csv')

train_len = data.shape[0]-1

data = pd.concat([data.iloc[:,:-1],pd.read_csv('test_set.csv')])


# feature engineering
# gender
data['first_name'] = data['creator_name'].str.split().str[0]
d = gender.Detector()
data['creator_gender'] = data['first_name'].apply(lambda x: d.get_gender(x))
data.loc[(data.creator_gender == 'mostly_male'), 'creator_gender'] = "male"
data.loc[(data.creator_gender == 'mostly_female'), 'creator_gender'] = "female"
data.loc[(data.creator_gender == 'andy'), 'creator_gender'] = "unknown"

# quarter
def add_quarter(file, date_col):
    colname_quarter = str(date_col) + "_Q"
    colname_year = str(date_col) + "_YR"
    colname_full = str(date_col+"_quarter")
    file[colname_quarter]=pd.to_datetime(file[date_col]).dt.quarter.astype(str).apply(lambda x:x[0]+'Q')
    file[colname_year]=pd.to_datetime(file[date_col]).dt.year.astype(str).apply(lambda x:x+' ')
    file[colname_full]=file[colname_year]+file[colname_quarter]
    return file

data = add_quarter(data, 'launched_at')

# submission_num for each creator_id
data1 = data.iloc[:(train_len+1),:]
data2 = data.iloc[(train_len+1):,:]

data1['deadline_2'] = pd.to_datetime(data1['deadline'])
data1  = data1.sort_values(['creator_id', 'deadline_2'], ascending=[True, True])
data1['submission_num'] = data1.groupby(['creator_id'])['deadline'].transform(lambda x: list(map(lambda y: dict(map(reversed, dict(enumerate(x.unique())).items()))[y]+1,x)))

data2['deadline_2'] = pd.to_datetime(data2['deadline'])
data2  = data2.sort_values(['creator_id', 'deadline_2'], ascending=[True, True])
data2['submission_num'] = data2.groupby(['creator_id'])['deadline'].transform(lambda x: list(map(lambda y: dict(map(reversed, dict(enumerate(x.unique())).items()))[y]+1,x)))

data = pd.concat([data1.sort_index(),data2.sort_index()])


# covert all currencies to USD
c = CurrencyConverter(fallback_on_missing_rate=True)

USD_norm = []
for i in range(data.shape[0]):
    x = data.iloc[i,:]
    USD_norm.append(c.convert(x['goal'], x['currency'], 'USD', date=datetime.strptime(x['launched_at'], '%m/%d/%y %H:%M')))

data['USD_norm'] = USD_norm
data = data[data['USD_norm'] < 100000000] # remove outlier, also good for scaling

# generate features
data['create2launch']=(pd.to_datetime(data['launched_at'])-pd.to_datetime(data['created_at'])).dt.days
data['launch2deadline']=(pd.to_datetime(data['deadline'])-pd.to_datetime(data['launched_at'])).dt.days

data = data.fillna(0)

feature_list = ['creator_register','USD_norm','sub_category','country','location_type','staff_pick','show_feature_image','create2launch','launch2deadline','disable_communication','duration','submission_num','launched_at_quarter','creator_gender']

# one hot encoding non-numerical features
X = np.array(data['creator_register']).reshape(-1, 1)
for i in feature_list[1:]:
    if data[i].dtype == data['main_category'].dtype:
        enc = preprocessing.OneHotEncoder()
        enc_i = enc.fit(np.array(data[i]).reshape(-1, 1))
        enc_i = enc.transform(np.array(data[i]).reshape(-1, 1)).toarray()
        X = np.append(X,enc_i,axis=1)
    elif i == 'creator_register':
        continue
    else:
        data[i] = data[i]/np.linalg.norm(data[i])
        X = np.append(X,np.array(data[i]).reshape(-1, 1),axis=1)

# create truth labels
train_data = pd.read_csv('train_set.csv')
train_data = train_data[train_data['id']!='ks2083255961']
y_train = np.array(train_data['outcome'])

X_train = X[:train_len,:]
X_test = X[train_len:,:]

# train tuned model
clf5 = xgb.XGBClassifier(objective="binary:logistic", n_estimators = 500, random_state=42, reg_alpha=0.02)
clf5 = clf5.fit(X_train, y_train)
y_pred = clf5.predict_proba(X_test)

# introduce rules on the feature "create_register"
y_pred[X_test[:,0]==0,0]=1
y_pred[:,1]=1-y_pred[:,0]


# write final output
test_file_id = pd.read_csv('test_set.csv')['id']
outfile = open('submission_team5.csv','w')  
outfile.write('Id,Predicted\n')
for i in range(y_pred.shape[0]):
    outfile.write('%s,%.6f\n' %(test_file_id[i],y_pred[i,1]))

outfile.close()