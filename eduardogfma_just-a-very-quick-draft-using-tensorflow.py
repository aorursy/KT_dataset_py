import numpy as np
import pandas as pd
raw_data = pd.read_csv("../input/online_sex_work.csv")

print('Initial shape:')
print(raw_data.shape)

raw_data=raw_data.dropna(axis=0, how='any').reset_index(drop=True)
print('\n Shape neglecting NaN:')
print(raw_data.shape)

raw_data.head()
raw_data = raw_data.drop(['User_ID', 'Verification', 'Looking_for', 'Location', 'Friends_ID_list'], axis=1)
raw_data.head()
import re

m=len(raw_data.Last_login)

rep=np.zeros(m)

for i in range(m):
    s = raw_data.Last_login[i]
    r = re.split('_+',s)
    rep[i] = int(r[1])
    
raw_data['Last_login']=pd.DataFrame(rep)
raw_data.head()
m = len(raw_data.Member_since)

# Create 3 vectors: day, month, year
day = np.zeros(m)
month = day
year = day

for i in range (m):
    s = raw_data.Member_since[i]
    r = re.split('\.', s)
    day[i] = r[0]
    month[i] = r[1]
    year[i] = r[2]
i
raw_data[i:i+2]
raw_data = raw_data.drop(raw_data.index[[i]]).reset_index(drop=True)
raw_data[773:775]
m = len(raw_data.Member_since)

# Create 3 vectors: day, month, year
day = np.zeros(m)
month = day
year = day

for i in range (m):
    s = raw_data.Member_since[i]
    r = re.split('\.', s)
    day[i] = r[0]
    month[i] = r[1]
    year[i] = r[2]
date=np.stack((np.transpose(day),np.transpose(month),np.transpose(year)), axis=1)
date=pd.DataFrame(date)
date=date.astype(int)
date.columns = ['Day', 'Month', 'Year']
date.head()
date.loc[date['Year'].idxmin()]
date.loc[date['Year'].idxmax()]
raw_data['Member_since'] = raw_data['Member_since'].str.replace('.', '-')

from datetime import datetime
raw_data['Member_since'] = pd.to_datetime(raw_data.Member_since, dayfirst=True)

# Subtracting 1/11/2009
raw_data['Member_since'] = raw_data['Member_since'] - pd.to_datetime('1/11/2009', dayfirst=True)
raw_data['Member_since'] = raw_data['Member_since'].astype('timedelta64[D]')
raw_data['Member_since'] = raw_data['Member_since'].astype(int) # Making it integer values

raw_data.head()
raw_data.rename(columns={'Time_spent_chating_H:M': 'Time_spent_chating'}, inplace=True)
raw_data.head()
m = len(raw_data.Time_spent_chating)

# Create 3 vectors: day, month, year
hours = np.zeros(m)
minutes = np.zeros(m)

for i in range (m):
    split = re.split('\:',raw_data.Time_spent_chating[i])
    hours[i] = int(split[0].replace(" ", ""))
    minutes[i] = int(split[1].replace(" ", ""))

t = hours+minutes/60
time = pd.DataFrame(t)
raw_data['Time_spent_chating'] = time
raw_data.head()
raw_data.Risk.astype(str).unique()
mapping = {'No_risk': 0, 'High_risk': 1, 'unknown_risk': 2}
raw_data = raw_data.replace({'Risk': mapping})
raw_data.Risk.astype(int).unique()
raw_data.columns
raw_data['Age'] = raw_data['Age'].str.replace(',', '.').astype(float)
raw_data['Points_Rank'] = raw_data['Points_Rank'].astype(float)
raw_data = raw_data[raw_data.Points_Rank.str.contains("a") == False].reset_index(drop=True)
raw_data['Points_Rank'] = raw_data['Points_Rank'].str.replace(" ", "").astype(float)
raw_data['Number_of_Comments_in_public_forum'] = raw_data['Number_of_Comments_in_public_forum'].str.replace(' ', '').astype(float)
raw_data['Number_of_advertisments_posted'] = raw_data['Number_of_advertisments_posted'].astype(float)
raw_data['Number_of_offline_meetings_attended'] = raw_data['Number_of_offline_meetings_attended'].astype(float)
raw_data['Profile_pictures'] = raw_data['Profile_pictures'].astype(float)
raw_data.isnull().values.any()
cols2norm = ['Age', 'Points_Rank', 'Last_login', 'Member_since', 'Number_of_Comments_in_public_forum',
       'Time_spent_chating', 'Number_of_advertisments_posted', 'Number_of_offline_meetings_attended', 'Profile_pictures']
raw_data[cols2norm] = raw_data[cols2norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
raw_data.head()
import tensorflow as tf
gender = tf.feature_column.categorical_column_with_hash_bucket('Gender', hash_bucket_size=10)
age = tf.feature_column.numeric_column('Age')
sex_or = tf.feature_column.categorical_column_with_hash_bucket('Sexual_orientation', hash_bucket_size=10)
rank = tf.feature_column.numeric_column('Points_Rank')
last_log = tf.feature_column.numeric_column('Last_login')
memb_since = tf.feature_column.numeric_column('Member_since')
comments = tf.feature_column.numeric_column('Number_of_Comments_in_public_forum')
time_spent = tf.feature_column.numeric_column('Time_spent_chating')
advertisements = tf.feature_column.numeric_column('Number_of_advertisments_posted')
meetings = tf.feature_column.numeric_column('Number_of_offline_meetings_attended')
pictures = tf.feature_column.numeric_column('Profile_pictures')
risk = tf.feature_column.categorical_column_with_hash_bucket('Risk', hash_bucket_size=5)
feat_cols = [gender, age, sex_or, rank, last_log, memb_since, comments, time_spent, advertisements, meetings, pictures]
features = raw_data.drop('Risk', axis=1)

labels = raw_data['Risk']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=101)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,
                                                 batch_size=10, num_epochs=1000, shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=3)
model.train(input_fn=input_func, steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = model.evaluate(eval_input_func)
results