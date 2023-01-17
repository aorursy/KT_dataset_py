import re

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

import seaborn as sns

from tqdm.notebook import tqdm

sns.set_style('darkgrid')

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
train = pd.read_csv('/kaggle/input/Data_Train.csv')

test = pd.read_csv('/kaggle/input/Data_Test.csv')

sub = pd.read_csv('/kaggle/input/Sample_Submission.csv')
train.shape, test.shape, sub.shape
train.duplicated().sum(), test.duplicated().sum()
train.head(2)
train.info()
train.isnull().sum()
train.nunique()
test.nunique()
train['Timestamp'] = pd.to_datetime(train['Timestamp'])

test['Timestamp'] = pd.to_datetime(test['Timestamp'])



train = train.sort_values('Timestamp').reset_index(drop = True)

test = test.sort_values('Timestamp').reset_index(drop = True)
df = train.append(test, ignore_index=True, sort=False)

df.shape
df.info()
df.head(2)
df['Other_artist'] = df['Song_Name'].str.count('feat|Feat|FEAT')
df['Year'] = pd.to_datetime(df['Timestamp']).dt.year

df['Month'] = pd.to_datetime(df['Timestamp']).dt.month

df['Day'] = pd.to_datetime(df['Timestamp']).dt.day

df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour

df['Minutes'] = pd.to_datetime(df['Timestamp']).dt.minute

df['Seconds'] = pd.to_datetime(df['Timestamp']).dt.second

df['Dayofweek'] = pd.to_datetime(df['Timestamp']).dt.dayofweek

df['DayOfyear'] = pd.to_datetime(df['Timestamp']).dt.dayofyear

df['WeekOfyear'] = pd.to_datetime(df['Timestamp']).dt.weekofyear
df['Likes'] = df['Likes'].str.replace(',','')

df['Likes'] = df['Likes'].replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval) 
df['Popularity'] = df['Popularity'].str.replace(',','')

df['Popularity'] = df['Popularity'].replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval) 
df.loc[df['Comments'] == 0, 'Comments'] = df['Comments'].mean()

df.loc[df['Likes'] == 0, 'Likes'] = df['Likes'].mean()

df.loc[df['Popularity'] == 0, 'Popularity'] = df['Popularity'].mean()
agg_func = {

    'Comments': ['sum'],

    'Likes': ['sum'],

    'Popularity': ['sum'],

    'Followers': ['sum']

}

agg_name = df.groupby(['Year','Name']).agg(agg_func)

agg_name.columns = [ 'YN_' + ('_'.join(col).strip()) for col in agg_name.columns.values]

agg_name.reset_index(inplace=True)

df = df.merge(agg_name, on=['Year','Name'], how='left')

del agg_name
agg_func = {

    'Comments': ['mean','min','max','sum','median'],

    'Likes': ['mean','min','max','sum','median'],

    'Popularity': ['mean','min','max','sum','median'],

    'Followers': ['mean','sum']

}

agg_name = df.groupby('Name').agg(agg_func)

agg_name.columns = [ 'Name_' + ('_'.join(col).strip()) for col in agg_name.columns.values]

agg_name.reset_index(inplace=True)

df = df.merge(agg_name, on=['Name'], how='left')

del agg_name
df['Followers / Popularity'] = df['Followers'] / df['Popularity']

df['Followers / Comments'] = df['Followers'] / df['Comments']

df['Followers / Likes'] = df['Followers'] / df['Likes']



df['Popularity / Followers'] = df['Popularity'] / df['Followers']

df['Popularity / Comments'] = df['Popularity'] / df['Comments']

df['Popularity / Likes'] = df['Popularity'] / df['Likes']



df['Likes / Followers'] = df['Likes'] / df['Followers']

df['Likes / Popularity'] = df['Likes'] / df['Popularity']

df['Likes / Comments'] = df['Likes'] / df['Comments']



df['Comments / Followers'] = df['Comments'] / df['Followers']

df['Comments / Popularity'] = df['Comments'] / df['Popularity']

df['Comments / Comments'] = df['Comments'] / df['Likes']
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Name'] = le.fit_transform(df['Name'])
df = pd.get_dummies(df, columns=['Genre'], drop_first=True)
df.drop(['Country','Song_Name','Timestamp'], axis=1, inplace=True)
train_df = df[df['Views'].isnull()!=True]

test_df = df[df['Views'].isnull()==True]

test_df.drop('Views', axis=1, inplace=True)
train_df = train_df.replace([np.inf, -np.inf], np.nan)

train_df = train_df.fillna(0)



test_df = test_df.replace([np.inf, -np.inf], np.nan)

test_df = test_df.fillna(0)
train_df.shape, test_df.shape
X = train_df.drop(labels=['Views'], axis=1)

y = train_df['Views'].values



from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.shape, y_train.shape, X_cv.shape, y_cv.shape
X_train.tail(2)
from math import sqrt 

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(verbose=1, learning_rate=0.2, n_estimators=1500, random_state=42, subsample=0.8)

gb.fit(X_train, y_train)

y_pred = gb.predict(X_cv)

print('RMSE', sqrt(mean_squared_error(y_cv, y_pred)))
feature_imp = pd.DataFrame(sorted(zip(gb.feature_importances_, X.columns), reverse=True)[:60], columns=['Value','Feature'])

plt.figure(figsize=(12,10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('Gradient Boosting Features')

plt.tight_layout()

plt.show()
Xtest = test_df
from sklearn.model_selection import KFold



errgb = []

y_pred_totgb = []



fold = KFold(n_splits=20, shuffle=True, random_state=101)



for train_index, test_index in fold.split(X):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

             

    gb = GradientBoostingRegressor(learning_rate=0.2, n_estimators=1500, random_state=42, subsample=0.8)

    gb.fit(X_train, y_train)

    y_pred = gb.predict(X_test)



    print('RMSE', sqrt(mean_squared_error(y_test, y_pred)))



    errgb.append(sqrt(mean_squared_error(y_test, y_pred)))

    p = gb.predict(Xtest)

    y_pred_totgb.append(p)
np.mean(errgb) 
final = np.mean(y_pred_totgb,0).round()

final
for i in range(20):

    sub = pd.DataFrame({'Unique_ID':test['Unique_ID'],'Views': y_pred_totgb[i]})

    sub.to_excel('fold_'+str(i)+'_Output.xlsx', index=False)
sub = pd.DataFrame({'Unique_ID':test['Unique_ID'],'Views': final})

sub.to_excel('Output.xlsx', index=False)

sub.head()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(sub)