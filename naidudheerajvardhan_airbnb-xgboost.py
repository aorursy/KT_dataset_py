# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD,PCA
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df_train = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip') # Training Set
df_test = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/test_users.csv.zip') # Testing Set
df_original = df_train.copy()
df_train.head() # Head of the dataframe
print("The Dimensions of the training set is " , df_train.shape)
print("The Dimensions of the testing set is ",df_test.shape)
labels = df_train['country_destination'].values

df_train.drop('country_destination',axis=1,inplace=True) # Droping Target Variable
# Concatenating training and testing sets for further use
df_all = pd.concat([df_train,df_test],axis=0,ignore_index=True) 
print("The Dimensions of Total Set is ",df_all.shape)
# Function that returns the missing percentage of each column in the dataset.
def missing_percentage(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = total/len(df)*100
    df = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
    return df.sort_values(by='Percent',ascending=False)
missing_percentage(df_all) # No need to store in a variable
# Droping 'date_first_booking' column as it is having roughly 67% missing data 
df_all.drop(['id','date_first_booking'],axis=1,inplace=True)
df_all.head()
df_all['date_dac'] = df_all['date_account_created'].apply(lambda x:x.split('-')[2])
df_all['date_dac'] = df_all['date_dac'].astype('int')

# We can also do this following like above "date_dac" column.
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'])
df_all['month_dac'] = df_all['date_account_created'].apply(lambda x:x.month)
df_all['year_dac'] = df_all['date_account_created'].apply(lambda x:x.year)

# No further use of 'date_account_created' column because we retrieved every information from it.
df_all.drop('date_account_created',axis=1,inplace=True)
# Retrieving year,month and date from timestamp.
tfa = np.vstack(df_all.timestamp_first_active.astype(str).
                apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_date'] = tfa[:,2]
df_all = df_all.drop('timestamp_first_active',axis=1)
# Function that returns the Categorical columns
def Categorical(df):
    categories = []
    for column in df.columns:
        if df[column].dtype =='object' or len(df[column].value_counts()) < 20:
            categories.append(column)
    return categories
Categories = Categorical(df_all)

# Some of the Categorical columns are int-type. So, Changed them to object.
df_all[Categories] = df_all[Categories].astype('object')

Categories
missing_percentage(df_all)
order1 = df_original['country_destination'].value_counts()
order2 = order1.index
plt.figure(figsize=(10,7))
sns.countplot(df_original['country_destination'],order=order2)
plt.xlabel('Country Destination')
plt.ylabel('Country Destination Count')
for i in range(order1.shape[0]):
    count = order1[i]
    strg = '{:0.2f}%'.format(100*count/df_all.shape[0])
    plt.text(i,count+1000,strg,ha='center')
sns.set_style('whitegrid')
plt.figure(figsize=(10,7))
sns.distplot(df_all['age'].dropna(),kde=False,bins=50)
plt.xlabel('Age')
plt.ylabel('Age Count')
plt.title('The Distribution of Age')
order1 = df_all['gender'].value_counts()
order2 = order1.index
plt.figure(figsize=(10,7))
sns.countplot(df_all['gender'],order=order2)
plt.xlabel('Gender')
for i in range(order1.shape[0]):
    count = order1[i]
    strg = '{:0.2f}%'.format(100*count/df_all.shape[0])
    plt.text(i,count+1000,strg,ha='center')
order1 = df_all['signup_method'].value_counts()
order2 = order1.index
plt.figure(figsize=(10,7))
sns.countplot(df_all['signup_method'],order=order2)
plt.xlabel('Signup Method')
for i in range(order1.shape[0]):
    count = order1[i]
    strg = '{:0.2f}%'.format(100*count/df_all.shape[0])
    plt.text(i,count+1000,strg,ha='center')
plt.figure(figsize=(12,7))
sns.countplot(df_original['country_destination'],hue=df_original['gender'])
# Filling 'first_affiliate_tracked' column using Random Forest Classifier

def completing_fa_tracked(df):
    age = df['age']
    df = df.drop('age',axis=1)
    y = df['first_affiliate_tracked']
    df.drop('first_affiliate_tracked',axis=1,inplace=True)
    
    cato = Categorical(df)
    onehot = pd.get_dummies(df[cato],drop_first = True)
    df = df.drop(cato,axis=1)
    df = pd.concat([df,onehot],axis=1)
    df = pd.concat([df,y],axis=1)
    
    temp_train = df.loc[df.first_affiliate_tracked.notnull()] 
    temp_test = df.loc[df.first_affiliate_tracked.isnull()]
    
    X_train= temp_train.drop('first_affiliate_tracked',axis=1)
    y_train = temp_train['first_affiliate_tracked']
    X_test = temp_test.drop('first_affiliate_tracked',axis=1)
    
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    
    pca = PCA(n_components=12)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    
    rfc = RandomForestClassifier(n_estimators=400, n_jobs=-1)
    rfc.fit(X_train_pca, y_train)
    predicted_fa = rfc.predict(X_test_pca)
    
    print("score:" ,rfc.score(X_train_pca,y_train))
    
    df.loc[df.first_affiliate_tracked.notnull(),"first_affiliate_tracked"] = y_train
    df.loc[df.first_affiliate_tracked.isnull(), "first_affiliate_tracked"] = predicted_fa
    df['age'] = age
    
    df['first_affiliate_tracked'] = df['first_affiliate_tracked'].astype('object')
    fa_t = pd.get_dummies(df['first_affiliate_tracked'],drop_first=True)
    df = df.drop('first_affiliate_tracked',axis=1)
    df = pd.concat([df,fa_t],axis=1)
    
    return df
df_all = completing_fa_tracked(df_all)
df_all.head()
missing_percentage(df_all)
# Filling 'age' column using Random Forest Regressor

def completing_age(df):
    
    temp_train = df.loc[df.age.notnull()] 
    temp_test = df.loc[df.age.isnull()]
    
    X_train= temp_train.drop('age',axis=1)
    y_train = temp_train['age']
    X_test = temp_test.drop('age',axis=1)

    pca = PCA(n_components=12)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Truncated SVD Can work efficiently with Sparse dataset. So it's better to use.
    
    t_svd = TruncatedSVD(n_components=12) 
    t_svd.fit(X_train)
    X_train_tsvd = t_svd.transform(X_train)
    X_test_tsvd = t_svd.transform(X_test)
    
    X_train_new = np.concatenate([X_train_pca,X_train_tsvd],axis=1)
    X_test_new = np.concatenate([X_test_pca,X_test_tsvd],axis=1)
    
    rfr = RandomForestRegressor(n_estimators=400, n_jobs=-1)
    rfr.fit(X_train_new, y_train)
    predicted_age = rfr.predict(X_test_new)
    
    print("score:" ,rfr.score(X_train_new,y_train))
    
    df.loc[df.age.notnull(),"age"] = y_train
    df.loc[df.age.isnull(), "age"] = predicted_age
    
    return df
df_all = completing_age(df_all)
missing_percentage(df_all)
df_all['age'] = df_all['age'].astype('int32') # Converting float to int.
df_all.head()
train = df_all.iloc[0:len(df_train),:]
test = df_all.iloc[len(df_train):,:]
le = LabelEncoder()
# Label Encoding the labels 

y = le.fit_transform(labels)
xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  
xgb.fit(train, y)
# Predictions using predict_proba to get the probabilities of classes

y_pred = xgb.predict_proba(test)
# Considering the 5 Classes with Highest Probabilities
id_test = df_test['id']
ids = []
cts = []
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()
# Submission 
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)
filename = 'Airbnb.pkl'

# Saving model using pickle
pickle.dump(xgb, open(filename, 'wb'))