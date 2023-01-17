import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/train.csv')
print(train.shape)
train.head()
test = pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/test.csv')
print(test.shape)
test.head()
train.isnull().mean()
test.isnull().mean()
print(train['condition'].unique())
print(test['condition'].unique())
train.fillna(999,inplace = True)
# doing the same for test data
test.fillna(999,inplace = True)
train.groupby('condition')['breed_category'].value_counts()
train.groupby('condition')['pet_category'].value_counts()
train.head()
train['pet_id'].nunique()
train['id_nums'] = train['pet_id'].map(lambda x:x[5:7])
test['id_nums'] = test['pet_id'].map(lambda x:x[5:7])
train.groupby('id_nums')['breed_category'].value_counts()
train.groupby('id_nums')['pet_category'].value_counts()
train.dtypes
train['issue_date'] = train['issue_date'].map(lambda x:x[:11])
train['listing_date'] = train['listing_date'].map(lambda x:x[:11])
test['issue_date'] = test['issue_date'].map(lambda x:x[:11])
test['listing_date'] = test['listing_date'].map(lambda x:x[:11])
train.head()
train['issue_year'] = train['issue_date'].map(lambda x:x[:4])
train['listing_year'] = train['listing_date'].map(lambda x:x[:4])
test['issue_year'] = test['issue_date'].map(lambda x:x[:4])
test['listing_year'] = test['listing_date'].map(lambda x:x[:4])
train.head()
# lets see how the target depends on year
train.groupby('issue_year')['breed_category'].value_counts()
train.groupby('listing_year')['breed_category'].value_counts().plot.bar()
train['issue_month'] = train['issue_date'].map(lambda x:x[5:7])
train['listing_month'] = train['listing_date'].map(lambda x:x[5:7])
test['issue_month'] = test['issue_date'].map(lambda x:x[5:7])
test['listing_month'] = test['listing_date'].map(lambda x:x[5:7])
train.head()
train.groupby('listing_month')['breed_category'].value_counts()
train.groupby('listing_month')['pet_category'].value_counts()
train.dtypes
# lets change the datatypes of the newly created cols to int
train['id_nums'] = train['id_nums'].astype('int')
train['issue_year'] = train['issue_year'].astype('int')
train['issue_month'] = train['issue_month'].astype('int')
train['listing_month'] = train['listing_month'].astype('int')
train['listing_year'] = train['listing_year'].astype('int')
test['id_nums'] = test['id_nums'].astype('int')
test['issue_year'] = test['issue_year'].astype('int')
test['issue_month'] = test['issue_month'].astype('int')
test['listing_month'] =test['listing_month'].astype('int')
test['listing_year'] = test['listing_year'].astype('int')
train.dtypes
#now lets see whether the target variables are realated to the amount of years it has between issue and listing
train['difference_years'] = train['listing_year'] - train['issue_year']
train.groupby('difference_years')['breed_category'].value_counts()
test['difference_years'] = test['listing_year'] - test['issue_year']

train.groupby('difference_years')['pet_category'].value_counts()
# drop the isuue data and listing data col now
train.drop(['issue_date','listing_date'],axis = 1,inplace = True)
test.drop(['issue_date','listing_date'],axis = 1,inplace = True)
train.head()
train['color_type'].nunique()
# there are 56 different types of colors
train['color_type'].unique()
train['color_tabby'] = train['color_type'].map(lambda x: 1 if ('Tabby' in x) else 0)
#fig,(ax1,ax2) = plt.subplots(1,2,figsize = (16,4))
print(train.groupby('color_tabby')['breed_category'].value_counts())
print(train.groupby('color_tabby')['pet_category'].value_counts()) # it gives much information about pet_category
test['color_tabby'] = test['color_type'].map(lambda x: 1 if ('Tabby' in x) else 0)

for specific_word in ['Brindle','Tick','Point','Cream','Merle','Tiger','Smoke']:
    train['color ' + str(specific_word)] = train['color_type'].map(lambda x: 1 if (specific_word in x) else 0)
    print(specific_word)
    print(train.groupby('color '+ str(specific_word))['pet_category'].value_counts()) # it gives much information about pet_category

    
for specific_word in ['Brindle','Tick','Point','Cream','Merle','Tiger','Smoke']:
    test['color ' + str(specific_word)] = test['color_type'].map(lambda x: 1 if (specific_word in x) else 0)
#     print(specific_word)
#     print(train.groupby('color '+ str(specific_word))['pet_category'].value_counts()) # it gives much information about pet_category

    
train['color_cat_1'] = train['color_tabby'] + train['color Point'] + train['color Smoke']
train['color_cat_1'].value_counts()
test['color_cat_1'] = test['color_tabby'] + test['color Point'] + test['color Smoke']
# train['color_cat_1'].value_counts()
train['color_cat_2'] = train['color Brindle'] + train['color Tick'] + train['color Point']
print(train['color_cat_2'].value_counts())
test['color_cat_2'] = test['color Brindle'] + test['color Tick'] + test['color Point']

train.drop(['color_tabby', 'color Brindle', 'color Tick',
       'color Point', 'color Cream', 'color Merle', 'color Tiger',
       'color Smoke'],axis = 1,inplace = True)
test.drop(['color_tabby', 'color Brindle', 'color Tick',
       'color Point', 'color Cream', 'color Merle', 'color Tiger',
       'color Smoke'],axis = 1,inplace = True)
print(train.shape)
train.head()
#lets look at test data
test.head()
plt.figure(figsize = (15,6))
temp_df = pd.Series(train['color_type'].value_counts() / len(train) )
temp_df.sort_values(ascending=False).plot.bar()
# for encoding it, i am going to first use rare category for the cols that 
def find_non_rare_labels(df, variable, tolerance):
    
    temp = df.groupby([variable])[variable].count() / len(df)
    
    non_rare = [x for x in temp.loc[temp>tolerance].index.values]
    
    return non_rare
frequent_cols = find_non_rare_labels(train,'color_type',0.02)
frequent_cols
#encoding the variables
train['color_encoded'] = np.where(train['color_type'].isin(frequent_cols),train['color_type'],'Rare')
test['color_encoded'] = np.where(test['color_type'].isin(frequent_cols),test['color_type'],'Rare')
train.head()
train['color_encoded'].value_counts()
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Myencoder(BaseEstimator, TransformerMixin):
   
    def __init__(self,drop = 'first',sparse=False):
        self.encoder = OneHotEncoder(drop = drop,sparse = sparse)
        self.drop = True if drop == 'first' else False
        self.features_to_encode = []
        self.columns = []
    
    def fit(self,X_train,features_to_encode):
        
        data = X_train.copy()
        self.features_to_encode = features_to_encode
        data_to_encode = data[self.features_to_encode]
        self.columns = pd.get_dummies(data_to_encode,drop_first = self.drop).columns
        self.encoder.fit(data_to_encode)
        return self.encoder
    
    def transform(self,X_test):
        
        data = X_test.copy()
        data.reset_index(drop = True,inplace =True)
        data_to_encode = data[self.features_to_encode]
        data_left = data.drop(self.features_to_encode,axis = 1)
        
        data_encoded = pd.DataFrame(self.encoder.transform(data_to_encode),columns = self.columns)
        
        return pd.concat([data_left,data_encoded],axis = 1)
my_encoder = Myencoder(drop = None)
my_encoder.fit(train,['color_encoded'])
len(my_encoder.columns)
train = my_encoder.transform(train)
test = my_encoder.transform(test)
train.head()
train['length(cm)'] = train['length(m)']*100
train.drop('length(m)',axis = 1,inplace = True)
test['length(cm)'] = test['length(m)']*100
test.drop('length(m)',axis = 1,inplace = True)
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize = (16,16))
sns.violinplot(train['breed_category'],train['length(cm)'],ax = ax1)
sns.violinplot(train['pet_category'],train['length(cm)'],ax = ax2)
sns.violinplot(train['breed_category'],train['height(cm)'],ax = ax3)
sns.violinplot(train['pet_category'],train['height(cm)'],ax = ax4)
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (16,6))
sns.violinplot(train['breed_category'],(train['length(cm)']*train['height(cm)']) / 100,ax = ax1)
sns.violinplot(train['pet_category'],(train['length(cm)']*train['height(cm)'] / 100),ax = ax2)
#there are some obs for which value of length is 0
temp = train[train['length(cm)'] == 0]
temp.shape
#lets see this data for which len if 0
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (14,4))
train['breed_category'].value_counts().plot.bar(ax = ax1)
train['pet_category'].value_counts().plot.bar(ax = ax2)
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize = (16,16))
sns.violinplot(train['breed_category'],train['X1'],ax = ax1)
sns.violinplot(train['pet_category'],train['X1'],ax = ax2)
sns.violinplot(train['breed_category'],train['X2'],ax = ax3)
sns.violinplot(train['pet_category'],train['X2'],ax = ax4)
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (16,6))
sns.violinplot(train['breed_category'],(train['X1']*train['X2']),ax = ax1)
sns.violinplot(train['pet_category'],(train['X1']*train['X2'] ),ax = ax2)
train['lh'] = (train['length(cm)']*train['height(cm)']) / 100
train['X1_X2'] = train['X1']*train['X2']
train['l_zero'] = np.where(train['length(cm)'] == 0,1,0) # where length == 0
test['lh'] = (test['length(cm)']*test['height(cm)']) / 100
test['X1_X2'] = test['X1']*test['X2']
test['l_zero'] = np.where(test['length(cm)'] == 0,1,0)
train.shape
train.head()
#drop some unnecessay features
train.drop('color_type',axis = 1,inplace = True)
test.drop('color_type',axis = 1,inplace = True)
print(len(train.columns))
train.columns
print(len(test.columns))
test.columns
train.to_csv('train_processed.csv')
test.to_csv('test_processed.csv')
