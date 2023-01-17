import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/avguided-community-hackathon2020term-deposit/Train_eP48B9k.csv')

test = pd.read_csv('../input/avguided-community-hackathon2020term-deposit/Test_jPKyvmK.csv')
train.head()
test.head()
id_col, target_col = 'id', 'term_deposit_subscribed'
print('Train contains',train.shape[0],'samples and ',train.shape[1],'variables')

print('Test contains',test.shape[0],'samples and ',test.shape[1],'variables')



features = [c for c in train.columns if c not in [id_col, target_col]]

print('There are',len(features),'number of features')
#Normalize the data to get ratio instead of raw count

train[target_col].value_counts(normalize=True)
sns.countplot(train[target_col])

plt.title('Target distribution')

plt.show()
train.info()
null_value_percentage = (train.isnull().sum()/train.shape[0])*100

null_value_percentage.sort_values(ascending = False)
train.nunique()
#looping through the columns

#check if datatype is object('O')

#if yes add to list

cat_cols = [train.columns[i] 

            for i in range(1, train.shape[1]-1)  

            if train.iloc[:,i].dtype=='O']

cat_cols
num_cols = [c for c in features if c not in cat_cols]

num_cols
fig, axes = plt.subplots(5, 2, figsize=(18,30))

axes = [ax for axes_rows in axes for ax in axes_rows]



for i, c in enumerate(train[cat_cols]):

    train[c].value_counts()[::-1].plot(kind='pie',

                                          ax=axes[i],

                                          title=c,

                                          autopct='%.0f%%',

                                          fontsize=12)

    axes[i].set_ylabel('')
fig, axes = plt.subplots(3, 3, figsize=(20,16))

axes = [ax for axes_rows in axes for ax in axes_rows]



for i, c in enumerate(train[cat_cols]):

    train[c].value_counts()[::-1].plot(kind='barh',

                                          ax=axes[i],

                                          title=c,

                                          fontsize=12)
fig, axes = plt.subplots(5, 2, figsize=(16,24))

axes = [ax for axes_rows in axes for ax in axes_rows]



for i, c in enumerate(train[cat_cols]):

    #index of rows where target_col value is 0

    fltr = train[target_col]==0

    

    #dataframe conraining rows and columns where target_col value is 0

    #fltr-index of rows where target_col value is 0

    #c-column name

    #taking the value count

    #resetting index as column name

    vc_a=train[fltr][c].value_counts(normalize=True).reset_index().rename({'index':c,c:'count'}, axis=1)

    

    #dataframe conraining rows and columns where target_col value is 1

    vc_b=train[~fltr][c].value_counts(normalize=True).reset_index().rename({'index':c,c:'count'}, axis=1)

    

    #setting target_col value to 0 and 1 respectively

    vc_a[target_col]=0

    vc_b[target_col]=1

    

    #combining into single dataframe

    df = pd.concat([vc_a, vc_b]).reset_index(drop=True)

    

    #plotting

    sns.barplot(y=c, x='count', data=df, hue=target_col, ax=axes[i])

    
vc_a
vc_b
df
fig, axes = plt.subplots(7,1,figsize=(8,20))

for i,c in enumerate(train[num_cols]):

    train[[c]].boxplot(ax=axes[i], vert=False)
fig, axes = plt.subplots(4, 2, figsize=(18,14))

axes = [ax for axes_rows in axes for ax in axes_rows]



for i, c in enumerate(num_cols):

    plot = train[[c]].plot(kind='kde', ax=axes[i])
sns.set(font_scale=1.3)

fig, axes = plt.subplots(4, 2, figsize=(18, 20))

axes = [ax for axes_row in axes for ax in axes_row]



for i, c in enumerate(num_cols):

    train.groupby(target_col)[c].median().plot(kind = 'barh', title=f'Median_{c}', ax=axes[i])
#create a new column called is_old and fill with true

train['is_old'] = True



#in each row see of age is less 50

#if yes make old_age value as False fo that row

train.loc[train['customer_age'] <= 50, 'is_old'] = False



#group by old_age and plot the count

_ = train.groupby('is_old')[target_col].mean().sort_values().plot(kind = 'barh', title='Probability of subscribing to a term deposit')
#old_age column is no longer needed

train=train.drop(['is_old'],axis=1)
plt.figure(figsize=(14, 8))

_ = sns.heatmap(train[num_cols].corr(), annot=True)