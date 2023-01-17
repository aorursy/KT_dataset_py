import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import MinMaxScaler, Imputer
train = pd.read_csv('../input/home-credit-default-risk/application_train.csv', sep=',', header=0)

test = pd.read_csv('../input/home-credit-default-risk/application_test.csv', sep=',', header=0)

bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv', sep=',', header=0)

prev = pd.read_csv('../input/home-credit-default-risk/previous_application.csv', sep=',', header=0)



y_train = train['TARGET']



print('Train shape ',train.shape)

print('Test shape ',test.shape)

print('Bureau shape ',bureau.shape)

print('Previous application shape ',prev.shape)

train.head()
# cek sebaran target

train['TARGET'].astype(int).plot.hist()
# cek null

train.isnull().sum()
# dari bureau ambil jumlah pinjaman sebelumnya per id

tbureau = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'COUNT_PREV'})



train = train.join(tbureau.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')

test = test.join(tbureau.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')



print('Train shape',train.shape)

print('Test shape', test.shape)
# Group by sk id curr, hitung agg

bureau_agg = bureau.drop(columns=['SK_ID_BUREAU'])

bureau_agg = bureau_agg.groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()



columns = ['SK_ID_CURR']

for col in bureau_agg.columns.levels[0]:

    if col != 'SK_ID_CURR':

        for agg in bureau_agg.columns.levels[1][:-1]:

            columns.append('BUREAU_%s_%s' % (col, agg))

            

bureau_agg.columns = columns

bureau_agg.head()



train = train.join(bureau_agg.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')

test = test.join(bureau_agg.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')



print('Train shape',train.shape)

print('Test shape', test.shape)



train.head()
bureau_cat = pd.get_dummies(bureau.select_dtypes('object'))

bureau_cat['SK_ID_CURR'] = bureau['SK_ID_CURR']



bureau_cat = bureau_cat.groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean']).reset_index()



columns = ['SK_ID_CURR']

for col in bureau_cat.columns.levels[0]:

    if col != 'SK_ID_CURR':

        for agg in bureau_cat.columns.levels[1][:-1]:

            columns.append('BUREAU_%s_%s' % (col, agg))

            

bureau_cat.columns = columns



train = train.join(bureau_cat.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')

test = test.join(bureau_cat.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')



print('Train shape',train.shape)

print('Test shape', test.shape)



train.head()
# dari previous application ambil jumlah pinjaman sebelumnya per id

tprev = prev.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count().rename(columns = {'SK_ID_PREV': 'COUNT_PREVS'})



train = train.join(tprev.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')

test = test.join(tprev.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')



print('Train shape',train.shape)

print('Test shape', test.shape)
# Group by sk id curr, hitung agg

prev_agg = prev.drop(columns=['SK_ID_PREV'])

prev_agg = prev_agg.groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()



columns = ['SK_ID_CURR']

for col in prev_agg.columns.levels[0]:

    if col != 'SK_ID_CURR':

        for agg in prev_agg.columns.levels[1][:-1]:

            columns.append('PREV_%s_%s' % (col, agg))

            

prev_agg.columns = columns

prev_agg.head()



train = train.join(prev_agg.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')

test = test.join(prev_agg.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')



print('Train shape',train.shape)

print('Test shape', test.shape)



train.head()
prev_cat = pd.get_dummies(prev.select_dtypes('object'))

prev_cat['SK_ID_CURR'] = prev['SK_ID_CURR']



prev_cat = prev_cat.groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean']).reset_index()



columns = ['SK_ID_CURR']

for col in prev_cat.columns.levels[0]:

    if col != 'SK_ID_CURR':

        for agg in prev_cat.columns.levels[1][:-1]:

            columns.append('BUREAU_%s_%s' % (col, agg))

            

prev_cat.columns = columns



train = train.join(prev_cat.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')

test = test.join(prev_cat.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')



print('Train shape',train.shape)

print('Test shape', test.shape)



train.head()
train.dtypes.value_counts()
train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
train = pd.get_dummies(train)

test = pd.get_dummies(test)
#fitur tambahan (persentase)



train['CREDIT_INCOME_PERCENT'] = train['AMT_CREDIT'] / train['AMT_INCOME_TOTAL']

train['ANNUITY_INCOME_PERCENT'] = train['AMT_ANNUITY'] / train['AMT_INCOME_TOTAL']

train['CREDIT_TERM'] = train['AMT_ANNUITY'] / train['AMT_CREDIT']

train['DAYS_EMPLOYED_PERCENT'] = train['DAYS_EMPLOYED'] / train['DAYS_BIRTH']

train['INCOME_PER_PERSON'] = train['AMT_INCOME_TOTAL'] / train['CNT_FAM_MEMBERS']



test['CREDIT_INCOME_PERCENT'] = test['AMT_CREDIT'] / test['AMT_INCOME_TOTAL']

test['ANNUITY_INCOME_PERCENT'] = test['AMT_ANNUITY'] / test['AMT_INCOME_TOTAL']

test['CREDIT_TERM'] = test['AMT_ANNUITY'] / test['AMT_CREDIT']

test['DAYS_EMPLOYED_PERCENT'] = test['DAYS_EMPLOYED'] / test['DAYS_BIRTH']

test['INCOME_PER_PERSON'] = test['AMT_INCOME_TOTAL'] / test['CNT_FAM_MEMBERS']
print(train.shape)

print(test.shape)
# Match dataframe train and test

train, test = train.align(test, join = 'inner', axis = 1)

train['TARGET'] = y_train

train.columns
def checknull(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% Missing Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table = mis_val_table[

            mis_val_table.iloc[:,1] != 0].sort_values(

        '% Missing Values', ascending=False).round(1)

        

        # Return the dataframe with missing information

        return mis_val_table



checknull(train).head()
checknull(train)['% Missing Values'].astype(int).plot.hist()
missing_col = list(checknull(train).index[checknull(train)['% Missing Values'] > 50])

train = train.drop(columns=missing_col)

test = test.drop(columns=missing_col)



print('Remove %d columns with more than 30%% missing values' % len(missing_col))

print('Train shape', train.shape)

print('Test shape', test.shape)
corrs = []



for col in train.columns:

    if col != 'TARGET' and col != 'SK_ID_CURR':

        corr = train['TARGET'].corr(train[col])

        corrs.append((col,corr))



corrs = pd.DataFrame(corrs)

corrs = corrs.rename(columns = {0 : 'Columns', 1 : 'Correlation'})

corrs = corrs[

            corrs.iloc[:,1] != 0].sort_values(

        'Correlation', ascending=False)

corrs.set_index('Columns',inplace=True)



corrs.head()
corrs['Correlation'].plot.hist()
lowcorr_col = list(corrs.index[abs(corrs['Correlation']) < 0.01])



train = train.drop(columns=lowcorr_col)

test = test.drop(columns=lowcorr_col)



print('Remove %d columns with <0.01 correlation to target' % len(lowcorr_col))

print('Train shape', train.shape)

print('Test shape', test.shape)
train = train.drop(columns=['TARGET'])
imputer = Imputer(strategy='median')

imputer.fit(train)



ntrain = imputer.transform(train)

ntest = imputer.transform(test)
scaler = MinMaxScaler(feature_range=(0,1))

scaler.fit(ntrain)



ntrain = scaler.transform(ntrain)

ntest = scaler.transform(ntest)
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(C=1)

logreg.fit(ntrain,y_train)
pred = logreg.predict_proba(ntest)[:,1]
submit = test[['SK_ID_CURR']]

submit['TARGET']=pred



submit
submit.to_csv('submission.csv',sep=',',index=False)