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
train = pd.read_csv('/kaggle/input/course-material-walmart-challenge/train.csv')

test = pd.read_csv('/kaggle/input/course-material-walmart-challenge/test.csv')

sample_submission = pd.read_csv('/kaggle/input/course-material-walmart-challenge/sample_submission.csv')
print('Train-----------------------------------------------')

print(train.head())

print('Test------------------------------------------------')

print(test.head())

print('SampleSubmission------------------------------------')

print(sample_submission.head())
train.shape,test.shape
train.head(5)
train.describe(include='all')
def levels(df):

    return (pd.DataFrame({'dtype':df.dtypes, 

                         'levels':df.nunique(), 

                         'levels':[df[x].unique() for x in df.columns],

                         'null_values':df.isna().sum(),

                         'unique':df.nunique()}))

levels(train)
train.isnull().sum()
print(train['MarkDown1'].isnull().sum()/train.shape[0])

print(train['MarkDown2'].isnull().sum()/train.shape[0])

print(train['MarkDown3'].isnull().sum()/train.shape[0])

print(train['MarkDown4'].isnull().sum()/train.shape[0])

print(train['MarkDown5'].isnull().sum()/train.shape[0])
train[(train['Store']==1)&(train['Dept']==1)].sort_values(['Dept','Date'],ascending=True).head(10)
train['MarkDown1'][(train['Store']==3)&(train['Dept']==2)].mean()
train['MarkDown1'][(train['Store']==3)&(train['Dept']==2)].median()
store=train['Store'].unique()

dept=train['Dept'].unique()
tstore=test['Store'].unique()

tdept=test['Dept'].unique()
store
dept
train['key'] = train['Store'].astype(str)+'_'+train['Dept'].astype(str)+'_'+train['Date'].astype(str)

test['key'] = test['Store'].astype(str)+'_'+test['Dept'].astype(str)+'_'+test['Date'].astype(str)
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])
train['WeekNo.'] = train['Date'].dt.strftime('%U')

test['WeekNo.'] = test['Date'].dt.strftime('%U')
import warnings

warnings.filterwarnings("ignore")
train['Weekly_Sales'][(train['Store']==1)&(train['Dept']==1)&(train['Date']=='2010-02-12')]
%%time

train['median'] = pd.Series()

test['median'] = pd.Series()

train['std'] = pd.Series()

test['std'] = pd.Series()



for i in store:

    for j in dept:

        train['median'][(train['Store']==i)&(train['Dept']==j)] = train['Weekly_Sales'][(train['Store']==i)&(train['Dept']==j)].median()

        test['median'][(test['Store']==i)&(test['Dept']==j)] = train['Weekly_Sales'][(train['Store']==i)&(train['Dept']==j)].median()

        train['std'][(train['Store']==i)&(train['Dept']==j)] = train['Weekly_Sales'][(train['Store']==i)&(train['Dept']==j)].std()

        test['std'][(test['Store']==i)&(test['Dept']==j)] = train['Weekly_Sales'][(train['Store']==i)&(train['Dept']==j)].std()
for i in store:

    train['median'][(train['Store']==i)&(train['median'].isnull())] = train['Weekly_Sales'][(train['Store']==i)&(train['Weekly_Sales']<=300)].median()

    test['median'][(test['Store']==i)&(test['median'].isnull())] = train['Weekly_Sales'][(train['Store']==i)&(train['Weekly_Sales']<=300)].median()

    train['std'][(train['Store']==i)&(train['std'].isnull())] = train['Weekly_Sales'][(train['Store']==i)&(train['Weekly_Sales']<=300)].std()

    test['std'][(test['Store']==i)&(test['std'].isnull())] = train['Weekly_Sales'][(train['Store']==i)&(train['Weekly_Sales']<=300)].std()
train.head(5)
%%time

train['roll_mean'] = (train.groupby(['Store','Dept','Date'],sort=True)['Weekly_Sales']

                        .rolling(3, min_periods=1).mean()

                        .reset_index(drop=True))
train.head()
train.isnull().sum()
train.groupby(['Store','Dept','WeekNo.']).agg({'roll_mean':'mean'}).reset_index()
roll_mean = train.groupby(['Store','Dept','WeekNo.']).agg({'roll_mean':'mean'}).reset_index()
test = pd.merge(test,roll_mean, left_on = ['Store','Dept','WeekNo.'], right_on=['Store','Dept','WeekNo.'],how='left')
test.head()
test[test['roll_mean'].isnull()].shape
test[test['roll_mean'].notnull()].shape
test1 = test[test['roll_mean'].isnull()]

test2 = test[test['roll_mean'].notnull()]

test1.drop('roll_mean',axis=1,inplace=True)
roll_mean1 = train.groupby(['Store','WeekNo.']).agg({'roll_mean':'mean'}).reset_index()
roll_mean1.head()
test1 = pd.merge(test1,roll_mean1, left_on = ['Store','WeekNo.'], right_on=['Store','WeekNo.'],how='left')
test_n = test2.append(test1).sort_index(axis=0)
test_n.head(2)
test.head(2)
test_n.shape,test.shape
%%time

for i in store:

    for j in dept:

        train['MarkDown1'][(train['MarkDown1'].isnull())&

                           (train['Store']==i)&

                           (train['Dept']==j)] = train['MarkDown1'][(train['Store']==i)&

                                                                    (train['Dept']==j)].median()

        train['MarkDown2'][(train['MarkDown2'].isnull())&

                           (train['Store']==i)&

                           (train['Dept']==j)] = train['MarkDown2'][(train['Store']==i)&

                                                                    (train['Dept']==j)].median()

        train['MarkDown3'][(train['MarkDown3'].isnull())&

                           (train['Store']==i)&

                           (train['Dept']==j)] = train['MarkDown3'][(train['Store']==i)&

                                                                    (train['Dept']==j)].median()

        train['MarkDown4'][(train['MarkDown4'].isnull())&

                           (train['Store']==i)&

                           (train['Dept']==j)] = train['MarkDown4'][(train['Store']==i)&

                                                                    (train['Dept']==j)].median()

        train['MarkDown5'][(train['MarkDown5'].isnull())&

                           (train['Store']==i)&

                           (train['Dept']==j)] = train['MarkDown5'][(train['Store']==i)&

                                                                    (train['Dept']==j)].median()

        test_n['MarkDown1'][(test_n['MarkDown1'].isnull())&

                           (test_n['Store']==i)&

                           (test_n['Dept']==j)] = train['MarkDown1'][(train['Store']==i)&

                                                                    (train['Dept']==j)].median()

        test_n['MarkDown2'][(test_n['MarkDown2'].isnull())&

                           (test_n['Store']==i)&

                           (test_n['Dept']==j)] = train['MarkDown2'][(train['Store']==i)&

                                                                    (train['Dept']==j)].median()

        test_n['MarkDown3'][(test_n['MarkDown3'].isnull())&

                           (test_n['Store']==i)&

                           (test_n['Dept']==j)] = train['MarkDown3'][(train['Store']==i)&

                                                                    (train['Dept']==j)].median()

        test_n['MarkDown4'][(test_n['MarkDown4'].isnull())&

                           (test_n['Store']==i)&

                           (test_n['Dept']==j)] = train['MarkDown4'][(train['Store']==i)&

                                                                    (train['Dept']==j)].median()

        test_n['MarkDown5'][(test_n['MarkDown5'].isnull())&

                           (test_n['Store']==i)&

                           (test_n['Dept']==j)] = train['MarkDown5'][(train['Store']==i)&

                                                                    (train['Dept']==j)].median()
for i in store:

    train['MarkDown1'][(train['MarkDown1'].isnull())&

                           (train['Store']==i)] = train['MarkDown1'][(train['Store']==i)&

                                                                    (train['Weekly_Sales']<=300)].median()

    train['MarkDown2'][(train['MarkDown2'].isnull())&

                           (train['Store']==i)] = train['MarkDown2'][(train['Store']==i)&

                                                                    (train['Weekly_Sales']<=300)].median()

    train['MarkDown3'][(train['MarkDown3'].isnull())&

                           (train['Store']==i)] = train['MarkDown3'][(train['Store']==i)&

                                                                    (train['Weekly_Sales']<=300)].median()

    train['MarkDown4'][(train['MarkDown4'].isnull())&

                           (train['Store']==i)] = train['MarkDown4'][(train['Store']==i)&

                                                                    (train['Weekly_Sales']<=300)].median()

    train['MarkDown5'][(train['MarkDown5'].isnull())&

                           (train['Store']==i)] = train['MarkDown5'][(train['Store']==i)&

                                                                    (train['Weekly_Sales']<=300)].median()

    test_n['MarkDown1'][(test_n['MarkDown1'].isnull())&

                           (test_n['Store']==i)] = train['MarkDown1'][(train['Store']==i)&

                                                                    (train['Weekly_Sales']<=300)].median()

    test_n['MarkDown2'][(test_n['MarkDown2'].isnull())&

                           (test_n['Store']==i)] = train['MarkDown2'][(train['Store']==i)&

                                                                    (train['Weekly_Sales']<=300)].median()

    test_n['MarkDown3'][(test_n['MarkDown3'].isnull())&

                           (test_n['Store']==i)] = train['MarkDown3'][(train['Store']==i)&

                                                                    (train['Weekly_Sales']<=300)].median()

    test_n['MarkDown4'][(test_n['MarkDown4'].isnull())&

                           (test_n['Store']==i)] = train['MarkDown4'][(train['Store']==i)&

                                                                    (train['Weekly_Sales']<=300)].median()

    test_n['MarkDown5'][(test_n['MarkDown5'].isnull())&

                           (test_n['Store']==i)] = train['MarkDown5'][(train['Store']==i)&

                                                                    (train['Weekly_Sales']<=300)].median()
test_n.isnull().sum()
import matplotlib.pylab as plt

import seaborn as sns

sns.boxplot(train['MarkDown1'],orient='v')
sns.boxplot(train['MarkDown2'],orient='v')
sns.boxplot(train['MarkDown3'],orient='v')
sns.boxplot(train['MarkDown4'],orient='v')
sns.boxplot(train['MarkDown5'],orient='v')
p99 = np.nanpercentile(train['MarkDown1'][(train['Store']==3)&(train['Dept']==2)],99)

print(p99)

p01 = np.nanpercentile(train['MarkDown1'][(train['Store']==3)&(train['Dept']==2)],1)

print(p01)

print(train[(train['Store']==3)&(train['Dept']==2)&(train['MarkDown1']>=p99)].shape)

print(train[(train['Store']==3)&(train['Dept']==2)&(train['MarkDown1']<=p01)].shape)
for i in store:

    md1_p99 = np.percentile(train['MarkDown1'][(train['Store']==i)],99)

    md1_p01 = np.percentile(train['MarkDown1'][(train['Store']==i)],1)

    md2_p99 = np.percentile(train['MarkDown2'][(train['Store']==i)],99)

    md2_p01 = np.percentile(train['MarkDown2'][(train['Store']==i)],1)

    md3_p99 = np.percentile(train['MarkDown3'][(train['Store']==i)],99)

    md3_p01 = np.percentile(train['MarkDown3'][(train['Store']==i)],1)

    md4_p99 = np.percentile(train['MarkDown4'][(train['Store']==i)],99)

    md4_p01 = np.percentile(train['MarkDown4'][(train['Store']==i)],1)

    md5_p99 = np.percentile(train['MarkDown5'][(train['Store']==i)],99)

    md5_p01 = np.percentile(train['MarkDown5'][(train['Store']==i)],1)



    train['MarkDown1'][(train['MarkDown1']>=md1_p99)&(train['Store']==i)] = md1_p99

    train['MarkDown1'][(train['MarkDown1']<=md1_p01)&(train['Store']==i)] = md1_p01



    train['MarkDown2'][(train['MarkDown2']>=md2_p99)&(train['Store']==i)] = md2_p99

    train['MarkDown2'][(train['MarkDown2']<=md2_p01)&(train['Store']==i)] = md2_p01



    train['MarkDown3'][(train['MarkDown3']>=md3_p99)&(train['Store']==i)] = md3_p99

    train['MarkDown3'][(train['MarkDown3']<=md3_p01)&(train['Store']==i)] = md3_p01



    train['MarkDown4'][(train['MarkDown4']>=md4_p99)&(train['Store']==i)] = md4_p99

    train['MarkDown4'][(train['MarkDown4']<=md4_p01)&(train['Store']==i)] = md4_p01



    train['MarkDown5'][(train['MarkDown5']>=md5_p99)&(train['Store']==i)] = md5_p99

    train['MarkDown5'][(train['MarkDown5']<=md5_p01)&(train['Store']==i)] = md5_p01

    

    test_n['MarkDown1'][(test_n['MarkDown1']>=md1_p99)&(test_n['Store']==i)] = md1_p99

    test_n['MarkDown1'][(test_n['MarkDown1']<=md1_p01)&(test_n['Store']==i)] = md1_p01

    

    test_n['MarkDown2'][(test_n['MarkDown2']>=md2_p99)&(test_n['Store']==i)] = md2_p99

    test_n['MarkDown2'][(test_n['MarkDown2']<=md2_p01)&(test_n['Store']==i)] = md2_p01

    

    test_n['MarkDown3'][(test_n['MarkDown3']>=md3_p99)&(test_n['Store']==i)] = md3_p99

    test_n['MarkDown3'][(test_n['MarkDown3']<=md3_p01)&(test_n['Store']==i)] = md3_p01

    

    test_n['MarkDown4'][(test_n['MarkDown4']>=md4_p99)&(test_n['Store']==i)] = md4_p99

    test_n['MarkDown4'][(test_n['MarkDown4']<=md4_p01)&(test_n['Store']==i)] = md4_p01

    

    test_n['MarkDown5'][(test_n['MarkDown5']>=md5_p99)&(test_n['Store']==i)] = md5_p99

    test_n['MarkDown5'][(test_n['MarkDown5']<=md5_p01)&(test_n['Store']==i)] = md5_p01
train['cel_week']=pd.Series()

test_n['cel_week'] = pd.Series()

train['cel_week'][(train['WeekNo.']==48)|(train['WeekNo.']==52)] = 1

train['cel_week'][(train['WeekNo.']!=48)&(train['WeekNo.']!=52)] = 0

test_n['cel_week'][(test_n['WeekNo.']==48)|(test_n['WeekNo.']==52)] = 1

test_n['cel_week'][(test_n['WeekNo.']!=48)&(test_n['WeekNo.']!=52)] = 0
train[train['MarkDown5']>=60000].shape
train['MarkDown5'][train['MarkDown5']>=60000]=60000

test_n['MarkDown5'][test_n['MarkDown5']>=60000]=60000
train['MarkDown4'][train['MarkDown4']>=30000]=30000

test_n['MarkDown4'][test_n['MarkDown4']>=30000]=30000
train['MarkDown3'][train['MarkDown3']>=11000]=11000

test_n['MarkDown3'][test_n['MarkDown3']>=11000]=11000
train['MarkDown2'][train['MarkDown2']>=40000]=40000

test_n['MarkDown2'][test_n['MarkDown2']>=40000]=40000
train['MarkDown1'][train['MarkDown1']>=40000]=40000

test_n['MarkDown1'][test_n['MarkDown1']>=40000]=40000
sns.boxplot(train['Weekly_Sales'],orient='v')
train[train['Weekly_Sales']>=200000].shape
train[train['Weekly_Sales']<1].shape
train_n = train[(train['Weekly_Sales']>=1)&(train['Weekly_Sales']<=200000)]
train_n.shape,train.shape
test_n.shape,test.shape
b = sns.distplot(np.sqrt(train_n['Weekly_Sales']))

b.set_title('Histogram of WeeklySales',fontsize = 16)

b.set_xlabel("WeeklySales",fontsize=14)

plt.show()
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True, figsize =(15,15)) #sharey -> share 'Price' as y

ax1.scatter(train_n['Date'][(train_n['Store']==1)&(train_n['Dept']==1)].sort_values(),

            train_n['Weekly_Sales'][(train_n['Store']==1)&(train_n['Dept']==1)])

ax1.set_title('WeeklySales and Date for Store1, Dept1')

ax2.scatter(train_n['Date'][(train_n['Store']==1)&(train_n['Dept']==2)].sort_values(),

            train_n['Weekly_Sales'][(train_n['Store']==1)&(train_n['Dept']==2)])

ax2.set_title('WeeklySales and Date for Store1, Dept2')

ax3.scatter(train_n['Date'][(train_n['Store']==1)&(train_n['Dept']==3)].sort_values(),

            train_n['Weekly_Sales'][(train_n['Store']==1)&(train_n['Dept']==3)])

ax3.set_title('WeeklySales and Date for Store1, Dept3')

plt.show()
f, (ax1) = plt.subplots(1, 1, sharey=True, figsize =(15,5))

ax1.scatter(train_n['Date'][train_n['Store']==2].sort_values(),

            train_n['Weekly_Sales'][train_n['Store']==2])

plt.show()
print(train_n.Date.max())

print(train_n.Date.min())

print(test_n.Date.max())

print(test_n.Date.min())
train_n = train_n.sort_values(['Store','Dept','Date'])

test_n = test_n.sort_values(['Store','Dept','Date'])
train_n.head(20)
test_n.head(20)
print(train_n['Weekly_Sales'].max())

print(train_n['Weekly_Sales'].min())

print(train_n['Weekly_Sales'].median())
import seaborn as sns



sns.set_style('whitegrid')



train_n['Weekly_Sales'].plot(kind='hist')
plt.figure(figsize=(15,7))

bins_list = [0,3999,7999,12999,24999,37999,49999,74999,99999,149999,199999]

plt.hist(train_n['Weekly_Sales'], bins=bins_list, alpha=0.5)
train_n['bin']=pd.Series()

train_n['bin']=pd.cut(train_n['Weekly_Sales'],bins_list)
train_n['bin'].value_counts()
bins = train_n.groupby(['Store','Dept']).agg({'bin':lambda x:x.value_counts().index[0]}).reset_index()
test_n = pd.merge(test_n,bins,left_on=['Store','Dept'],right_on=['Store','Dept'],how='left')
bins1 = train_n.groupby(['Store']).agg({'bin':lambda x:x.value_counts().index[0]}).reset_index()

bins1
store = test_n['Store'][test_n['bin'].isnull()].unique()

store
test_n['bin'][(test_n['bin'].isnull())&(test_n['Store']==3)]
for st in store:

    #print(bins1['bin'][bins1['Store']==st])

    test_n['bin'][(test_n['bin'].isnull())&(test_n['Store']==st)] = bins1['bin'][bins1['Store']==st].values
train_n.isnull().sum()
test_n.isnull().sum()
%%time

tr = pd.DataFrame()

val = pd.DataFrame()

for i in store:

    for j in dept:

        df = train_n[(train_n['Store']==i)&(train_n['Dept']==j)]

        #print(df.shape)

        c = int(df.shape[0]*0.8)

        tr = tr.append(df.iloc[0:c,:])

        val = val.append(df.iloc[c:,:])

        #print(tr.shape,val.shape)
train_n.shape, tr.shape,val.shape,test_n.shape
tr.set_index('key',inplace = True)

val.set_index('key', inplace = True)

test_n.set_index('key',inplace=True)

tr.drop('Date',axis=1,inplace=True)

val.drop('Date',axis=1,inplace=True)

test_n.drop('Date',axis=1,inplace=True)
x_train = tr.drop('Weekly_Sales',axis=1)

x_val = val.drop('Weekly_Sales',axis=1)

y_train = tr['Weekly_Sales']

y_val = val['Weekly_Sales']
x_train.columns
num_cols = ['Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment','Size','median','std','roll_mean']

cat_cols = ['Dept', 'IsHoliday', 'Store', 'Type', 'WeekNo.','cel_week','bin']
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

scale.fit(x_train[num_cols])

x_train[num_cols] = scale.transform(x_train[num_cols])

x_val[num_cols] = scale.transform(x_val[num_cols])

test_n[num_cols] = scale.transform(test_n[num_cols])
for col in cat_cols:

    x_train[col] = x_train[col].astype('category')

    x_val[col] = x_val[col].astype('category')

    test_n[col] = test_n[col].astype('category')
x_train = pd.get_dummies(x_train,columns=cat_cols,drop_first=True)

x_val = pd.get_dummies(x_val,columns=cat_cols,drop_first=True)

test_n = pd.get_dummies(test_n,columns=cat_cols,drop_first=True)
x_train.shape,x_val.shape,test_n.shape
test_n.columns.intersection(x_train.columns)
x_train.columns.difference(test_n.columns)
set(x_train.columns)-set(test_n.columns)
set(test_n.columns)-set(x_train.columns)
x_train.head(2)
test_n.rename(columns={'bin_(7999.0, 12999.0]':'bin_(7999, 12999]',

                      'bin_(3999.0, 7999.0]':'bin_(3999, 7999]',

                      'bin_(12999.0, 24999.0]':'bin_(12999, 24999]',

                      'bin_(24999.0, 37999.0]':'bin_(24999, 37999]',

                       'bin_(37999.0, 49999.0]':'bin_(37999, 49999]',

                      'bin_(49999.0, 74999.0]':'bin_(49999, 74999]',

                       'bin_(74999.0, 99999.0]':'bin_(74999, 99999]',

                      'bin_(99999.0, 149999.0]':'bin_(99999, 149999]',

                      'bin_(149999.0, 199999.0]':'bin_(149999, 199999]'},inplace=True)
test_n.head(2)
test_n.columns.difference(x_train.columns)
test_n.drop(test_n.columns.difference(x_train.columns),axis=1,inplace=True)
x_train.shape,x_val.shape,test_n.shape,y_train.shape,y_val.shape,test.shape
from sklearn.linear_model import LinearRegression

linreg = LinearRegression() 

linreg.fit(x_train, y_train)

train_pred = linreg.predict(x_train)

val_pred = linreg.predict(x_val)
from sklearn.metrics import mean_squared_error, r2_score

print("The R2 value on val dataset: {} \n".format(r2_score(y_pred=(val_pred), y_true=y_val)))

print("The R2 value on train dataset: {} \n".format(r2_score(y_pred=(train_pred), y_true=y_train)))

print("The Mean Squared Error on val dataset: {} \n".format(mean_squared_error(y_pred=(val_pred),y_true=y_val)))

print("The Mean Squared Error on train dataset: {} \n".format(mean_squared_error(y_pred=(train_pred),y_true=y_train)))
MAE=np.mean(np.abs(y_train - train_pred))

print(MAE)
MAE_val=np.mean(np.abs(y_val - val_pred))

print(MAE_val)
x_train.columns
feat = x_train[['Temperature', 'Fuel_Price', 'MarkDown1',

       'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI',

       'Unemployment', 'Size','median','std','roll_mean']]
feat.shape
from statsmodels.stats.outliers_influence import variance_inflation_factor

Vif = pd.DataFrame()

Vif["VIF Factor"] = [variance_inflation_factor(feat.values,i) for i in range(feat.shape[1])]

Vif["features"] = feat.columns

Vif
pd.DataFrame(y_train).shape
from sklearn.preprocessing import MinMaxScaler

target_scaler = MinMaxScaler()

y_train1 = pd.DataFrame(y_train)

y_val1 = pd.DataFrame(y_val)

target_scaler.fit(y_train1)

train_y = target_scaler.transform(y_train1)

val_y = target_scaler.transform(y_val1)
linreg = LinearRegression() 

linreg.fit(x_train, train_y)

train_pred_t = linreg.predict(x_train)

val_pred_t = linreg.predict(x_val)
train_pred_t = target_scaler.inverse_transform(train_pred_t)

val_pred_t = target_scaler.inverse_transform(val_pred_t)
from sklearn.metrics import mean_squared_error, r2_score

print("The R2 value on val dataset: {} \n".format(r2_score(y_pred=val_pred_t, y_true=y_val)))

print("The R2 value on train dataset: {} \n".format(r2_score(y_pred=train_pred_t, y_true=y_train)))

print("The Mean Squared Error on val dataset: {} \n".format(mean_squared_error(y_pred=val_pred_t,y_true=y_val)))

print("The Mean Squared Error on train dataset: {} \n".format(mean_squared_error(y_pred=train_pred_t,y_true=y_train)))
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(max_depth=20,max_features=15,min_samples_split=2,min_samples_leaf=1)

regressor.fit(x_train,train_y)
train_pred_t = regressor.predict(x_train)

val_pred_t = regressor.predict(x_val)
train_pred_t.shape
train_pred_t = target_scaler.inverse_transform(pd.DataFrame(train_pred_t))

val_pred_t = target_scaler.inverse_transform(pd.DataFrame(val_pred_t))
print("The R2 value on val dataset: {} \n".format(r2_score(y_pred=val_pred_t, y_true=y_val)))

print("The R2 value on train dataset: {} \n".format(r2_score(y_pred=train_pred_t, y_true=y_train)))

print("The Mean Squared Error on val dataset: {} \n".format(mean_squared_error(y_pred=val_pred_t,y_true=y_val)))

print("The Mean Squared Error on train dataset: {} \n".format(mean_squared_error(y_pred=train_pred_t,y_true=y_train)))

print("The RMSE on val dataset: {} \n".format(np.sqrt(mean_squared_error(y_pred=val_pred_t,y_true=y_val))))

print("The RMSE on train dataset: {} \n".format(np.sqrt(mean_squared_error(y_pred=train_pred_t,y_true=y_train))))
regressor1 = DecisionTreeRegressor(max_depth=14,max_features=40,min_samples_split=2,min_samples_leaf=1)

regressor1.fit(x_train,y_train)

train_pred = regressor1.predict(x_train)

val_pred = regressor1.predict(x_val)

print("The R2 value on val dataset: {} \n".format(r2_score(y_pred=val_pred, y_true=y_val)))

print("The R2 value on train dataset: {} \n".format(r2_score(y_pred=train_pred, y_true=y_train)))

print("The Mean Squared Error on val dataset: {} \n".format(mean_squared_error(y_pred=val_pred,y_true=y_val)))

print("The Mean Squared Error on train dataset: {} \n".format(mean_squared_error(y_pred=train_pred,y_true=y_train)))

print("The RMSE on val dataset: {} \n".format(np.sqrt(mean_squared_error(y_pred=val_pred,y_true=y_val))))

print("The RMSE on train dataset: {} \n".format(np.sqrt(mean_squared_error(y_pred=train_pred,y_true=y_train))))

print("The Mean Absolute Error on val dataset: {} \n".format(np.mean(np.abs(val_pred-y_val))))

print("The Mean Absolute Error on train dataset: {} \n".format(np.mean(np.abs(train_pred-y_train))))
np.mean(np.abs(val_pred-y_val)/y_val)*100
np.mean(np.abs(train_pred-y_train)/y_train)*100
%%time

from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor()

rfr



rfr.fit(X=x_train,y=train_y)
train_pred_t = rfr.predict(x_train)

val_pred_t = rfr.predict(x_val)

train_pred_t = target_scaler.inverse_transform(pd.DataFrame(train_pred_t))

val_pred_t = target_scaler.inverse_transform(pd.DataFrame(val_pred_t))
rfr
print("The R2 value on val dataset: {} \n".format(r2_score(y_pred=val_pred_t, y_true=y_val)))

print("The R2 value on train dataset: {} \n".format(r2_score(y_pred=train_pred_t, y_true=y_train)))

print("The Mean Squared Error on val dataset: {} \n".format(mean_squared_error(y_pred=val_pred_t,y_true=y_val)))

print("The Mean Squared Error on train dataset: {} \n".format(mean_squared_error(y_pred=train_pred_t,y_true=y_train)))

print("The RMSE on val dataset: {} \n".format(np.sqrt(mean_squared_error(y_pred=val_pred_t,y_true=y_val))))

print("The RMSE on train dataset: {} \n".format(np.sqrt(mean_squared_error(y_pred=train_pred_t,y_true=y_train))))
%%time



rfr1=RandomForestRegressor(n_estimators=300, max_features='auto',max_depth=9,bootstrap=False)

rfr1



rfr1.fit(X=x_train,y=y_train)

train_pred = rfr1.predict(x_train)

val_pred = rfr1.predict(x_val)





print("The R2 value on val dataset: {} \n".format(r2_score(y_pred=val_pred, y_true=y_val)))

print("The R2 value on train dataset: {} \n".format(r2_score(y_pred=train_pred, y_true=y_train)))

print("The Mean Squared Error on val dataset: {} \n".format(mean_squared_error(y_pred=val_pred,y_true=y_val)))

print("The Mean Squared Error on train dataset: {} \n".format(mean_squared_error(y_pred=train_pred,y_true=y_train)))

print("The RMSE on val dataset: {} \n".format(np.sqrt(mean_squared_error(y_pred=val_pred,y_true=y_val))))

print("The RMSE on train dataset: {} \n".format(np.sqrt(mean_squared_error(y_pred=train_pred,y_true=y_train))))

print("The Mean Absolute Error on val dataset: {} \n".format(np.mean(np.abs(val_pred-y_val))))

print("The Mean Absolute Error on train dataset: {} \n".format(np.mean(np.abs(train_pred-y_train))))
test_pred = rfr1.predict(test_n)
test3 = test_n.reset_index()
test3['Weekly_Sales'] = test_pred
test3.head(2)
sample_submission.drop(['Weekly_Sales'],axis=1,inplace=True)

sample_submission.head()
test3.shape,sample_submission.shape
sample = sample_submission.merge(test3[['key','Weekly_Sales']], left_on='id',right_on='key', how='left')
sample.drop('key',axis=1,inplace=True)

sample.head()
sample.to_csv('sample_submission.csv',index=False)