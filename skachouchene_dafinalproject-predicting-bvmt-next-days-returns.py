import pandas as pd # pandas

import seaborn as sns # seaborn package to generate nice plots

import matplotlib.pyplot as plt # matplotlib

import numpy as np # numpy

import scipy.stats as stats

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

sns.set(style="white", palette="muted", color_codes=True)

#Setting seaborn style
# read data

train=pd.read_csv('../input/data-analysis-2020/train.csv')

test=pd.read_csv('../input/data-analysis-2020/test.csv')
# convert date from string to datetime type

train.date=pd.to_datetime(train.date,format='%Y-%m-%d')
train.head()
ids=test.Id

train=train.dropna()
all_data = pd.concat((train, test)).reset_index(drop = True)
all_data=all_data.replace([np.inf, -np.inf], np.nan)
all_data['currReturn']=all_data.groupby(['company'])['close'].pct_change()
print("Missing values percentage in the Training Set")

missing_percentage=(train.isna().sum()/len(train)).sort_values(ascending=False)

print(missing_percentage)

print("Missing values percentage in the Testing Set")

missing_percentage=(test.isna().sum()/len(test)).sort_values(ascending=False)

print(missing_percentage)
needed_data=all_data[['yesterday_price', 'open', 'last', 'close', 'low', 'high', 'qty_traded', 'num_trades', 'value','currReturn','next_day_ret']]

corr_mat=needed_data.corr(method='pearson')

plt.figure(figsize=(20,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
to_drop=["yesterday_price","open","last","high","low","company_code"]

all_data=all_data.drop(to_drop,axis=1)
new=all_data.groupby('company').agg(next_day_ret_per_company=('next_day_ret','median'))

all_data = pd.merge(all_data,new,how='left',left_on='company',right_on='company')



all_data['year'] = pd.DatetimeIndex(all_data['date']).year

all_data['month'] = pd.DatetimeIndex(all_data['date']).month

all_data['day'] = pd.DatetimeIndex(all_data['date']).day

new=all_data.groupby('month').agg(next_day_ret_per_month=('next_day_ret','mean'))

all_data = pd.merge(all_data,new,how='left',left_on='month',right_on='month')
comps=train.company.unique()

mc=[]

for i in range(len(comps)):

  m=train[train["company"]==comps[i]].close.median()

  mc.append(m)



cat40=[]

cat25=[]

cat10=[]

cat5=[]

cat0=[]



for i in range(len(comps)):

  if mc[i]>40:

    cat40.append(comps[i])

  if mc[i]>25 and mc[i]<40:

    cat25.append(comps[i])

  if mc[i]>10 and mc[i]<25:

    cat10.append(comps[i])

  if mc[i]>5 and mc[i]<10:

    cat5.append(comps[i])

  if mc[i]<5:

    cat0.append(comps[i])



co=all_data.company.tolist()

a40=[0 for i in range(len(co))]

a25=[0 for i in range(len(co))]

a10=[0 for i in range(len(co))]

a5=[0 for i in range(len(co))]

a0=[0 for i in range(len(co))]

for i in range(len(comps)):

  if co[i]in cat40:

    a40[i]=1

  if co[i]in cat25:

    a25[i]=1

  if co[i]in cat10:

    a10[i]=1

  if co[i]in cat5:

    a5[i]=1

  if co[i] in cat0:

    a0[i]=1



all_data["over40"]=a40

all_data["25to40"]=a25

all_data["10to25"]=a10

all_data["5to10"]=a5

all_data["0to5"]=a0



cols=['skyblue','orange']

c=['qty_traded', 'num_trades']

for i in range(2):

    yy=np.log1p(all_data[c[i]])

    sns.distplot(yy,color=cols[i])

    plt.title('Distribution of the log-transformation of {}'.format(c[i]))

    plt.show()

print("Applying the log-transform on qty_traded makes it closer to a normal distribution")
all_data.qty_traded=np.log1p(all_data.qty_traded)

#all_data.num_trades=np.log1p(all_data.num_trades)
yy=stats.boxcox(all_data['close'])

plt.hist(yy, color=['orange','orange'])

plt.title("BoxCox transformation of 'close'")
#all_data.close=stats.boxcox(all_data.close)[0] #BoxCox transform for close as it is strictly positive
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

n=all_data.qty_traded.to_numpy().reshape(-1, 1)

scaler.fit(n)

scaler.transform(n)

all_data.dty_traded=n

train= all_data[:train.shape[0]]

test= all_data[train.shape[0]:]
test.value=test.qty_traded*test.close

print("filling missing values with the right formula instead of 'mean'")
sns.distplot(train.next_day_ret)
train=train[train["next_day_ret"]>-1000]

#Removing outliers
train=train.dropna()

Y=train.next_day_ret
to_drop=["Id","next_day_ret"]

train=train.drop(to_drop,axis=1)

test=test.drop(to_drop,axis=1)
test['value'].fillna((test['value'].mean()), inplace=True)

test['currReturn'].fillna((test['currReturn'].mean()), inplace=True)
ret_data=train.pivot(index='date',columns='company',values='currReturn').dropna(axis=1)
ret_data.head()
from sklearn.decomposition import PCA

num_pc = 1



X = np.asarray(train.drop(["company","date"],axis=1))



[n,m] = X.shape

print ('The number of timestamps is {}.'.format(n))

print ('The number of stocks is {}.'.format(m))



pca = PCA(n_components=num_pc) # number of principal components

pca.fit(X)



percentage =  pca.explained_variance_ratio_

percentage_cum = np.cumsum(percentage)

print ('{0:.2f}% of the variance is explained by the first PC'.format(percentage_cum[-1]*100))



pca_components = pca.components_
to_drop=["date","company"]

train=train.drop(to_drop,axis=1)

test=test.drop(to_drop,axis=1)
train.head()
train_X, val_X, train_y, val_y = train_test_split(train,Y , test_size=0.20, random_state=4)
lr = LinearRegression()

lr.fit(train_X, train_y)

val_pred=lr.predict(val_X)

val_mae = np.sqrt(mean_squared_error(val_pred, val_y))

print(val_mae)
# make predictions which we will submit.

lr.fit(train, Y)

test_preds = lr.predict(test)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.

output = pd.DataFrame({'Id': ids, 'next_day_ret': (test_preds)})

output.to_csv('submission.csv', index=False)