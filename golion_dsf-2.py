# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# TODO: show visualization

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
df1 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
corrmat = df1.corr(method = 'pearson')

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df1[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, square=True, fmt='.2f', yticklabels=cols.values, xticklabels=cols.values)

plt.show()
print (corrmat['SalePrice'].sort_values(ascending=False)[:10], '\n') #top 10 values

print ('----------------------')

print (corrmat['SalePrice'].sort_values(ascending=False)[-10:]) #last 10 values`
# TODO: code to generate Plot 1

sns.distplot(df1['SalePrice'])
# TODO: code to generate Plot 2

pivot = df1.pivot_table(index='GarageCars', values='SalePrice', aggfunc=np.median)

pivot.plot(kind='bar', color='blue')
# TODO: code to generate Plot 3

sp_pivot = df1.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)

sp_pivot.plot(kind='bar', color='red')
# TODO: code to generate Plot 4

data = pd.concat([df1['SalePrice'], df1['GrLivArea']], axis=1)

data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
# TODO: code to generate Plot 5

data = pd.concat([df1['SalePrice'], df1['TotalBsmtSF']], axis=1)

data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));
var = 'OverallQual'

data = pd.concat([df1['SalePrice'], df1[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

ntrain = df1.shape[0]

ntest = test.shape[0]

y_train = df1.SalePrice.values

all_data = pd.concat((df1, test), sort=False).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(35)
all_data = all_data.drop((missing_data[missing_data['Total'] > 1]).index,1)

all_data = all_data.drop((missing_data[missing_data['Total'] == 1]).index,1)

all_data.isnull().sum().max()
train = all_data[:ntrain]

test = all_data[ntrain:]
train['Neighborhood']
# TODO: code for scoring function

df2 = train

f = df2.groupby(['GrLivArea']).mean()

# g = f.rank(1,'SalePrice')

g1 = df1.groupby('Neighborhood').mean()['SalePrice'].rank(ascending=True)

g2 = df1.groupby('GrLivArea').mean()['SalePrice'].rank(ascending=True)

g3 = df1.groupby('OverallQual').mean()['SalePrice'].rank(ascending=True)

g4 = df1.groupby('GarageCars').mean()['SalePrice'].rank(ascending=True)

g5 = df1.groupby('GarageArea').mean()['SalePrice'].rank(ascending=True)

g6 = df1.groupby('TotalBsmtSF').mean()['SalePrice'].rank(ascending=True)

g7 = df1.groupby('1stFlrSF').mean()['SalePrice'].rank(ascending=True)

g8 = df1.groupby('FullBath').mean()['SalePrice'].rank(ascending=True)

g9 = df1.groupby('TotRmsAbvGrd').mean()['SalePrice'].rank(ascending=True)

g10 = df1.groupby('YearBuilt').mean()['SalePrice'].rank(ascending=True)



t1 = g1.to_dict()

t2 = g2.to_dict()

t3 = g3.to_dict()

t4 = g4.to_dict()

t5 = g5.to_dict()

t6 = g6.to_dict()

t7 = g7.to_dict()

t8 = g8.to_dict()

t9 = g9.to_dict()

t10 = g10.to_dict()



train1 = pd.DataFrame()



cols = ['Neighborhood','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath',

        'TotRmsAbvGrd','YearBuilt'] 

for x in cols:

    train1[x] = df1[x]

train1.head()
t90 = pd.DataFrame()

def score_fn():

    c = train1['Neighborhood'].count()

    lst = list(range(c))

    print(train1)

    for i in lst:

        j = int(i) 

        t = train1.iloc[j]['Neighborhood']

        f = t1[t]

        train1.set_value(j, 'Neighborhood', f)

    t90 = train1

    print(t90)

    train1['Sum'] = train1.sum(axis=1)

    #train1['Neighborhood']

    t3 = train1.nlargest(10, ['Sum'])

    t5 = train1.nsmallest(10, ['Sum'])

    for index, row in train1.iterrows():

        x1 = row['Neighborhood']

        for item in g1.items():

            if item[1] == x1:

                t3.set_value(index, 'Neighborhood', item[0])

                

    for index, row in train1.iterrows():

        x1 = row['Neighborhood']

        for item in g1.items():

            if item[1] == x1:

                t5.set_value(index, 'Neighborhood', item[0])

    t4 = t3.nlargest(10, ['Sum'])

    t6 = t5.nlargest(10, ['Sum'])

    return t4,t6

    

    
t4,t6 = score_fn()

t4
t90
t6
# TODO: code for distance function

cols = ['Neighborhood','OverallQual','GrLivArea','1stFlrSF','FullBath',

        'TotRmsAbvGrd','YearBuilt'] 

for x in cols:

    train1[x] = train[x]

train1.head()
train['Neighborhood'].unique()
from sklearn.preprocessing import LabelEncoder

train3 = train

cols = ('Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',

        'HouseStyle','RoofStyle','RoofMatl','ExterQual','ExterCond','Foundation','Heating','HeatingQC','CentralAir',

        'PavedDrive','SaleCondition')



# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(train3[c].values)) 

    train3[c] = lbl.transform(list(train3[c].values))



# shape        

print('Shape all_data: {}'.format(train3.shape))


#train90 = train1

def pairwise(a,b):

    #print(np.linalg.norm(a-b))

    return np.linalg.norm(a-b)

train90 = train1

train90.rename(columns={ train90.columns[0]: "Neighborhood" }, inplace = True)

c = train90['Neighborhood'].count()

lst = list(range(c))

print(train90)

for i in lst:

    j = int(i) 

    t = train90.iloc[j]['Neighborhood']

    f = t1[t]

    train90.set_value(j, 'Neighborhood', f)



train90
train3 = train

from sklearn.preprocessing import LabelEncoder

cols = ('Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',

        'HouseStyle','RoofStyle','RoofMatl','ExterQual','ExterCond','Foundation','Heating','HeatingQC','CentralAir',

        'PavedDrive','SaleCondition')



# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(train3[c].values)) 

    train3[c] = lbl.transform(list(train3[c].values))



# shape        

print('Shape all_data: {}'.format(train3.shape))
train3 = train

from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=280, min_samples=2, metric = pairwise).fit(train90)

clust = clustering.labels_

clust
clust1 = pd.DataFrame(clust)

clust1[0].unique()

ytrain_pca = pd.DataFrame(clust)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(train90)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, ytrain_pca], axis = 1)

print(type(finalDf))

finalDf.rename(columns={ finalDf.columns[2]: "target" }, inplace = True)

finalDf
type(clust)
fig = plt.figure(figsize = (10,10))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)



targets = clust

#colors = ['r', 'g', 'b']

for target in targets:

    indicesToKeep = finalDf['target'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               , finalDf.loc[indicesToKeep, 'principal component 2']

               , s = 50)

ax.legend(clust1[0].unique())

ax.grid()
from sklearn.preprocessing import LabelEncoder

cols = ('Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',

        'HouseStyle','RoofStyle','RoofMatl','ExterQual','ExterCond','Foundation','Heating','HeatingQC','CentralAir',

        'PavedDrive','SaleCondition')



# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(test[c].values)) 

    test[c] = lbl.transform(list(test[c].values))



# shape        

print('Shape all_data: {}'.format(test.shape))
# TODO: code for linear regression

train3 = train

from sklearn.linear_model import LinearRegression

model1 = LinearRegression()  

model1.fit(train3,y_train)

predict1 = model1.predict(train3)

predict = model1.predict(test)

from sklearn.metrics import mean_squared_error

from math import sqrt

rms = sqrt(mean_squared_error(y_train, predict1))



coeff_df = pd.DataFrame(model1.coef_, train3.columns, columns=['Coefficient'])  
y_train.shape
rms
coeff_df
# TODO: code to import external dataset and test

import random
lst =[]

for i in range(1460):

    lst.append(random.randint(1,5))

i = pd.DataFrame(lst)

i.rename(columns={ i.columns[0]: "Schools" }, inplace = True)

i

# random.randint(3,4)
train4 = train

len(train4.columns)

train4.shape

ext1 = np.ones(1460)

ext1 = ext1*3

ext1 = np.transpose(ext1)

ext1 = np.reshape(ext1, (1460,1))

ext1

ext1 = pd.DataFrame(ext1)

#ext1

#ext1.rename(columns={ ext1.columns[0]: "Schools" }, inplace = True)

# train4 = pd.concat([train4,ext1],axis=1)

train4 = pd.concat([train4,i],axis=1)

y_train = pd.DataFrame(y_train)

y_train.rename(columns={ y_train.columns[0]: "Y" }, inplace = True)

train4 = pd.concat([train4,y_train],axis=1)

from scipy.stats import pearsonr

corr, _ = pearsonr(train4['Schools'], train4['Y'])

corr

#train4['Schools'].isnull().sum()
train['Neighborhood'].unique()
# TODO: code for all permutation tests

train4 = pd.DataFrame()

train4['OverallQual'] = train['OverallQual']

train5 = pd.DataFrame()

train5['GrLivArea'] = train['GrLivArea']

train6 = pd.DataFrame()

train6['Street'] = train['Street']

train7 = pd.DataFrame()

train7['KitchenAbvGr'] = train['KitchenAbvGr']

train8 = pd.DataFrame()

train8['SaleCondition'] = train['SaleCondition']

train9 = pd.DataFrame()

train9['YrSold'] = train['YrSold']

train10 = pd.DataFrame()

train10['LotArea'] = train['LotArea']

train11 = pd.DataFrame()

train11['CentralAir'] = train['CentralAir']

train12 = pd.DataFrame()

train12['FullBath'] = train['FullBath']

train13 = pd.DataFrame()

train13['HalfBath'] = train['HalfBath']

model2 = LinearRegression()

model3 = LinearRegression()

model4 = LinearRegression()

model5 = LinearRegression()

model6 = LinearRegression()

model7 = LinearRegression()

model8 = LinearRegression()

model9 = LinearRegression()

model10 = LinearRegression()

model11 = LinearRegression()





model2.fit(train4,y_train)

model3.fit(train5,y_train)

model4.fit(train6,y_train)

model5.fit(train7,y_train)

model6.fit(train8,y_train)

model7.fit(train9,y_train)

model8.fit(train10,y_train)

model9.fit(train11,y_train)

model10.fit(train12,y_train)

model11.fit(train13,y_train)





test1 = pd.DataFrame()

test1['OverallQual'] = test['OverallQual']

test2 = pd.DataFrame()

test2['GrLivArea'] = test['GrLivArea']

test3 = pd.DataFrame()

test3['Street'] = test['Street']

test3 = pd.get_dummies(test3)

test4 = pd.DataFrame()

test4['KitchenAbvGr'] = test['KitchenAbvGr']

test5 = pd.DataFrame()

test5['SaleCondition'] = test['SaleCondition']

test5 = pd.get_dummies(test5)

test6 = pd.DataFrame()

test6['YrSold'] = test['YrSold']

test7 = pd.DataFrame()

test7['LotArea'] = test['LotArea']

test8 = pd.DataFrame()

test8['CentralAir'] = test['CentralAir']

test9 = pd.DataFrame()

test9['FullBath'] = test['FullBath']

test10 = pd.DataFrame()

test10['HalfBath'] = test['HalfBath']

#test5 = pd.get_dummies(test5)





predicted_prices2 = model2.predict(test1)

predicted_prices3 = model3.predict(test2)

predicted_prices4 = model4.predict(test3)

predicted_prices5 = model5.predict(test4)

predicted_prices6 = model6.predict(test5)

predicted_prices7 = model7.predict(test6)

predicted_prices8 = model8.predict(test7)

predicted_prices9 = model9.predict(test8)

predicted_prices10 = model10.predict(test9)

predicted_prices11 = model11.predict(test10)





y_train_com = y_train.sample(100,random_state=1)



predicted_prices2_comp = pd.DataFrame(predicted_prices2).sample(100) 

predicted_prices3_comp = pd.DataFrame(predicted_prices3).sample(100) 

predicted_prices4_comp = pd.DataFrame(predicted_prices4).sample(100) 

predicted_prices5_comp = pd.DataFrame(predicted_prices5).sample(100) 

predicted_prices6_comp = pd.DataFrame(predicted_prices6).sample(100)

predicted_prices7_comp = pd.DataFrame(predicted_prices7).sample(100) 

predicted_prices8_comp = pd.DataFrame(predicted_prices8).sample(100) 

predicted_prices9_comp = pd.DataFrame(predicted_prices9).sample(100) 

predicted_prices10_comp = pd.DataFrame(predicted_prices10).sample(100) 

predicted_prices11_comp = pd.DataFrame(predicted_prices11).sample(100)



from sklearn.metrics import mean_squared_error

#y_train_com = pd.DataFrame(y_train_com).fillna('0')

predicted_prices2_comp = predicted_prices2_comp.reset_index()

#y_train_com = y_train_com.reset_index()

np.reshape(y_train_com, (100,1))

predicted_prices2_comp.rename(columns={ predicted_prices2_comp.columns[1]: "SalePrice" }, inplace = True)

predicted_prices21_comp = pd.DataFrame()

predicted_prices21_comp['SalesPrice'] = predicted_prices2_comp['SalePrice']

print("MSE1",mean_squared_error(np.log(y_train_com), np.log(predicted_prices21_comp)))

print("MSE2",mean_squared_error(np.log(y_train_com), np.log(predicted_prices3_comp)))

print("MSE3",mean_squared_error(np.log(y_train_com), np.log(predicted_prices4_comp)))

print("MSE4",mean_squared_error(np.log(y_train_com), np.log(predicted_prices5_comp)))

print("MSE5",mean_squared_error(np.log(y_train_com), np.log(predicted_prices6_comp)))

print("MSE6",mean_squared_error(np.log(y_train_com), np.log(predicted_prices7_comp)))

print("MSE7",mean_squared_error(np.log(y_train_com), np.log(predicted_prices8_comp)))

print("MSE8",mean_squared_error(np.log(y_train_com), np.log(predicted_prices9_comp)))

print("MSE9",mean_squared_error(np.log(y_train_com), np.log(predicted_prices10_comp)))

print("MSE10",mean_squared_error(np.log(y_train_com), np.log(predicted_prices11_comp)))





# We will look at the predicted prices to ensure we have something sensible.

#print(predicted_prices2)

score_dict={}

pvalue_dict={}

pvalue_lst = []

from sklearn.model_selection import permutation_test_score

score1,perm_score, pvalue= permutation_test_score(model2, train4, y_train)

print("OverallQual", score1)

t=train4.columns.values

pvalue_dict[str(t)]=pvalue

score2,perm_score, pvalue= permutation_test_score(model3, train5, y_train)

print("GrLiveArea", score2)

t=train5.columns.values

pvalue_dict[str(t)]=pvalue

score3,perm_score, pvalue= permutation_test_score(model4, train6, y_train)

print("Steet", score3)

t=train6.columns.values

pvalue_dict[str(t)]=pvalue

score4,perm_score, pvalue= permutation_test_score(model5, train7, y_train)

print("KitchenAbvGrd", score4)

t=train7.columns.values

pvalue_dict[str(t)]=pvalue

score5,perm_score, pvalue= permutation_test_score(model6, train8, y_train)

t=train8.columns.values

print("SaleCondition",score5)

pvalue_dict[str(t)]=pvalue

#pvalue_dict


predicted_prices21_comp
y_train2 = pd.DataFrame(y_train)

y_train2.isnull().sum()
train4.isnull().sum()
import random

c1=0

for i in range(100):

#     print(train4.isnull().sum())

#     print(y_train1.isnull().sum())

    train40 = pd.concat([train4, y_train2], axis=1)

#     print(train40.isnull().sum())

    train41 = train40.sample(1000)

    train41.rename(columns={ train41.columns[1]: "SalePrice" }, inplace = True)

    y_train1 = train41['SalePrice']

    train41 = train41.drop('OverallQual',axis=1)

    score,perm_score, pvalue= permutation_test_score(model2, train41, y_train1)

    c1=c1+perm_score

c1=c1/100

#c1

s = sns.distplot(c1)

plt.xticks(rotation=45)
train40.isnull().sum()
c2=0

for i in range(100):

    train50 = pd.concat([train5, y_train2], axis=1)

    train51 = train50.sample(750)

    train51.rename(columns={ train51.columns[1]: "SalePrice" }, inplace = True)

    y_train1 = train51['SalePrice']

    train51 = train51.drop('GrLivArea',axis=1)

    score,perm_score, pvalue= permutation_test_score(model2, train51, y_train1)

    c2=c2+perm_score

c2=c2/100

c2

s = sns.distplot(c2)

plt.xticks(rotation=45)
c3=0

for i in range(100):

    train60 = pd.concat([train6, y_train2], axis=1)

    train61 = train60.sample(1000)

    train61.rename(columns={ train61.columns[1]: "SalePrice" }, inplace = True)

    y_train1 = train61['SalePrice']

    train61 = train61.drop('Street',axis=1)

    score,perm_score, pvalue= permutation_test_score(model2, train61, y_train1)

    c3=c3+perm_score

c3=c3/100

c3

s = sns.distplot(c3)

plt.xticks(rotation=45)
c4=0

for i in range(100):

    train70 = pd.concat([train7, y_train2], axis=1)

    train71 = train70.sample(1000)

    train71.rename(columns={ train71.columns[1]: "SalePrice" }, inplace = True)

    y_train1 = train71['SalePrice']

    train71 = train71.drop('KitchenAbvGr',axis=1)

    score,perm_score, pvalue= permutation_test_score(model2, train71, y_train1)

    c4=c4+perm_score

c4=c4/100

c4

s = sns.distplot(c4)

plt.xticks(rotation=45)
c5=0

for i in range(100):

    train80 = pd.concat([train8, y_train2], axis=1)

    train81 = train80.sample(1000)

    train81.rename(columns={ train81.columns[1]: "SalePrice" }, inplace = True)

    y_train1 = train81['SalePrice']

    train81 = train81.drop('SaleCondition',axis=1)

    score,perm_score, pvalue= permutation_test_score(model2, train81, y_train1)

    c5=c5+perm_score

c5=c5/100

c5

s = sns.distplot(c5)

plt.xticks(rotation=45)
c6=0

for i in range(100):

    train100 = pd.concat([train9, y_train2], axis=1)

    train101 = train100.sample(1000)

    train101.rename(columns={ train101.columns[1]: "SalePrice" }, inplace = True)

    y_train1 = train101['SalePrice']

    train101 = train101.drop('YrSold',axis=1)

    score,perm_score, pvalue= permutation_test_score(model2, train101, y_train1)

    c6=c6+perm_score

c6=c6/100

c6

s = sns.distplot(c6)

plt.xticks(rotation=45)
c7=0

for i in range(100):

    train110 = pd.concat([train10, y_train2], axis=1)

    train111 = train110.sample(1000)

    train111.rename(columns={ train111.columns[1]: "SalePrice" }, inplace = True)

    y_train1 = train111['SalePrice']

    train111 = train111.drop('LotArea',axis=1)

    score,perm_score, pvalue= permutation_test_score(model2, train111, y_train1)

    c7=c7+perm_score

c7=c7/100

c7

s = sns.distplot(c7)

plt.xticks(rotation=45)
c8=0

for i in range(100):

    train120 = pd.concat([train11, y_train2], axis=1)

    train121 = train120.sample(1000)

    train121.rename(columns={ train121.columns[1]: "SalePrice" }, inplace = True)

    y_train1 = train121['SalePrice']

    train121 = train121.drop('CentralAir',axis=1)

    score,perm_score, pvalue= permutation_test_score(model2, train121, y_train1)

    c8=c8+perm_score

c8=c8/100

c8

s = sns.distplot(c8)

plt.xticks(rotation=45)
c9=0

for i in range(100):

    train130 = pd.concat([train12, y_train2], axis=1)

    train131 = train130.sample(1000)

    train131.rename(columns={ train131.columns[1]: "SalePrice" }, inplace = True)

    y_train1 = train131['SalePrice']

    train131 = train131.drop('FullBath',axis=1)

    score,perm_score, pvalue= permutation_test_score(model2, train131, y_train1)

    c9=c9+perm_score

c9=c9/100

c9

s = sns.distplot(c9)

plt.xticks(rotation=45)
c10=0

for i in range(100):

    train140 = pd.concat([train13, y_train2], axis=1)

    train141 = train140.sample(1000)

    train141.rename(columns={ train141.columns[1]: "SalePrice" }, inplace = True)

    y_train1 = train141['SalePrice']

    train141 = train141.drop('HalfBath',axis=1)

    score,perm_score, pvalue= permutation_test_score(model2, train141, y_train1)

    c10=c10+perm_score

c10=c10/100

c10

s = sns.distplot(c10)

plt.xticks(rotation=45)
from sklearn.ensemble import GradientBoostingRegressor

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=5, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

GBoost.fit(train3, y_train)

predicted_prices = GBoost.predict(test)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})



my_submission.to_csv('submission.csv', index=False)