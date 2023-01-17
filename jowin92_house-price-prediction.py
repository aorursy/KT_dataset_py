# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_id = train['Id']
test_id = test['Id']

train.drop('Id', axis = 1)
test.drop('Id', axis = 1)

train.head()

sns.distplot(train['SalePrice'])

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

var  = 'GrLivArea'
data = pd.concat([train['SalePrice'], train['GrLivArea']],axis = 1 )
data.plot.scatter(x=var, y='SalePrice',ylim=(0,800000));
plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
g = sns.regplot(x=train[var], y=train['SalePrice'],fit_reg=False).set_title("Before")

plt.subplot(1,2,2)
train = train.drop(train[((train.GrLivArea>4000) & (train.SalePrice<600000))].index)
g = sns.regplot(x=train[var],y=train['SalePrice'],fit_reg=False).set_title("After")

# Categorical variable

var = 'YearBuilt'
data = pd.concat([train['SalePrice'],train[var]], axis = 1)
f, ax = plt.subplots(figsize=(8,6))
fig =sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)

corrmat = train.corr()
f,ax = plt.subplots(figsize=(14,9))
sns.heatmap(corrmat, vmax=0.8, square=True)
k = 14 # No of columns to show
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True,square=True, fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
plt.rcParams["figure.figsize"] = [16,9]
plt.show


f = pd.melt(train, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")

test['FireplaceQu'] = test['FireplaceQu'].fillna('NA')
test['MasVnrType'] = test['MasVnrType'].fillna('None')
test['GarageCars'] = test['GarageCars'].fillna(0)

train['FireplaceQu'] = train['FireplaceQu'].fillna('NA')
train['MasVnrType'] = train['MasVnrType'].fillna('None')
train['GarageCars'] = train['GarageCars'].fillna(0)

for c in qualitative:
 train[c]= train[c].astype('category')
 if train[c].isnull().any():
    train[c] = train[c].cat.add_categories(['MISSING'])
    train[c] = train[c].fillna('MISSING')
        
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
    
f=  pd.melt(train, id_vars=['SalePrice'], value_vars=qualitative)
g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")
from scipy import stats
def anova(frame):
    anv=pd.DataFrame()
    anv['feature'] = qualitative
    pvals =[]
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
        
    anv['pval'] = pvals
    return anv.sort_values('pval')

a = anova(train)
a['disparity']= np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature',y='disparity')
x=plt.xticks(rotation=90)
    

def encode(frame, feature):
    ordering =pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature,'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = 0
        
def encode2(frame, feature):
    ordering =pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val 
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = 0
        
qual_encoded = []
for q in qualitative:
    encode(train,q)
    encode2(test,q)
    qual_encoded.append(q+'_E')
    
print(qual_encoded)

def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data = spr, y='feature', x='spearman', orient='h')
    
features = quantitative + qual_encoded
spearman(train, features)



def log_transform(feature):
    train[feature] = np.log1p(train[feature].values)
    
def quadratic(feature):
    train[feature+'2']=train[feature]**2
    
log_transform('GrLivArea')
log_transform('1stFlrSF')
log_transform('2ndFlrSF')
log_transform('TotalBsmtSF')
log_transform('LotArea')
log_transform('LotFrontage')
log_transform('KitchenAbvGr')
log_transform('GarageArea')

quadratic('OverallQual')
quadratic('YearBuilt')
quadratic('YearRemodAdd')
quadratic('TotalBsmtSF')
quadratic('2ndFlrSF')
quadratic('Neighborhood_E')
quadratic('RoofMatl_E')
quadratic('GrLivArea')

qdr = ['OverallQual2', 'YearBuilt2', 'YearRemodAdd2', 'TotalBsmtSF2',
        '2ndFlrSF2', 'Neighborhood_E2', 'RoofMatl_E2', 'GrLivArea2']

train['HasBasement'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasGarage'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
train['Has2ndFloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasMasVnr'] = train['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
train['HasWoodDeck'] = train['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasPorch'] = train['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasPool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train['IsNew'] = train['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

boolean = ['HasBasement', 'HasGarage', 'Has2ndFloor', 'HasMasVnr', 'HasWoodDeck',
            'HasPorch', 'HasPool', 'IsNew']

features = quantitative + qual_encoded + boolean + qdr
def log_transform_test(feature):
    test[feature] = np.log1p(test[feature].values)
    
def quadratic_test(feature):
    test[feature+'2']=test[feature]**2
    
log_transform_test('GrLivArea')
log_transform_test('1stFlrSF')
log_transform_test('2ndFlrSF')
log_transform_test('TotalBsmtSF')
log_transform_test('LotArea')
log_transform_test('LotFrontage')
log_transform_test('KitchenAbvGr')
log_transform_test('GarageArea')

quadratic_test('OverallQual')
quadratic_test('YearBuilt')
quadratic_test('YearRemodAdd')
quadratic_test('TotalBsmtSF')
quadratic_test('2ndFlrSF')
quadratic_test('Neighborhood_E')
quadratic_test('RoofMatl_E')
quadratic_test('GrLivArea')

test['HasBasement'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasGarage'] = test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
test['Has2ndFloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasMasVnr'] = test['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
test['HasWoodDeck'] = test['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasPorch'] = test['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasPool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
test['IsNew'] = test['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

    
    

#sns.set()
#cols=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd','MasVnrType', 'FireplaceQu']
#sns.pairplot(train[cols], size = 2.5)
#plt.show()

#train['FireplaceQu'] = train['FireplaceQu'].fillna('NA')
#train['MasVnrType'] = train['MasVnrType'].fillna('None')

#total = train.isnull().sum().sort_values(ascending=False)
#percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
#missing_data.head(20)





#train = train.drop((missing_data[missing_data['Total']>1]).index,1)
#train = train.drop(train.loc[train['Electrical'].isnull()].index) 
#train.isnull().sum().max()  
#train.head()
                                
                                



from sklearn.preprocessing import StandardScaler
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)






from scipy.stats import norm

var1= 'YearBuilt'
sns.distplot(train[var1], fit=norm);
fig = plt.figure()
res = stats.probplot(train[var1],plot=plt)


import scipy.stats as st
y =  train['SalePrice']
plt.figure(1);plt.title('Johnson SU')
sns.distplot(y,fit = st.johnsonsu)
plt.figure(2);plt.title('Normal')
sns.distplot(y, fit = st.norm)
plt.figure(3);plt.title('Log Normal')
sns.distplot(y, fit = st.lognorm)




# The above graph shows that Johnson Su distribution is the best transformation before running our 
#regression algorithm
import scipy.stats as st

y = train['SalePrice'].values
def johnson(y):
    gamma, eta, epsilon, lbda = stats.johnsonsu.fit(y)
    yt = gamma + eta*np.arcsinh((y-epsilon)/lbda)
    return yt, gamma, eta, epsilon, lbda

def johnson_inverse(y, gamma, eta, epsilon, lbda):
    return lbda*np.sinh((y-gamma)/eta) + epsilon

yt, g, et, ep, l = johnson(y)
yt2 = johnson_inverse(yt, g, et, ep, l)
plt.figure(1)
sns.distplot(yt)
plt.figure(2)
sns.distplot(yt2)
#train = train[cols]
#train.head()
train = pd.get_dummies(train)
train.head()


from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
labels = train['SalePrice']
labels = np.array(train['SalePrice'])
train = train.drop('SalePrice',axis = 1)
test=pd.get_dummies(test)
# Get missing columns in the training test
missing_cols = set( train.columns ) - set( test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test = test[train.columns]
train = train.fillna(train.mean())
train = np.array(train)
rf = RandomForestRegressor (n_estimators = 1000, random_state = 42)
rf.fit(train, np.log(labels))


#cols_test=[ 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd','MasVnrType', 'FireplaceQu']
#test = test[cols_test]
#test.head()
test = test.fillna(test.mean())
#total = test.isnull().sum().sort_values(ascending=False)
#percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
#missing_data.head(20)

test = pd.get_dummies(test)
test.head()


test = np.array(test)
predictions = rf.predict(test)

predictions = (np.exp(predictions))
submission = pd.DataFrame({
        "Id": test_id,
        "SalePrice": predictions
})

submission.to_csv('submission_rf.csv', index=False)


