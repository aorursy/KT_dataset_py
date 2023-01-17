import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score , train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import stats
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("train : " + str(train.shape))
print("test : " + str(test.shape))
quan = [c for c in train.columns if train[c].dtype != 'object']    #quantitative variables
qual = [c for c in train.columns if train[c].dtype == 'object']    #qualitative variables

#for quantitative variables we will perform pearson correlation test

corr_mat = train[quan].corr()
plt.figure(figsize = (12,9))
sns.heatmap(corr_mat)
#let's select top 20 features based on correlation with SalePrice and observe them.
top_20_quant = corr_mat.nlargest(20,'SalePrice')['SalePrice'].index
corr_coff_q = train[top_20_quant].corr()       #correlation cofficients of top 20 quantitative features
plt.figure(figsize = (12,9))
sns.heatmap(corr_coff_q , annot = True)
#selected quantitative features

quan_s = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt','YearRemodAdd', 'MasVnrArea', 'Fireplaces',
       'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF','OpenPorchSF', 'HalfBath']

# to analyse qualitative features , we will first fill the missing values
def fill_missing(data , features):
    for col in features:
        data[col] = data[col].astype('category')
        data[col] = data[col].cat.add_categories('MISSING')
        data[col] = data[col].fillna('MISSING')
    
df = pd.concat([train.drop('SalePrice',axis=1),test])
print ("missing values in qualitative features before "+ str(df[qual].isnull().sum().sum()))
fill_missing(df , qual)
print ("missing values in qualitative features after "+ str(df[qual].isnull().sum().sum()))

y = train['SalePrice']
train = df[0:1460]    #updating train and test after filing missing values
test = df[1460:]

def annov(df , features):
    ann = pd.DataFrame()
    ann['feat'] = features
    pvals = []
    for col in features:
        var = []
        for val in df[col].unique():
            var.append(df[df[col] == val].SalePrice.values)
        pvals.append(stats.f_oneway(*var).pvalue)
    ann['pvals'] = pvals
    return ann

train['SalePrice'] = y
result = annov(train , qual)
result['disp'] = np.log(1. / result['pvals'])
result = result.sort_values('disp' ,ascending = False)
plt.figure(figsize = (12,9))
sns.barplot(y = result['feat'] , x = result['disp'] ,orient = 'h')
qual_s = result.feat[0:20]      #already sorted

#we need to encode the qualitative features and convert to int to perform statistics tests on them 
df = pd.concat([train,test])

def encode(data , features):
    encoder=LabelEncoder()
    qual_E = []
    for col in features:
        encoder.fit(data[col].unique())
        data[col+'_E'] = encoder.transform(data[col])
        qual_E.append(col+'_E')
    return qual_E
    
qual_E_s = encode(df,qual_s)

df_s = df[quan_s + qual_E_s]    
train = df_s[0:1460]
test = df_s[1460:]

#chi square test between categorical features

def chi_sq(data , features):
    chi = pd.DataFrame(index = features , columns = features)
    for col1 in features:
        for col2 in features:
            #reshape beacause chi2 accepts dataframe of shape(n_samples,n_fetures)
            chi.loc[col1,col2] = chi2(data[col1].values.reshape(-1,1),data[col2])[1]
    return chi


chi = chi_sq(train , qual_E_s)
plt.figure(figsize = (12,9))
chi = chi.astype(float)
sns.barplot( y = chi.index , x= chi['Neighborhood_E'])
qual_E_final = ['Neighborhood_E' , 'SaleCondition_E' , 'SaleType_E' , 'GarageCond_E']
X = train[quan_s + qual_E_final]
X_test = test[quan_s + qual_E_final]

#filling missing values
data = pd.concat([X,X_test])
data.isnull().sum().sort_values(ascending = False)[0:10]
missing_lot = data.groupby('Neighborhood_E').LotFrontage.mean()

data.index = (range(0,data.shape[0]))

for index in range(0,data.shape[0]):
    if np.isnan(data.loc[index,'LotFrontage']):
        data.loc[index,'LotFrontage'] = missing_lot[data.loc[index,'Neighborhood_E']]
data.GarageCars =data.GarageCars.fillna(2.)          #mode
data.TotalBsmtSF =data.TotalBsmtSF.fillna(data.TotalBsmtSF.mean())
data.BsmtFinSF1 =data.BsmtFinSF1.fillna(data.BsmtFinSF1.mean())
data.MasVnrArea.describe()
#null value in MasVnrArea indicates absence of MasVnr , so we will fill it with zero

data['MasVnrArea'] = data['MasVnrArea'].fillna(0.)

print('Total missing values in dataset = ' + str(data.isnull().sum().sum()))
X_train = data[0:1460]
X_test = data[1460:]

f = pd.melt(pd.concat([X_train,y],axis=1) , id_vars = ['SalePrice'] , value_vars = X_train.columns )
g = sns.FacetGrid(f , col = 'variable' , col_wrap = 3 , size = 4,sharex=False , sharey=False)
g = g.map(plt.scatter, "value", "SalePrice")
X_tr = X_train
X_train = pd.concat([X_train,y],axis=1)

X_train = X_train.drop(X_train[(X_train['GrLivArea'] > 4000) & (X_train['SalePrice'] < 300000)].index)
X_train = X_train.drop(X_train[(X_train['TotalBsmtSF'] > 5000) & (X_train['SalePrice'] < 300000)].index)
X_train = X_train.drop(X_train[(X_train['MasVnrArea'] > 1500) & (X_train['SalePrice'] < 300000)].index)
X_train = X_train.drop(X_train[(X_train['BsmtFinSF1'] > 4000) & (X_train['SalePrice'] < 300000)].index)
X_train = X_train.drop(X_train[(X_train['LotFrontage'] > 250) & (X_train['SalePrice'] < 300000)].index)
X_train = X_train.drop(X_train[(X_train['OpenPorchSF'] > 400) & (X_train['SalePrice'] < 100000)].index)

y=X_train['SalePrice']
X_tr , X_ts , y_tr , y_ts = train_test_split( X_train , y, test_size = 0.3 )

reg = GradientBoostingRegressor()
reg.fit(X_tr,y_tr)
y_tr_pre = reg.predict(X_tr)
y_ts_pre = reg.predict(X_ts)
res = y_tr_pre - y_tr
res_val = y_ts_pre - y_ts

#for col in X_tr.columns:
#    plt.scatter(X_tr[col] , res ,c='blue')
 #   plt.hlines(y=0 , xmin = X_tr[col].min() , xmax = X_tr[col].max() , color = 'red')
  #  plt.xlabel(col)
   # plt.ylabel("Residuals")
    #plt.show()
    
plt.scatter(y_tr_pre , res ,c='blue')
plt.hlines(y=0 , xmin = y_tr_pre.min() , xmax = y_tr_pre.max() , color = 'red')
plt.xlabel('y_pred')
plt.ylabel("Residuals")
    
    

for col in X_train.columns:
     plt.scatter(X_train[col] , X_train['SalePrice'] ,c='blue')
     #plt.hlines(y=0 , xmin = X_tr[col].min() , xmax = X_train[col].max() , color = 'red')
     plt.xlabel(col)
     plt.ylabel("SalePrice")
     plt.show()
def make_quad(model , col):
    model[col + '_sq' ] = (model[col])**4
    
make_quad(X_train , 'YearBuilt')
make_quad(X_train , 'YearRemodAdd')
make_quad(X_train , 'BsmtFinSF1')

make_quad(X_test , 'YearBuilt')
make_quad(X_test , 'YearRemodAdd')
make_quad(X_test , 'BsmtFinSF1')


reg = GradientBoostingRegressor()
reg.fit(X_train.drop('SalePrice',axis=1),X_train['SalePrice'])
sample = pd.read_csv('../input/sample_submission.csv')
sample['SalePrice'] = reg.predict(X_test)
sample.to_csv('quad.csv',index=False)

