#Import library

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve

from sklearn.model_selection import KFold,RandomizedSearchCV

from sklearn.decomposition import PCA

seed =45

% matplotlib inline

plt.style.use('fivethirtyeight')
path = '../input/'

#path = ''

train = pd.read_csv(path+'train.csv',na_values=-1)

test = pd.read_csv(path+'test.csv',na_values=-1)

print('Number rows and columns:',train.shape)

print('Number rows and columns:',test.shape)
train.head(5).T
plt.figure(figsize=(12,6))

sns.distplot(train['y'],bins=120)

plt.xlabel('y')



#train['y'].value_counts()
cor = train.corr()

plt.figure(figsize=(16,10))

sns.heatmap(cor,cmap='viridis')
train.isnull().sum().sum()
test.isnull().sum().sum()
train_len = train.shape[0]

df = pd.concat([train,test],axis=0)
bin_col = [c for c in df.columns if (df[c].nunique()==2)]

len(bin_col)
other_col = [c for c in df.columns if c not in bin_col]

other_col
df[other_col].nunique()
def category_type(df):

    col = df.columns

    for i in col:

        if (2< df[i].nunique() <=53):

            df[i] = df[i].astype('category')

category_type(df)
fig ,ax = plt.subplots(2,2,figsize=(14,8))

ax1,ax2,ax3,ax4 = ax.flatten()

sns.countplot(df['X0'],palette='rainbow',ax=ax1)

sns.countplot(df['X1'],palette='summer',ax=ax2)

sns.countplot(df['X2'],palette='rainbow',ax=ax3)

sns.countplot(df['X3'],palette='magma',ax=ax4)
fig,ax = plt.subplots(2,2,figsize=(14,8))

ax1,ax2,ax3,ax4 = ax.flatten()

sns.countplot(df['X4'],palette='magma',ax=ax1)

sns.countplot(df['X5'],palette='rainbow',ax=ax2)

sns.countplot(df['X6'],palette='summer',ax=ax3)

sns.countplot(df['X8'],palette='magma',ax=ax4)
plt.figure(figsize=(14,80))

k = df[bin_col].sum().sort_values()

sns.barplot(k,k.index,orient='h',color='b')
def OHE(df,columns):

    print('Categorical features',len(columns))

    c2,c3 = [],{}

    for c in columns:

        c2.append(c)

        c3[c] = 'ohe_'+c

    df1 = pd.get_dummies(df,prefix=c3,columns=c2,drop_first=True)

    print('Size',df1.shape)

    return df1
col_ohe = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

df1 = OHE(df,col_ohe)
df1.head()
pca = PCA(n_components=None,random_state=seed)

pca.fit(df1.drop(['y','ID'],axis=1))
pca_var = pca.explained_variance_ratio_

fig,ax = plt.subplots(1,2,figsize=(16,8))

ax1,ax2, = ax.flatten()

ax1.plot(pca_var)

pca_var_cum = np.cumsum(pca_var)

ax2.plot(pca_var_cum,color='r')
pca = PCA(n_components=300,random_state=seed)

df_pca = pca.fit_transform(df1.drop(['y','ID'],axis=1))
X = df_pca[:train_len,]

y = df1[:train_len]['y']

x_test = df_pca[train_len:,]

X.shape,y.shape,x_test.shape
kf = KFold(n_splits=3,random_state=seed,shuffle=True)

pred_test_full=0

cv_score=[]

i=1

for train_index,test_index in kf.split(X,y):    

    print('\n{} of kfold {}'.format(i,kf.n_splits))

    xtr,xvl = X[train_index], X[test_index]

    ytr,yvl = y[train_index], y[test_index]

    

    lr = LinearRegression()

    lr.fit(xtr, ytr)

    pred_test = lr.predict(xvl)

    score = lr.score(xvl,yvl)

    print('R square score',score)

    cv_score.append(score)

    pred_test_full += lr.predict(x_test)

    i+=1
print('Cv',cv_score,'\nMean cv Score',np.mean(cv_score),'\n std',np.std(cv_score))
y_pred = pred_test_full/3

submit = pd.DataFrame({'ID':test['ID'],'y':y_pred})

#submit.to_csv('lr_benz.csv.gz',index=False,compression='gzip') 

submit.to_csv('lr_benz.csv',index=False) 
submit.head()