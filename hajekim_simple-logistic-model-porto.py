# 라이브러리 호출하기

import numpy as np

import pandas as pd

import seaborn as sns # 사용자 데이터 분포 시각화 패키지, Matplotlib를 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지

import matplotlib.pyplot as plt # 시각화 패키지 Matplotlib



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc

from sklearn.model_selection import StratifiedKFold,GridSearchCV

import missingno as mssno

seed =45

% matplotlib inline
path = '../input/'

# path = 'dataset/'

train = pd.read_csv(path+'train.csv',na_values=-1)

test = pd.read_csv(path+'test.csv',na_values=-1)

print('Number rows and columns:',train.shape)

print('Number rows and columns:',test.shape)
train.head(3).T
plt.figure(figsize=(10,3))

sns.countplot(train['target'],palette='rainbow')

plt.xlabel('Target')



train['target'].value_counts()
cor = train.corr()

plt.figure(figsize=(16,10))

sns.heatmap(cor,cmap='Set1')
ps_cal = train.columns[train.columns.str.startswith('ps_calc')] 

train = train.drop(ps_cal,axis =1)

test = test.drop(ps_cal,axis=1)

train.shape
k= pd.DataFrame()

k['train']= train.isnull().sum()

k['test'] = test.isnull().sum()

fig,ax = plt.subplots(figsize=(16,5))

k.plot(kind='bar',ax=ax)
mssno.bar(train,color='y',figsize=(16,4),fontsize=12)
mssno.bar(test,color='b',figsize=(16,4),fontsize=12)
mssno.matrix(train)
def missing_value(df):

    col = df.columns

    for i in col:

        if df[i].isnull().sum()>0:

            df[i].fillna(df[i].mode()[0],inplace=True)
missing_value(train)

missing_value(test)
def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = df.isnull().sum()

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(train)
def category_type(df):

    col = df.columns

    for i in col:

        if df[i].nunique()<=104:

            df[i] = df[i].astype('category')

category_type(train)

category_type(test)
cat_col = [col for col in train.columns if '_cat' in col]

print(cat_col)
fig ,ax = plt.subplots(1,2,figsize=(14,2))

ax1,ax2, = ax.flatten()

sns.countplot(train['ps_ind_02_cat'],palette='rainbow',ax=ax1)

sns.countplot(train['ps_ind_04_cat'],palette='summer',ax=ax2)

fig,ax = plt.subplots(figsize=(14,2))

sns.countplot(train['ps_ind_05_cat'],palette='rainbow',ax=ax)
fig,ax = plt.subplots(2,2,figsize=(14,4))

ax1,ax2,ax3,ax4 = ax.flatten()

sns.countplot(train['ps_car_01_cat'],palette='rainbow',ax=ax1)

sns.countplot(train['ps_car_02_cat'],palette='summer',ax=ax2)

sns.countplot(train['ps_car_03_cat'],palette='summer',ax=ax3)

sns.countplot(train['ps_car_04_cat'],palette='rainbow',ax=ax4)
fig,ax = plt.subplots(2,2,figsize = (14,4))

ax1,ax2,ax3,ax4 = ax.flatten()

sns.countplot(train['ps_car_05_cat'],palette='summer',ax=ax1)

sns.countplot(train['ps_car_06_cat'],palette='rainbow',ax=ax2)

sns.countplot(train['ps_car_07_cat'],palette='summer',ax=ax3)

sns.countplot(train['ps_car_08_cat'],palette='rainbow',ax=ax4)

fig, ax = plt.subplots(1,2,figsize=(14,2))

ax1,ax2 = ax.flatten()

sns.countplot(train['ps_car_09_cat'],palette='rainbow',ax=ax1)

sns.countplot(train['ps_car_10_cat'],palette='gist_rainbow',ax=ax2)

fig,ax = plt.subplots(figsize=(14,3))

sns.countplot(train['ps_car_11_cat'],palette='rainbow',ax=ax)
bin_col = [col for col in train.columns if 'bin' in col]

print(bin_col)
fig,ax = plt.subplots(3,3,figsize=(15,14),sharex='all')

ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9 = ax.flatten()

sns.countplot(train['ps_ind_06_bin'],palette='rainbow',ax=ax1)

sns.countplot(train['ps_ind_07_bin'],palette='summer',ax=ax2)

sns.countplot(train['ps_ind_08_bin'],palette='gist_rainbow',ax=ax3)

sns.countplot(train['ps_ind_09_bin'],palette='summer',ax=ax4)

sns.countplot(train['ps_ind_10_bin'],palette='rainbow',ax=ax5)

sns.countplot(train['ps_ind_11_bin'],palette='gist_rainbow',ax=ax6)

sns.countplot(train['ps_ind_12_bin'],palette='coolwarm',ax=ax7)

sns.countplot(train['ps_ind_13_bin'],palette='gist_rainbow',ax=ax8)

sns.countplot(train['ps_ind_16_bin'],palette='rainbow',ax=ax9)
fig,ax = plt.subplots(1,2,figsize=(14,6))

ax1,ax2 = ax.flatten()

sns.countplot(train['ps_ind_17_bin'],palette='coolwarm',ax=ax1)

sns.countplot(train['ps_ind_18_bin'],palette='gist_rainbow',ax=ax2)
tot_cat_col = list(train.select_dtypes(include=['category']).columns)



other_cat_col = [c for c in tot_cat_col if c not in cat_col+ bin_col]

other_cat_col
fig,ax = plt.subplots(2,2,figsize=(14,6))

ax1,ax2,ax3,ax4 = ax.flatten()

sns.countplot(data=train,x='ps_ind_01',palette='rainbow',ax=ax1)

sns.countplot(data=train,x='ps_ind_03',palette='gist_rainbow',ax=ax2)

sns.countplot(data=train,x='ps_ind_14',palette='gist_rainbow',ax=ax3)

sns.countplot(data=train,x='ps_ind_15',palette='rainbow',ax=ax4)
fig,ax = plt.subplots(2,2,figsize=(14,6))

ax1,ax2,ax3,ax4 =ax.flatten()

sns.countplot(data=train,x='ps_reg_01',palette='gist_rainbow',ax=ax1)

sns.countplot(data=train,x='ps_reg_02',palette='rainbow',ax=ax2)

sns.countplot(data=train,x='ps_car_11',palette='summer',ax=ax3)

sns.countplot(data=train,x='ps_car_15',palette='gist_rainbow',ax=ax4)

plt.xticks(rotation=90)
num_col = [c for c in train.columns if c not in tot_cat_col]

num_col.remove('id')

num_col
train['ps_reg_03'].describe()
fig,ax = plt.subplots(2,2,figsize=(14,8))

ax1,ax2,ax3,ax4 = ax.flatten()

sns.distplot(train['ps_reg_03'],bins=100,color='red',ax=ax1)

sns.boxplot(x ='ps_reg_03',y='target',data=train,ax=ax2)

sns.violinplot(x ='ps_reg_03',y='target',data=train,ax=ax3)

sns.pointplot(x= 'ps_reg_03',y='target',data=train,ax=ax4)
train['ps_car_12'].describe()
fig,ax = plt.subplots(2,2,figsize=(14,8))

ax1,ax2,ax3,ax4 = ax.flatten()

sns.distplot(train['ps_car_12'],bins=50,ax=ax1)

sns.boxplot(x='ps_car_12',y='target',data=train,ax=ax2)

sns.violinplot(x='ps_car_12',y='target',data=train,ax=ax3)

sns.pointplot(x='ps_car_12',y='target',data=train,ax=ax4)
train['ps_car_13'].describe()
fig,ax = plt.subplots(2,2,figsize=(14,8))

ax1,ax2,ax3,ax4 = ax.flatten()

sns.distplot(train['ps_car_13'],bins=120,ax=ax1)

sns.boxplot(x='ps_car_13',y='target',data=train,ax=ax2)

sns.violinplot(x='ps_car_13',y='target',data=train,ax=ax3)

sns.pointplot(x='ps_car_13',y='target',data=train,ax=ax4)
train['ps_car_14'].describe()
fig,ax = plt.subplots(2,2,figsize=(14,8))

ax1,ax2,ax3,ax4 = ax.flatten()

sns.distplot(train['ps_car_14'],bins=120,ax=ax1)

sns.boxplot(x='ps_car_14',y='target',data=train,ax=ax2)

sns.violinplot(x='ps_car_14',y='target',data=train,ax=ax3)

sns.pointplot(x='ps_car_14',y='target',data=train,ax=ax4)
def descrictive_stat_feat(df):

    df = pd.DataFrame(df)

    dcol= [c for c in train.columns if train[c].nunique()>=10]

    dcol.remove('id')   

    d_median = df[dcol].median(axis=0)

    d_mean = df[dcol].mean(axis=0)

    q1 = df[dcol].apply(np.float32).quantile(0.25)

    q3 = df[dcol].apply(np.float32).quantile(0.75)

    

    #Add mean and median column to data set having more then 10 categories

    for c in dcol:

        df[c+str('_median_range')] = (df[c].astype(np.float32).values > d_median[c]).astype(np.int8)

        df[c+str('_mean_range')] = (df[c].astype(np.float32).values > d_mean[c]).astype(np.int8)

        df[c+str('_q1')] = (df[c].astype(np.float32).values < q1[c]).astype(np.int8)

        df[c+str('_q3')] = (df[c].astype(np.float32).values > q3[c]).astype(np.int8)

    return df
train = descrictive_stat_feat(train)

test = descrictive_stat_feat(test)
cor = train[num_col].corr()

plt.figure(figsize=(10,4))

sns.heatmap(cor,annot=True)

plt.tight_layout()
def outlier(df,columns):

    for i in columns:

        quartile_1,quartile_3 = np.percentile(df[i],[25,75])

        quartile_f,quartile_l = np.percentile(df[i],[1,99])

        IQR = quartile_3-quartile_1

        lower_bound = quartile_1 - (1.5*IQR)

        upper_bound = quartile_3 + (1.5*IQR)

        print(i,lower_bound,upper_bound,quartile_f,quartile_l)

                

        df[i].loc[df[i] < lower_bound] = quartile_f

        df[i].loc[df[i] > upper_bound] = quartile_l

        

outlier(train,num_col)

outlier(test,num_col) 
def OHE(df1,df2,column):

    cat_col = column

    #cat_col = df.select_dtypes(include =['category']).columns

    len_df1 = df1.shape[0]

    

    df = pd.concat([df1,df2],ignore_index=True)

    c2,c3 = [],{}

    

    print('Categorical feature',len(column))

    for c in cat_col:

        if df[c].nunique()>2 :

            c2.append(c)

            c3[c] = 'ohe_'+c

    

    df = pd.get_dummies(df, prefix=c3, columns=c2,drop_first=True)



    df1 = df.loc[:len_df1-1]

    df2 = df.loc[len_df1:]

    print('Train',df1.shape)

    print('Test',df2.shape)

    return df1,df2
train1,test1 = OHE(train,test,tot_cat_col)
X = train1.drop(['target','id'],axis=1)

y = train1['target'].astype('category')

x_test = test1.drop(['target','id'],axis=1)

del train1,test1
#Grid search

"""logreg = LogisticRegression(class_weight='balanced')

param = {'C':[0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,1]}

clf = GridSearchCV(logreg,param,scoring='roc_auc',refit=True,cv=3)

clf.fit(X,y)

print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C'])) """
kf = StratifiedKFold(n_splits=5,random_state=seed,shuffle=True)

pred_test_full=0

cv_score=[]

i=1

for train_index,test_index in kf.split(X,y):    

    print('\n{} of kfold {}'.format(i,kf.n_splits))

    xtr,xvl = X.loc[train_index],X.loc[test_index]

    ytr,yvl = y[train_index],y[test_index]

    

    lr = LogisticRegression(class_weight='balanced',C=0.003)

    lr.fit(xtr, ytr)

    pred_test = lr.predict_proba(xvl)[:,1]

    score = roc_auc_score(yvl,pred_test)

    print('roc_auc_score',score)

    cv_score.append(score)

    pred_test_full += lr.predict_proba(x_test)[:,1]

    i+=1
print('Confusion matrix\n',confusion_matrix(yvl,lr.predict(xvl)))

print('Cv',cv_score,'\nMean cv Score',np.mean(cv_score))
proba = lr.predict_proba(xvl)[:,1]

fpr,tpr, threshold = roc_curve(yvl,proba)

auc_val = auc(fpr,tpr)



plt.figure(figsize=(14,8))

plt.title('Reciever Operating Charactaristics')

plt.plot(fpr,tpr,'b',label = 'AUC = %0.2f' % auc_val)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')
y_pred = pred_test_full/5

submit = pd.DataFrame({'id':test['id'],'target':y_pred})

#submit.to_csv('lr_porto.csv.gz',index=False,compression='gzip') 

submit.to_csv('lr_porto.csv',index=False) 
submit.head()