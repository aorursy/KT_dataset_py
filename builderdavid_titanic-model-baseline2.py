import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import warnings

from tqdm import tqdm
trainfile_path = '../input/train.csv'

testfile_path = '../input/test.csv'

tr_data = pd.read_csv(trainfile_path,index_col='PassengerId')

tst_data = pd.read_csv(testfile_path,index_col='PassengerId')
#tr_data.info() # have some missing value at Age and Embarked and many at Cabin
#tst_data.info() # Also have one missing value at Fare
#print(tr_data.groupby('Pclass').describe()['Age'])

#print(type(tr_data.groupby('Pclass').describe()['Age']))

#print(tr_data.groupby('Pclass')['Age'].info())
df = tr_data.groupby(['Pclass','Survived']).size()

df = df.unstack()

ax = df.plot.bar(stacked=True)

df = tst_data.groupby(['Pclass']).size()

plt.figure()

ax = df.plot.bar()
df = tr_data.groupby(['SibSp','Survived']).size()

df = df.unstack()

ax = df.plot.bar(stacked = True)

df = tst_data.groupby(['SibSp']).size()

plt.figure()

ax = df.plot.bar()
df = tr_data.groupby(['Parch','Survived']).size()

df = df.unstack()

ax = df.plot.bar(stacked = True)

df = tst_data.groupby(['Parch']).size()

plt.figure()

ax = df.plot.bar()
df = tr_data.groupby(['Embarked','Survived']).size()

df = df.unstack()

ax = df.plot.bar(stacked=True)

df = tst_data.groupby(['Embarked']).size()

plt.figure()

ax = df.plot.bar()
tst_data[tst_data['Fare'].isna()] # this is person we don't know his Fare Value. I guess he didn't survive
overall_data = pd.concat([tr_data,tst_data],axis = 0,sort=False)

overall_data.info()
import re

def Ticketcleaner(s):

    p = re.compile(r'\.|/| ')

    s=p.sub('',s)

    return s.replace('STON','SOTON').upper()

overall_data['Ticket']=overall_data['Ticket'].apply(Ticketcleaner)
title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev','Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess','Don', 'Jonkheer'] # We have all the titles for Passengers

def passager_title(name):

    for title in title_list:

        if title in name: return title

    return None # for cases no title in name

    

overall_data['title'] = overall_data['Name'].apply(passager_title)
df = overall_data.groupby(['title','Survived']).size()

df = df.unstack()

df = df.div(df.sum(axis=1),axis=0)

df.plot.barh(stacked=True)
def Cabin_class_extractor(cabinstr):

    if pd.isna(cabinstr) == True:

        return 'Unkonwn'

    return cabinstr[0]



overall_data['CabinClass'] = overall_data['Cabin'].apply(Cabin_class_extractor)
df = overall_data.groupby(['CabinClass','Survived']).size()

df = df.unstack()

df = df.div(df.sum(axis=1),axis=0)

df.plot.bar(stacked=True)
overall_data['Family_size'] = overall_data['Parch']+overall_data['SibSp']+1
df = overall_data.groupby(['Family_size','Survived']).size()

df = df.unstack()

df = df.div(df.sum(axis=1),axis=0)

df.plot.bar(stacked=True)
def ticket_type(s):# s will be the string of the ticket

    if s.isdigit():

        if len(s)==6: return 1

        if len(s)==5: return 2

        if len(s)==4: return 3

        else: return -1

    else:

        if    s.startswith('A'): return 4

        elif  s.startswith('CA'):return 5

        elif  s.startswith('PC'):return 6

        elif  s.startswith('SOTON'):return 7

        elif  s.startswith('SC'):return 8

        else: return -1

        

overall_data['Ticket_Type']=overall_data['Ticket'].apply(ticket_type)
df = overall_data.groupby(['Ticket_Type','Survived']).size()

df = df.unstack()

df.plot.bar(stacked=True)
overall_data['Missing_Age']=overall_data.Age.isna().astype('int')
from sklearn.impute import SimpleImputer
def impute1(s,method= 'mean'):

    if method == 'mean':return s.fillna(s.mean())

    elif method == 'median':return s.fillna(s.median())

    elif method == 'most frequent': return s.fillna(s.value_counts().index[0])

    else :return s
overall_data['ImputedAge']=overall_data.groupby('title')['Age'].transform(impute1,method = 'mean')

#overall_data['ImputedAge2'] = overall_data.groupby('title')['Age'].transform(impute1,method = 'most frequent')

overall_data['Imputed_Age_Range'] = pd.cut(overall_data['ImputedAge'],range(0, 105, 10))
overall_data['ImputedEmbarked'] = overall_data['Embarked'].transform(impute1, method = 'most frequent')
# overwrite Fare because only one missing record

overall_data['Fare'] = overall_data.groupby(['title','Imputed_Age_Range'])['Fare'].transform(impute1, method = 'median')
overall_data['Fare/Person']=overall_data.groupby('Ticket').Fare.transform(lambda s: s/len(s))
overall_data['Family_Name'] = overall_data.Name.apply(lambda s: s.split(', ')[0])
Prev_Family_Name=set() # A set store the family names in previous Companion group

prev_df = pd.DataFrame([0],columns=['Companion_Group_Num'])

overall_data['Companion_Group_Num'] = 0 # Initial all Companion group as 0 which means they all travel independently

Group_Num=1#initial number

for _,group in overall_data.groupby('Ticket',sort=True): #name should be their Ticket, Group is a dataframe of that Ticket group

    # step 1 check the group

    Have_Companion=False

    Current_Family_Name = set(group.Family_Name.unique())

    if Prev_Family_Name.isdisjoint(Current_Family_Name) == False: #current Ticket have common Family Name as the Previous

        Have_Companion = True

        Current_Family_Name |= Prev_Family_Name # put these family together for next iteration

        Group_Num -=1 # use the previous group_num

        #check previous group, need to adjust if it was judged alone

        if (prev_df.Companion_Group_Num==0).all()==True:

            Group_Num+=1

            overall_data.loc[prev_df.index,'Companion_Group_Num']=Group_Num

    if len(group) >= 2:

        Have_Companion = True

       #step2 set the companion group num

    if Have_Companion:

        overall_data.loc[group.index,'Companion_Group_Num'] = Group_Num

        Group_Num += 1

    Prev_Family_Name = Current_Family_Name

    prev_df=overall_data.loc[group.index]
overall_data['WC_group']=False

for idx,record in overall_data.iterrows():

    if record.Sex=='female' or record.ImputedAge<18: overall_data.loc[idx,'WC_group']=True
overall_data['Group_Size']=1

idx=overall_data.Companion_Group_Num!=0

overall_data.loc[idx,'Group_Size']=overall_data[idx].groupby('Companion_Group_Num').Companion_Group_Num.transform(len)

overall_data['WC_Group_Size']=np.nan

idx=(overall_data.Companion_Group_Num!=0)&(overall_data.WC_group)

overall_data.loc[idx,'WC_Group_Size']=overall_data[idx].groupby('Companion_Group_Num').Companion_Group_Num.transform(len)
def checkAll_TorF(s):

    global count_dead_group,count_live_group

    s=s.dropna()

    size = len(s)

    if size <2 :return np.nan

    if s.sum()==0:

        count_dead_group+=1

        return 1

    if s.sum()==size:

        count_live_group+=1

        return 1

    else:

        return 0

total_group_num = len(overall_data.Companion_Group_Num.unique())-1

covered_people = len(overall_data[overall_data.Companion_Group_Num!=0])

print(f'all passengers are divided in to {total_group_num} groups')

print(f'{covered_people} people are travel with companions')



count_live_group=0

count_dead_group=0

s = overall_data[(overall_data.Companion_Group_Num>=1)&(overall_data.WC_group==True)].groupby('Companion_Group_Num')['Survived'].agg(checkAll_TorF)

s1=s.dropna()

print(f'the proportion of the all dead or all alive women or child of companion Group : {s1.sum()/s1.size} which is {s1.sum()} in {s1.size}')

print(f'{s.isna().sum()}groups are all test data and no reference')

print(f'{count_live_group} groups are live,  {count_dead_group} groups are dead')



count_live_group=0

count_dead_group=0

s = overall_data[(overall_data.Companion_Group_Num>=1)&(overall_data.WC_group==False)].groupby('Companion_Group_Num')['Survived'].agg(checkAll_TorF)

s1=s.dropna()

print(f'the proportion of the all dead or all alive men of companion Group : {s1.sum()/s1.size} which is {s1.sum()} in {s1.size}')

print(f'{s.isna().sum()}groups are all test data and no reference')

print(f'{count_live_group} groups are live,  {count_dead_group} groups are dead')

num = len(overall_data[overall_data.Companion_Group_Num==0])

num2 = len(overall_data)

print(f'not covered by companion group people : {num/num2} which is {num} in {num2}')

del num,num2
## the Woman and Childs in the same group but both dead and survived records

idx=(overall_data.Companion_Group_Num!=0)&(overall_data.WC_group==True)#&(overall_data.Survived.notna())

def checkmixedsurvived(df):

    s=df.Survived.dropna()

    size=len(s)

    return ((s==1).sum()!=size) & ((s==1).sum()!=0)

overall_data.loc[idx].groupby('Companion_Group_Num').filter(checkmixedsurvived).sort_values('Companion_Group_Num')[['Ticket','Family_Name','Name','Sex','Age','Companion_Group_Num','Survived']]
### Groups that have one Woman_Child and one adult male

idxlst=[]

def check11(df):

    tmp=pd.Series(False,index=df.index)

    if ((df.WC_group==1).sum()==1): tmp.loc[df[df.WC_group==1].index]=True 

    if((df.WC_group==0).sum()==1): tmp.loc[df[df.WC_group==0].index]=True

    return tmp

idx = (overall_data.Companion_Group_Num!=0)

idx = overall_data[idx].groupby('Companion_Group_Num').apply(check11)

idx.reset_index('Companion_Group_Num',drop=True,inplace=True)

idx = idx[idx].index

df = overall_data.loc[idx].sort_values('Companion_Group_Num')

df[['Ticket','Pclass','Family_Name','Name','Sex','Age','Companion_Group_Num','Survived']]

#df.groupby(['WC_group','Survived']).size()

idx = df.WC_group==True

df[idx].groupby(['Pclass','Survived']).size().unstack()
idx = (overall_data.Companion_Group_Num!=0)&(overall_data.WC_group==True)

overall_data[idx].groupby(['Pclass','Survived']).size().unstack()
idx =(overall_data.Companion_Group_Num!=0)&(overall_data['WC_group']==False)# & (overall_data.Pclass==1)

#overall_data['survived_cat'] = overall_data['Survived'].to

import seaborn as sns

sns.relplot(data=overall_data[idx],x='Fare/Person',y='Family_size',hue='Survived')

sns.relplot(data=overall_data[idx],x='Fare',y='Family_size',hue='Survived')
idx=(overall_data.Companion_Group_Num!=0)&(overall_data['WC_group']==False)

#overall_data['survived_cat'] = overall_data['Survived'].to

import seaborn as sns

sns.relplot(data=overall_data[idx],x='Fare/Person',y='ImputedAge',hue='Survived')

sns.relplot(data=overall_data[idx],x='Fare',y='ImputedAge',hue='Survived')

overall_data[idx].groupby(['Pclass','Survived']).size()
idx=(overall_data.Companion_Group_Num!=0)&(overall_data['WC_group']==False)# & (overall_data.Pclass==1)

#overall_data['survived_cat'] = overall_data['Survived'].to

import seaborn as sns

sns.relplot(data=overall_data[idx],x='Fare/Person',y='ImputedAge',hue='Survived')

overall_data[idx].groupby(['Survived']).size()#.unstack() #
print(overall_data.columns)

print(overall_data.select_dtypes('object').columns)

# columns to code Let's try sklearn ordinal encode for Sex and onehot encode for other

columns_to_encode_ordinal = ['Sex','ImputedEmbarked','title','CabinClass']

columns_to_encode_onehot = ['ImputedEmbarked','title','CabinClass','Pclass']
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder

enc = OrdinalEncoder()

enc2 = OneHotEncoder(sparse=True) # exercise a little with sparse matrix

codedmatrix1 = enc.fit_transform(overall_data[columns_to_encode_ordinal])

codedmatrix2 = enc2.fit_transform(overall_data[columns_to_encode_onehot])



SparseColumns=[]

for idx,featurename in enumerate(columns_to_encode_onehot):

    for item in enc2.categories_[idx]:

        if featurename == 'ImputedEmbarked': featurename='Embarked' # shorten the column name for convenience

        if featurename == 'CabinClass' : featurename = 'Cabin'

        column_name = f'{featurename}_{item}'

        SparseColumns.append(column_name)

        

Encoded_df2 = pd.DataFrame.sparse.from_spmatrix(codedmatrix2,index=overall_data.index,columns=SparseColumns).sparse.to_dense()

Encoded_df = pd.DataFrame(codedmatrix1,columns=['Male?','EmbarkedCode','titleCode','CabinCode'],index=overall_data.index)

overall_data = pd.concat([overall_data,Encoded_df,Encoded_df2],axis = 1)

enc.categories_
tr_data = overall_data.loc[tr_data.index]

tst_data = overall_data.loc[tst_data.index]
idx =(tr_data['Sex'] == 'female')

idx2 =(tr_data['Sex'] == 'female')&( tr_data['Pclass'] == 3)

df = tr_data[idx].groupby(['Pclass','Survived']).size()

df = df.unstack()

print('-'*10,'female passenger in different class','-'*10)

print(df)

df = tr_data[idx2].groupby(['Embarked','Survived']).size()

df = df.unstack()

print('-'*10,'female passenger in 3rd class at different Embarked','-'*10)

print(df)
idx =(overall_data['WC_group']==0)

df = overall_data[idx].groupby('Survived').size()

df# predict all adult male dead have 83% acc on training set
def func1(df):

    return (df.Companion_Group_Num!=0).all() & df.Survived.isna().all() & (df.WC_group==True).all()

overall_data.groupby('Companion_Group_Num').filter(func1).loc[:,['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket','Fare', 'Cabin', 'Age_Range', 'title', 'Family_size','Companion_Group_Num']].sort_values('Companion_Group_Num')
from sklearn.model_selection import RepeatedStratifiedKFold

import lightgbm as lgb
cv_splitter = RepeatedStratifiedKFold(n_splits=10, n_repeats=5,random_state=100)

using_feature = ['Fare/Person', 'Ticket_Type', 'Age', 'Group_Size','Pclass']

categorical_feature=['Pclass', 'Ticket_Type']

predict_feature = 'Survived'

idx=(tr_data.WC_group==False) #&(tr_data.Companion_Group_Num!=0)

#TR_matrix = lgb.Dataset(tr_data.loc[idx,using_feature],label=tr_data.loc[idx,predict_feature], 

#                        categorical_feature=['Pclass','Male?','EmbarkedCode','CabinCode','Missing_Age','Ticket_Type'],

#                        free_raw_data=False

#                       )

#from scipy.stats import randint as sp_randint

param={

    'boosting_type':'gbdt', 

    'learning_rate':0.01, ##

    'num_iteration':500, ##

    'objective':'binary',

    'n_jobs':-1,

#    'num_leaves':sp_randint(10,30), 

    'max_depth':-1,

    'min_child_samples':15,

    'subsample':1, 

    'subsample_freq':0,

    'colsample_bytree':1, 

    'reg_alpha':0.0, ##

    'reg_lambda':0.1, ##

    'random_state':0,

    'is_unbalance':True,

#    'early_stopping_round':50,

#    'first_metric_only':True,    

    'metric':['auc','binary_logloss']

    }



#evalhist = lgb.cv(param,TR_matrix,folds = cv_splitter)

#for k,v in evalhist.items():

#    print(k,v[-1])    

#print(len(v))
def acc_at_threshold(preds,train_data):

    y_true=train_data.get_label()

    pred=preds>threshold

    tmp= y_true==pred

    return 'acc2',tmp.sum()/len(tmp),True
warnings.filterwarnings('ignore',category=UserWarning)

tr_x=tr_data.loc[idx,using_feature]

tr_y=tr_data.loc[idx,predict_feature]

result=pd.DataFrame([],columns=['threshold','binary_logloss_mean','binary_logloss_std','auc_mean','auc_std','acc2_mean','acc2_std'])

#for threshold in np.linspace(0.7,1,16):

threshold=0.95

binary_logloss=[]

auc=[]

acc2=[]

for tr_idx, cv_idx in  cv_splitter.split(tr_x,tr_y):

    TR_matrix = lgb.Dataset(tr_x.iloc[tr_idx],label=tr_y.iloc[tr_idx],free_raw_data=True)

    CV_matrix = lgb.Dataset(tr_x.iloc[cv_idx],label=tr_y.iloc[cv_idx],free_raw_data=True)

    model = lgb.train(param,TR_matrix,valid_sets=CV_matrix,verbose_eval=0,feval=acc_at_threshold,keep_training_booster=True,categorical_feature=categorical_feature)

    modelresult= model.eval_valid(acc_at_threshold)

    binary_logloss.append(modelresult[0][2])

    auc.append(modelresult[1][2])

    acc2.append(modelresult[2][2])

#print(model.eval_valid(acc_at_threshold))

tmpdf=pd.DataFrame({'binary_logloss':binary_logloss,'auc':auc,'acc2':acc2})

tmpdf=tmpdf.agg(['mean','std'])



result=result.append({'threshold':threshold,

                      'binary_logloss_mean': tmpdf.loc['mean','binary_logloss'], 

                      'binary_logloss_std':  tmpdf.loc['std','binary_logloss'], 

                      'auc_mean':            tmpdf.loc['mean','auc'], 

                      'auc_std':             tmpdf.loc['std','auc'], 

                      'acc2_mean':           tmpdf.loc['mean','acc2'], 

                      'acc2_std':            tmpdf.loc['std','acc2']},

                      ignore_index=True

                    )

result
TR_matrix = lgb.Dataset(tr_data.loc[idx,using_feature],label=tr_data.loc[idx,predict_feature])

model = lgb.train(param,TR_matrix,categorical_feature=categorical_feature)
lgb.plot_importance(model,importance_type='gain')
lgb.create_tree_digraph(model,2,show_info='leaf_count')
tst_data['Survived'] = pd.Series(np.nan,index = tst_data.index) # use the value in test data,did not modify value in overall_data

## step1 process the WC_group with companion

print(f'total test record num: {tst_data.Survived.isna().sum()}')

idx = (overall_data.Companion_Group_Num!=0)&(overall_data.WC_Group_Size>1)

grouped_df = overall_data[idx].groupby('Companion_Group_Num')

for name,df in grouped_df:

    survived_count = (df.Survived==1).sum()

    dead_count = (df.Survived==0).sum()

    na_index = df[df.Survived.isna()].index

    if na_index.empty: continue

    if dead_count>survived_count: tst_data.loc[na_index,'Survived']=0 #predict dead if all people are dead in this group

    elif survived_count>dead_count: tst_data.loc[na_index,'Survived']=1 # similar to the above

    elif df.Pclass.iloc[0]!=3: tst_data.loc[na_index,'Survived']=1 #remaining judge by Pclass these should include the groups with all member are in test data

    else:tst_data.loc[na_index,'Survived']=0

print(f'remaining record num: {tst_data.Survived.isna().sum()}')

## step2 using model to predict all male group

idx = (tst_data.WC_group==False)&(tst_data.Survived.isna())

#threshold=0.86

#pred= model.predict(tst_data.loc[idx,using_feature])>=threshold

tst_data.loc[idx,'Survived']=0

print(f'remaining record num: {tst_data.Survived.isna().sum()}')
## step3 alone female

idx = (tst_data.Companion_Group_Num==0)&(tst_data.WC_group==True)|(tst_data.WC_Group_Size==1)

tst_data.loc[idx,'Survived']=1

##submit checkpoint 2

print(tst_data.Survived.isna().sum())
resultDF = pd.DataFrame(tst_data.Survived.astype('int'),columns=['Survived'],index=tst_data.index)

print(resultDF.head(5))
resultDF.to_csv('Submission1.csv',header=True)