# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Please use inspect function in the lagou website to get your cookie, user-agent. the website url should also depends on your searching.

# I add a break after loop for the sample practice



# import requests

# from lxml import etree

# import pandas as pd

# from time import sleep

# import random

# import json

# import sys 

# cookie= 'YOUR-COOKIE'



# headers={

#     'User-Agent':'YOUR-USER-AGENT',

#     'Cookie': cookie

# }



# for i in (1,18):

#     sleep(random.randint(3,10))

#     url= 'https://www.lagou.com/zhaopin/shujufenxishi/{}/?filterOption=3&sid=50a903238ad44fa589cd5ad354d47f80'.format(i)

#     print('getting ',format(i),url)

#     con=etree.HTML(requests.get(url=url,headers=headers).text)

#     job_name=[i for i in con.xpath("//a[@class='position_link']/h3/text()")]

#     job_address=[i for i in con.xpath("//span[@class='add']/em/text()")]

#     job_company=[i for i in con.xpath("//div[@class='company_name']/a/text()")]

#     job_salary=[i for i in con.xpath("//span[@class='money']/text()")]

#     job_exp_edu=[i for i in con.xpath("//div[@class='li_b_l']/text()")]

#     job_exp_edu2=[i for i in [i.strip() for i in job_exp_edu] if i != '']

#     job_industry=[i for i in con.xpath("//div[@class='industry']/text()")]

#     job_tempation=[i for i in con.xpath("//div[@class='list_item_bot']/div[@class='li_b_r']/text()")]

#     job_links=[i for i in con.xpath("//div[@class='p_top']/a/@href")]

#     job_tags=[i for i in con.xpath("//div[@class='list_item_bot']/div[@class='li_b_l']")]

    

#     ingredients = []

#     for item in job_tags:

#         item = item.xpath("span/text()")

#         item = " ".join([" ".join(elem.split()) for elem in item])

#         ingredients.append(item)



#     job_tags = ingredients

    

#     job_names = pd.Series(job_name,name='job_name')

#     job_addresss = pd.Series(job_address,name='job_address')

#     job_companys = pd.Series(job_company,name='job_company')

#     job_salarys = pd.Series(job_salary,name='job_salary')

#     job_exp_edus = pd.Series(job_exp_edu2,name='job_exp_edu')

#     job_industrys = pd.Series(job_industry,name='job_industry')

#     job_tempations = pd.Series(job_tempation,name='job_tempation')

#     job_tagss = pd.Series(job_tags,name='job_tags')

    

#     job_des=[]

#     for link in job_links:

#         sleep(random.randint(3,10))

#         con2=etree.HTML(requests.get(url=url,headers=headers).text)

#         des=[[i.xpath('string(.)') for i in con2.xpath("//dd[@class='job_bt']/div/p")]]

#         job_des +=des

        

#     job_dess = pd.Series(job_des,name='job_des')    

#     break

# df = pd.concat([job_names,job_addresss,job_companys,job_salarys,job_exp_edus,job_industrys,job_tempations,job_tagss], axis=1)

# df.head()
df=pd.read_csv('/kaggle/input/lagou-data-analysis/lagou.csv')

df.head()
df.shape
df.info()
# Data preprocess

del df['Unnamed: 0']

df.head()
for i,j in enumerate(df['job_industry']):

    j=j.replace('\n','')

    df['job_industry'][i]=j

df.head()    
for i,j in enumerate(df['job_tempation']):

    j=j.replace('“','').replace('”','')

    df['job_tempation'][i]=j

df.head()   
j1=[]

j2=[]

for i,j in enumerate(df['job_salary']):

    j=j.replace('k','').replace('K','').replace('以上','-0')

    j1.append(int(j.split('-')[0])*1000)

    j2.append(int(j.split('-')[1])*1000)

df['job_salarymin']=j1

df['job_salarymax']=j2

df['job_salarymean']=(df['job_salarymin']+df['job_salarymax'])/2

df['job_salarymean']=df['job_salarymean'].astype(int)

df.head()  
del df['job_salary']
j1=[]

j2=[]

for i,j in enumerate(df['job_exp_edu']):

    j1.append(j.split('/')[0])

    j2.append(j.split('/')[1])

df['job_exp']=j1

df['job_edu']=j2

df.head()  
del df['job_exp_edu']
foo = lambda x: pd.Series([i for i in reversed(x.split('/'))])

rev = df['job_industry'].apply(foo)

df['empolyees']=rev[0]

df['company_stage']=rev[1]

df['job_industry']=rev[2]

df.head()
foo = lambda x: pd.Series([i for i in reversed(x.split(','))])

rev = df['job_industry'].apply(foo)

df['job_industry1']=rev[0]

df['job_industry2']=rev[1]

df.head()
del df['job_industry']
print(df.groupby('job_edu')['job_salarymean'].mean().sort_values(ascending=False))
print(df.groupby('job_exp')['job_salarymean'].mean().sort_values(ascending=False))
print(df.groupby('empolyees')['job_salarymean'].mean().sort_values(ascending=False))
df['job_name'].value_counts()
df.loc[df['job_name'].str.contains('高级'), 'job_name'] = '高级分析师'

df.loc[df['job_name'].str.contains('资深'), 'job_name'] = '高级分析师'

df.loc[df['job_name'].str.contains('数据'), 'job_name'] = '数据分析师'

df['job_name'].value_counts()
print(df.groupby('job_name')['job_salarymean'].mean().sort_values(ascending=False))
k=[]

for i in df['job_industry1']:

    j = i.replace(' ','')

    k.append(j)

df['job_industry1']=k    
df['job_industry1'].value_counts()
df.groupby(['job_industry1'])['job_salarymean'].mean().astype(int).sort_values(ascending=False).head()
k=[]

for i in df['job_tempation']:

    j = i.replace('，','').replace('、','').replace('-','').replace('；','')

    k.append(j)

df['job_tempation']=k 
# one hot 

cat_features=['job_name','job_exp','job_edu','empolyees','company_stage','job_industry1']

for col in cat_features:

    temp=pd.get_dummies(df[col])

    df=pd.concat([df,temp],axis=1)

print(df.shape ) 

df.head()    
plt.hist(df['job_salarymean'])
sns.boxplot(df['job_edu'],df['job_salarymean'])
sns.boxplot(df['job_exp'],df['job_salarymean'])
sns.boxplot(df['company_stage'],df['job_salarymean'])
sns.boxplot(df['empolyees'],df['job_salarymean'])
# content analysis

adv=[]

for i in df['job_tempation']:

    adv.append(i)

adv_text=''.join(adv)                       
import jieba

jieba.suggest_freq(('六险一金'),True)

jieba.suggest_freq(('五险一金'),True)

jieba.suggest_freq(('带薪年假'),True)

jieba.suggest_freq(('带薪年假'),True)

jieba.suggest_freq(('大牛'),True)

jieba.suggest_freq(('年终奖'),True)

jieba.suggest_freq(('500强'),True)

result=jieba.cut(adv_text)  
word_lst = []

word_dict= {}

for i in result:

    word_lst.append(i)

    for item in word_lst:

        if item not in word_dict: 

            word_dict[item] = 1

        else:

            word_dict[item] += 1

 
word=pd.DataFrame(list(word_dict.items()))

word=word[word[0].map(len) >= 3]
word.sort_values(1,ascending=False).head(10)
adv=[]

for i in df['job_tags']:

    adv.append(i)

adv_text=''.join(adv)  
import jieba

result=jieba.cut(adv_text) 

word_lst = []

word_dict= {}

for i in result:

    word_lst.append(i)

    for item in word_lst:

        if item not in word_dict: 

            word_dict[item] = 1

        else:

            word_dict[item] += 1

tags=pd.DataFrame(list(word_dict.items()))

tags=tags[tags[0].map(len) >= 2]

tags.sort_values(1,ascending=False).head(10)
from sklearn.feature_extraction.text import CountVectorizer

taglist=['SQL','电商','互联网','数据库','金融','移动','商业','数据挖掘']

vect = CountVectorizer(vocabulary=[v.lower() for v in taglist])    

X = vect.fit_transform(df['job_tags'].str.replace(r'\d+', ''))    

r = pd.DataFrame(X.A, columns=vect.get_feature_names(), index=df.index)

df=pd.concat([df,r],axis=1)

df.head()
datest=df.iloc[:,7:]

datest=pd.concat([df.iloc[:,7],df.iloc[:,14:]],axis=1)

datest.head()
# Modeling-GBR

X=datest.drop(['job_salarymean'],axis=1)

x=datest.drop(['job_salarymean'],axis=1).values

y=datest[['job_salarymean']].values.reshape((-1,1))

print(x.shape,y.shape)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
from sklearn.ensemble import GradientBoostingRegressor

params={'n_estimators':80,'max_depth':5}

model=GradientBoostingRegressor(**params)

model.fit(x_train,y_train)
from sklearn.metrics import mean_squared_error

y_pred=model.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))
test_score=np.zeros((params['n_estimators'],),dtype=np.float64)

for i,y_pred in enumerate(model.staged_predict(x_test)):

    test_score[i]=model.loss_(y_test,y_pred)

plt.title('deviance')

plt.plot(np.arange(params['n_estimators'])+1,model.train_score_,'b-',label='Train_set_deviance')

plt.plot(np.arange(params['n_estimators'])+1,test_score,'r-',label='Test_set_deviance')

plt.legend(loc='upper left')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')

#####

feature_importance=model.feature_importances_

feature_importance=100* (feature_importance/feature_importance.max())

sorted_idx=np.argsort(feature_importance)

fp= pd.DataFrame({'feature_importance': feature_importance, 'features': X.columns[sorted_idx]})

print(fp.sort_values('feature_importance',ascending=False).head(10))
plt.plot(y_pred)

plt.plot(y_test)

plt.legend(['y_pred','y_test'])

plt.show()
#XGBoost

import xgboost as xgb

import time

# cross-validation 5folds





xgb_params={

        'eta':1,

        'max_depth':5,

        'sub_sample':0.9,

        'colsample_bytree':0.9,

        'objective':'reg:linear',

        'eval_metric':'rmse',

        'seed':99,

        'slient':True

    }

d_train=xgb.DMatrix(x_train,label=y_train)

d_valid=xgb.DMatrix(x_test,label=y_test)

num_round = 1

model = xgb.train(xgb_params, d_train, num_round)

# make prediction

y_pred = model.predict(d_valid)

print(np.sqrt(mean_squared_error(y_test,y_pred)))

plt.plot(y_pred)

plt.plot(y_test)

plt.legend(['y_pred','y_test'])

plt.show()
