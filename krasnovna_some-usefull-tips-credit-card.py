import pickle,os
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
np.random.seed(50)
df_app = pd.read_csv('/kaggle/input/application_train.csv').sample(50000)
#Prev application
df_prev = pd.merge(df_app[['SK_ID_CURR','TARGET']], pd.read_csv('/kaggle/input/previous_application.csv'),
                   on = 'SK_ID_CURR', how = 'inner')
#Credit cards 
df_cc = pd.merge(df_prev[['SK_ID_PREV','TARGET']], pd.read_csv('/kaggle/input/credit_card_balance.csv'),
                   on = 'SK_ID_PREV', how = 'inner')
df_cc.head()
filter_dict = {
'p0_12m' : "(DF['MONTHS_BALANCE'] > -12) & (DF['MONTHS_BALANCE'] <= 0)",
'nz_draw': "(DF['AMT_DRAWINGS_CURRENT'].fillna(0) > 0)",
'nz_bal' : "(DF['AMT_BALANCE'].fillna(0) > 0)",
'nz_pos': "(DF['AMT_DRAWINGS_POS_CURRENT'].fillna(0) > 0)",
'nz_pay': "(DF['AMT_PAYMENT_CURRENT'].fillna(0) > 0)",
'nz_atm': "(DF['AMT_DRAWINGS_ATM_CURRENT'].fillna(0) > 0)"   
}
DF = df_cc.copy()
def nvl(x,y):
    if (not pd.isnull(x)):
        res = x 
    else :
        res = y 
    return res 

df_res = DF[['SK_ID_PREV','TARGET']].drop_duplicates() 

df_res = pd.merge(df_res, DF[eval(filter_dict['p0_12m'] + "&" + filter_dict['nz_pos'])].\
groupby('SK_ID_PREV').aggregate({'MONTHS_BALANCE' : 'count'}).reset_index().\
rename(columns = {'MONTHS_BALANCE':'CNT_MON_NZ_POS'}), on = 'SK_ID_PREV', how = 'left')

df_res = pd.merge(df_res, DF[eval(filter_dict['p0_12m'] + "&" + filter_dict['nz_atm'])].\
groupby('SK_ID_PREV').aggregate({'MONTHS_BALANCE' : 'count'}).reset_index().\
rename(columns = {'MONTHS_BALANCE':'CNT_MON_NZ_ATM'}), on = 'SK_ID_PREV', how = 'left')

df_res['CC_DRAW_HABIT'] = df_res[['CNT_MON_NZ_ATM','CNT_MON_NZ_POS']].\
apply(lambda x : (nvl(x[0],0)-nvl(x[1],0))/max(nvl(x[0],0),nvl(x[1],0)) if (not pd.isnull(x[0])) or (not pd.isnull(x[1])) else np.nan, axis = 1)
df_res2 = df_res[[not pd.isnull(x) for x in df_res['CC_DRAW_HABIT'].values]]
df_res2 = df_res2.sort_values(by = 'CC_DRAW_HABIT')
df_res2['rank'] = df_res2['CC_DRAW_HABIT'].rank(method = 'first')
print('The number of clients with spending on credit cards for last 12 month is about {x},\
\nThat is about {y}% of total'.format(x = df_res2['CC_DRAW_HABIT'].count(),
                                                     y = 100*round(df_res2['CC_DRAW_HABIT'].count()/50000,2)))
sns.distplot(df_res2['CC_DRAW_HABIT'].values, bins = 10,kde = False)
plt.title('CC_DRAW_HABIT Distribution')
#######################################
def get_bins(x, bin_intervals  ):
    pair_bin = []
    pair_bin.append((-np.inf,bin_intervals[0]))
    for i in range(len(bin_intervals)-1):
        pair_bin.append((bin_intervals[i],bin_intervals[i+1]))
    pair_bin.append((bin_intervals[-1],np.inf))
    
    bin_num = np.asarray([x > bin_[0] and x <= bin_[1] for bin_ in pair_bin]).argmax()
    return bin_num
bins_ = []
alpha = 10
while( alpha < 100):
        bins_.append( np.percentile(df_res2['rank'],alpha))
        alpha = alpha + 10
fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
plt.title('Draw habbit distribution')
#ax1.axvline(x= df_res['CC_DRAW_HABIT'].mean(), color='r', linestyle='dashed', linewidth=2)
df_res2['groups'] = df_res2['rank'].\
apply(lambda x:  get_bins(x, bin_intervals = bins_ ))
sns.countplot(x =df_res2['groups'].values,color = 'gray')
#####
df_res3 = pd.merge( df_res2, 
                    df_res2.groupby('groups').aggregate({'CC_DRAW_HABIT':max}).reset_index().\
rename(columns = {'CC_DRAW_HABIT': 'max_CC_DRAW_HABIT'}),
                   on = 'groups')
ax2 = ax1.twinx()
agg_ = pd.merge(df_res3[['groups','max_CC_DRAW_HABIT']].drop_duplicates(),
                df_res3.groupby('max_CC_DRAW_HABIT').aggregate({'TARGET':'mean'}).reset_index ().\
rename(columns = {'TARGET': 'Default rate'}),
               on = 'max_CC_DRAW_HABIT')
plt.plot(agg_['groups'].values,agg_['Default rate'].values,  'b--', marker = 'o',ms =4)
labels_ = df_res3[['groups','max_CC_DRAW_HABIT']].drop_duplicates()['max_CC_DRAW_HABIT'].tolist()
ax1.set_xticklabels([round(x,2) for x in labels_])
ax1.set_xlabel('maximum of CC_DRAW_HABIT in each percentile group(10% step)')
#ax1.legend('Default rate',)

import matplotlib.lines as mlines
blue_line = mlines.Line2D([], [],ls = '--', marker = 'o',ms =4, color='blue', label='Default Rate')
plt.legend(handles=[blue_line])
print("For each group I make a label with it's maximum of CC_DRAW_HABIT\
\nIt is easy to see the groups with same Default Rate\
\nOf course, we can mention a fact, that client with only ATM months have a bit more chance\
 to get a default on their credit ~12% comparing to  ~9% for 'POS clients'\
 \nI will not calcaulate importance of this variable here for purpose of this challenge, you can\
 make it for yourself ;)")
def max_(x):
    if(x.dropna().shape[0]):
        res = max(x.dropna())
    else :
        res = np.nan
    return res 

DF['CC_LOAD_RATE'] = DF[['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL']].\
apply(lambda x : x[0]/(1+x[1]), axis = 1)
df_res = DF[['SK_ID_PREV','TARGET']].drop_duplicates() 

df_res = pd.merge(df_res, DF[eval(filter_dict['p0_12m'])].\
groupby('SK_ID_PREV').aggregate({'CC_LOAD_RATE' : lambda x : max_(x) }).reset_index().\
rename(columns = {'CC_LOAD_RATE':'MAX_CC_LOAD_RATE'}), on = 'SK_ID_PREV', how = 'left')
df_res2 = df_res[[not pd.isnull(x) for x in df_res['MAX_CC_LOAD_RATE'].values]]
df_res2 = df_res2.sort_values(by = 'MAX_CC_LOAD_RATE')
df_res2['rank'] = df_res2['MAX_CC_LOAD_RATE'].rank(method = 'first')
bins_ = []
alpha = 10
while( alpha < 100):
        bins_.append( np.percentile(df_res2['rank'],alpha))
        alpha = alpha + 10
print('The number of clients with debt exist on credit cards for last 12 month is about {x},\
\nThat is about {y}% of total'.format(x = df_res2['MAX_CC_LOAD_RATE'].count(),
                                                     y = 100*round(df_res2['MAX_CC_LOAD_RATE'].count()/50000,2)))
fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
plt.title('Distribution of maximum debt to credit card limit rate')
#ax1.axvline(x= df_res['CC_DRAW_HABIT'].mean(), color='r', linestyle='dashed', linewidth=2)
df_res2['groups'] = df_res2['rank'].\
apply(lambda x:  get_bins(x, bin_intervals = bins_ ))
#####
sns.countplot(x =df_res2['groups'].values,color = 'gray')
#####
df_res3 = pd.merge( df_res2, 
                    df_res2.groupby('groups').aggregate({'MAX_CC_LOAD_RATE':max}).reset_index().\
rename(columns = {'MAX_CC_LOAD_RATE': 'max_MAX_CC_LOAD_RATE'}),
                   on = 'groups')

ax2 = ax1.twinx()
agg_ = pd.merge(df_res3[['groups','max_MAX_CC_LOAD_RATE']].drop_duplicates(),
                df_res3.groupby('max_MAX_CC_LOAD_RATE').aggregate({'TARGET':'mean'}).reset_index ().\
rename(columns = {'TARGET': 'Default rate'}),
               on = 'max_MAX_CC_LOAD_RATE')
plt.plot(agg_['groups'].values,agg_['Default rate'].values,  'b--', marker = 'o',ms =4)
labels_ = df_res3[['groups','max_MAX_CC_LOAD_RATE']].drop_duplicates()['max_MAX_CC_LOAD_RATE'].tolist()
ax1.set_xticklabels([round(x,2) for x in labels_])
ax1.set_xlabel('maximum of MAX_CC_LOAD_RATE in each percentile group(10% step)')
#ax1.legend('Default rate',)

import matplotlib.lines as mlines
blue_line = mlines.Line2D([], [],ls = '--', marker = 'o',ms =4, color='blue', label='Default Rate')
plt.legend(handles=[blue_line])
print("There we have high chance for default for clients with high debt to limit rate.\
\nOf course it is strange to have this rate more than 1, but I think it is  because\
 AMT_BALANCE consider a principal debt and interest debt and even smth more, \
for exmplу penalties for delinquency;).\nAnyway, this variable works really good for this competition.\
and helps to detach these 20% of 'clients' who has debt for last 12 month on credit card")


