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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
%%time

train_df=pd.read_csv('../input/santander-customer-satisfaction/train.csv').set_index('ID')

test_df=pd.read_csv('../input/santander-customer-satisfaction/test.csv').set_index('ID')
target=pd.DataFrame(train_df['TARGET'])
train_df.info()
def reduce_mem_usage(df):

    """ 

    iterate through all the columns of a dataframe and 

    modify the data type to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print(('Memory usage of dataframe is {:.2f}''MB').format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max <np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max <np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max <np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max <np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                #if c_min > np.finfo(np.float16).min and c_max <np.finfo(np.float16).max:

                    #df[col] = df[col].astype(np.float16)

                if c_min > np.finfo(np.float32).min and c_max <np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2

    print(('Memory usage after optimization is: {:.2f}''MB').format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
#Reducing memory

train_df=reduce_mem_usage(train_df)
train_df.info()
train_df.describe()
#Custom Code to Display properties of each column

def ldf_f(train_df):

    l=[]

    for i in train_df.columns:

        l.append([i,len(train_df[i].unique()),round(max(train_df[i].unique()),2),round(min(train_df[i].unique()),2),train_df[i].var(),train_df[i].astype(bool).sum(axis=0),train_df[i].count(),sorted(list(train_df[i].unique()))])

    return pd.DataFrame(l, columns=['Features', 'No_Unique_Values', 'Max_Value','Min_Value','Variance','Non-Zero','Total_Values','Unique_Values'])

ldf=ldf_f(train_df)
ldf.shape
#pd.set_option('display.max_rows',None)

#pd.set_option('display.precision',2)
ldf[['Features','No_Unique_Values','Max_Value','Min_Value','Unique_Values','Non-Zero']].astype({'Max_Value': int,'Min_Value':int})
max_impute=list(ldf[(ldf.Max_Value==9999999999) | (ldf.Max_Value==10000000000)].Features)

for i in max_impute:

    if i in train_df.columns:

        print (train_df[(train_df[i]==9999999999) | (train_df[i]==10000000000)].shape,'\t',i)

    else:

        print ('Column Removed')

#missing values in max_impute Columns - 307 rows each - Impute the values - Use mode w.r.t. target

#max_impute
min_impute=list(ldf[ldf.Min_Value==-999999.00].Features)

for i in min_impute:

    if i in train_df.columns:

        print (train_df[train_df[i]==-999999.00].shape,'\t',i)

    else:

        print ('Column Removed')

#missing values in min_values Columns - 116 rows - Impute the values - Use mode w.r.t. target

#min_impute
#replacing missing values

train_df.replace({9999999999:np.NaN,-999999:np.NaN, 10000000000:np.NaN},inplace=True)

train_df['var36'].replace({99:np.NaN},inplace=True)
miss=pd.DataFrame(train_df.isnull().sum(),columns=['miss']).reset_index()

miss_features=list(miss[miss.miss!=0]['index'])

#All the features with missing values and frequency of it.
train_df=pd.concat([train_df.groupby('TARGET').transform(lambda x: x.fillna(x.value_counts().idxmax())),target],axis=1,sort=False)

#Replacing missing_values wth mode/most_frequent values w.r.t. TARGET
del_features=list(ldf[ldf.No_Unique_Values==1].Features)

ldf[ldf.No_Unique_Values==1].shape

#Remove these Columns : No Unique Value/Columns with same values throughout - 34 Columns
del_ID=list(train_df[train_df.duplicated()==True].index)

train_df[train_df.duplicated()==True].shape

#Duplicated Records - To be removed - 4961 rows
columns=list(train_df.columns)
fea_del_set=set()

for i in range(len(columns)):

    for j in range(i+1,len(columns)):

        if train_df[columns[i]].equals(train_df[columns[j]]) and columns[j] not in fea_del_set:

            fea_del_set.add(columns[j])

#creating a set of all duplicate columns.     
#Alternate for above cell.

"""fea_del={}

for i in range(len(columns)):

    l=[]

    for j in range(i+1,len(columns)):

        if train_df[columns[i]].equals(train_df[columns[j]]):

            l.append(columns[j])

    if l!=[]:

        fea_del[columns[i]]=l

##Check accuracy

set1=set()

for i in list(fea_del.values()):

    for j in i:

        set1.add(j)

"""
del_col=fea_del_set.union(del_features)
corr=pd.DataFrame(abs(train_df[list(ldf.Features)].corr())['TARGET']).sort_values(by=['TARGET'],ascending=False)

#dataframe containing correlation between columns
#HeatMap of correlation between Feature & Target with less than 300 columns to eliminate extra features

#I chose a correlation of 0.015 as the threshold value

#===HEATMAP===

plt.figure(figsize = (200 ,200))

corrmat = train_df[list(ldf[ldf['Non-Zero']<300].Features) + ['TARGET']].corr()

top_corr_features = corrmat.index

sns.heatmap(corrmat[top_corr_features].corr(), annot = True, cmap = 'RdYlGn')



crr_df=pd.DataFrame(abs(train_df[list(ldf[ldf['Non-Zero']<300].Features) + ['TARGET']].corr()['TARGET']))

crr_015=tuple(crr_df[crr_df.TARGET>0.015].index)
#Removing Features with less than 10% of non-zer values and with less than 0.015 correlation.

rem_features=set(train_df[list(ldf[ldf['Non-Zero']<300].Features)])

for i in crr_015:

    rem_features.discard(i)
np.array([i in corr['TARGET'][:100] for i in rem_features]).any()
del_col.update(rem_features)

del_col.update(('imp_trasp_var33_out_ult1','imp_reemb_var33_ult1'))
#del_col - All the columns to be deleted - 204 columns



"""

{'delta_imp_amort_var18_1y3',

 'delta_imp_amort_var34_1y3',

 'delta_imp_aport_var17_1y3',

 'delta_imp_aport_var33_1y3',

 'delta_imp_compra_var44_1y3',

 'delta_imp_reemb_var13_1y3',

 'delta_imp_reemb_var17_1y3',

 'delta_imp_reemb_var33_1y3',

 'delta_imp_trasp_var17_in_1y3',

 'delta_imp_trasp_var17_out_1y3',

 'delta_imp_trasp_var33_in_1y3',

 'delta_imp_trasp_var33_out_1y3',

 'delta_imp_venta_var44_1y3',

 'delta_num_aport_var17_1y3',

 'delta_num_aport_var33_1y3',

 'delta_num_compra_var44_1y3',

 'delta_num_reemb_var13_1y3',

 'delta_num_reemb_var17_1y3',

 'delta_num_reemb_var33_1y3',

 'delta_num_trasp_var17_in_1y3',

 'delta_num_trasp_var17_out_1y3',

 'delta_num_trasp_var33_in_1y3',

 'delta_num_trasp_var33_out_1y3',

 'delta_num_venta_var44_1y3',

 'imp_amort_var18_hace3',

 'imp_amort_var18_ult1',

 'imp_amort_var34_hace3',

 'imp_amort_var34_ult1',

 'imp_aport_var17_hace3',

 'imp_aport_var17_ult1',

 'imp_aport_var33_hace3',

 'imp_aport_var33_ult1',

 'imp_compra_var44_hace3',

 'imp_compra_var44_ult1',

 'imp_op_var40_comer_ult1',

 'imp_op_var40_ult1',

 'imp_reemb_var13_hace3',

 'imp_reemb_var13_ult1',

 'imp_reemb_var17_hace3',

 'imp_reemb_var17_ult1',

 'imp_reemb_var33_hace3',

 'imp_reemb_var33_ult1',

 'imp_sal_var16_ult1',

 'imp_trasp_var17_in_hace3',

 'imp_trasp_var17_in_ult1',

 'imp_trasp_var17_out_hace3',

 'imp_trasp_var17_out_ult1',

 'imp_trasp_var33_in_hace3',

 'imp_trasp_var33_in_ult1',

 'imp_trasp_var33_out_hace3',

 'imp_trasp_var33_out_ult1',

 'imp_var7_emit_ult1',

 'imp_var7_recib_ult1',

 'imp_venta_var44_hace3',

 'imp_venta_var44_ult1',

 'ind_var1',

 'ind_var13_medio',

 'ind_var13_medio_0',

 'ind_var17',

 'ind_var17_0',

 'ind_var18',

 'ind_var18_0',

 'ind_var2',

 'ind_var20',

 'ind_var20_0',

 'ind_var25',

 'ind_var26',

 'ind_var27',

 'ind_var27_0',

 'ind_var28',

 'ind_var28_0',

 'ind_var29',

 'ind_var29_0',

 'ind_var2_0',

 'ind_var31',

 'ind_var32',

 'ind_var32_0',

 'ind_var32_cte',

 'ind_var33',

 'ind_var33_0',

 'ind_var34',

 'ind_var34_0',

 'ind_var37',

 'ind_var39',

 'ind_var40',

 'ind_var41',

 'ind_var44',

 'ind_var44_0',

 'ind_var46',

 'ind_var46_0',

 'ind_var6',

 'ind_var6_0',

 'ind_var7_emit_ult1',

 'ind_var7_recib_ult1',

 'num_aport_var17_hace3',

 'num_aport_var17_ult1',

 'num_aport_var33_hace3',

 'num_aport_var33_ult1',

 'num_compra_var44_hace3',

 'num_compra_var44_ult1',

 'num_meses_var13_medio_ult3',

 'num_meses_var17_ult3',

 'num_meses_var29_ult3',

 'num_meses_var33_ult3',

 'num_meses_var44_ult3',

 'num_op_var40_comer_ult1',

 'num_op_var40_hace2',

 'num_op_var40_hace3',

 'num_op_var40_ult1',

 'num_op_var40_ult3',

 'num_reemb_var13_hace3',

 'num_reemb_var13_ult1',

 'num_reemb_var17_hace3',

 'num_reemb_var17_ult1',

 'num_reemb_var33_hace3',

 'num_reemb_var33_ult1',

 'num_sal_var16_ult1',

 'num_trasp_var17_in_hace3',

 'num_trasp_var17_in_ult1',

 'num_trasp_var17_out_hace3',

 'num_trasp_var17_out_ult1',

 'num_trasp_var33_in_hace3',

 'num_trasp_var33_in_ult1',

 'num_trasp_var33_out_hace3',

 'num_trasp_var33_out_ult1',

 'num_var1',

 'num_var13_medio',

 'num_var13_medio_0',

 'num_var17',

 'num_var17_0',

 'num_var18',

 'num_var18_0',

 'num_var20',

 'num_var20_0',

 'num_var25',

 'num_var26',

 'num_var27',

 'num_var27_0',

 'num_var28',

 'num_var28_0',

 'num_var29',

 'num_var29_0',

 'num_var2_0_ult1',

 'num_var2_ult1',

 'num_var31',

 'num_var32',

 'num_var32_0',

 'num_var33',

 'num_var33_0',

 'num_var34',

 'num_var34_0',

 'num_var37',

 'num_var39',

 'num_var40',

 'num_var41',

 'num_var44',

 'num_var44_0',

 'num_var46',

 'num_var46_0',

 'num_var6',

 'num_var6_0',

 'num_var7_emit_ult1',

 'num_var7_recib_ult1',

 'num_venta_var44_hace3',

 'num_venta_var44_ult1',

 'saldo_medio_var13_largo_hace3',

 'saldo_medio_var13_medio_hace2',

 'saldo_medio_var13_medio_hace3',

 'saldo_medio_var13_medio_ult1',

 'saldo_medio_var13_medio_ult3',

 'saldo_medio_var17_hace2',

 'saldo_medio_var17_hace3',

 'saldo_medio_var17_ult1',

 'saldo_medio_var17_ult3',

 'saldo_medio_var29_hace2',

 'saldo_medio_var29_hace3',

 'saldo_medio_var29_ult1',

 'saldo_medio_var29_ult3',

 'saldo_medio_var33_hace2',

 'saldo_medio_var33_hace3',

 'saldo_medio_var33_ult1',

 'saldo_medio_var33_ult3',

 'saldo_medio_var44_hace2',

 'saldo_medio_var44_hace3',

 'saldo_medio_var44_ult1',

 'saldo_medio_var44_ult3',

 'saldo_var1',

 'saldo_var13_medio',

 'saldo_var17',

 'saldo_var18',

 'saldo_var20',

 'saldo_var27',

 'saldo_var28',

 'saldo_var29',

 'saldo_var2_ult1',

 'saldo_var31',

 'saldo_var32',

 'saldo_var33',

 'saldo_var34',

 'saldo_var40',

 'saldo_var41',

 'saldo_var44',

 'saldo_var46',

 'saldo_var6'}

"""



len(del_col)
train_df.drop(del_col, axis=1,inplace=True)

#deleting columns
train_df.drop_duplicates(keep='first',inplace=True)

#deleting duplicates
train_df.shape
ldf=ldf_f(train_df)
f_u2=list(ldf[ldf.No_Unique_Values==2].Features)

f_u3=list(ldf[ldf.No_Unique_Values==3].Features)
def unique_percent(df,column):

    unique=list(df[column].unique())

    total=df[column].count()

    count=[]

    percent=[]

    for i in unique:

        count.append((i,(df[column]==i).sum()))

        percent.append((i,(df[column]==i).sum()/total*100))

    return count,percent
count=[]

percent=[]

for i in f_u2:

    c,p=unique_percent(train_df[train_df.TARGET==1],i)

    count.append((i,c))

    percent.append((i,p))

count
count=[]

percent=[]

for i in f_u3:

    c,p=unique_percent(train_df[train_df.TARGET==1],i)

    count.append((i,c))

    percent.append((i,p))

count
#train_df[(train_df.num_var13_corto==train_df.num_var24)].count()/train_df.count()

train_df[(train_df.num_var1_0==train_df.num_var40_0) & (train_df.TARGET==1)]
corr2=pd.DataFrame(abs(train_df[list(ldf.Features)].corr())['TARGET']).sort_values(by=['TARGET'],ascending=False).drop(['TARGET'])

#Correlation of Target with every feature
corr2['TARGET'].sort_values(ascending=False)
corr2['TARGET'].mean()
plt.figure(figsize = (50,50))

plt.xticks(rotation=90)

plt.axhline(0.04,label='Threshold',color='r')

plt.axhline(corr2['TARGET'].mean(),label='Mean',color='g')

sns.lineplot(y=corr2['TARGET'],x=corr2.index, label='Correlation with Target',)
above_threshold=list(corr2[corr2['TARGET']>0.04].index)

for i in above_threshold:

    print (ldf[ldf.Features==i])
corr_thresh=train_df[above_threshold+['TARGET']].corr()

mask = np.zeros_like(corr_thresh, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(corr_thresh, mask=mask, cmap="RdYlGn",annot=True, square=True, linewidths=.5, center=0, vmax=1);

fig=plt.gcf()

fig.set_size_inches(50,50)

plt.show()
train_df[train_df.num_var8_0!=train_df.ind_var8_0].shape[0]

"""fea_corr=[]

for i in corr_thresh:

    fea_corr.append((i,corr_thresh.loc[[corr_thresh[i]>0.9],i]))

fea_corr"""
plt.figure(figsize = (50,50))

plt.xticks(rotation=90)



sns.lineplot(y=ldf['Max_Value'],x=ldf.Features, label='Maximum Value',color='g')

sns.lineplot(y=ldf['Min_Value'],x=ldf.Features, label='Minimum Value',color='r')



plt.show()
plt.figure(figsize = (50,50))

plt.xticks(rotation=90)



sns.lineplot(y=ldf['No_Unique_Values'],x=ldf.Features, label='Unique_Values',color='g')
div=set()

for i in train_df.columns:

    div.add((i[:3]))

div.remove('TAR')

delta=[];imp=[];ind=[];saldo=[];var=[]

for i in train_df.columns:

    if i[:3]=='del':

        delta.append(i)

    elif i[:3]=='imp':

        imp.append(i)

    elif i[:3]=='ind':

        ind.append(i)

    elif i[:3]=='sal':

        saldo.append(i)

    elif i[:3]=='var':

        var.append(i)
ldf[ldf.Features=='var15']
"""plt.figure(figsize = (50,50))

plt.xticks(rotation=90)



sns.distplot(train_df['var15'])"""



fig, ax = plt.subplots()

sns.distplot((train_df[train_df.TARGET==1]['var15']),hist=False,kde_kws={'label':'Unhappy Customers',"linewidth": 3},color='r')



sns.distplot((train_df[train_df.TARGET==0]['var15']),hist=False,kde_kws={'label':'Happy Customers',"linewidth": 3,"color":'g'},norm_hist=True,ax=ax,label ='Happy Customers')



sns.distplot((train_df['var15']),hist=False,kde_kws={'label':'All Customers',"linewidth": 3, 'color':'b'},ax=ax,label ='All Customers')



#hist_kws={"histtype": "bar", "linewidth": 3}



fig=plt.gcf()

fig.set_size_inches(20,20)

plt.show()
"""plt.figure(figsize = (15,15))

plt.xticks(rotation=90)

"""

fig,ax=plt.subplots()

sns.distplot(train_df[train_df.TARGET==0]['var15'],bins=np.arange(0,120,10))



x = ax.lines[0].get_xdata() # Get the x data of the distribution

y = ax.lines[0].get_ydata()



plt.axvline(x[np.argmax(y)],label='Threshold',color='r')



fig.set_size_inches(20,20)



plt.show()
x[np.argmax(y)]
plt.figure(figsize = (15,15))

plt.xticks(rotation=90)



sns.countplot(train_df[train_df.TARGET==1]['var15'])
np.unique(train_df[train_df.TARGET==1]['var15'].values,return_counts=True)
train_df[(train_df.TARGET==1) & (train_df.var15>22) & (train_df.var15<50)]['var15'].sum()/((train_df[train_df.TARGET==1]['var15']).sum())*100
train_df[(train_df.TARGET==1) & (train_df.var15>25) & (train_df.var15<45)]['var15'].sum()/((train_df[train_df.TARGET==1]['var15']).sum())*100
train_df[(train_df.TARGET==0) & (train_df.var15>0) & (train_df.var15<36)]['var15'].sum()/((train_df[train_df.TARGET==0]['var15']).sum())*100
train_df[(train_df.TARGET==1) & (train_df.var15>22) & (train_df.var15<50)]['var15'].sum()/train_df[(train_df.var15>22) & (train_df.var15<50)]['var15'].sum()*100
ldf[ldf.Features=='num_var4']
prod_list=[[i,train_df[(train_df.num_var4==i)]['num_var4'].count(),train_df[(train_df.TARGET==1) & (train_df.num_var4==i)]['num_var4'].count(), round((train_df[(train_df.TARGET==1) & (train_df.num_var4==i)]['num_var4'].count()/train_df[(train_df.num_var4==i)]['num_var4'].count())*100,2),round((train_df[(train_df.TARGET==1) & (train_df.num_var4==i)]['num_var4'].count()/train_df[(train_df.TARGET==1)]['num_var4'].count())*100,2)] for i in (train_df.num_var4.unique())]

prod_df=pd.DataFrame(prod_list,columns=['prod_no','Total_Customer','Unhappy_Customer','Unhappy_Percent_All_Prod','Percent_Prod_Unhappy']).set_index('prod_no')

prod_df
fig,ax = plt.subplots()

sns.barplot(x=prod_df.index,y=prod_df.Total_Customer,ax=ax)

fig.set_size_inches(10,10)

plt.show()
fig,ax = plt.subplots()

sns.barplot(x=prod_df.index,y=prod_df.Unhappy_Customer,ax=ax)

fig.set_size_inches(10,10)

plt.show()
fig,ax = plt.subplots()

sns.barplot(x=prod_df.index,y=prod_df.Unhappy_Percent_All_Prod,ax=ax)

fig.set_size_inches(10,10)

plt.show()
fig,ax = plt.subplots()

sns.barplot(x=prod_df.index,y=prod_df.Percent_Prod_Unhappy,ax=ax)

fig.set_size_inches(10,10)

plt.show()
fig,ax=plt.subplots()

sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==0)]['var15'],ax=ax, label="Unhappy Customers-0")

sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==0)]['var15'],ax=ax,label="Happy Customers-0")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==1)]['var15'],ax=ax, label="Unhappy Customers-1")

sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==1)]['var15'],ax=ax,label="Happy Customers-1")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==2)]['var15'],ax=ax, label="Unhappy Customers-2")

sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==2)]['var15'],ax=ax,label="Happy Customers-2")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==3)]['var15'],ax=ax, label="Unhappy Customers-3")

sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==3)]['var15'],ax=ax,label="Happy Customers-3")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==4)]['var15'],ax=ax, label="Unhappy Customers-4")

sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==4)]['var15'],ax=ax,label="Happy Customers-4")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==5)]['var15'],ax=ax, label="Unhappy Customers-5")

sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==5)]['var15'],ax=ax,label="Happy Customers-5")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==6)]['var15'],ax=ax, label="Unhappy Customers-6")

sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==6)]['var15'],ax=ax,label="Happy Customers-6")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==7)]['var15'],ax=ax, label="Unhappy Customers-7")

sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==7)]['var15'],ax=ax,label="Happy Customers-7")

fig.set_size_inches(10,10)

plt.show()
fig,ax=plt.subplots()

sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==0)]['var15'],ax=ax, label="Unhappy Customers-0")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==1)]['var15'],ax=ax, label="Unhappy Customers-1")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==2)]['var15'],ax=ax, label="Unhappy Customers-2")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==3)]['var15'],ax=ax, label="Unhappy Customers-3")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==4)]['var15'],ax=ax, label="Unhappy Customers-4")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==5)]['var15'],ax=ax, label="Unhappy Customers-5")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==6)]['var15'],ax=ax, label="Unhappy Customers-6")



sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==7)]['var15'],ax=ax, label="Unhappy Customers-7")

fig.set_size_inches(10,10)

plt.show()
fig,ax=plt.subplots()

sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==0)]['var15'],ax=ax, label="Happy Customers-0")



sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==1)]['var15'],ax=ax, label="Happy Customers-1")



sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==2)]['var15'],ax=ax, label="Happy Customers-2")



sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==3)]['var15'],ax=ax, label="Happy Customers-3")



sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==4)]['var15'],ax=ax, label="Happy Customers-4")



sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==5)]['var15'],ax=ax, label="Happy Customers-5")



sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==6)]['var15'],ax=ax, label="Happy Customers-6")



sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==7)]['var15'],ax=ax, label="Happy Customers-7")

fig.set_size_inches(10,10)

plt.show()
fig,ax=plt.subplots()

sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==0)]['var15'],ax=ax, label="Unhappy Customers-0")

sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==0)]['var15'],ax=ax,label="Happy Customers-0")

fig.set_size_inches(10,10)

plt.show()
fig,ax=plt.subplots(1,8,sharey='row')

for i in range(8):

    sns.kdeplot(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==i)]['var15'],ax=ax[i], label="Unhappy Customers-%d"%(i))    

    

    if i<6:

        x = ax[i].lines[0].get_xdata()

        y = ax[i].lines[0].get_ydata()

        ax[i].axvline(x[np.argmax(y)],label="Unhappy Customer Age Probability: %d"%(x[np.argmax(y)]),color='r')

        ax[i].axvline(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==i)]['var15'].mean(),color='g',label="Unhappy Customer Age Mean: %d"%(train_df[(train_df['TARGET']==1) & (train_df['num_var4']==i)]['var15'].mean()))

        

    sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==i)]['var15'],ax=ax[i],label="Happy Customers-%d"%(i))



    ax[i].xaxis.set_ticks(np.arange(0, 120+1,10))



fig.set_size_inches(50,8)

plt.show()
fig,ax=plt.subplots(1,8,sharey='row')

for i in range(8):

    sns.kdeplot(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==i)]['var15'],ax=ax[i], label="Happy Customers-%d"%(i))    

    x = ax[i].lines[0].get_xdata()

    y = ax[i].lines[0].get_ydata()

    ax[i].axvline(x[np.argmax(y)],color='r',label="Happy Customer Age Probability: %d"%(x[np.argmax(y)]))

    ax[i].axvline(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==i)]['var15'].mean(),color='g',label="Happy Customer Age Mean: %d"%(train_df[(train_df['TARGET']==0) & (train_df['num_var4']==i)]['var15'].mean()))

    

    ax[i].legend()

    ax[i].xaxis.set_ticks(np.arange(0, 120+1,10))

fig.set_size_inches(50,8)

plt.show()

for i in range(0,8):

    print ('Product No. %d - Percentage Over Age 40 : %f'%(i, train_df[(train_df['num_var4']==i) & (train_df['var15']>40)]['var15'].count()/train_df[(train_df['num_var4']==i)]['var15'].count()*100))
for i in range(0,8):

    print ('Product No. %d - Percentage Under Age 30 : %f'%(i, train_df[(train_df['num_var4']==i) & (train_df['var15']<30)]['var15'].count()/train_df[(train_df['num_var4']==i)]['var15'].count()*100))
for i in range(0,8):

    print ('Product No. %d - Percentage between Age 30 and 40 : %f'%(i, train_df[(train_df['num_var4']==i) & (train_df['var15']>30) & (train_df['var15']<40)]['var15'].count()/train_df[(train_df['num_var4']==i)]['var15'].count()*100))
above_threshold
train_df.shape
ldf[ldf.Features=='var36']
np.unique(train_df[train_df.TARGET==1]['var36'],return_counts=True)[1]
var36_cat=[[i,train_df[(train_df.var36==i)]['var36'].count(),train_df[(train_df.TARGET==1) & (train_df.var36==i)]['var36'].count(), round((train_df[(train_df.TARGET==1) & (train_df.var36==i)]['var36'].count()/train_df[(train_df.var36==i)]['var36'].count())*100,2),round((train_df[(train_df.TARGET==1) & (train_df.var36==i)]['var36'].count()/train_df[(train_df.TARGET==1)]['var36'].count())*100,2)] for i in (train_df.var36.unique())]

var36_df=pd.DataFrame(var36_cat,columns=['Cat','Total_Customer','Unhappy_Customer','Unhappy_Percent_All','Percent_Unhappy']).set_index('Cat')
plt.figure(figsize = (50,50))

plt.xticks(rotation=90)



sns.countplot(train_df['var36'],hue=train_df['TARGET'])
var36_df
fig,ax=plt.subplots()

sns.kdeplot(train_df[train_df.TARGET==1]['var36'],bw=0.05,ax=ax,shade=True,label='Unhappy Customers')

sns.kdeplot(train_df[train_df.TARGET==0]['var36'],bw=0.05,ax=ax,label='Happy Customers')

plt.show()
fig,ax=plt.subplots()

sns.violinplot(x='var36',y='var15',hue='TARGET',data=train_df,ax=ax)

fig.set_size_inches(15,15)

plt.show()
fig,ax=plt.subplots()

sns.violinplot(train_df['var36'],train_df['num_var4'])

fig.set_size_inches(15,15)

plt.show()
fig,ax=plt.subplots()

sns.violinplot(train_df['var36'],train_df['num_var4'],hue=train_df.TARGET)

fig.set_size_inches(15,15)

plt.show()
def var36_Count(train_df,var2,val,var,tar):

    l=[]

    for i in (np.unique(train_df[var])):

        l.append(train_df[train_df[var2]==val][train_df.TARGET==tar][train_df[var]==i].shape[0])

    return l

plt.figure(figsize=(10,10))

col0=var36_Count(train_df,'num_var4',0,'var36',1)

col1=var36_Count(train_df,'num_var4',1,'var36',1)

col2=var36_Count(train_df,'num_var4',2,'var36',1)

col3=var36_Count(train_df,'num_var4',3,'var36',1)

col4=var36_Count(train_df,'num_var4',4,'var36',1)

col5=var36_Count(train_df,'num_var4',5,'var36',1)

col6=var36_Count(train_df,'num_var4',6,'var36',1)

col7=var36_Count(train_df,'num_var4',7,'var36',1)



indx=np.arange(4)

indy=np.arange(0,2501,100)



p0=plt.bar(indx,col0)

p1=plt.bar(indx,col1,bottom=col0)

p2=plt.bar(indx,col2,bottom=[col0[j] + col1[j] for j in range(len(col0))])

p3=plt.bar(indx,col3,bottom=[col0[j] + col1[j] + col2[j] for j in range(len(col0))])

p4=plt.bar(indx,col4,bottom=[col0[j] + col1[j] + col2[j] + col3[j] for j in range(len(col0))])

p5=plt.bar(indx,col5,bottom=[col0[j] + col1[j] + col2[j] + col3[j] + col4[j] for j in range(len(col0))])

p6=plt.bar(indx,col6,bottom=[col0[j] + col1[j] + col2[j] + col3[j] + col4[j] + col5[j] for j in range(len(col0))])

p7=plt.bar(indx,col7,bottom=[col0[j] + col1[j] + col2[j] + col3[j] + col4[j] + col5[j] + col6[j] for j in range(len(col0))])



plt.xticks(indx, ('0', '1', '2', '3'))

plt.yticks(indy)





plt.legend((p0[0], p1[0], p2[0],p3[0], p4[0], p5[0], p6[0], p7[0]), ('P0', 'P1','P2','P3','P4','P5','P6','P7'))



plt.show()
plt.figure(figsize=(10,10))

col0=var36_Count(train_df,'num_var4',0,'var36',0)

col1=var36_Count(train_df,'num_var4',1,'var36',0)

col2=var36_Count(train_df,'num_var4',2,'var36',0)

col3=var36_Count(train_df,'num_var4',3,'var36',0)

col4=var36_Count(train_df,'num_var4',4,'var36',0)

col5=var36_Count(train_df,'num_var4',5,'var36',0)

col6=var36_Count(train_df,'num_var4',6,'var36',0)

col7=var36_Count(train_df,'num_var4',7,'var36',0)



indx=np.arange(4)

indy=np.arange(0,50001,2000)



p0=plt.bar(indx,col0)

p1=plt.bar(indx,col1,bottom=col0)

p2=plt.bar(indx,col2,bottom=[col0[j] + col1[j] for j in range(len(col0))])

p3=plt.bar(indx,col3,bottom=[col0[j] + col1[j] + col2[j] for j in range(len(col0))])

p4=plt.bar(indx,col4,bottom=[col0[j] + col1[j] + col2[j] + col3[j] for j in range(len(col0))])

p5=plt.bar(indx,col5,bottom=[col0[j] + col1[j] + col2[j] + col3[j] + col4[j] for j in range(len(col0))])

p6=plt.bar(indx,col6,bottom=[col0[j] + col1[j] + col2[j] + col3[j] + col4[j] + col5[j] for j in range(len(col0))])

p7=plt.bar(indx,col7,bottom=[col0[j] + col1[j] + col2[j] + col3[j] + col4[j] + col5[j] + col6[j] for j in range(len(col0))])



plt.xticks(indx, ('0', '1', '2', '3'))

plt.yticks(indy)





plt.legend((p0[0], p1[0], p2[0],p3[0], p4[0], p5[0], p6[0], p7[0]), ('P0', 'P1','P2','P3','P4','P5','P6','P7'))



plt.show()
ldf[ldf.Features=='var21']
train_df[train_df.var21==900][train_df.TARGET==1].shape
plt.figure(figsize = (50,10))

plt.xticks(rotation=90)



sns.countplot(train_df['var21'],hue=train_df['TARGET'])
plt.figure(figsize = (50,10))

plt.xticks(rotation=90)



sns.countplot(train_df[train_df.var21!=0]['var21'],hue=train_df['TARGET'])
ldf[ldf.Features=='var15']
ldf[ldf.Features=='var38']
plt.figure(figsize = (50,50))

plt.xticks(rotation=90)

plt.axhline(train_df['var38'][train_df.TARGET==0].mean(),label='Happy Mean - %f'%train_df['var38'][train_df.TARGET==0].mean())

plt.axhline(train_df['var38'][train_df.TARGET==1].mean(),color='#FF7F00',label='Unhappy Mean - %f'%train_df['var38'][train_df.TARGET==1].mean())

plt.axhline(train_df['var38'].mean(),color='r',label='Overall Mean - %f'%train_df['var38'].mean())





sns.lineplot(y=train_df['var38'],x=train_df['var15'],hue=train_df['TARGET'])
plt.figure(figsize = (10,10))

plt.xticks(rotation=90)

#plt.axhline()



sns.barplot(y=train_df['var38'],x=train_df['num_var4'])
plt.figure(figsize = (10,10))

plt.xticks(rotation=90)

#plt.axhline()



sns.barplot(y=train_df['var38'],x=train_df['var36'])
#plt.axhline(train_df[train_df['TARGET']==0]['var38'].mean(),label='Mean - Happy %s'%train_df[train_df['TARGET']==0]['var38'].mean(),color='b')

#plt.axhline(train_df[train_df['TARGET']==1]['var38'].mean(),label='Mean - Unhappy %s'%train_df[train_df['TARGET']==1]['var38'].mean(),color='#FF7F00')

#plt.axhline(train_df['var38'].mean(),label='Mean - Unhappy %s'%train_df['var38'].mean(),color='r')

fig, ax = plt.subplots()

sns.distplot(np.log(train_df[train_df.TARGET==1]['var38']),hist_kws={"histtype": "step", "linewidth": 3},kde_kws={'label':'Unhappy Customers',"linewidth": 3},color='r')

#sns.distplot(np.log(train_df[train_df.TARGET==0]['var38']),norm_hist=True,ax=ax,label ='Happy Customers')

sns.distplot(np.log(train_df['var38']),hist_kws={"histtype": "step", "linewidth": 3},kde_kws={'label':'All Customers',"linewidth": 3},ax=ax,label ='All Customers')

fig=plt.gcf()

fig.set_size_inches(20,20)

plt.show()