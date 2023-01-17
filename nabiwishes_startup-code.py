# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
SEED = 123
import warnings
warnings.filterwarnings('ignore')
import random
#from sklearn.externals import joblib
import qgrid
def gview(data):
    return(qgrid.show_grid(data,show_toolbar=True,grid_options={'forceFitColumns': False,'highlightSelectedCell': True,'highlightSelectedRow': True}))
from time import time
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

random.seed(123)
# -----------FUNCTIONS ------------
#%run start_up.py


curr_dir = os.getcwd()
print("current directory:\n\t\t",curr_dir)

# os.chdir(curr_dir)

# folder_list=['input_data','temp_data','submission']

# for iin in folder_list:
#     if not os.path.exists(f'{iin}'):
#         os.makedirs(f'{iin}')   
    
# data_locn = 'C:/Users/nshaikh2/Desktop/live/full_data/ml_data'

from datetime import datetime
t_stamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SETTING BASIC ENVIORMENT :

# LOADING LIBRARY ---

import qgrid
import os
import pandas as pd 
import numpy as np 
import scipy.stats as stats
# import qgrid
from scipy.stats import kurtosis, skew
import seaborn as sns
import matplotlib.pyplot as plt

# import pyarrow.parquet as pq
# import pyarrow as pa
import pandas as pd
import numpy as np
from scipy import interp

# from dplython import select, DplyFrame, X, arrange, count, sift, head, summarize, group_by, tail, mutate

from sklearn.model_selection import validation_curve
import sklearn.metrics as metrics

from sklearn.metrics import auc, accuracy_score
from sklearn.metrics  import plot_roc_curve
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold

from catboost import CatBoostClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from itertools import product, chain
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report


def gview(data):
    print('DIMENSION',data.shape)
    return(qgrid.show_grid(data,show_toolbar=True,grid_options={'forceFitColumns': False,'highlightSelectedCell': True,
        'highlightSelectedRow': True}))

def trim_all_columns(df):
        trim_strings = lambda x: x.strip() if isinstance(x, str) else x
        return df.applymap(trim_strings)


# Basic data cleaning
def data_basic_clean(fsh):
        fsh.columns = [c.strip() for c in fsh.columns]
        fsh.columns = [c.replace(' ', '_') for c in fsh.columns]
        fsh.columns = map(str.lower, fsh.columns)
        fsh.replace(['None','nan','Nan',' ','NaT','#REF!'],np.nan,inplace=True)
        fsh = trim_all_columns(fsh)
        fsh=fsh.drop_duplicates(keep='last')
        df = pd.DataFrame(fsh)
        return(df)

def call_napercentage(data_train):
    op = pd.DataFrame(data_train.isnull().sum()/data_train.shape[0]*100)
    op = op.reset_index()
    op.rename(columns={'index':'variable_name'},inplace=True)
    op.rename(columns={0:'na%'},inplace=True)
    op=op.sort_values(by='na%',ascending=False)
    return(op)

def get_numcolnames(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdf = df.select_dtypes(include=numerics)
    cols_nums =numdf.columns
    cols_nums = cols_nums.tolist()
    return(cols_nums)

def get_catcolnames(df):
    categoric = ['object']
    catdf = df.select_dtypes(include=categoric)
    cols_cat = catdf.columns
    cols_cat = cols_cat.tolist()
    return(cols_cat)
    
    
def get_partialcol_match(final_df,txt):
    date_colist = final_df[final_df.columns[final_df.columns.to_series().str.contains(f'{txt}')]].columns
    date_colist = date_colist.tolist()
    return(date_colist)  
    
    
def summarise_yourdf(df,leveli):
    print(f"Dataset Shape: {df.shape}")    

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing_Count'] = df.isnull().sum().values
    summary['Missing_Perct'] = round(summary['Missing_Count']/df.shape[0]*100,2)
    summary['Uniques_Count'] = df.nunique().values
    summary['Uniques_Perct'] = round(summary['Uniques_Count']/df.shape[0]*100,2)
    
    #summary['First Value'] = df.loc[0].values
    #summary['Second Value'] = df.loc[1].values
    #summary['Third Value'] = df.loc[2].values
    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 
    summary['Zeros_count'] = df[df == 0].count(axis=0).values
    summary['Zeros_Perct'] = round(summary['Zeros_count']/df.shape[0]*100,2)
    
    
    summary['Levels']= 'empty'
    for i, m in enumerate (summary['Name']):
            #print(i,m)
            if len(df[f'{m}'].value_counts()) <= leveli:
                #print(df[f'{m}'].value_counts())
                tab = df[f'{m}'].unique()
                summary.ix[i,'Levels']=f'{tab}'
    summary['N'] = df.shape[0]
    
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdf = df.select_dtypes(include=numerics)
    cols_nums =numdf.columns
    categoric = ['object']
    catdf = df.select_dtypes(include=categoric)
    cols_cat = catdf.columns#names of all the columns
    

    desc=df[cols_nums].describe().T
    desc = desc.reset_index()
    desc = pd.DataFrame(desc)
    desc.rename(columns={f'{desc.columns[0]}':'Name'}, inplace=True)
    desc.drop(['count'], axis=1,inplace=True)
    desc = round(desc,2)
   # desc

    merged_inner=pd.merge(summary, desc, on='Name', how='outer')
    merged_inner = merged_inner.replace(np.nan, '', regex=True)
    merged_inner = merged_inner.sort_values('Missing_Perct',ascending=False)
    merged_inner.to_excel('profiling.xlsx',index=False)
    return merged_inner
    
    
def get_dataprofile(data,leveli):
    print('DATA SHAPE',data.shape)
    tempi = pd.DataFrame()
    tempi = pd.DataFrame(data.dtypes,columns=['dtypes'])
    tempi = tempi.reset_index()
    tempi['Name'] = tempi['index']
    tempi = tempi[['Name','dtypes']]
    tempi['Missing_Count'] = data.isnull().sum().values
    tempi['Missing_Perct'] = round(tempi['Missing_Count']/data.shape[0]*100,2)
    tempi['Uniques_Count'] = data.nunique().values
    tempi['Uniques_Perct'] = round(tempi['Uniques_Count']/data.shape[0]*100,2)

    tempi['Zeros_count'] = data[data == 0].count(axis=0).values
    tempi['Zeros_Perct'] = round(tempi['Zeros_count']/data.shape[0]*100,2)

    tempi['Ones_count'] = data[data == 1].count(axis=0).values
    tempi['Ones_Perct'] = round(tempi['Ones_count']/data.shape[0]*100,2)

    tempi['mcp'] = np.nan

    def mode_perC(data,coli):
        #i =  'status_6'
        xi = data[f'{i}'].value_counts(dropna=False)
        xi = pd.DataFrame(xi)
        xi.reset_index(inplace=True)
        xi.rename(columns= {'index':'colanme',0:f'{i}'},inplace=True)
        xi.sort_values(by=f'{i}')
        mode_name = xi.iloc[0,0]
        mode_count = xi.iloc[0,1]
        mode_perC = round((mode_count/data.shape[0])*100,3)
        m = f'{mode_name}/ {mode_count}/ %{mode_perC}'
        return m
    

# Computing MCp
    for i in tempi['Name'].unique():
        #print(mode_perC(data,f'{i}'))
        idi = tempi[tempi['Name'] == f'{i}'].index
        tempi.ix[idi,'mcp'] = mode_perC(data,f'{i}')
        
# Computing Levels
    
    tempi['Levels']= 'empty'
    for i, m in enumerate (tempi['Name']):
            #print(i,m)
            if len(data[f'{m}'].value_counts()) <= leveli:
                #print(data[f'{m}'].value_counts())
                tab = data[f'{m}'].unique()
                tempi.ix[i,'Levels']=f'{tab}'
    tempi['N'] = data.shape[0]
    
    
    
    # Numerical describe func
    
    num_cols =get_numcolnames(data)


    di =data[num_cols].describe().T
    di.reset_index(inplace=True)
    di.rename(columns={'index':'Name'},inplace=True)
    
    
    ret_df =pd.merge(tempi, di, on='Name', how='outer')
    ret_df = ret_df.replace(np.nan, '', regex=True)
    ret_df = ret_df.sort_values('Missing_Perct',ascending=False)
    
    
    ret_df = ret_df.round(3)
    print('-'*50)
    print('DATA TYPES:\n',tempi['dtypes'].value_counts(normalize=False))
    
    
    a = call_napercentage(data)
    miss_col_great30 = a[a['na%']>30].shape[0]
    miss_col_less30 = a[a['na%'] <=30].shape[0]
    miss_col_equal2_0 = a[a['na%'] ==0].shape[0]
    miss_col_equal2_100 = a[a['na%'] ==100].shape[0]
    print('\nMISSING REPORT:-')
    print('-'*100,
           '\nTotal Observation                       :',tempi.shape[0],
          '\nNo of Columns with >30% data missing    :',miss_col_great30,
          '\nNo of Columns with <30% data missing    :',miss_col_less30,

          '\nNo of Columns with =0% data missing     :',miss_col_equal2_0,
         '\nNo of Columns with =100% data missing   :',miss_col_equal2_100,'\n','-'*100)
    
    ret_df.to_excel('profile.xlsx',index=False)  

    return(ret_df)


# outlier treatment ----

def get_quatile_df(data):
    return(pd.DataFrame.quantile(data,[0,0.01,0.02,0.03,0.04,.05,0.25,0.50,0.75,0.85,.95,0.99,0.991,0.992,0.993,0.994,0.995,0.995,0.996,0.997,0.998,0.8999]))

def get_box_plots(data):
    for coli in get_numcolnames(data):
        plt.figure(figsize=(15,9))
        plt.boxplot(data[f'{coli}'],0,'gD')
        plt.title(f'{coli}')
        plt.show()
        
def get_outliers(df,l_band , u_band):
    Q1 = df.quantile(l_band)
    Q3 = df.quantile(u_band)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return(df)

def get_binary_distribution(data,target):
    f,ax=plt.subplots(1,2,figsize=(15,6))
    data[f'{target}'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
    ax[0].set_title(f'{target}')
    ax[0].set_ylabel('')
    sns.countplot(f'{target}',data=data,ax=ax[1])
    ax[1].set_title(f'{target}')
    plt.show()
    
def get_plot_catVs_target(df,cat_col,y_lab,Top_n,thres,ytag,prt):
    i = cat_col#"Age_Of_Vehicle"
    y2 = y_lab#"Renewed2"
    Top_n = Top_n #15
    ytag = ytag
    col_count  = df[f'{i}'].value_counts()
    #print(col_count)
    col_count = col_count[:Top_n,]

    col_count1 = df[f'{i}'].value_counts(normalize=True)*100
    col_count1 = col_count1[:Top_n,]
    vol_inperc = col_count1.sum()
    vol_inperc = round(vol_inperc,2)

    tmp = pd.crosstab(df[f'{i}'], df[f'{y2}'], normalize='index') * 100
    tmp = pd.merge(col_count, tmp, left_index=True, right_index=True)
    tmp.rename(columns={0:'NotRenwed%', 1:'Renewed%'}, inplace=True)
    if 'NotRenwed%' not in tmp.columns:
        print("NotRenwed% is not present in ",i)
        tmp['NotRenwed%'] = 0
    if 'Renewed%' not in tmp.columns:
        print("Renewed% is not present in ",i)
        tmp['Renewed%'] = 0

    tmp1 = pd.crosstab(df[f'{i}'], df[f'{y2}'])
    tmp1.rename(columns={0:'NR_count', 1:'R_count'}, inplace=True)
    if 'NR_count' not in tmp1.columns:
        print("NR_count is not present in ",i)
        tmp1['NR_count'] = 0
    if 'R_count' not in tmp1.columns:
        print("R_count is not present in ",i)
        tmp1['R_count'] = 0

    tmpz=pd.merge(tmp,tmp1,
        left_index=True,
        right_index=True)
    tmpz['Tot'] = tmpz['NR_count'] + tmpz['R_count'] 
    tmpz['Renewed%'] =  round(tmpz['Renewed%'],2)
    tmpz['Mean'] = (tmpz['R_count']/tmpz['Tot'])*100
    tmpz['Nperformer'] = np.where(tmpz['Renewed%'] < tmpz['Mean'] ,1,0)
    #tmpz.sort_index(inplace=True)
    if prt == 'Y':
        gview(tmpz)
        tmpzi = tmpz.reset_index()  
        #tmpzii = tmpzi .join(pd.DataFrame(tmpzi.index.str.split('-').tolist()))
        #tmpzi = pd.concat([tmpzi,DataFrame(tmpzi.index.tolist())], axis=1, join='outer')
        #tmpzi.to_excel("tmpz.xlsx")
      
   
    plt.figure(figsize=(16,7))
    g=sns.barplot(tmpz.index, tmpz[f'{i}'], alpha=0.8,order = col_count.index)
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.title(f'{i} with {vol_inperc}%', fontsize = 16,color='blue')
    #g.set_title(f'{i}')
    plt.ylabel('Volume', fontsize=12)
    plt.xlabel(f'{i}', fontsize=12)
    plt.xticks(rotation=90)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{}\n{:1.2f}%'.format(round(height),height/len(df)*100),
            ha="center", fontsize=10, color='blue')

    gt = g.twinx()

    if ytag == 1:
        values = tmpz['Renewed%'].values # <--- store the values in a variable for easy access
        gt = sns.pointplot(x=tmpz.index, y='Renewed%', data=tmpz, color='green', legend=True,order=tmpz.index)
    if thres is np.nan:
        gt.set_ylim(0,100)
   

    if ytag == 0:
        values = tmpz['NotRenwed%'].values # <--- store the values in a variable for easy access
        gt = sns.pointplot(x=tmpz.index, y='NotRenwed%', data=tmpz, color='red', legend=True,order=tmpz.index)
        #gt.set_ylim(tmp['NotRenwed%'].min()-1,tmp['NotRenwed%'].max()+5)
    if thres is np.nan:
        gt.set_ylim(0,100)
        
        
        
    if thres is np.nan and ytag ==1: 
        gt.axhline(y=(tmpz['R_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    else:
        gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
        
    if thres is np.nan and ytag ==0: 
        gt.axhline(y=(tmpz['NR_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    else:
        gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
        
    #values = tmpz['Renewed%'].values # <--- store the values in a variable for easy access
    j=0
    for c in gt.collections:
        for i, of in zip(range(len(c.get_offsets())), c.get_offsets()):
            gt.annotate(values[j], of, color='brown', fontsize=12, rotation=45)
            j += 1
    plt.show()
    
    
def get_plot_ordVs_target(df,cat_col,y_lab,Top_n,thres,ytag,prt):
    i = cat_col#"Age_Of_Vehicle"
    y2 = y_lab#"Renewed2"
    Top_n = Top_n #15
    ytag = ytag
    
    #df[f'{i}'] = df[f'{i}'].astype(int)
    col_count  = df[f'{i}'].value_counts()
    #print(col_count)
    col_count.sort_index(inplace=True)
    col_count = col_count[:Top_n,]

    col_count1 = df[f'{i}'].value_counts(normalize=True)*100
    col_count1 = col_count1[:Top_n,]
    vol_inperc = col_count1.sum()
    vol_inperc = round(vol_inperc,2)

    tmp = pd.crosstab(df[f'{i}'], df[f'{y2}'], normalize='index') * 100
    tmp = pd.merge(col_count, tmp, left_index=True, right_index=True)
    tmp.rename(columns={0:'NotRenwed%', 1:'Renewed%'}, inplace=True)
    if 'NotRenwed%' not in tmp.columns:
        print("NotRenwed% is not present in ",i)
        tmp['NotRenwed%'] = 0
    if 'Renewed%' not in tmp.columns:
        print("Renewed% is not present in ",i)
        tmp['Renewed%'] = 0

    tmp1 = pd.crosstab(df[f'{i}'], df[f'{y2}'])
    tmp1.rename(columns={0:'NR_count', 1:'R_count'}, inplace=True)
    if 'NR_count' not in tmp1.columns:
        print("NR_count is not present in ",i)
        tmp1['NR_count'] = 0
    if 'R_count' not in tmp1.columns:
        print("R_count is not present in ",i)
        tmp1['R_count'] = 0

    tmpz=pd.merge(tmp,tmp1,
        left_index=True,
        right_index=True)
    tmpz['Tot'] = tmpz['NR_count'] + tmpz['R_count'] 
    tmpz['Renewed%'] =  round(tmpz['Renewed%'],2)
    tmpz['Mean'] = tmpz['Renewed%'].mean()
    tmpz['Nperformer'] = np.where(tmpz['Renewed%'] < tmpz['Mean'] ,1,0)
    #tmpz.sort_index(inplace=True)
    
    
    if prt == 'Y':
        print(tmpz)
        tmpzi = tmpz.reset_index()  
        #tmpzii = tmpzi .join(pd.DataFrame(tmpzi.index.str.split('-').tolist()))
        #tmpzi = pd.concat([tmpzi,DataFrame(tmpzi.index.tolist())], axis=1, join='outer')
        #tmpzi.to_excel("tmpz.xlsx")
      
   
    plt.figure(figsize=(16,7))
    g=sns.barplot(tmpz.index, tmpz[f'{i}'], alpha=0.8)
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.title(f'{i} with {vol_inperc}%', fontsize = 16,color='blue')
    #g.set_title(f'{i}')
    plt.ylabel('Volume', fontsize=12)
    plt.xlabel(f'{i}', fontsize=12)
    plt.xticks(rotation=90)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{}\n{:1.2f}%'.format(round(height),height/len(df)*100),
            ha="center", fontsize=10, color='blue')

    gt = g.twinx()

    if ytag == 1:
        values = tmpz['Renewed%'].values # <--- store the values in a variable for easy access
        gt = sns.pointplot(x=tmpz.index, y='Renewed%', data=tmpz, color='green', legend=True)
    
    if ytag == 0:
        values = tmpz['NotRenwed%'].values # <--- store the values in a variable for easy access
        gt = sns.pointplot(x=tmpz.index, y='NotRenwed%', data=tmpz, color='red', legend=True)
        gt.set_ylim(tmp['NotRenwed%'].min()-1,tmp['NotRenwed%'].max()+5)
        
        
    if thres is np.nan:
        gt.set_ylim(0,100)
        gt.axhline(y=(tmpz['R_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    else:
        gt.set_ylim(0,100)
        gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
   


        
        
    if thres is np.nan and ytag ==1: 
        gt.axhline(y=(tmpz['R_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    else:
        gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
        
    if thres is np.nan and ytag ==0: 
        gt.axhline(y=(tmpz['NR_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    else:
        gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
        
    #values = tmpz['Renewed%'].values # <--- store the values in a variable for easy access
    j=0
    for c in gt.collections:
        for i, of in zip(range(len(c.get_offsets())), c.get_offsets()):
            gt.annotate(values[j], of, color='brown', fontsize=12, rotation=45)
            j += 1
    plt.show()
    
from statsmodels.stats.outliers_influence import variance_inflation_factor    

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

# Confusion Matrix function

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def get_distplot(df,icolumns):
    plt.figure(figsize=(12,10))
    sns.distplot(df[f'{icolumns}'], color='g', label = "close%") 
    plt.xlabel(f"{icolumns}")
    plt.ylabel("Frequency")
    plt.title(f"Distribution {icolumns}", fontsize=14)
    plt.legend()
    plt.show()
data_locn = '/kaggle/input/stock-market-next-day-close-change-prediction/train_24_june_2020.csv'
df=pd.read_csv(f'{data_locn}')
df.head()
df.columns
id_col = 'tday_stock_name'
y_name = 'target'
x_name = [i for i in df.columns if i not in [f'{id_col}',f'{y_name}','close_diff%','close_diff','close_prev%']]
x =  df[x_name]
y =  df[y_name]
x_tr , x_te , y_tr , y_te = train_test_split(x , y , 
                                               test_size = 0.30,stratify = y,
                                               random_state =  SEED )
cat_features_names = get_catcolnames(x_tr) # here we specify names of categorical features
cat_features = [x_tr.columns.get_loc(col) for col in cat_features_names]
print(cat_features)
params = {'loss_function':'MultiClass', # objective function
          'eval_metric':'TotalF1', # metric
          'cat_features': cat_features,
          'early_stopping_rounds': 500,
          'verbose': 200, # output to stdout info about training process every 200 iterations
          'random_seed': SEED,
          'iterations': 1000
         }
cbc_1 = CatBoostClassifier(**params)
cbc_1.fit(x_tr, y_tr, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)
          eval_set=(x_te, y_te), # data to validate on
          use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score
          plot=True # True for visualization of the training process (it is not shown in a published kernel - try executing this code)
         );
feat_importances = pd.Series(cbc_1.feature_importances_, index=x_tr.columns)
feat_importances.nlargest(30).sort_values(ascending = True).plot(kind='barh',figsize=(15,7))
vi = pd.DataFrame(feat_importances.sort_values(ascending=False)).reset_index()
vi.rename(columns= {'index':'varname',0:'score'},inplace=True)
vi
ilist = vi[0:20]['varname'].tolist()
len(ilist)
ilist
x =  df[ilist]
y =  df[y_name]
x_tr , x_te , y_tr , y_te = train_test_split(x , y , 
                                               test_size = 0.30,stratify = y,
                                               random_state =  SEED )
cat_features_names = get_catcolnames(x_tr) # here we specify names of categorical features
cat_features = [x_tr.columns.get_loc(col) for col in cat_features_names]
print(cat_features)
params = {'loss_function':'MultiClass', # objective function
          'eval_metric':'TotalF1', # metric
          'cat_features': cat_features,
          'early_stopping_rounds': 500,
          'verbose': 200, # output to stdout info about training process every 200 iterations
          'random_seed': SEED,
          'iterations': 2000
         }
cbc_2 = CatBoostClassifier(**params)
cbc_2.fit(x_tr, y_tr, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)
          eval_set=(x_te, y_te), # data to validate on
          use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score
          plot=True # True for visualization of the training process (it is not shown in a published kernel - try executing this code)
         );
feat_importances = pd.Series(cbc_2.feature_importances_, index=x_tr.columns)
feat_importances.nlargest(30).sort_values(ascending = True).plot(kind='barh',figsize=(15,7))
y_pred_val = cbc_2.predict(x_te)
y_pred_test = cbc_2.predict(x_te)
y_prob_test = cbc_2.predict_proba(x_te)
y_prob_test
y_pred_test
