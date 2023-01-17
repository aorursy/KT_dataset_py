%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import zipfile

%matplotlib inline 

zf = zipfile.ZipFile('../input/train.csv.zip')
df = pd.read_csv(zf.open('train.csv'))

sns.set_style('whitegrid')
sns.despine()

df['Dates'] = df['Dates'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df['Months'] = df['Dates'].apply(lambda x: dt.datetime(x.year, x.month, 1) )
def crime_res(df, n, cat, d):
    df_c2 = df[['Category', 'Months', 'Resolution']][df['PdDistrict'] == d]
    df_c3 = df_c2[df_c2['Resolution'] != 'NONE']
    df_c2 = df_c2.groupby('Months').size()
    df_c3 = df_c3.groupby('Months').size()    
    
    title = "Top %d %s CRIMES (%s) per Month" %(n, d, cat)
    label1, label2= 'crimes', 'resolved'
    
    return title, label1, label2, df_c2, df_c3

def top_c_r(df, n):
    
    lyst = df['PdDistrict'].unique().tolist()
    d_lyst = len(lyst)
    tot_sp = d_lyst * n
    
    fig, axarr = plt.subplots(nrows=tot_sp, ncols=1, sharex=True, figsize=(12,tot_sp*2))
    
    for dd, d in enumerate(lyst):
        
        top_n_cats = df[df['PdDistrict'] == d]
        top_n_cats = top_n_cats.groupby('Category').size().order(ascending=False).index[:n].tolist()
        
        for ii, cat in enumerate(top_n_cats):
            
            title, label1, label2, data1, data2 = crime_res(df, n, cat, d)
                        
            axarr[ii+(dd*n)].plot(data1, color='k', alpha=.3, label=label1)
            axarr[ii+(dd*n)].plot(data2, color='b', alpha=.5, label=label2)
            axarr[ii+(dd*n)].set_title(title)
            axarr[ii+(dd*n)].legend(loc=1)
            axarr[ii+(dd*n)].set_ylabel('Crimes')
    
    fig.show()
c = 1
print("Top %d Categories of Crimes by District" % c)
top_c_r(df, c)
def crime_res(df, cat, d):
    df2 = df[['Category', 'Months']][df['PdDistrict'] == d]
    df2 = df2[df2['Category'] == cat]
    dfg = df2.groupby('Months').size()
    return dfg    

def top_c_r(df, n):
    li = df['PdDistrict'].unique().tolist()
    li_len = len(li)
    
    f, ax = plt.subplots(nrows=li_len, sharex=True, figsize=(14,li_len*2))
    
    for ii,d in enumerate(li):
        top_n_cats = df[df['PdDistrict'] == d]
        top_n_cats = top_n_cats.groupby('Category').size().order(ascending=False).index[:n].tolist()
        
        for cat in top_n_cats[:n]:
            ax[ii].plot(crime_res(df, cat,d), alpha=.5, label=cat)
            ax[ii].set_ylabel("Crimes")
        ax[ii].set_title("Crimes per Month in %s" % d)
        ax[ii].legend(bbox_to_anchor=[1.2,1])
        
    f.show()
top_c_r(df, 6)