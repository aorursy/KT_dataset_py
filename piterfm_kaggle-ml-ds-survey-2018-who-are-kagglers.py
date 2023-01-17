import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import os
import re

print('Сontent of output directory:')
print(os.listdir("../input"))
df = pd.read_csv('../input/multipleChoiceResponses.csv', skiprows=(1,1))
df.head()
df.shape
df.info()
df.columns
output_col = [i for i in list(df.columns) if re.search('Q\d{1,2}$', i)]
len(output_col)
df = df[output_col]
df.columns
dcol = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False).iloc[:1]
dcol = dcol[output_col]
dcol
def barplot_top20(data, col, v_cnt, xlabel, ylabel, fs, top, title=''):
    plt.figure(figsize=(10,10))
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    ax = sns.countplot(y=col, data=data, order=v_cnt.iloc[:top].index)
    plt.title(title, fontsize=fs)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)
    sns.despine()
    rel_size='small' if top > 10 else 'large'
    
    new_ytickslabel = []
    for i in ax.get_yticklabels():
        new_ytickslabel.append('{} ...'.format(i.get_text()[:45]))  if len(i.get_text())>50 else new_ytickslabel.append(i)
    ax.set_yticklabels(new_ytickslabel)
    
    for patch, value in zip(ax.patches, v_cnt):  
        ax.text(patch.get_width() + 100, patch.get_y() + (patch.get_bbox().y1-patch.get_y())/2,
            value,
            ha="left", va='center',
            fontsize=rel_size)
xlabel='# of Respondents'
ylabel=''
fntsz=20
top=20
print(dcol.iloc[0].Q1)
col='Q1'
v_cnt=df.Q1.value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
print(dcol.iloc[0].Q2)
col='Q2'
v_cnt=df.Q2.value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
country_dict = {'United States of America':'USA', 
                'United Kingdom of Great Britain and Northern Ireland':'UK',
                'I do not wish to disclose my location':'Hidden Loction',
                'Iran, Islamic Republic of...':'Iran'}
df.Q3 = df.Q3.replace(country_dict)
column = 'Q3'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q4'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q5'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q5'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q6'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q7'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q8'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q9'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q10'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q17'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q18'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q20'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q22'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q23'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q24'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q25'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q26'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q32'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q37'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q40'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q43'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q46'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
column = 'Q48'
print(dcol.iloc[0][column])
col=column
v_cnt=df[column].value_counts()
barplot_top20(data=df, col=col, v_cnt=v_cnt, xlabel=xlabel, ylabel=ylabel, fs=fntsz, top=top)
