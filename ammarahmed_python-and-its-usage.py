# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

path1='../input/python developer survey 2018 external sharing/Python Developer Survey 2018 external sharing/python_psf_external_18.csv'

path2='../input/python developers survey 2017_ raw data sharing/Python Developers Survey 2017_ Raw Data Sharing/pythondevsurvey2017_raw_data.csv'



df_2017=pd.read_csv(path1)

df_2018=pd.read_csv(path2)

def plot_onepie(cols,col_num):

    labels=[]

    values=[]

    title=cols[col_num]

    for a in df_2017.iloc[:,col_num].unique():

        if pd.isna(a):

            percentage=df_2017.iloc[:,col_num].isna().sum() * 100 / df_2017.iloc[:,col_num].count()

        else:

            percentage=(df_2017.iloc[:,col_num]==a).sum() * 100 / df_2017.iloc[:,col_num].count()

        

        labels.append(a)

        values.append(percentage)

        #print(cols[1],a,percentage)

    

    fig1, ax1 = plt.subplots()

    ax1.pie(values, labels=labels, autopct='%1.1f%%',

            shadow=True, startangle=90)

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    ax1.set_title(title)  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()



# Any results you write to the current directory are saved as output.
df_2017.head(1)

#print(len(df_2017.columns))#284

cols=[]

cols=list(df_2017.columns)

print(df_2017.iloc[:,1].unique())

plot_onepie(cols,1)
plot_onepie(cols,26)
ques_bank_index=pd.DataFrame({'Question':cols , 'index':[a for a in range(len(cols))]} )

ques_bank_index.head(5)
ques_bank_index
plot_onepie(cols,282)
count_dict={}

lang=''

lang_list=[]

val_list=[]

#print(df_2017.iloc[:,a].isna().count())

for a in range(3,24):

    lang=cols[a].split(':')[0]

    #print(lang)

    val=(df_2017.iloc[:,a]==lang).sum()

    lang_list.append(lang)

    val_list.append(val)

#print(count_dict.keys(),[count_dict[a] for a in count_dict.keys()])

count_dict['language']=lang_list

count_dict['value']=val_list

df_lang_count=pd.DataFrame(count_dict)

df_lang_count=df_lang_count.sort_values(by='value',ascending=False)

df_lang_count