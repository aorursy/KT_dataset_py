import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
l_cat = ['Survived', 'Pclass','Sex','SibSp','Parch','Cabin', 'Embarked']

df_train[l_cat].nunique()
df_train['Cabin'].unique()
df_train.isnull().sum(axis=0)
def bubblechart(ip,op,count_col,data):

    f, ax= plt.subplots(len(ip),1, figsize=(10,5*len(ip)), squeeze=False)

    

    # Find unique output values 

    y_tick_label = pd.Series(data[op[0]].unique()).sort_values().tolist()

    y_tick_value = pd.Series(np.arange(1,len(y_tick_label)+ 1,1)).tolist()

    

    for i,ip_param in enumerate(ip):

        

        # Find unique input values

        x_tick_label = pd.Series(data[ip_param].unique()).sort_values().tolist()

        x_tick_value = pd.Series(np.arange(1,len(x_tick_label)+ 1,1)).tolist()

         

        #Calculate parameters required for bubblechart

        gr_ip_op = data.groupby([ip_param]+op)[count_col].count()

        gr_ip = data.groupby([ip_param])[count_col].count()

        df1 = pd.DataFrame(gr_ip_op).reset_index()

        df2 = pd.DataFrame(gr_ip).reset_index()

        df3 = df1.merge(df2,how='inner',on=ip_param)

        df3['proportion'] = df3[count_col+'_x']/df3[count_col+'_y']*3000.0

        df3['x_tick_value'] = df3[ip_param].apply(lambda x : x_tick_value[x_tick_label.index(x)] )

        df3['y_tick_value'] = df3[op[0]].apply(lambda x : y_tick_value[y_tick_label.index(x)] )

        

        #Plot the bubblechart

        x=df3['x_tick_value']

        y=df3['y_tick_value']

        s=df3['proportion']

        c=df3['proportion']/3000.0

        ax[i,0].scatter(x=x,y=y,s=s,c=c, cmap='RdYlGn')

        ax[i,0].grid(b=True)

        

        ax[i,0].set_xlabel(ip[i])

        ax[i,0].set_xticks(x_tick_value)

        ax[i,0].set_xticklabels(x_tick_label,rotation=45)

        

        ax[i,0].set_ylabel(op[0])

        ax[i,0].set_yticks(y_tick_value)

        ax[i,0].set_yticklabels(y_tick_label)

        

        

    plt.tight_layout 
bubblechart(['Sex'],['Survived'],'PassengerId',df_train)
bubblechart(['Pclass','SibSp','Parch'],['Survived'],'PassengerId',df_train,)
import re

def findtitle(str_name):

    m = re.search(r', (?P<Title>.*)\.',str_name)

    return m.group('Title')
df_train['Title'] = df_train['Name'].apply(findtitle)
bubblechart(['Title'],['Survived'],'PassengerId',df_train)
df_train['Cabin'] = df_train['Cabin'].apply(lambda x : 'X' if pd.isnull(x) else x[:1])
bubblechart(['Cabin'],['Survived'],'PassengerId',df_train)