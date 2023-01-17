%matplotlib inline

import pandas as pd

from matplotlib import pyplot as plt



import numpy as np

import seaborn as sns



#import joypy

import re

#from IPython.display import display, HTML

#import ipywidgets as widgets # for later







sns.set(style="darkgrid", color_codes=True)

pd.options.display.float_format = '{:.2f}'.format
df = pd.read_csv('train.csv')
# Based on asindico kernel - https://www.kaggle.com/asindico/porto-seguro-the-essential-kickstarter

# Classifying the variables in the data

variables = []

for variable in df.columns:

    for types in ['ind','reg','car', 'calc','target','id']:

        ty = "None"

        if df[variable].dtype == int:

            tybin = "ordinal"

        elif df[variable].dtype == float:

            tybin = "continuous"

        match = re.search('^.*'+types+'.*$',variable)

        if match:

            ty = types

            if re.search('^.*bin.*$',variable):

                tybin='binary'

            if re.search('^.*cat.*$',variable):

                tybin='categorical'

            if 'target' in variable:

                tybin = 'binary'

            break

    variables.append([variable,ty,tybin])



# Creating dataframe containing variables

variablesdf = pd.DataFrame(variables,columns=['name','type','bin'])
# Showing the number of variables per type

print('Total number of variables',len(variablesdf))

variablesdf.pivot_table(values='name',index='type',columns='bin',aggfunc="count",fill_value=0)
variablesdf = variablesdf.drop(0)
sns.countplot(x=df.target,data=df)

print(df.shape)

print('Percentage of Target equals 1 =',np.round(sum(df.target)/len(df)*100,2),("%"))
emptyvalue = df[(df==-1)].count()/(len(df))

emptyvalue = emptyvalue[emptyvalue>0]



#plot variables with empty values

plt.figure(figsize=(25,7))

sns.barplot(x=emptyvalue.index,y=emptyvalue)
targetdata = df[df.target==1].copy() # Create database containing only cases in which the target is equal to 1
emptyvalue = targetdata[(targetdata==-1)].count()/(len(targetdata))

emptyvalue = emptyvalue[emptyvalue>0]

#plot variables with empty values

plt.figure(figsize=(25,7))

sns.barplot(x=emptyvalue.index,y=emptyvalue)
binarydata = pd.DataFrame(df[variablesdf.name[(variablesdf.bin=='binary')]].sum().copy(),columns=['1s'])

binarydata['1s'] =binarydata['1s']/len(df)

binarydata['0s'] =(1-binarydata['1s'])



plt.figure(figsize=(26,10))

plt.subplot(211)

plt.xticks(range(len(binarydata)),binarydata.index)

plt.bar(left=range(len(binarydata)),height=binarydata['1s'].values)

plt.bar(left=range(len(binarydata)),height=binarydata['0s'].values,bottom=binarydata['1s'].values)
binarydata_t = pd.DataFrame(targetdata[variablesdf.name[(variablesdf.bin=='binary')]].sum().copy(),columns=['1s_t'])

binarydata['1s_t'] = binarydata_t['1s_t']/len(targetdata)

binarydata['0s_t'] = (1-binarydata['1s_t'])



plt.figure(figsize=(26,5))

plt.xticks(range(len(binarydata)),binarydata.index)

plt.bar(left=range(len(binarydata)),height=binarydata['1s_t'].values)

plt.bar(left=range(len(binarydata)),height=binarydata['0s_t'].values,bottom=binarydata['1s_t'].values)
plt.figure(figsize=(26,5))

binarydata['dif'] = binarydata['0s']-binarydata['0s_t']

plt.xticks(range(len(binarydata)),binarydata.index)

sns.barplot(x=binarydata.index,y=binarydata['dif'])
plt.figure(figsize=(17,7))

plt.subplot(121)

plt.title('All dataset')

sns.heatmap(df[variablesdf.name[(variablesdf.bin=='binary')]].corr(),cmap="coolwarm", linewidths=0.1)

plt.subplot(122)

plt.title('Only target=1')

sns.heatmap(targetdata[variablesdf.name[(variablesdf.bin=='binary')]].corr(),cmap="coolwarm", linewidths=.1)
uniquecat = pd.DataFrame(df[variablesdf.name[variablesdf.bin=='categorical']]

                              .T.apply(lambda x: x.nunique(), axis=1),columns=['val_unicos'])

uniquecat
for i in variablesdf.name[variablesdf.bin=='categorical']:

    plt.figure(figsize=(20,3))

    plt.subplot(121)

    g= sns.countplot(x=i,data=df)

    plt.subplot(122)

    g= sns.countplot(x=i,data=targetdata)
sns.pairplot(df[variablesdf.name[

    (variablesdf.bin=='ordinal')|(variablesdf.name=="target")]][0:3000],hue='target')
sns.pairplot(df[variablesdf.name[

    (variablesdf.bin=='continuous')|(variablesdf.name=="target")]][0:1000],hue='target')
sns.pairplot(df[variablesdf.name[

    (variablesdf.type=='calc')&(variablesdf.bin=='ordinal')|(variablesdf.bin=="continuous")|

    (variablesdf.name=="target")]][0:500],hue='target')
plt.figure(figsize=(17,17))

plt.title('All dataset')

sns.heatmap(df.dropna().corr(),cmap="coolwarm", linewidths=0.1)
plt.figure(figsize=(17,17))

plt.title('Only target=1')

sns.heatmap(targetdata.dropna().corr(),cmap="coolwarm", linewidths=.1)