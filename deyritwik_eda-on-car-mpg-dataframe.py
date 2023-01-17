#Import all the necessary modules
#Import all the necessary modules
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
%matplotlib inline

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

import gc
gc.enable()
df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",header=None, delimiter=r"\s+")
df.head()
cdf=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names", header=None,sep=":",skiprows=32,nrows=9)
cdf.rename(columns={0:'Row',1:"datatype"},inplace=True)
cdf = pd.DataFrame(cdf["Row"].str.split(' ',10).tolist())
cdf.rename(columns={5:'Row'},inplace=True)
def rename(df1,df2):
    t=df1.columns.values
    s=df2['Row'].tolist()
    for x in range(df.shape[1]):
        if x!=9:
            df1.rename(columns={t[x]:s[x]},inplace=True)  
            #print("_____________")
            #print(t[x])
            #print(s[x])
    return df1

Car_df=rename(df,cdf)
Car_df.head()
Car_df.tail()
Car_df.sample(5)
Car_df.shape
def datatypes_insight(data):
    display(data.dtypes.to_frame())
    data.dtypes.value_counts().plot(kind="barh")
#Car_df.apply(lambda x: len(x.unique()))
datatypes_insight(Car_df)
def datatypes_insight(data):
    display(data.apply(lambda x: len(x.unique())).to_frame())
    data.apply(lambda x: len(x.unique())).plot(kind="barh")
datatypes_insight(Car_df)
Car_df.describe().T
Car_df = Car_df.replace('?', np.nan)
Car_df.isnull().sum()
Car_df = Car_df.drop('car', axis=1)
Car_df["horsepower"] = Car_df.astype('float64')
pd.DataFrame({'count' : Car_df.groupby(["horsepower","origin"] ).size()}).head().reset_index()
Car_df["horsepower"] = Car_df.groupby(["origin"])["horsepower"]\
    .transform(lambda x: x.fillna(x.median()))
Car_df.isnull().sum()
Car_df.head()
def distploting(df):
    col_value=df.columns.values.tolist()
    sns.set(context='notebook',style='whitegrid', palette='dark',font='sans-serif',font_scale=1.2,color_codes=True)
    
    fig, axes = plt.subplots(nrows=2, ncols=4,constrained_layout=True)
    count=0
    for i in range (2):
        for j in range (4):
            s=col_value[count+j]
            #axes[i][j].hist(df[s].values,color='c')
            sns.distplot(df[s].values,ax=axes[i][j],bins=30,color="b")
            axes[i][j].set_title(s,fontsize=17)
            fig=plt.gcf()
            fig.set_size_inches(15,10)
            plt.tight_layout()
        count=count+j+1
        
             
distploting(Car_df)
def boxplot(df):
    col_value=['mpg',
 'cylinders',
 'displacement',
 'horsepower',
 'weight',
 'acceleration',
 'model',"origin"]
    sns.set(context='notebook', palette='pastel',font='sans-serif',font_scale=1.5,color_codes=True,style='whitegrid')
    fig, axes = plt.subplots(nrows=2, ncols=4,constrained_layout=True)
    count=0
    for i in range (2):
        for j in range (4):
            s=col_value[count+j]
            #axes[i][j].boxplot(df[s])
            sns.boxplot(df[s],ax=axes[i][j],orient="v")
            fig=plt.gcf()
            fig.set_size_inches(15,20)
            plt.tight_layout()
        count=count+j+1
        
             
boxplot(Car_df)
fig, ax = plt.subplots(nrows=1, ncols=2,squeeze=True)
fig.set_size_inches(20,8)
Age_frequency_colums= pd.crosstab(index=Car_df["origin"],columns="count")
Age_frequency_colums.plot(kind='bar',ax=ax[0],color="c",legend=False)
Age_frequency_colums.plot(kind='pie',ax=ax[1],subplots=True,legend=False,autopct='%.2f')
ax[0].set_title('Frequency Distribution of Dependent variable: origin')
ax[1].set_title('Pie chart representation of Dependent variable: origin')

#adding the text labels
rects = ax[0].patches
labels = Age_frequency_colums["count"].values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax[0].text(rect.get_x() + rect.get_width()/2, height +1,label, ha='center', va='bottom',fontsize=15)
plt.show()
def diffplot(df):
    sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
    column_names=df.describe().columns.values.tolist()
    number_of_column=len(column_names)

    fig,ax=plt.subplots(nrows=2,ncols=4,figsize=(15,15))

    counter=0
    for i in range(2):
        for j in range(4):
            sns.boxplot(x='origin', y=column_names[counter],data=df,ax=ax[i][j])
            plt.tight_layout()
            counter+=1
            if counter==(number_of_column-1,):
                break
                 
diffplot(Car_df)
def diffplot(df):
    sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
    column_names=df.describe().columns.values.tolist()
    number_of_column=len(column_names)

    fig,ax=plt.subplots(nrows=2,ncols=4,figsize=(15,15))

    counter=0
    for i in range(2):
        for j in range(4):
            sns.violinplot(x='origin', y=column_names[counter],data=df,ax=ax[i][j])
            plt.tight_layout()
            counter+=1
            if counter==(number_of_column-1,):
                break
                 
diffplot(Car_df)
sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.jointplot(x=Car_df.mpg,y=Car_df["horsepower"], kind="scatter", color="c")
sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.jointplot(x=Car_df.mpg,y=Car_df["cylinders"], kind="scatter", color="g")
sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.jointplot(x=Car_df.mpg,y=Car_df["displacement"], kind="scatter", color="b")
sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.jointplot(x=Car_df.mpg,y=Car_df["weight"], kind="scatter", color="r")
sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.jointplot(x=Car_df.mpg,y=Car_df.acceleration, kind="scatter", color="b")
sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.jointplot(x=Car_df.mpg,y=Car_df.model, kind="scatter", color="g")

sns.pairplot(Car_df,hue="origin")
# Draw a heatmap with the numeric values in each cell
cor_mat= Car_df.corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(28,28)
sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True,cmap="coolwarm",linewidths=1)