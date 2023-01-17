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
df=pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
df.head()
def show_nan(df):
    total=len(df)
    nan_sum=df.isnull().sum()
    nan_perc=[i/total for i in nan_sum]
    df_nan=pd.DataFrame(data=nan_perc,index=df.columns,columns=["Nan values"])
    return df_nan

def show_unique(df,th):
    total=len(df)
    unique=[(df[col].unique()) for col in df.columns]
    unique_perc=[len(col) for col in unique ]
    
    df_unique=pd.DataFrame(data=unique_perc,index=df.columns,columns=["unique vals count"])
    
    cat_col,non_cat,uniques=[],[],[]
    for index,col in enumerate(unique):
        if len(col)<th:
            u_val=col
            cat_col.append(df.columns[index])
        else:
            u_val="more than"+ str(th)
            non_cat.append(df.columns[index])
        uniques.append(u_val)
        
    df_unique["unique vals"]=uniques
    return df_unique,cat_col,non_cat

def get_info(df,unique_val_threshold,only_df=False):
    df_unique,cat_col,non_cat=show_unique(df,unique_val_threshold)
    df_nan=show_nan(df)
    df_info=df_nan.join(df_unique)
    df_info["Dtypes"]=df.dtypes
    if only_df:
        return df_info.style.\
            background_gradient(cmap='Greens',axis=0).\
            applymap(lambda x: "color:red" if x>0 else "color:black",subset=["Nan values"]).\
            applymap(lambda x: "background-color:lightgreen" if x[0]!="m" else "background-color:pale green",subset=["unique vals"])
    return cat_col,non_cat,df_info.style.\
            background_gradient(cmap='Greens',axis=0).\
            applymap(lambda x: "color:red" if x>0 else "color:black",subset=["Nan values"]).\
            applymap(lambda x: "background-color:lightgreen" if x[0]!="m" else "background-color:pale green",subset=["unique vals"])

cat_col,non_cat,df_info=get_info(df,10)
df_info
df.describe().style.background_gradient(axis=0)
non_cat
cat_col
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
def count_catcols(df,catcols,width,height):
    fig,ax=plt.subplots(1,len(catcols),figsize=(width,height))
    i=0
    for cols in catcols:
        x=df[cols]
        percentage = lambda i: len(i) / float(len(x)) 
        sns.barplot(x=x,y=x,ax=ax[i],estimator=percentage).set(ylabel="percent")
        plt.title(cols)
        i+=1

from matplotlib.pyplot import figure
def show_dist(df,target,ax):
    # Add lines for mean, median and mode:
    # For Grid (for better mapping of heights)
    target=target
    sns.kdeplot(df[target].dropna(),color="blue",ax=ax)

count_catcols(df,cat_col,25,5)
_,ax_non_cat=plt.subplots(3,2,figsize=(25,12))

show_dist(df,"age",ax_non_cat[0][0])
show_dist(df,'creatinine_phosphokinase',ax_non_cat[0][1])
show_dist(df,'ejection_fraction',ax_non_cat[1][0])
show_dist(df,"platelets",ax_non_cat[1][1])
show_dist(df,"serum_creatinine",ax_non_cat[2][0])
show_dist(df,"time",ax_non_cat[2][1])
# countplots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
def catplot(df,col,target_col):
    x,t= col, target_col

    df1 = df.groupby(x)[t].value_counts(normalize=True)
    df1 = df1.rename('percent').reset_index()
    p=sns.catplot(x=x,y='percent',hue=t,kind='bar',data=df1)
    return p
print(cat_col[:-1])
# see relationship with target var
cat_col=cat_col[:-1]
for i,j in enumerate(cat_col):
    catplot(df,j,"DEATH_EVENT")
    plt.title(j.upper())
    plt.show()
sns.pairplot(vars=non_cat,data=df,hue="DEATH_EVENT",kind="scatter")
df[df["DEATH_EVENT"]==1][non_cat].describe().iloc[:3,:].style.background_gradient()
df[df["DEATH_EVENT"]==0][non_cat].describe().iloc[:3,:].style.background_gradient()
# observation : seems like platelets is not a good dertimining feature as it doesn't show significant difference in both classes
# binarization
df=pd.get_dummies(columns=cat_col,data=df)
# raw train( just OHE no outlier removal,feature selection,engineering,tuning,etc)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X=df.drop(["DEATH_EVENT"],axis=1)
y=df["DEATH_EVENT"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
clf=LogisticRegression()
clf.fit(X_train,y_train)
clf.score(X_train,y_train)
print(classification_report(y_test,clf.predict(X_test)))
