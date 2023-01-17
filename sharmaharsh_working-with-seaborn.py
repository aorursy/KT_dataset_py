# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
input="../input/500_Person_Gender_Height_Weight_Index.csv"
df=pd.read_csv(input)
df.columns=['Gender','Height','Weight','Health']
df.head()
df.info()
corr=df.corr()
corr
def healthy(x):
    if(x==0):
        return "Extremely Weak"
    elif(x==1):
        return "Weak"
    elif(x==2):
        return "Normal"
    elif(x==3):
        return "OverWeight"
    elif(x==4):
        return "Obesity"
    else:
        return "Extreme Obesity"
    
    
df["Health"]=df['Health'].apply(healthy)
df.head()
mask=np.zeros_like(corr)
mask[np.triu_indices_from(mask)]=True
fig,ax = plt.subplots(figsize=(10, 10))
with sns.axes_style("white"):
    sns.heatmap(corr,mask=mask,cmap="Dark2",annot=True)
    plt.title("Correlation")
    plt.show()
sns.lmplot(data=df,x="Weight",y="Height",fit_reg=False,hue="Health") ## Hue is for differentiating between healths
#this is seaborn's inbuilt scatter function
fig,ax = plt.subplots(figsize=(10, 10))
with sns.axes_style("white"):
    sns.boxplot(data=df,x='Health',y='Weight',hue='Gender',color="Green")
    sns.swarmplot(data=df,x='Health',y='Weight',hue='Gender',color="Yellow")
    plt.title("Gender wise Health Distribution based on weight")
    plt.plot()
fig,ax = plt.subplots(figsize=(10, 10))
with sns.axes_style("white"):
    sns.boxplot(data=df,x='Health',y='Height',hue='Gender',color="Pink",notch=True,width=.5)
    sns.swarmplot(data=df,x='Health',y='Height',hue='Gender',color="Blue",alpha=.7)
    plt.title("Gender wise Health Distribution based on weight")
    plt.plot()
df["BMI"]=df['Weight']/(df['Height']/100)**2
sns.lmplot(data=df,x="Weight",y="BMI",fit_reg=False,hue="Health") ## Hue is for differentiating between healths
plt.title("BMI vs Weight")
plt.show()
fig=plt.subplots(figsize=(10,10))
sns.violinplot(data=df,y="BMI",x="Health",hue="Gender",color="Red")
sns.swarmplot(data=df,x='Health',y='BMI',hue='Gender',color="Green",alpha=.7) ##transparency
plt.show()
sns.kdeplot(df.Weight)
sns.kdeplot(df.Height)
#sns.distplot(df.BMI,color="Red")
plt.subplot(2,1,1)
plt.title("Distribution of height and weight")
sns.distplot(df.Weight,color="Red")
plt.subplot(2,1,2)
sns.distplot(df.Height,color="Blue")
sns.countplot(x="Health",data=df)
plt.xticks(rotation=45)
plt.title("Density Plot")
sns.kdeplot(df.Height,df.Weight)
sns.jointplot(x='Weight',y='Height',data=df,color="Blue")
