import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

import math

import warnings

warnings.filterwarnings('ignore')

%matplotlib.inline
#Load Dataset

data=load_iris()

target_names=data.target_names

target=pd.DataFrame(data.target,columns=["Class"])

fe=data.feature_names

df=pd.DataFrame(data.data,columns=fe)

df.columns=df.columns.str.strip().str.lower().str.replace(" ","_").str.replace("(","").str.replace(")","").str.replace('_cm',"")

df=pd.concat([df,target], axis=1)

df.head()
#Datatype of the variables

df.dtypes
#Convert Class to Categorical

pd.Categorical(df.Class)
#Number of unique categories in Class

pd.unique(df.Class)
#Number of observation of each class

df.groupby(["Class"])["sepal_length"].count()
#Check for null orna in the dataset

df.isna().sum() #So there is no na or null in the dataset
feature=list(set(df.columns.values).difference({"Class"}))

X=df[feature].values

y=df[["Class"]].values

print(f'Shape of X is {X.shape}')

print(f'Shape of y is {y.shape}')
#Label Dictionary

l_dict=dict()

for k,v in zip(range(4),target_names):

    l_dict.update({k:v})

print(l_dict)
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(20,10))

for ax,cnt in zip(axes.ravel(),range(4)):

    #Set bin size

    min_b=math.floor(np.min(X[:,cnt]))

    max_b=math.ceil(np.max(X[:,cnt]))

    bins=np.linspace(min_b,max_b,25)

    

    #Plot the histogram

    for lab,col in zip(range(4),('blue',"red","green")):

        ax.hist(X[np.where(y==lab)[0],cnt],

            color=col,

            alpha=0.5,

            bins=bins,

            label=f'class {l_dict[lab]}'

           )

    ylims=ax.get_ylim()

    

    #plot annotations

    leg=ax.legend(loc="upper right")

    leg.get_frame()

    ax.set_xlabel(feature[cnt])

    ax.set_ylim(0,max(ylims)+2)

    ax.set_title(f"Iris Histogram-{str(cnt + 1)}")

    

    #remove axis spine(hide the border around the figures)

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_visible(False)

    

    #hide axis ticks

    ax.tick_params(axis="both",which="both",bottom="off", top="off",left="off",right="off",labelbottom="on",labelright="off", labelleft="on")

    

axes[0][0].set_ylabel("Count")

axes[1][0].set_ylabel("Count")



fig.tight_layout()

plt.show()