# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Data Manipulation

import pandas as pd

import numpy as np



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix

import math

#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 20,20
data= pd.read_csv("../input/cities.csv")
x0,y0=data.iloc[0,1],data.iloc[0,2]

def distance(x):

    return math.sqrt((math.pow((x[1]-x0),2) + math.pow((x[2]-y0),2)))
def Rel_Distance(df):

    dist=[]

    prev=0

    for i in df.index:

        if i!=0:

            dist.append(math.sqrt((math.pow(df.get_value(i,"X")-df.get_value(prev,'X'),2) + math.pow(df.get_value(i,"Y")-df.get_value(prev,'Y'),2))))

            prev=i

    return dist


data['0_Dist']=data.apply(distance,axis=1)

df=data.sort_values("0_Dist")
x0,y0
def theta(df):

    return (df.Y-y0)/(df.X-x0)

df["theta"]=df.apply(theta,axis=1)
df['theta'][0]=1
df.head()
above=df[df['theta']>0]

below=df[df['theta']<0]

print(above.info()+below.info())
below=below.iloc[::-1]
final=pd.concat([above,below],axis=0)
submission=pd.DataFrame({'Path':final.index})

submission.set_value(index=197769,col="Path",value=int(0))



submission.tail()

submission=submission.astype(int)
submission.tail()
submission.to_csv("submission.csv",index=False)
type(final.index)