



import numpy as np 

import pandas as pd

import os

print(os.listdir("../input"))



heart_disease=pd.read_csv('../input/heart.csv')
heart_disease.head(10)
heart_disease.isnull().values.any()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
heart_disease.columns
#ax = sns.barplot(x = 'target',y='age',data = heart_disease)

def bar_plot(x,y,hues=None,ci=95):

    ax = sns.barplot(x,y,data = heart_disease,hue = hues)
import warnings

warnings.filterwarnings("ignore")

bar_plot(heart_disease['target'],heart_disease['age'],heart_disease['sex'],ci=78)
bar_plot(heart_disease['target'],heart_disease['trestbps'],heart_disease['cp'])
bar_plot(heart_disease['target'],heart_disease['chol'],heart_disease['sex'])
sns.barplot(x = heart_disease['target'],y = heart_disease['fbs'],data= heart_disease,hue='cp')
ax = sns.barplot(x='target',y = 'thalach',data= heart_disease,hue='thal')
ax = sns.barplot(x = 'cp',y = 'chol',data = heart_disease,hue='target',palette='husl')
x = 'target'

y = ['age','trestbps','chol','thalach','oldpeak']

hues = ['sex','cp','fbs']

def violin_plot(x,y,hues):

    ax= sns.violinplot(x,y,data = heart_disease,hue=hues)

    plt.show()
violin_plot(x,y[0],hues[0])
violin_plot(x,y[2],hues[1])
violin_plot(x,y[2],hues[2])
violin_plot(heart_disease['fbs'],y[1],hues[1])