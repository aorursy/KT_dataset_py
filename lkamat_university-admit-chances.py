# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #For visualization

from matplotlib import rcParams #add styling to the plots

from matplotlib.cm import rainbow #for colors

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import everything we need for the tabular application

from fastai.tabular import *
# read the data file

df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.shape # number of rows and columns
df.info()
df.describe()
df.sample(5)
df.columns
df=df.rename(columns = {'Chance of Admit ':'Admit'})
#Visualize

plt.figure(figsize=(15,8))

sns.heatmap(df.corr(), annot=True)
# Understanding the data

# Visualizations to better understand data and do any processing if needed.

df["GRE Score"].plot(kind = 'hist',bins = 200,figsize = (6,6))

plt.title("GRE Scores")

plt.xlabel("GRE Score")

plt.ylabel("Frequency")

plt.show()
df["TOEFL Score"].plot(kind = 'hist',bins = 200,figsize = (6,6))

plt.title("TOEFL Scores")

plt.xlabel("TOEFL Score")

plt.ylabel("Frequency")

plt.show()
s = df[df["Admit"] >= 0.80]["GRE Score"].value_counts().head(5)

plt.title("GRE Scores of Candidates with an 80% acceptance chance")

s.plot(kind='bar',figsize=(20, 10))

plt.xlabel("GRE Scores")

plt.ylabel("Candidates")

plt.show()
# Process the data 

df = df.drop(['Serial No.'], axis=1)
df.head()
df.loc[df.Admit <= 0.8, 'Admitted'] = '0' 

df.loc[df.Admit > 0.8, 'Admitted'] = '1' 

#df
df['Admit'] = df['Admitted']

df.Admit=df.Admit.astype(int)

df = df.drop(['Admitted'], axis=1)



df.head()
dep_var = 'Admit'

cat_names = []

cont_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research']

procs = [FillMissing, Categorify, Normalize]
#save the model

path="../kaggle/working"

#np.random.rand()
#valid_idx = range(len(df)-100, len(df))

#valid_idx 



random.seed(33148690)

valid_idx = random.sample(list(df.index.values), int(len(df)*0.2) )

#valid_idx 


data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)

#print(data.train_ds.cont_names)
data.show_batch(rows=5)
learn = tabular_learner(data, layers=[200,100], metrics=accuracy)

learn.fit_one_cycle(1, 1e-2)


#df.loc[df['Admit'] == 1]
learn.predict(df.iloc[150])
learn.predict(df.iloc[50])