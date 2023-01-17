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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df= pd.read_csv('../input/Admission_Predict.csv')
df.head()
df.describe()
#getting information about the data set

df.info()
#check null values

# we cansee from info that there are no null values in the data set but if there are any null values we can visualize them with seaborn

sns.heatmap(df.isnull(), cmap= 'viridis')
#since data set is relatively small we can visualize how each column is realated to other columns in the data set.

df.drop('Serial No.', axis = 1, inplace=True)
sns.set_style('whitegrid')

sns.pairplot(data = df, palette= 'rainbow')
#from above pair plot we can see that Chance of Admit share linear realation with GRE Score, TOEFL score  and CGPA

# we confirm this with following joint plots

features  = ['GRE Score', 'TOEFL Score', 'CGPA']

for fea in features:

    sns.jointplot(df[(fea)], y = df[('Chance of Admit ')],kind ='reg')
# lets explore data more

for fea in features:

    plt.figure(figsize=(10,4))

    sns.distplot(df[fea], color= 'red', kde = False, bins = 50)
features2 = ['University Rating', 'SOP', 'LOR ', 'Research']

for fea in features2:

    sns.jointplot(df[(fea)], y = df['Chance of Admit '],kind ='hex')
#checking the correlation

plt.figure(figsize=(10, 10))

sns.heatmap(df.corr(), annot=True, linewidths=0.05, fmt= '.2f',cmap="plasma",)
df.loc[df['Chance of Admit ']>=0.9, 'Target'] = "Highly Likely"

df.loc[(df['Chance of Admit ']>=0.8) &(df['Chance of Admit ']<0.9), 'Target'] = "Likely"

df.loc[(df['Chance of Admit ']>=0.7) &(df['Chance of Admit ']<0.8), 'Target'] = "Reach"

df.loc[df['Chance of Admit ']<.7, 'Target'] = "Not Possible"
df.head()
from sklearn.preprocessing import LabelEncoder
lbl_enc = LabelEncoder()

df['Target_encoder'] = lbl_enc.fit_transform(df['Target'])
df.info()
#lets predict chances

X = df.drop(['Chance of Admit ', 'Target'], axis=1)

y = df['Target_encoder']
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#lets predict using logistic Regression

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()

log_reg.fit(X_train,y_train)
log_prediction = log_reg.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, log_prediction))