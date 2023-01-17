# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/voicegender/voice.csv")

df=pd.read_csv("../input/voicegender/voice.csv")
data.info()
data.tail()
sns.pairplot(df[['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label']], hue='label', height=2)

plt.show()
df_m=df[df["label"]=="male"]

df_f=df[df["label"]=="female"]

m_m=np.mean(df_m)

f_m=np.mean(df_f)
data.label=[1 if i == "male" else 0 for i in data.label]



y=data.label.values

x_data=data.drop(["label"],axis=1)
x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)



from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(solver='lbfgs')



lr.fit(x_train,y_train)



print("test accuracy {} %".format(lr.score(x_test,y_test)*100))