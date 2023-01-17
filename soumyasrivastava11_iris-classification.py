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
train_df=pd.read_csv("/kaggle/input/iris/Iris.csv")
train_df
train_df["Species"].unique()
from matplotlib import pyplot as plt
plt.figure(figsize=(16,9))
import seaborn as sns
sns.scatterplot(x="PetalLengthCm",y="PetalWidthCm",data=train_df)
from matplotlib import pyplot as plt
plt.figure(figsize=(16,9))
import seaborn as sns
sns.scatterplot(x="SepalLengthCm",y="PetalWidthCm",data=train_df)
from sklearn import preprocessing 
  

label_encoder = preprocessing.LabelEncoder() 
  

train_df['Species']= label_encoder.fit_transform(train_df['Species']) 
  
train_df['Species'].unique()
target=train_df["Species"]
del train_df["Species"]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train,X_test,y_train,y_test=train_test_split(train_df,target,random_state=0)

X_train
X_test
reg = LogisticRegression()
reg.fit(X_train,y_train)

print(reg.score(X_test,y_test)*100)
print(reg.score(X_train,y_train)*100)
