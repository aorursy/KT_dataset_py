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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
train_df=pd.read_csv("/kaggle/input/csvfiles/train_data.csv")
test_df=pd.read_csv("/kaggle/input/csvfiles/train_data.csv")
train_df.isna().sum()
train_df.set_index('id', inplace=True)
train_df
plt.figure(figsize=(15,15))
sns.heatmap(train_df.corr(),annot=True)
plt.show()
features=['battery_power', 'dual_sim','int_memory', 'px_height','ram','sc_h']
train_DF=train_df[features]
y=train_df['price_range']
test_DF=test_df[features]
[(col,train_df[col].nunique() )for col in train_DF ]
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(random_state=0)
my_pipe=Pipeline(steps=[('model',model)])
for s in [50,100,200,300,500]:
    my_pipe=Pipeline(steps=[('model',RandomForestClassifier(n_estimators=s,random_state=0))])
    scores=cross_val_score(my_pipe, train_DF, y, cv=5, scoring='f1_micro')
    print(scores.mean())
for s in [100,150,200,250,300]:
    my_pipe=Pipeline(steps=[('model',RandomForestClassifier(n_estimators=s,random_state=0))])
    scores=cross_val_score(my_pipe, train_DF, y, cv=5, scoring='f1_micro')
    print(scores.mean())
my_model=RandomForestClassifier(n_estimators=150)
my_model.fit(train_DF, y)
predictions=my_model.predict(test_DF)
my_sub=pd.DataFrame({"id":test_DF.index, 'price_range':predictions})
my_sub
my_sub.to_csv('submission.csv',index=False)
