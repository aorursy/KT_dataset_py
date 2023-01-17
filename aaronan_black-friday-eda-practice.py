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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

bf_df = pd.read_csv('../input/BlackFriday.csv')
bf_df.head()
bf_df.info()
bf_df.shape
bf_df.describe() # column level, has to be numeric
df_1 = bf_df.groupby('User_ID').agg(
    {'Product_ID':[pd.Series.nunique, 'count']
    }
)

df_1.head()
df_1.info()
df_1.reset_index().head()
## aggregate data into user level data. 

bf_user_df = bf_df.groupby('User_ID').agg(
    {'Product_ID':pd.Series.nunique,
     'Gender':lambda x:x.value_counts().index[0],
     'Age':lambda x:x.value_counts().index[0],
     'Occupation':lambda x:x.value_counts().index[0],
     'City_Category':lambda x:x.value_counts().index[0],
     'Stay_In_Current_City_Years':lambda x:x.value_counts().index[0],
     'Marital_Status':lambda x:x.value_counts().index[0],
     'Product_Category_1':'sum',
     'Product_Category_2':'sum',
     'Product_Category_3':'sum',
     'Purchase':'sum'
    }
)
bf_df.head(3)
bf_user_df.head(3)
bf_user_df.shape
bf_user_df['Gender'].value_counts()
bf_user_df[bf_user_df['Gender']=='F']

bf_user_df.query("Gender=='F'").head()
for col in bf_user_df[
    ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
]:
    print(bf_user_df[col].value_counts(sort=True)/bf_user_df.shape[0])
sns.countplot(bf_user_df['Gender'], color="skyblue")
for col in bf_user_df[
    ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
]:
    plt.figure()
    sns.countplot(bf_user_df[col], color="skyblue")
tgt_cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']

fig, axes =plt.subplots(2,3, figsize=(15,15))
axes = axes.flatten() # hard/

for ax, col in zip(axes, bf_user_df[tgt_cols]):
    sns.countplot(bf_user_df[col], color="skyblue", ax=ax)
fig, axes =plt.subplots(3,2, figsize=(20,8), sharex=True)
axes = axes.flatten()

tgt_cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']

for ax, col in zip(axes, bf_user_df[tgt_cols]):
    smy_df = bf_user_df[col].value_counts()/bf_user_df.shape[0]
    smy_df.reset_index().pivot_table(columns='index').plot.barh(stacked=True,ax=ax)
    
    for i in ax.patches:
        width, height  = i.get_width(), i.get_height()
        x, y = i.get_xy()
        ax.annotate(s = str(round((i.get_width())*100, 1))+'%', xy = (i.get_x()+.2*width, i.get_y()+.4*height), fontsize=10, rotation=70)
smy_df = bf_user_df['Age'].value_counts()/bf_user_df.shape[0]
smy_df
smy_df.reset_index().pivot_table(columns='index').plot.barh(stacked=True)
