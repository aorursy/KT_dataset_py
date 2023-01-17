import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../input/CableTVSubscribersData.csv")
data.groupby('subscribe').size()
data.head()
data.shape
data.dtypes
data.describe()
data[['gender','ownHome','subscribe','Segment']].describe()
import missingno as msno
msno.matrix(data)
catdf = pd.get_dummies(data[['gender','ownHome','Segment']])
data = pd.concat([data, catdf],axis=1, sort=False)
data.head()
data.drop(columns = ['gender', 'ownHome', 'Segment'], inplace = True)
data['subscribe'] = pd.Categorical(data['subscribe'])
data['subscribe'] = data['subscribe'].cat.codes
data.columns
sns.countplot(data['subscribe'])
plt.xticks([0,1], ['subNo', 'subYes'])
plt.show()
data.groupby(['subscribe']).size()
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
x_train, x_test, y_train, y_test = train_test_split(data, data['subscribe'])
df_class_0 = x_train[x_train['subscribe'] == 0]
df_class_1 = x_train[x_train['subscribe'] == 1]
df_class_1_over = df_class_1.sample(len(df_class_0.index), replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

df_test_over['subscribe'].value_counts()
sns.countplot(df_test_over['subscribe'])
plt.xticks([0,1], ['subNo', 'subYes'])
plt.show()
data_train = df_test_over
plt.figure(figsize = (12,9))
sns.heatmap(data_train.corr(), cmap='YlGnBu', annot=True)
plt.show()
