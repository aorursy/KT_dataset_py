!pip install jovian opendatasets --upgrade --quiet
# Change this
dataset_url = 'https://www.kaggle.com/sheenabatra/facebook-data' 
import opendatasets as od
od.download(dataset_url)
data_dir = './facebook-data'
import os
os.listdir(data_dir)
project_name = "facebook_data_analysis" 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv('./facebook-data/pseudo_facebook.csv')
df.head()
df.shape
df.describe()
df['age'].value_counts()
df = pd.get_dummies(df,columns=['gender'])
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['figure.figsize'] = (20,5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
sns.countplot(x='age',data=df)
matplotlib.rcParams['figure.figsize'] = (9,5)
sns.scatterplot(x="age", y="likes", hue="gender_male",data=df)
plt.subplots(figsize=(14,12))
sns.heatmap(df.corr(), annot=True)
plt.show()
maximum_likes = df['likes'].max()
minimum_likes = df['likes'].min()
print("maximum likes:", maximum_likes)
print("minimum likes:", minimum_likes)
labels=['10-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','101-110','111-120']
df['age_group'] = pd.cut(df.age,bins=np.arange(10,121,10),labels=labels,right=True)
sns.barplot(df['age_group'],df['mobile_likes'])
plt.title("Mobile users")
sns.barplot(df['age_group'],df['www_likes'])
plt.title("Website users")
sns.barplot(df['age_group'],df['friend_count'])
sns.jointplot(x='age',y='friend_count',data=df)
total_likes = df['likes'].sum()
print("Total number of likes over: ", total_likes)
total_likes_male = df.groupby("gender_male")["likes"].sum()
total_likes_male
df_updated = pd.read_csv('./facebook-data/pseudo_facebook.csv')
labels=['10-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','101-110','111-120']
df_updated['age_group'] = pd.cut(df_updated.age,bins=np.arange(10,121,10),labels=labels,right=True)
sns.barplot(x=df_updated['age_group'],y=df_updated['friend_count'],hue=df_updated.gender)
