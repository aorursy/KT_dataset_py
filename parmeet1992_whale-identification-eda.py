import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
print(os.listdir("../input"))
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
print("Number of images in the train images")
print(len(os.listdir("../input/train")))
print("Number of images in the test images")
print(len(os.listdir("../input/test")))
df = pd.read_csv('../input/train.csv')
df.head()
#histogram of 
df['Id'].value_counts().hist(bins=50)
df['Id'].value_counts().head(7)
print("Total values that are null")
print(df.isnull().sum().sort_values(ascending=False))
print("Total values that are ")
print(df.isnull().count())
total = df.isnull().sum().sort_values(ascending=False)
percent = df.isnull().sum().sort_values(ascending=False)/df.isnull().count().sort_values(ascending=False)
pd.concat([total,percent],axis=1,keys=['Total','Percent'])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Ids = le.fit_transform(df['Id'])
sns.distplot(Ids)
plt.title('Categorical Distribution')
plt.show()
temp = pd.DataFrame(df['Id'].value_counts().head(8))
temp.reset_index(inplace=True)
temp.columns = ['Id','Counts']
plt.figure(figsize=(9,8))
sns.barplot(x='Id',y='Counts',data = temp)
plt.show()
temp = pd.DataFrame(df['Id'].value_counts().tail())
temp.reset_index(inplace=True)
temp.columns = ['Whale_id','Counts']
sns.barplot(x='Whale_id',y='Counts',data=temp)
