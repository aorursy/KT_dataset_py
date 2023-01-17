# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

%matplotlib inline
df=pd.read_csv('/kaggle/input/wine-quality/winequalityN.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
plt.figure(figsize=(6,10))

plt.title('NaN Distribution')

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
for i in ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','pH','sulphates']:

    df[i]=df[i].fillna(df[i].mean(),inplace=False)
df.isnull().sum()
sns.pairplot(df,hue='type',palette={'red':'crimson','white':'skyblue'})
corr=df.corr(method='pearson')

mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr, mask=mask, cmap='Spectral', vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
plt.figure(figsize=(8,5))

sns.countplot(x='quality',hue='type',data=df,palette={'red':'crimson','white':'skyblue'})

sns.despine()
plt.figure(figsize=(8,5))

sns.boxplot(x='alcohol',y='type',data=df,palette={'red':'crimson','white':'skyblue'})

sns.despine()
sns.jointplot('alcohol','quality',data=df)
plt.figure(figsize=(10,5))

sns.violinplot(x='quality',y='alcohol',hue='type',data=df,palette={'red':'crimson','white':'skyblue'},

               split=True,inner='quart')

sns.despine()
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
df1=df.drop('quality',axis=1).copy()

df1.head()
df1=df1.apply(LabelEncoder().fit_transform)

df1.head()
std_sclr=StandardScaler().fit(df1)

X=std_sclr.transform(df1)

y=df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
svc=SVC(kernel='rbf',C=10,gamma=1).fit(X,y)

svc
print("Accuracy on training set: {:.4f}".format(svc.score(X_train, y_train)))

print("Accuracy on test set: {:.4f}".format(svc.score(X_test, y_test)))