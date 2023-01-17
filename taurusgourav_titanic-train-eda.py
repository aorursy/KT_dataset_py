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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('../input/titanicdataset-traincsv/train.csv')
df.head()
df.shape
df.isnull()
df.notnull()
df.describe()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(15,10))
sns.set_style('darkgrid')
sns.countplot(x='Survived', data=df)
plt.figure(figsize=(15,10))
sns.countplot(x='Survived',hue='Sex', data=df, palette='rainbow')
sns.countplot(x='Survived',hue='Pclass',data=df,palette='rainbow')
sns.distplot(df['Age'].dropna(),kde=True, rug=True,color='darkred',bins=40)
sns.jointplot(x=df['Age'], y=df['Survived'], kind='kde')
sns.violinplot(x='Age', y='Sex', data=df)

sns.pairplot(data=df)
df['Age'].hist(bins=30,color='darkred',alpha=0.3)

sns.countplot(x='SibSp',data=df)
df['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 35

        elif Pclass == 2:
            return 28

        else:
            return 22

    else:
        return Age
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.drop('Cabin', axis=1, inplace=True)
df.columns
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
features = ['Fare', 'Pclass', 'Sex', 'Embarked']
X = df[features]
y = df.Survived
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=1)
rfc_model.fit(train_X, train_y)

from sklearn.metrics import mean_absolute_error
val_predictions = rfc_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
