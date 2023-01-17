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
import matplotlib as mpt
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/iris/Iris.csv')
df.head()
df.describe()
g=sns.pairplot(df, hue='Species')
df=df.drop(['Id'] ,axis =1)
df.head()
## Correlation
import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
sns.pairplot(df, hue = 'Species')

df.info()
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
df = datasets.load_iris()
X = df.data
y = df.target
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
clf = LogisticRegression(random_state=0, multi_class='ovr')
model = clf.fit(X_std, y)
new_observation = [[.5, .5, .5, .5]]
model.predict(new_observation)
model.predict_proba(new_observation)
arr=np.array(model.predict_proba(new_observation))
arr.mean()
## using kmn features
