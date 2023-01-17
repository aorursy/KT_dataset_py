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
import pandas as pd

import pandas.util.testing as tm

import seaborn as sns

from sklearn.model_selection import cross_val_score,train_test_split,StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('/kaggle/input/iris/Iris.csv',sep = ',',index_col=['Id'])
# first 5 rows of the Dataset

data.head()
sns.heatmap(data.isnull())
sns.countplot(x = 'Species',data = data)
sns.heatmap(data.corr(),annot = True)
sns.boxplot(y = 'SepalLengthCm',x = 'Species',data = data)

sns.boxplot(y = 'SepalWidthCm',x = 'Species',data = data)
sns.boxplot(y = 'PetalWidthCm',x = 'Species',data = data)
sns.boxplot(y = 'PetalLengthCm',x = 'Species',data = data)
x = data.iloc[:,:4]

y = data.iloc[:,4]
encode = LabelEncoder()

y = encode.fit_transform(y)

df = pd.DataFrame({'Categorical':data['Species'][:10],'Numerical':y[:10]})

df
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 42)
nb = GaussianNB()

cross_val_score(nb,xtrain,y = ytrain,cv = StratifiedKFold(n_splits=5,shuffle = True)).mean()
nb.fit(xtrain,ytrain)

pred = nb.predict(xtest)
accuracy_score(ytest,pred)