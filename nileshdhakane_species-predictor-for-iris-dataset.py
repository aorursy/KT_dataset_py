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
from sklearn.datasets import load_iris
iris = load_iris()

#print('Printing iris dataset',iris)
data = iris.data

feature = iris.feature_names

df = pd.DataFrame(data,columns = feature)

print(df.head())

target = iris.target

print('\n\n target',target)

target_names = iris.target_names

print('\n\ntarget names',target_names)
import seaborn as sns

%matplotlib inline
df['target']  = target

df['Species'] = target_names[target]

df.head()

# now we can visualize data 
sns.pairplot(df,hue = 'Species',height = 2)
from sklearn.model_selection import train_test_split



train_data,test_data,train_target,test_target = train_test_split(data,target,test_size = 0.3,random_state = 1)

print('train data shape',train_data.shape)

print('test data shape',test_data.shape)

print('train target data',train_target.shape)

print('test target data',test_target.shape)

print(test_data[:3])
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics



classifier  =  KNeighborsClassifier(n_neighbors=3)

kn_model = classifier.fit(train_data,train_target)

species_pred = kn_model.predict(train_data)

print(species_pred)



## checking accuracy score

accuracy_score = metrics.accuracy_score(train_target,species_pred)

print('Accuracy score =',accuracy_score)
## now we give sample to the model

sample = [[1,5,6,3],[4,2,8,9],[2,8,9,3]]

sample_pred = kn_model.predict(sample)

for i in sample_pred:

    print(f'Species name for given sample is ',target_names[i] )

test_pred = kn_model.predict(test_data)

accuracy = metrics.accuracy_score(test_target,test_pred)

print('Accuracy score =',accuracy)
new_data = [[1,2,1,2],[4.7,3.2,1.3,0.2],[1,2,5,1],[3,2,6,3],[4,2,5,4],[5,3,6,2.1]]

df1 = pd.DataFrame(new_data,columns = feature)

new_pred = kn_model.predict(new_data)

print(new_data)

print(target_names[new_pred])

df1['species'] = target_names[new_pred]

df1
sns.pairplot(df1,hue='species')