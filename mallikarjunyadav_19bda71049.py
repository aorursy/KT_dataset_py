# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv (r'/kaggle/input/bda-2019-ml-test/Train_Mask.csv')
#print(df)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Y = np.array(df["flag"])

df = df.drop('flag', 1)
print(df.columns)
# l=df.columns

#not useful in normalizing it

# for i in l:
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x = df[[i]].values.astype(float)
#     # Create an object to transform the data to fit minmax processor
#     x_scaled = min_max_scaler.fit_transform(x)

#     # Run the normalizer on the dataframe
#     df[i] = pd.DataFrame(x_scaled)

#     df[i] = df[i] - df[i].min() / df[i].max() - df[i].min()
df['timeindex'].head()
X = df # contains all except flag variable  
Y = Y # contains flag variable
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1,test_size=0.2)

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 30)
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')


model.fit(X_train, y_train)
y_predict = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict)
from sklearn.metrics import confusion_matrix

sc = pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['0', '1'],
    index=['0', '1']
)
sc['0']['0'] / (sc['0']['0'] + sc['0']['1'])
sc['1']['1'] / (sc['1']['0'] + sc['1']['1'])
y_predict1 = model.predict(X_train)

from sklearn.metrics import accuracy_score

accuracy_score(y_train, y_predict1)
df1 = pd.read_csv (r'/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv')

y_predict2 = model.predict(df1)

sno = range(1,len(y_predict2)+1)

print(len(sno),len(y_predict2))
arr = np.vstack((sno, y_predict2)).T
from sklearn.metrics import accuracy_score

# accuracy_score(y_train, y_predict1)
df2 = pd.DataFrame({'Sl.No': arr[:, 0], 'flag': arr[:, 1]})

df2.to_csv('/kaggle/working/Submission.csv')
