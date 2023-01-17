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


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier



dados = pd.read_csv("../input/autentbancariacsv/dados_autent_bancaria.csv")

df = pd.DataFrame(data=dados)
df.columns =['x1','x2','x3','x4','cls']

df.head()   


df['cls'] = df['cls'].apply(lambda x: 'c'+str(x + 1))

df.describe()
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if df[feature_name].dtypes != "object":
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = ((df[feature_name] - min_value) / (max_value - min_value))*100
    return result


df_normalized = normalize(df)

df_normalized.describe()
train, test = train_test_split(df_normalized, test_size=0.2)

train.to_csv("./train_normalized.csv",index=False)
test.to_csv("./test_normalized.csv",index=False)
clf = MLPClassifier(hidden_layer_sizes=(100,),solver='lbfgs',max_iter=500)
#train.head()
x = train.iloc[:,0:4]
#x.head()
y = train['cls']
#y.head()
clf.fit(x,y)
x_test = test.iloc[:,0:4]
y_yest = test['cls']
predicts = clf.predict(x_test)
accuracy_array = y_yest == predicts
accuracy = (np.count_nonzero(accuracy_array.values == True)/len(accuracy_array.values))*100

print("Model accuracy = {0:f} %".format(accuracy))

