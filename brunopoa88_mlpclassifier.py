# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.metrics import classification_report, confusion_matrix



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/train.csv')

test_df = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/test.csv')





x=train_df.iloc[:,0:len(train_df.columns.values) -1]

y=train_df.Activity



x_testeDF = test_df.iloc[:,0:len(train_df.columns.values) -1]

y_testeDF = test_df.Activity
y.unique()
enc = LabelEncoder()



y = enc.fit_transform(y)

y_testeDF = enc.fit_transform(y_testeDF)

y
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size=0.8)
mlp = MLPClassifier(hidden_layer_sizes=(200))

mlpFit = mlp.fit(X_train, np.array(y_train))
MLPpred = mlp.predict(X_test)
CM = confusion_matrix(y_test,MLPpred)



sns.heatmap(CM, center=True)

plt.show()



print(classification_report(y_test,MLPpred))
print(x.shape)

print(y.shape)

print(x_testeDF.shape)

print(y_testeDF.shape)



mlpFit.score(x_testeDF,y_testeDF)