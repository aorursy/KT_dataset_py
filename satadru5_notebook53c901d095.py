# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/charlist.csv")
df.head(3)
df_data=pd.read_csv("../input/dataset.csv")
df_data.head(6)
level=df_data['1024']
df_data=df_data.drop(['1024'],axis=1)
df_data.shape


# convert to array, specify data type, and reshape

#target = target.astype(np.uint8)

#train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)

#test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)



data = np.array(df_data).reshape((-1, 1, 32, 32)).astype(np.uint8)


import matplotlib.pyplot as plt

import matplotlib.cm as cm



plt.imshow(data[2][0], cmap=cm.binary) # draw the picture
sns.countplot(level)
#Train-Test split

from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = train_test_split(df_data, level, test_size = 0.2, random_state = 42)
data.shape,df_data.shape
#Random forest classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(data_train, label_train)

rf_score_train = rf.score(data_train, label_train)

print("Training score: ",rf_score_train)

rf_score_test = rf.score(data_test, label_test)

print("Testing score: ",rf_score_test)
#decision tree

from sklearn import tree

dt = tree.DecisionTreeClassifier()

dt.fit(data_train, label_train)

dt_score_train = dt.score(data_train, label_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(data_test, label_test)

print("Testing score: ",dt_score_test)