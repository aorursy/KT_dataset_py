# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/iris-clf-dataset/IRIS_TYPE_CLF.csv")
df
df.dtypes
df.drop_duplicates() # no duplicate row
df["sepal_length"].plot.hist()
df["sepal_width"].plot.hist()
df["petal_length"].plot.hist()
df["petal_width"].plot.hist()
df.groupby("species")["sepal_length"].mean().plot.bar()
df.groupby("species")["sepal_width"].mean().plot.bar()
df.groupby("species")["petal_length"].mean().plot.bar()
df.groupby("species")["petal_width"].mean().plot.bar()
df.isnull().sum()
df[["sepal_length"]].plot.box() # plotting the boxplot to check the outliers
df["sepal_width"].plot.box()
df[["petal_length"]].plot.box()
df[["petal_width"]].plot.box()
# seprating the target and the features

x=df.drop(["species"],axis=1)

y=df["species"]
# splitting the test and the train data

from sklearn.model_selection import train_test_split as tts

train_x,test_x,train_y,test_y=tts(x,y,random_state=42,stratify=y)
test_y.value_counts()/len(test_y)   # checking the distribution
train_y.value_counts()/len(train_y)
# importing and creating instance of decision tree classifier

from sklearn.tree import DecisionTreeClassifier

dt_model=DecisionTreeClassifier(max_depth=1)

dt_model.fit(train_x,train_y)
dt_model.score(train_x,train_y)
dt_model.score(test_x,test_y)
#finding out the test and train accuracy putting them in dataframe

train_acc=[]

test_acc=[]

for depth in range(1,10):

    dt_model=DecisionTreeClassifier(max_depth=depth)

    dt_model.fit(train_x,train_y)

    train_acc.append(dt_model.score(train_x,train_y))

    test_acc.append(dt_model.score(test_x,test_y))

frame=pd.DataFrame({"max_depth":range(1,10),"train_acc":train_acc,"test_acc":test_acc})

frame
# plotting the plot to find the best fit model for the value of max_depth

plt.figure(figsize=[10,6] ,dpi=120)

plt.plot(frame["max_depth"],frame["train_acc"],color="yellow")

plt.plot(frame["max_depth"],frame["test_acc"],color="pink")

plt.xlabel("depth of tree")

plt.ylabel("accuracy score")
## after seeing the value of maxdepth the model underfits for max_depth=1 and the model overfits for max_depth>5



from sklearn.tree import DecisionTreeClassifier

dt_model=DecisionTreeClassifier(max_depth=4)   # applying the best maxdepth

dt_model.fit(train_x,train_y)
dt_model.score(train_x,train_y)   # checking the training score
dt_model.score(test_x,test_y)      # checking the test score