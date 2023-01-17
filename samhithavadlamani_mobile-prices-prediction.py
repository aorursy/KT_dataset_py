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
#libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



# from IPython.core.interactiveshell import InteractiveShell 

# InteractiveShell.ast_node_interactivity = "all"
#load data

df_train=pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')

df_test=pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')

df_train.info()

df_test.info()

# all numeric columns

# no null values
#cleaning



# first check for duplicated rows and remove them for test and train separately 

df_train.duplicated().any()

df_test.duplicated().any()

# no duplicated rows 



# next for outliers to remove rows

box1=df_train.boxplot()

box2=df_test.boxplot()

plt.setp(box1.get_xticklabels(), rotation=90)

plt.setp(box2.get_xticklabels(), rotation=90)

df_train=df_train[(np.abs(stats.zscore(df_train))<3).all(axis=1)]

df_test=df_test[(np.abs(stats.zscore(df_test))<3).all(axis=1)]

#removed outliers

print("train shape after outliers:", np.shape(df_train))

print("test shape after outliers:",np.shape(df_test))



# Now it is safe to join train and test to form the main dataset

target=df_train["price_range"]

df_train=df_train.drop(["price_range"], axis=1)

test_id=df_test["id"]

df_test=df_test.drop(["id"], axis=1)

df=pd.concat([df_train, df_test])

print("full data shape:", np.shape(df))











#visualization 



df.hist(figsize=(20,20), xlabelsize=8, ylabelsize=8)

plt.show()

target.hist(figsize=(20,20), xlabelsize=8, ylabelsize=8)

corr=df.corr()

sns.heatmap(corr[(corr>=0.5) | (corr<=-0.5)], vmin=-1, vmax=1, annot=True, annot_kws={'fontsize':8})



df_copy=df.copy()

# drop 4g/3g, fc/pc

df=df.drop(["three_g", "fc"], axis=1)
print(np.shape(df))

X_train=df.iloc[:1988]

X_test=df.iloc[1988:]

print(np.shape(X_train), np.shape(X_test))



#model

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

#create the train/test split

x_train, x_test, y_train, y_test = train_test_split(X_train, target, test_size=0.3, random_state=1)

#Create the model and train

model = RandomForestClassifier(random_state=1)

model.fit(x_train,y_train)

#predict the results for test

test_pred = model.predict(x_test)

#test the accuracy

acc=accuracy_score(y_test, test_pred)

print(acc)



# important features

feat_importances = pd.Series(model.feature_importances_, index=df.columns)

feat_importances.nlargest(7).plot(kind='barh')
# improving model

# 1 combine 4g/3g, fc/pc instead of discarding

print(np.shape(df_copy))

df_imp0=df_copy.copy()

df_imp0["g"]=df_copy["three_g"]+df_copy["four_g"]

df_imp0=df_imp0.drop(["three_g", "four_g"], axis=1)

df_imp0["c"]=df_copy["fc"]+df_copy["pc"]

df_imp0=df_imp0.drop(["fc", "pc"], axis=1)



X_train=df_imp0.iloc[:1988]

X_test=df_imp0.iloc[1988:]

print(np.shape(X_train), np.shape(X_test))



#model

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

#create the train/test split

x_train, x_test, y_train, y_test = train_test_split(X_train, target, test_size=0.3, random_state=1)

#Create the model and train

model = RandomForestClassifier(random_state=1)

model.fit(x_train,y_train)

#predict the results for test

test_pred = model.predict(x_test)

#test the accuracy

acc=accuracy_score(y_test, test_pred)

print(acc)



# important features

feat_importances = pd.Series(model.feature_importances_, index=df_imp0.columns)

feat_importances.nlargest(7).plot(kind='barh')