# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, f1_score

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



import matplotlib.pylab as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
zoo_file_path="../input/zoo-animal-classification/zoo.csv"

zoo_data=pd.read_csv(zoo_file_path)
zoo_data.head()
zoo_data.animal_name.unique()
len(zoo_data.animal_name.unique())
zoo_data.animal_name.value_counts()
target='class_type'

zoo_data[target].unique()
zoo_data[target].value_counts()
print("This ZOO dataset is consised of",len(zoo_data),"rows.")

zoo_data.columns
y=zoo_data[target]

# features = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator',

#        'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail',

#        'domestic', 'catsize']

# X=zoo_data[features]

X=zoo_data.drop(columns=[target,'animal_name'])
train_X, val_X, train_y, val_y = train_test_split(X,y)

para = list(range(3,12,2))

print(para)
result={}

for n in para:

    print('para= ',n)

    model=RandomForestClassifier(n_estimators=n,random_state=1)

    model.fit(train_X,train_y)

    preds=model.predict(val_X)

    accu=accuracy_score(y_true=val_y,y_pred=preds)

    f1=f1_score(y_true=val_y,y_pred=preds,average='micro')

    print(classification_report(y_true=val_y,y_pred=preds))

    print('-------------------------------------------------------')

    result[n]=f1
result
# sorted by key, return a list of tuples

lists = sorted(result.items()) 

# unpack a list of pairs into two tuples

p, a = zip(*lists) 

plt.plot(p,a)

plt.show()
# zoo_model=RandomForestClassifier()

# #Fil Model

# zoo_model.fit(train_X,train_y)
# #make Prediction

# val_predict=zoo_model.predict(val_X)

# #calculate mae

# accuracy_score(val_predict, val_y)
# #Support Vector Machines



# from sklearn import svm

# zoo_model_svc=svm.SVC()

# zoo_model_svc.fit(train_X,train_y)

# val_predict_svc=zoo_model_svc.predict(val_X)

# accuracy_score(val_predict_svc, val_y)

# #Desion Tree

# from sklearn import tree

# zoo_model_tree=tree.DecisionTreeClassifier()

# zoo_model_tree.fit(train_X,train_y)
# val_pred_tree=zoo_model_tree.predict(val_X)

# accuracy_score(val_pred_tree,val_y)