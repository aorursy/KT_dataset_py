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
Data = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
Data
Data.columns
for i in Data.columns:
    print('this {} column has unique_vlaues = \n'.format(i),Data[i].value_counts())
for i in Data.columns: 
    a = Data[i].isnull().sum()
    if a !=0:
        print(i,' has ', Data[i].isnull().sum())
    else:
        print('no null values')

color = {"b": 0, "c": 1,"e": 2, "g": 3,"n": 4, "p": 5,"r": 6, "u": 7,"w": 8, "y": 9}


for i, j in color.items():
    Data["cap-color"] = Data["cap-color"].replace(i, j)
Data['gill-color'].unique()

color = {"b": 0,"e": 2, "g": 3,"n": 4, "p": 5,"r": 6, "u": 7,"w": 8, "y": 9, 'h':10,'k':11,'o':12}


for i, j in color.items():
    Data["gill-color"] = Data["gill-color"].replace(i, j)
Data['veil-color'].unique()

color = {"n": 4,"w": 8, "y": 9,'o':12}


for i, j in color.items():
    Data["veil-color"] = Data["veil-color"].replace(i, j)
Data['spore-print-color'].unique()
color = {"b": 0,"n": 4,"r": 6, "u": 7,"w": 8, "y": 9, 'h':10,'k':11,'o':12}


for i, j in color.items():
    Data["spore-print-color"] = Data["spore-print-color"].replace(i, j)
Data['stalk-color-above-ring'].unique()

color = {"b": 0, "c": 1,"e": 2, "g": 3,"n": 4, "p": 5, "w": 8, "y": 9,'o':12}


for i, j in color.items():
    Data["stalk-color-above-ring"] = Data["stalk-color-above-ring"].replace(i, j)
Data['stalk-color-below-ring'].unique()

color = {"b": 0, "c": 1,"e": 2, "g": 3,"n": 4, "p": 5, "w": 8, "y": 9,'o':12}


for i, j in color.items():
    Data["stalk-color-below-ring"] = Data["stalk-color-below-ring"].replace(i, j)
Data['cap-shape'].unique()
color = {"b": 0,"c": 1,"f": 2, "k": 3,"s": 4, "x": 5}


for i, j in color.items():
    Data["cap-shape"] = Data["cap-shape"].replace(i, j)
Data['cap-surface'].unique()
color = {"f": 2,"s": 4, 'g':6,'y':7}


for i, j in color.items():
    Data["cap-surface"] = Data["cap-surface"].replace(i, j)
Data['bruises'].unique()
color = {"f": 0,"t": 1}


for i, j in color.items():
    Data["bruises"] = Data["bruises"].replace(i, j)
Data['odor'].unique()
color = {'a':0,'c':1,'f':2,'l':3,'m':4,'n':5,'p':6,'s':7,'y':8}


for i, j in color.items():
    Data["odor"] = Data["odor"].replace(i, j)
Data['gill-attachment'].unique()
color = {"f": 0,"a": 1}


for i, j in color.items():
    Data["gill-attachment"] = Data["gill-attachment"].replace(i, j)
Data['gill-spacing'].unique()
color = {"c": 0,"w": 1}


for i, j in color.items():
    Data["gill-spacing"] = Data["gill-spacing"].replace(i, j)
Data['gill-size'].unique()
color = {"n": 0,"b": 1}


for i, j in color.items():
    Data["gill-size"] = Data["gill-size"].replace(i, j)
Data['stalk-shape'].unique()
color = {"e": 0,"t": 1}


for i, j in color.items():
    Data["stalk-shape"] = Data["stalk-shape"].replace(i, j)
Data['stalk-root'].unique()
color = {"b": 0,"c": 1,'e':2,'r':3,'?':4}


for i, j in color.items():
    Data["stalk-root"] = Data["stalk-root"].replace(i, j)
Data['stalk-surface-above-ring'].unique()
color = {"f": 0,"k": 1,'s':2,'y':3}


for i, j in color.items():
    Data["stalk-surface-above-ring"] = Data["stalk-surface-above-ring"].replace(i, j)
Data['stalk-surface-below-ring'].unique()
color = {"f": 0,"k": 1,'s':2,'y':3}


for i, j in color.items():
    Data["stalk-surface-below-ring"] = Data["stalk-surface-below-ring"].replace(i, j)
Data['veil-type'].unique()
color = {"p": 0}


for i, j in color.items():
    Data["veil-type"] = Data["veil-type"].replace(i, j)
Data['ring-number'].unique()
color = {"o": 0,'t':1, 'n':2}


for i, j in color.items():
    Data["ring-number"] = Data["ring-number"].replace(i, j)
Data['ring-type'].unique()
color = {"e": 0,'f':1, 'l':2, 'n':3, 'p':4}


for i, j in color.items():
    Data["ring-type"] = Data["ring-type"].replace(i, j)
Data['population'].unique()
color = {"a": 0,'c':1, 'n':2, 's':3, 'v':4, 'y':5}


for i, j in color.items():
    Data["population"] = Data["population"].replace(i, j)
Data['habitat'].unique()
color = {"d": 0,'g':1, 'l':2, 'm':3, 'p':4, 'u':5, 'w':6}


for i, j in color.items():
    Data["habitat"] = Data["habitat"].replace(i, j)
Data['class'].unique()
color = {"e": 0,'p':1}


for i, j in color.items():
    Data["class"] = Data["class"].replace(i, j)
Data
Train = Data.iloc[:7000, :]
Test = Data.iloc[7000:, :]
Train
X = Train.drop(columns=['class'])
Y = Train['class']
test_data = Test.drop(columns='class')
test_class = Test['class']
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X, Y)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
prediction = model.predict(test_data)
print('accuracy_score is ', (accuracy_score(prediction, test_class)))
print('classification_report is ', (classification_report(prediction, test_class)))
print('confusion_matrix is \n', (confusion_matrix(prediction, test_class)))
