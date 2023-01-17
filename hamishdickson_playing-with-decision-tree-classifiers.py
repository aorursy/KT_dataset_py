# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

import graphviz
from subprocess import check_call

from IPython.display import Image as PImage
from PIL import Image, ImageDraw, ImageFont
df = pd.read_csv('../input/train.csv', error_bad_lines=False)
df.head(3)
df = df.dropna(axis=1)
df = df.drop(['Name', 'Ticket'], axis=1)
X = pd.get_dummies(df.drop('Survived', axis=1))
Y = pd.get_dummies(df['Survived'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
X_train.head(3)
clf = DecisionTreeClassifier(max_depth=3, min_samples_split=20)
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)
with open("tree1.dot", 'w') as f:
     f = export_graphviz(clf,
                              out_file=f,
                              max_depth = 3,
                              impurity = True,
                              feature_names = list(X_train),
                              class_names = ['Died', 'Survived'],
                              rounded = True,
                              filled= True )

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")
