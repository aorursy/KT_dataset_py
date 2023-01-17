# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from PIL import Image



import matplotlib.pyplot as plt



%matplotlib inline
test_url = "/kaggle/input/heart-disease-uci/heart.csv"

test = pd.read_csv(test_url)



print(test)
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.model_selection import train_test_split



X = np.array(pd.DataFrame(test, columns=['age','sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']))



print(X)

y = np.array(pd.DataFrame(test, columns=['target']))



print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y)

X_train
y_train
dt_clf = DecisionTreeClassifier()

dt_clf = dt_clf.fit(X_train, y_train)
dt_prediction = dt_clf.predict(X_test)
print("Train set Score : {:.2f}".format(dt_clf.score(X_train, y_train)))

print("Test set Score : {:.2f}".format(dt_clf.score(X_test, y_test)))
dTreeLimit = DecisionTreeClassifier(max_depth=3, random_state=0)



dTreeLimit = dTreeLimit.fit(X_train, y_train)
print("Train set Score : {:.2f}".format(dTreeLimit.score(X_train, y_train)))

print("Test set Score : {:.2f}".format(dTreeLimit.score(X_test, y_test)))
import os

from sklearn import tree



os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

feature_names = test.columns.tolist()



feature_names = feature_names[0:13]



target_name = np.array(['Play No', 'Play Yes'])



dt_dot_data = tree.export_graphviz(dt_clf, out_file = "exampleTest.dot",

                                  feature_names = feature_names,

                                  class_names = target_name,

                                  filled = True, rounded = True,

                                  special_characters = True)
import pydot

(graph,) = pydot.graph_from_dot_file('exampleTest.dot', encoding='utf8')



graph.write_png('exampeTest.png')
path = './exampeTest.png'



image_pil = Image.open(path)

image = np.array(image_pil)



print(image.shape)
plt.figure(figsize=(50,50))

plt.imshow(image)

plt.show()