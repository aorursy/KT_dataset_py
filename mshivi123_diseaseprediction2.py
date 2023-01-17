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
import pandas as pd

from sklearn import preprocessing

from sklearn.tree import DecisionTreeClassifier,_tree

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.tree import export_graphviz

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

training = pd.read_csv("/kaggle/input/diseasepredictio-n/Training.csv")

testing  = pd.read_csv("/kaggle/input/diseasepredictio-n/Testing.csv")

cols     = training.columns

cols     = cols[:-1]

x        = training[cols]

y        = training['prognosis']

y1       = y

reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers

le = preprocessing.LabelEncoder()

le.fit(y)

y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

testx    = testing[cols]

testy    = testing['prognosis']  

testy    = le.transform(testy)

clf1  = DecisionTreeClassifier()

clf = clf1.fit(x_train,y_train)

#print(clf.score(x_train,y_train))

#print ("cross result========")

#scores = cross_validation.cross_val_score(clf, x_test, y_test, cv=3)

#print (scores)

#print (scores.mean())

#print(clf.score(testx,testy))importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

features = cols

print("Please reply Yes or No for the following symptoms") 

def print_disease(node):

    #print(node)

    node = node[0]

    #print(len(node))

    val  = node.nonzero() 

    #print(val)

    disease = le.inverse_transform(val[0])

    return disease

def tree_to_code(tree, feature_names):

    tree_ = tree.tree_

    #print(tree_)

    feature_name = [

        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"

        for i in tree_.feature

    ]

    #print("def tree({}):".format(", ".join(feature_names)))

    symptoms_present = []

    def recurse(node, depth):

        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:

            name = feature_name[node]

            threshold = tree_.threshold[node]

            print(name + " ?")

            ans = input()

            ans = ans.lower()

            if ans == 'yes':

                val = 1

            else:

                val = 0

            if  val <= threshold:

                recurse(tree_.children_left[node], depth + 1)

            else:

                symptoms_present.append(name)

                recurse(tree_.children_right[node], depth + 1)

        else:

            present_disease = print_disease(tree_.value[node])

            print( "You may have " +  present_disease )

            red_cols = reduced_data.columns 

            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

            print("symptoms present  " + str(list(symptoms_present)))

            print("other symptoms "  +  str(list(symptoms_given)) )  

            confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)

            print("confidence level is " + str(confidence_level))

            

    recurse(0, 1)

    

tree_to_code(clf,cols)

    

    




