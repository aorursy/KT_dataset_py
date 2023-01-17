from math import log



def entropy(*probs):

  """Calculate information entropy"""

  try:

    total = sum(probs)

    return sum([-p / total * log(p / total, 2) for p in probs])

  except:

    return 0



age = {

    '<=30': entropy(2, 6),

    '31-40': entropy(6, 0),

    '>40': entropy(4, 2),

}



income = {

    'high': entropy(3, 2),

    'medium': entropy(5, 3),

    'low': entropy(4, 3),

}



student = {

    'yes': entropy(8, 1),

    'no': entropy(4, 7),

}



credit_ranking = {

    'excellent': entropy(5, 5),

    'fair': entropy(7, 3),

}





entropy_root = entropy(12, 8)

entropy_age = 8/20 * age['<=30'] + 6/20 * age['31-40'] + 6/20 * age['>40']

entropy_income = 5/20 * income['high'] + 8/20 * income['medium'] + 7/20 * income['low']

entropy_student = 9/20 * student['yes'] + 11/20 * student['no']

entropy_credit_ranking = 10/20 * credit_ranking['excellent'] + 10/20 * credit_ranking['fair'] 



print('The root entropy is H(T):')

print(entropy_root)

print('')

print('The resulting entropy is H(T, age)')

print(entropy_age)

print('Thus, information gain if the set is split according to age')

print(entropy_root - entropy_age)

print('')

print('The resulting entropy is H(T, income)')

print(entropy_income)

print('Thus, information gain if the set is split according to income')

print(entropy_root - entropy_income)

print('')

print('The resulting entropy is H(T, student)')

print(entropy_student)

print('Thus, information gain if the set is split according to student')

print(entropy_root - entropy_student)

print('')

print('The resulting entropy is H(T, credit_ranking)')

print(entropy_credit_ranking)

print('Thus, information gain if the set is split according to credit rating')

print(entropy_root - entropy_credit_ranking)
from graphviz import Digraph



tree = Digraph()



tree.edge("Age\nsamples = 20", "Student\nsamples = 8\nvalue = [6, 2]", "<=30")

tree.edge("Student\nsamples = 8\nvalue = [6, 2]", "Buying\nsamples = 2\nvalue = [0, 2]", "Yes")

tree.edge("Student\nsamples = 8\nvalue = [6, 2]", "Not Buying\nsamples = 6\nvalue = [6, 0]", "No")



tree.edge("Age\nsamples = 20", "Buying\nsamples = 6\nvalue = [0, 6]", "31-40")



tree.edge("Age\nsamples = 20", "Student\nsamples = 6", ">40")

tree.edge("Student\nsamples = 6", "Credit Rating\nsamples = 4", "Yes")

tree.edge("Credit Rating\nsamples = 4", "Buying\nsamples = 2\nvalue = [0,2]", "Fair")

tree.edge("Credit Rating\nsamples = 4", "Income\nsamples = 2\nvalue = [1, 1]", "Excellent")

tree.edge("Income\nsamples = 2\nvalue = [1, 1]", "Buying\nsamples = 1\nvalue = [0, 1]", "Low")

tree.edge("Income\nsamples = 2\nvalue = [1, 1]", "Not Buying\nsamples = 1\nvalue = [1, 0]", "Medium")



tree.edge("Student\nsamples = 6", "Credit Rating\nsamples = 2", "No")

tree.edge("Credit Rating\nsamples = 2", "Buying\nsamples = 1", "Fair")

tree.edge("Credit Rating\nsamples = 2", "Not Buying\nsamples = 1", "Excellent")



tree
import numpy as np

import pandas as pd

from sklearn.datasets import load_iris

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz



# translate values into unique integers to handle them



x = {

    'Age': [

        0, 0, 1, 2, 2, 

        2, 1, 0, 0, 2,

        0, 1, 1, 2, 0, 

        0, 0, 1, 2, 1 

    ], 

    'Income': [

        4, 4, 4, 5, 3, 

        3, 3, 5, 3, 5, 

        5, 5, 4, 5, 5,

        3, 3, 3, 5, 4

    ],

    'Student': [

        6, 6, 6, 6, 7, 

        7, 7, 6, 7, 7, 

        7, 6, 7, 6, 6, 

        6, 6, 7, 7, 6 

    ],

    'Credit_rating': [

        9, 8, 9, 9, 9,

        8, 8, 9, 9, 9,

        8, 8, 9, 8, 8,

        9, 8, 9, 8, 8

    ],

}



y = {

    'Buys Computer': [

        'Buys Not', 'Buys Not', 'Buys', 'Buys', 'Buys', 

        'Buys Not', 'Buys', 'Buys Not', 'Buys', 'Buys', 

        'Buys', 'Buys', 'Buys', 'Buys Not', 'Buys Not', 

        'Buys Not', 'Buys Not', 'Buys', 'Buys', 'Buys' 

    ]

}



x_df = pd.DataFrame(data=x)

y_df = pd.DataFrame(data=y)



X = x_df.to_numpy()

y = y_df.to_numpy()



tree_clf = DecisionTreeClassifier()

tree_clf.fit(X, y)



from sklearn.tree import export_graphviz



x_df = pd.DataFrame(data=x)

y_df = pd.DataFrame(data=y)



X = x_df.to_numpy()

y = y_df.to_numpy().astype('U10')

    

export_graphviz(

     tree_clf,

     feature_names=list(x_df.columns),

     class_names=np.unique(y),

     filled=True

 )