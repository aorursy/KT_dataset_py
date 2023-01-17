#===========================================================================

# import the libraries

#===========================================================================

import pandas  as pd

import numpy as np

import matplotlib.pyplot as plt



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/titanic/train.csv')

test_data  = pd.read_csv('../input/titanic/test.csv')



#===========================================================================

# features: Sex, and a bit of class...

#===========================================================================

features = ["Sex", "Pclass"]



#===========================================================================

# for the features that are categorical we use pd.get_dummies

#===========================================================================

X_train       = pd.get_dummies(train_data[features])

y_train       = train_data["Survived"]

X_test        = pd.get_dummies(test_data[features])



#===========================================================================

# perform the classification using a decision tree 

#===========================================================================

from sklearn import tree

classifier = tree.DecisionTreeClassifier(criterion='gini',

                                         splitter='best',

                                         max_depth=2,

                                         random_state=42)

classifier.fit(X_train, y_train)
from sklearn.tree import export_graphviz

import graphviz



dot_data = export_graphviz(classifier, 

                           feature_names=X_train.columns, 

                           class_names=['Died', 'Survived'], 

                           filled=True, 

                           rounded=True,

                           proportion=False)

graphviz.Source(dot_data)
classifier = tree.DecisionTreeClassifier(criterion='entropy',

                                         splitter='best',

                                         max_depth=2,

                                         random_state=42)

classifier.fit(X_train, y_train)



# now visualise the 'entropy' tree

dot_data = export_graphviz(classifier, 

                           feature_names=X_train.columns, 

                           class_names=['Died', 'Survived'], 

                           filled=True, 

                           rounded=True,

                           proportion=False)

graphviz.Source(dot_data)
plt.figure(figsize = (12,4))

limit = 1

x = np.linspace(0.01, limit-0.01, 50)

line_1 = 1 - (x**2) - (1-x)**2

line_2 = ( -1*x*np.log2(x) ) - ( (1-x)*np.log2(1-x) )

#------------------------------------------

plt.plot(x,line_1, color='darkorange', linestyle='solid', lw=3)

plt.plot(x,line_2, color='navy',       linestyle='dashed', lw=3)

#------------------------------------------

plt.title   ("Plot of the gini (orange) and entropy (blue) impurity criterion")

plt.xlabel  ("p")

plt.ylabel  ("impurity")

#------------------------------------------

plt.xlim    (0, limit)

plt.ylim    (0, limit+0.1)

#------------------------------------------

plt.grid(True)

#------------------------------------------

plt.show()
# first fill an array with zeros (i.e. initially there are no survivors at all)

predictions = np.zeros((418), dtype=int)



# now use our model

survived_df = X_test[((X_test["Pclass"] ==1)|(X_test["Pclass"] ==2)) & (X_test["Sex_female"]==1 )]



for i in survived_df.index:

    predictions[i] = 1 # the 1's are now the survivors

    

#===========================================================================

# write out CSV submission file

#===========================================================================

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('submission.csv', index=False)