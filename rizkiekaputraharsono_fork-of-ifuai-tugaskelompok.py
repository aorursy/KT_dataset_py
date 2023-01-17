import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt



flags = pd.read_csv('../input/inputt/zoo.data')

flags.head()
from sklearn.model_selection import train_test_split

X = flags.iloc[:,1:17]

y = flags.iloc[:,17]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


from sklearn.tree import export_graphviz

from IPython.display import Image

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import graphviz





# Declare and train the model

clf = DecisionTreeClassifier(random_state = 0,criterion='gini')

clf.fit(X_train, y_train)



y_pred_DecisionTreeClassifier = clf.predict(X_test)



scores = []

score = accuracy_score(y_pred_DecisionTreeClassifier,y_test)

scores.append(score)



global tree  

# Get the tree

tree = []

tree = clf



dot_data = export_graphviz(tree,

                           filled=True, 

                           rounded=True,

                           class_names=["1","2","3","4","5","6","7" ],

                           feature_names=X.columns,

                           out_file=None) 



graph = graphviz.Source(dot_data)  

graph 
#use cross validation score since this is a small size dataset 

from sklearn.model_selection import cross_val_score

score_tree=cross_val_score(clf, X,y, cv=10)

score_tree
cv_scores = []

print("Decision Tree Accuracy: %0.2f (+/- %0.2f)" % (score_tree.mean(), score_tree.std() * 2))

cv_score = score_tree.mean()

cv_scores.append(cv_score)


##############################################################################

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

##############################################################################



# Import CSV (Change filepath in speech marks to where CSV is saved)

Zoo = pd.read_csv("../input/inputt/ZooDataset.csv")



# Convert CSV to dataframe

Data = pd.DataFrame(Zoo)



# Feature matrix

X = np.array(Data.loc[:,["Hair","Feathers", "Eggs", "Milk", "Airborne", 

             "Aquatic","Predator", "Toothed", "Backbone", "Breathes", 

             "Venomous","Fins", "Legs", "Tail", "Domestic", "Catsize"]])



# Target we want to predict

y = np.array(Data.loc[:,"Type"])



# Array for predictions

prdList = [[]]



# Array for y[test]

yList = [[]]



# Creates array to store means

MeanList = []



# Mean array is currently empty

MeanTotal = 0



# Runs KFold 5 times

for i in range(5):

    

    # Creates array to store scores

    ScoreList = []



    # Score array is currently empty

    Total = 0



    # KFold instance, does 10 folds

    kf = KFold(n_splits=10, random_state=None, shuffle=True)



    # KNN instance using 3 neighbours (default) for each feature

    knn = KNeighborsClassifier()



    # Split data into training and testing within K-Fold

    for train, test in kf.split(X):

    

        # Fit test and train arrays

        fit = knn.fit(X[train], y[train])

    

        # Create prediction

        prd = knn.predict(X[test])

        

        # Append predictions to array

        prdList.append(prd)

        

        # Append y[test] to array

        yList.append(y[test])

    

        # Add score to array

        ScoreList.append(accuracy_score(y[test],prd))

    

        # Display accuracy score

        print("The accuracy score is", accuracy_score(y[test],prd))



    # Add each score as value to 'List' array

    for val in ScoreList:

        Total += val

    

    # Creates a list of 10 means

    MeanList.append(Total/len(ScoreList))

    

    # Total of all the means

    MeanTotal += (Total/len(ScoreList))

    

    # Display mean accuracy score     

    print ("\nThe mean accuracy score is", Total/len(ScoreList))



    # Display confusion matrix

    print ("\nConfusion matrix:\n", 

           confusion_matrix(train[y], knn.predict(train[X])), "\n")

    

    # Sort the list of means in ascending order

    MeanList.sort()

    

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



# Display list of all 100 means

print("\nList of means:")

for i in range(len(MeanList)):

    print(MeanList[i])

    

# Display mean of all 100 means    

print("\nThe grand mean is", MeanTotal/len(MeanList))



# Show lists of predicted and actual animal types

print("\nPredicted animal types:",prdList[1]) 

print("Actual animal types:",yList[1])



# Display graph showing predicted vs actual types

plt.title("Predicted vs actual animal types for one K-Fold sample\n")

plt.plot(prdList[1], 'b', label = 'prediction')

plt.plot(yList[1], 'r', label = 'y_test')

plt.xlabel("Sample")

plt.ylabel("Animal type")

plt.show()
from sklearn.neighbors import KNeighborsClassifier

# Declare the model

clf = KNeighborsClassifier(n_neighbors=5)



# Train the model

clf.fit(X_train, y_train)

y_pred_KNeighborsClassifier = clf.predict(X_test)

#Get Accuracy Score

score = accuracy_score(y_pred_KNeighborsClassifier,y_test)

scores.append(score)



#Get cross validation score of K-Nearest Neighbors

score_knn=cross_val_score(clf, X,y, cv=10)

print("Support Vector Machine Accuracy: %0.2f (+/- %0.2f)" % (score_knn.mean(), score_knn.std() * 2))

cv_score = score_knn.mean()

cv_scores.append(cv_score)
df = pd.read_csv('../input/inputt/zoo.data')

df.head()
df.describe