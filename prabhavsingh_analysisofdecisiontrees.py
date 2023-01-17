import pandas as pd

import numpy as np

import math

from sklearn import datasets
'''SkLearn Datasets is used to load the IRIS DATABASE that includes information about

different classes of the IRIS plant species. These species are SETOSA, VIRSICOLOR and 

VIRGINICA. The features that are used to classify them are SEPAL_WIDTH, SEPAL_LENGTH,

PETAL_WIDTH, PETAL_LENGTH.'''



'''All the data like petal length etc are stored in numpy array DATA and the target,

ie : category of species is stored in TARGET. Hence X, Y have made using them.'''



df = datasets.load_iris()

X1 = df.data

Y1 = df.target
class TreeNode:

    def __init__(self, data, output):

        # data - Represents the feature upon which the decision tree node was split.

        # data is NULL for leaf node.

        self.data = data

        # children of a node are stored as a dicticionary with key being the value of feature upon which the node was split.

        # and the corresponding value stores the child TreeNode.

        self.children = {}

        # output represents the class with current majority at this instance of the decision tree.

        self.output = output

        self.index = -1

        

    def add_child(self, feature_value, obj):

        self.children[feature_value] = obj

    # add_child is used add objects to the children dictionary corresponding to a particular feature.
class DecisionTreeClassifier:

    def __init__(self):

        # root represents the root node of the data after fitting data.

        self.root = None

    

    def count_unique(self, Y):

        # This function will take Y (Classes) as input and then return a dictionary with

        # keys as unique values of Y and corresponding value as its frequency in Y.

        dicti = {}

        for i in Y:

            if i not in dicti:

                dicti[i]=1

            else:

                dicti[i]+=1

        return dicti

    

    def entropy(self, Y):

        # This function will return the entropy of the node.

        freq_dicti = self.count_unique(Y)

        entropy = 0

        length = len(Y)

        for i in freq_dicti:

            prob = freq_dicti[i] / length

            entropy = entropy + ( (-prob) * math.log2(prob) )

        return entropy

    

    def gain_ratio(self, X, Y, feature):

        # Returns the gain ratio

        orig_entropy = self.entropy(Y) # orig_entropy represents entropy before splitting

        new_entropy = 0  # new_entropy represents entropy after splitting upon the selected feature

        split_info = 0

        values = set(X[:,feature])

        df = pd.DataFrame(X)

        # Adding Y values as the last column in the dataframe 

        df[df.shape[1]] = Y

        initial_size = df.shape[0] 

        for i in values:

            df1 = df[df[feature] == i]

            current_size = df1.shape[0]

            new_entropy += (current_size/initial_size)*self.entropy(df1[df1.shape[1]-1])

            split_info += (-current_size/initial_size)*math.log2(current_size/initial_size)

        # To handle the case when split info = 0 which leads to division by 0 error

        if split_info == 0 :

            return math.inf

        # Otherwise

        entropy_gain = orig_entropy - new_entropy

        gain_ratio = entropy_gain / split_info

        return gain_ratio

    

    def gini_index(self, Y):

        # Returns the gini index 

        freq_dicti = self.count_unique(Y)

        gini_index = 1

        length = len(Y)

        for i in freq_dicti:

            p = freq_dicti[i] / length

            gini_index = gini_index - p**2

        return gini_index

    

    def gini_gain(self, X, Y, feature):

        # Returns the gini gain

        gini_orig = self.gini_index(Y)   # gini_orig represents gini index before splitting

        gini_split_f = 0                 # gini_split_f represents gini index after splitting upon the selected feature

        values = set(X[:,feature])

        df = pd.DataFrame(X)

        # Adding Y values as the last column in the dataframe 

        df[df.shape[1]] = Y

        initial_size = df.shape[0] 

        for i in values:

            df1 = df[df[feature] == i]

            current_size = df1.shape[0]

            gini_split_f += (current_size/initial_size)*self.gini_index(df1[df1.shape[1]-1])

        gini_gain = gini_orig - gini_split_f

        return gini_gain

    

    def decision_tree(self, X, Y, features, level, metric, classes):

        '''Returns the root of the Decision Tree (which consists of Class TreeNodes) built after fitting the training data.

        Here Nodes are printed as in PREORDER traversal. Classes represents the different classes present in the classification problem. 

        Metric can take value gain_ratio or gini_index.

        Level represents depth of the tree.

        We split a node on a particular feature only once to avoid overlap.'''

           

        # If the node consists of only 1 class.

        if (len(set(Y)) == 1):

            print("Level", level)

            output = None

            for i in classes:

                if i in Y:

                    output = i

                    print("Count of",i,"=",len(Y))

                else:

                    print("Count of",i,"=",0)

            if metric == "gain_ratio":

                print("Current Entropy is =  0.0")

            elif metric == "gini_index":

                print("Current Gini Index is =  0.0")

            print("Reached leaf Node")

            print()

            return TreeNode(None,output)

        

        # If we have run out of features to split upon.

        # In this case we will output the class with maximum count. This will be the final result.

        if len(features) == 0:

            print("Level",level)

            freq_dicti = self.count_unique(Y)

            output = None

            max_count = -math.inf

            for i in classes:

                if i not in freq_dicti:

                    print("Count of",i,"=",0)

                else :

                    if freq_dicti[i] > max_count :

                        output = i

                        max_count = freq_dicti[i]

                    print("Count of",i,"=",freq_dicti[i])



            if metric == "gain_ratio":

                print("Current Entropy  is =",self.entropy(Y))

            elif metric == "gini_index":

                print("Current Gini Index is =",self.gini_index(Y))            

            print("Reached leaf Node")

            print()

            return TreeNode(None,output)

        

        # Finding the best feature to split upon further.

        max_gain = -math.inf

        final_feature = None

        for f in features :

            if metric == "gain_ratio":

                current_gain = self.gain_ratio(X,Y,f)

            elif metric =="gini_index":

                current_gain = self.gini_gain(X,Y,f)

            if current_gain > max_gain:

                max_gain = current_gain

                final_feature = f

                

        print("Level",level)

        freq_dicti = self.count_unique(Y)

        output = None

        max_count = -math.inf

        

        # Printin count of all features at that node.

        for i in classes:

            if i not in freq_dicti:

                print("Count of",i,"=",0)

            else :

                if freq_dicti[i] > max_count :

                    output = i

                    max_count = freq_dicti[i]

                print("Count of",i,"=",freq_dicti[i])

        # Using input metric to determine feature to be split on.    

        if metric == "gain_ratio" :        

            print("Current Entropy is =",self.entropy(Y))

            print("Splitting on feature  X[",final_feature,"] with gain ratio ",max_gain,sep="")

            print()

        elif metric == "gini_index":

            print("Current Gini Index is =",self.gini_index(Y))

            print("Splitting on feature  X[",final_feature,"] with gini gain ",max_gain,sep="")

            print()

            

        unique_values = set(X[:,final_feature]) # unique_values represents the unique values of the feature selected.

        df = pd.DataFrame(X)

        # Adding Y values as the last column in the dataframe

        df[df.shape[1]] = Y

        current_node = TreeNode(final_feature,output)



        # Now removing the selected feature from the list as we do not want to split on one feature more 

        # than once(in a given root to leaf node path).

        index  = features.index(final_feature)

        features.remove(final_feature)

        for i in unique_values:

            # Creating a new dataframe with value of selected feature = i

            df1 = df[df[final_feature] == i]

            # Segregating the X and Y values and recursively calling on the splits

            node = self.decision_tree(df1.iloc[:,0:df1.shape[1]-1].values,df1.iloc[:,df1.shape[1]-1].values,features,level+1,metric,classes)

            current_node.add_child(i,node)

        # Add the removed feature     

        features.insert(index,final_feature)

        return current_node

    

    def fit(self, X, Y, metric="gain_ratio"):

        # Fits to the given training data.

        # Metric can take value gain_ratio or gini_index. Default value will be gain_ratio.

        features = [i for i in range(len(X[0]))]

        classes = set(Y)

        level = 0

        if metric != "gain_ratio" :

            if metric != "gini_index":

                metric="gain_ratio"  # If user entered a value which was neither gini_index nor gain_ratio

        self.root = self.decision_tree(X,Y,features,level,metric,classes)

        

    def predict_helper(self, data, node):

        # predicts the class for a given testing point and returns the answer.

        # If we have reached a leaf node :      

        if len(node.children) == 0 :

            return node.output

        val = data[node.data] # represents the value of feature on which the split was made       

        if val not in node.children :

            return node.output

        # Recursively call on the splits.

        return self.predict_helper(data,node.children[val])



    def predict(self, X):

        # This function returns Y predicted.

        # X should be a 2-D np array.

        Y = np.array([0 for i in range(len(X))])

        for i in range(len(X)):

            Y[i] = self.predict_helper(X[i],self.root)

        return Y

    

    def score(self,X,Y):

        # Returns the mean accuracy of predictions.

        Y_pred = self.predict(X)

        count = 0

        for i in range(len(Y_pred)):

            if Y_pred[i] == Y[i]:

                count+=1

        return count/len(Y_pred)
clf1 = DecisionTreeClassifier()

x = np.array([[0,0],

              [0,1],

              [1,0],

              [1,1]])

y = np.array([0,

              1,

              1,

              1]) 

clf1.fit(x,y)

Y_pred = clf1.predict(x)

print("Predictions :",Y_pred)

print()

print("Score :",clf1.score(x,y)) # Score on training data

print()
'''USING MY DECISION TREE CLASSIFIER'''



from sklearn import datasets

# Generating a random dataset

X, Y = datasets.make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=0)

# To reduce the values a feature can take ,converting floats to int

for i in range(len(X)):

    for j in range(len(X[0])):

        X[i][j] = int(X[i][j])

        

clf2 = DecisionTreeClassifier()

clf2.fit(X,Y,metric='gini_index')

Y_pred2 = clf2.predict(X)

print("Predictions : ",Y_pred2)

print()

our_score = clf2.score(X,Y)

print("Score :",our_score) # score on training data

print()
'''USING SKLEARN INBUILT CLASSIFIER'''



import sklearn.tree

clf3 = sklearn.tree.DecisionTreeClassifier()

clf3.fit(X,Y)

Y_pred3 = clf3.predict(X)

print("Predictions",Y_pred3)

sklearn_score = clf3.score(X,Y)

print("Score :",sklearn_score)
'''SCORE OF BOTH CLASSFIERS IS SAME. BOTH PERFORM WELL ON TRAINING DATA'''
'''USING MY DECISION TREE CLASSIFIER'''

# Use train_test_split for IRIS Data.

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, random_state = 0)



# TRAINING DATA

clf = DecisionTreeClassifier()

clf.fit(X_train,Y_train,metric='gini_index')

Y_pred_train = clf.predict(X_train)

print("Predictions for train data : ",Y_pred_train)

print()

train_score = clf.score(X_train,Y_train)

print("Score :", train_score)

print()



# TEST DATA

Y_pred_test = clf.predict(X_test)

print("Predictions for test data : ",Y_pred_test)

print()

test_score = clf.score(X_test,Y_test)

print("Score :", test_score)

print()
'''USING SKLEARN INBUILT CLASSIFIER'''



import sklearn.tree

clfsk = sklearn.tree.DecisionTreeClassifier()

clfsk.fit(X_train,Y_train)

Y_pred_train_sk = clfsk.predict(X_train)

print("Predictions for train",Y_pred_train_sk)

sklearn_score_train = clfsk.score(X_train,Y_train)

print("Score for training :",sklearn_score_train)

Y_pred_test_sk = clfsk.predict(X_test)

print("Predictions for test",Y_pred_test_sk)

sklearn_score_test = clfsk.score(X_test,Y_test)

print("Score for training :",sklearn_score_test)