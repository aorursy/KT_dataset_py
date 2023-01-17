import numpy as np 

import pandas as pd





train = pd.read_csv('../input/train.csv') # Test data

test = pd.read_csv('../input/test.csv') # Test data



datasets = [train, test]



train.tail()
import re

epic_titles = set(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'])

female_titles = set(['Lady', 'Countess', 'Dona', 'Mlle', 'Ms', 'Mme'])

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}



def mapFare(fare):

    if fare <= 7.91:

        return 0

    elif fare > 7.91 and fare <= 14.454:

        return 1

    elif fare > 14.454 and fare <= 31:

        return 2

    elif fare > 31:

        return 3



def mapAge(age):

    if age <= 16:

        return 0

    elif age > 16 and age <= 32:

        return 1

    elif age > 32 and age <= 48:

        return 2

    elif age > 48 and age <= 64:

        return 3

    elif age > 64:

        return 4

    

for dataset in datasets:

    # Adding feature of family size

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    dataset.drop([ 'SibSp', 'Parch'], axis=1, inplace=True)

    

    

    # Adding title as a feature

    dataset['Title'] = dataset['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

    dataset['Title'] = dataset['Title'].apply(lambda x: x if x not in epic_titles else 'Epic')

    

    # Fill NaN for Sex based on Title 

    dataset['Sex'] = dataset['Sex'].fillna(dataset['Title'].apply(lambda x: 'female' if x in female_titles else 'male'))



    # Changing from old names to new names

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



    # Mapping title columns to int values

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Epic": 5}

    dataset['Title'] = dataset['Title'].apply(lambda x: title_mapping[x] if x in title_mapping else 5)



    # Mapping Sex columns

    dataset['Sex'] = dataset['Sex'].map( {"female": 0, "male": 1} ).astype(int)



    # Mapping Age from 0 to 4

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean()) # First replacing NaN with meanvalue

    dataset['Age'] = dataset['Age'].apply(lambda x: mapAge(x))

    dataset['Age'] = dataset['Age'].astype(int)



    # Mapping Fare

    #dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    #dataset['Fare'] = dataset['Fare'].apply(lambda x: mapFare(x))

    #dataset['Fare'] = dataset['Fare'].astype(int)



    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping deck feature

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)

    # we can now drop the cabin feature

    dataset.drop(['Cabin'], axis=1, inplace=True)

    

    familysizeDummies = pd.get_dummies(dataset['FamilySize'], prefix='familysize')

    dataset = pd.concat([dataset,familysizeDummies],axis=1)

    dataset = dataset.drop(columns=['FamilySize'])

    print(dataset.columns)

    
# Lookin at sex as a feature

train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean', 'count', 'sum'])
# Selecting columsn for splitting of data in next section

y = train[['Survived']]

x = train.drop(columns=['Name', 'Ticket', 'Survived', 'PassengerId'])

x.columns
import matplotlib.pyplot as plt

import seaborn as sns



colormap = plt.cm.viridis

plt.figure(figsize=(10,10))

plt.title('Titanic Correlation of Features', y=1.05, size=15)

sns.heatmap(train.select_dtypes([np.number]).astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler



# Splitting data for training and validation

train_x, test_x, train_y, test_y = train_test_split(x, np.ravel(y))
logreg_model = LogisticRegression()

logreg_model.fit(train_x, train_y)

preds = logreg_model.predict(test_x)

acc = accuracy_score(preds, test_y)

print(f'Accuracy: {acc}')

# Scaling and centering the data before performing pca

scaler = MinMaxScaler(feature_range=[0, 1])

train_x_scaled = scaler.fit_transform(train_x.iloc[1:, 0:8])

test_x_scaled = scaler.fit_transform(test_x.iloc[1:, 0:len(train_x.columns)])



train_x_centered = train_x - np.mean(train_x, axis = 0)

test_x_centered = train_x - np.mean(test_x, axis = 0)

train_x_centered.describe()
pca = PCA().fit(train_x_centered)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Pulsar Dataset Explained Variance')

plt.show()
# Testing with different # of pca components



for i in range(1, len(train_x.columns)):

    pca = PCA(n_components=i)

    pca.fit(train_x)

    train_x_pca = pca.transform(train_x)

    test_x_pca = pca.transform(test_x)

    

    logreg_model.fit(train_x_pca, train_y)

    preds_pca = logreg_model.predict(test_x_pca)

    acc_pca = accuracy_score(preds_pca, test_y)

    print(f'Accuracy with {i} principal components: {acc_pca}')
"""

Here I will make a decision-tree-algorithm which will trained by a set of training data,

and then used to classify a set of test data. The algorithm adopts a greedy divide-and-conquer

strategy: always test the most important attribute/feature first. Most important means making

the most difference to the classification.



"""



import pandas as pd

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning) # To ignore some future warnings from pandas

import math

import random





def plurality_value(parent_examples):

    # Chooses random if tie, else the most common value for 'y'

    # (value_counts() is sorted, therefore index = 0)

    r = 0

    #if parent_examples['Survived'].value_counts().index[0] == parent_examples['Survived'].value_counts().index[1]:

     #   r = random.randint(1)

    return parent_examples['Survived'].value_counts().index[r]



def allValuesEqual(examples):

    return len(examples['Survived'].value_counts()) == 1





def partition(examples, column, value):

    """  

    :param examples: input dataset/rows 

    :param column: column to match 

    :param value: condition value to match each row with, equals 0 or 1

    :return: matching rows and nonmatching rows

    """



    true_rows, false_rows = examples.copy(), examples.copy()

    for index, row in examples.iterrows():

        if row[column] == value:

            false_rows = false_rows.drop([index])

        else:

            true_rows = true_rows.drop([index])

    return true_rows, false_rows





def get_entropy(examples):

    # Calculating the number of the examples that have ouput (y) == 1

    number_of_ones = 0

    for index, row in examples.iterrows():

        if row['Survived'] == 1:

            number_of_ones += 1



    q = number_of_ones/len(examples) if len(examples) != 0 else 0

    return - ( q*math.log2(q) + (1-q)*math.log2((1-q)) ) if (q != 0 and q!= 1) else 0





def get_remainder(one_rows, two_rows, entropy1, entropy2):

    total = len(one_rows) + len(two_rows)

    p, n = len(one_rows), len(two_rows)

    return (p/total)*entropy1 + (n/total)*entropy2





def find_best_split(examples, attributes, current_entropy):

    """

    :param examples: examples to consider

    :param attributes: attributes to consider

    :param current_entropy: the entropy we are comparing the different new entropies against

    :return: the attributes that makes a split which results in the most information gain

    """



    # For each attribute, calculate info gain and choose attribute with highest info gain

    best_gain = 0

    best_attribute = attributes[0]

    info_gains = []

    for attribute in attributes:

        # Partitions the examples based on whether or not their value for the attribute equals 0

        ones, twoes = partition(examples, attribute, 0)

        # print(ones.shape



        # Calculating entropies for the two partitions

        entropy1, entropy2 = get_entropy(ones), get_entropy(twoes)



        # Calculating the remainder using the entropies

        remainder = get_remainder(ones, twoes, entropy1, entropy2)



        # Skip this split if it doesn't divide the dataset

        #if len(ones) == 0 and len(twoes) == 0:

         #   continue



        # Information gain

        info_gain = current_entropy - remainder

        info_gains.append(info_gain)

        if info_gain >= best_gain:

            best_gain, best_attribute = info_gain, attribute

    #print(best_gain)

    return (best_attribute, best_gain) if best_gain > 0 else (random.choice(attributes), best_gain)





def decision_tree_learning(examples, attributes, parent_examples, info_gain):

    """

    :param examples: examples to consider in this iteration

    :param attributes: attributes 'available' in this iteration, meaning not previously used in the path from 

    root to this node 

    :param parent_examples: the examples as they are before the split 

    :return: a complete Decision Tree (of class Tree) 

    """

    

    # If examples are empty return most common output value among the parent examples (before the last split)

    if len(examples) == 0:

        return plurality_value(parent_examples)



    # If all examples have the same output value, then the partition is pure and we return the classification

    elif allValuesEqual(examples):

        #print(examples.iloc[0]['y'])

        return examples.iloc[0]['Survived']



    # If attributes are empty (no more partition possible) return the plurality value of current examples

    elif len(attributes) == 0:

        return plurality_value(examples)

    

    elif info_gain < 0.005:

        return plurality_value(examples)



    # Else we continue the partition



    current_entropy = get_entropy(examples)

    best_attribute, info_gain = find_best_split(examples, attributes, current_entropy)[0], find_best_split(examples, attributes, current_entropy)[1] # Importance function version 1

    #best_attribute = random.choice(attributes) # Importance function version 2



    tree = Tree(best_attribute)

    # Making a copy and then removing the attribute from that copy,

    # as we need the attribute available in other nonsuccessor branches (when the recursion "comes back" again)

    attr_copy = attributes.copy()

    attr_copy = attr_copy.drop(labels=best_attribute)



    for value in train[best_attribute].unique():

        next_examples = examples.loc[examples[best_attribute] == value]

        subtree = decision_tree_learning(next_examples, attr_copy, examples, info_gain)

        tree.add_branch(value, subtree)

    return tree







class Tree:

    def __init__(self, root, branches=None):

        self.root = root

        if branches == None:

            branches = {}

        self.branches = branches



    def add_branch(self, label, branch):

        self.branches[label] = branch





def predict(tree, example):

    value = example[tree.root]

    if isinstance(tree.branches[value], Tree):

        return predict(tree.branches[value], example)

    else:

        return tree.branches[value]

    #else:

     #   if isinstance(tree.branches[1], Tree):

      #      return predict(tree.branches[1], example)

       # else:

        #    return tree.branches[1]





def print_tree(tree, value="", level=0):

    print("\t" * level + str(value), end = " -> ")

    attribute = tree.root if isinstance(tree, Tree) else str(tree)

    print(attribute, end = "\n")



    if isinstance(tree, Tree):

        for label, subtree in tree.branches.items():

            print_tree(subtree, label, level + 1)



def main():



    print("--------------------------------------------")



#     # Heatmap of correlations between attributes and 'y' (Class attribute)

#     import seaborn as sns

#     import matplotlib.pyplot as plt

#     hm = sns.heatmap(train.corr(), annot=True, linewidth=.5, cmap='Blues')

#     hm.set_title(label='Heatmap of correlations', fontsize=20)

#     # plt.show()

    train_new = train.copy()

    train_new = train_new.drop(columns=['Name', 'Ticket', 'PassengerId'])

    test_new = test.copy()

    y = train_new['Survived']

    train_x, test_x, train_y, test_y = train_test_split(train_new, y, test_size=0.25)

    train_x_attributes = train_x.drop(columns=['Survived']).columns

    

    

    # Build tree

    tree = decision_tree_learning(train_x, train_x_attributes, train_x, 1)



    # Visualize tree

    print_tree(tree)



    # Check accuracy

    from sklearn.metrics import accuracy_score

    predictions = []

    for index, example in test_x.iterrows():

        try:

            prediction = predict(tree, example)

        except:

            prediction = 0

            print("her")

        predictions.append(prediction)

    acc = accuracy_score(test_y, predictions)

    print("Accuracy with my greedy Importance Decision Tree: ", acc)

    

    real_tree = decision_tree_learning(train, train_x_attributes, train, 1)

    # Predict for the real test data

    predictions = []

    for index, example in test.iterrows():

        try:

            prediction = predict(real_tree, example)

        except: 

            prediction = 0

            print("her")

        predictions.append(prediction)

    

    pred_df = test[['PassengerId']].copy()

    pred_df['Survived'] = predictions

    pred_df.to_csv('submission_DTC.csv', index=False)



#main()
x.head()
# # Parameters for grid search

# parameters = {

#     'n_estimators'      : [50, 100] + [x for x in range(200, 250, 30)],

#     'max_depth'         : [x for x in range(5,10)] + [100],

#     'random_state'      : [0, 1],

#     'max_features'      : ['auto', 'log2'],

# }

# rf_vanilla = RandomForestClassifier(n_estimators=100, max_depth=100)

# rf_vanilla = rf_vanilla.fit(train_x, train_y)



# # Looking at importance of features

# fti = rf_vanilla.feature_importances_

# print("Feature imporantances out of the box:")

# for i, feat in enumerate(list(train_x.columns)):

#     print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))



    

# GS_rf = GridSearchCV(RandomForestClassifier(), parameters, cv=10, n_jobs=-1)

# GS_rf.fit(x, np.ravel(y))

# preds = GS_rf.predict(test_x)

# print('Accuracy with grid searched rf: ', accuracy_score(preds, test_y))

# print('Best params', GS_rf.best_params_)

# # Did return "Best params {'max_depth': 6, 'max_features': 'auto', 'n_estimators': 240, 'random_state': 0}"
# # Testing all classifiers

# rf = RandomForestClassifier(n_estimators=50, max_depth=100, n_jobs=-1)

# rf.fit(x, np.ravel(y))

# preds = rf.predict(test_x)

# print('Accuracy with rf: ', accuracy_score(preds, test_y))



# dt = DecisionTreeClassifier()

# dt.fit(train_x, train_y)

# preds = dt.predict(test_x)

# print('Accuracy with dt: ', accuracy_score(preds, test_y))



# neigh = KNeighborsClassifier(n_neighbors=5)

# neigh.fit(train_x, train_y)

# preds = neigh.predict(test_x)

# print('Accuracy with kneigh: ', accuracy_score(preds, test_y))
#final_model = RandomForestClassifier(max_depth=5, n_estimators=800)

def predict_test(model, name):

    test.fillna(0, inplace=True)

    model.fit(x, np.ravel(y))

    test_data = test.drop(columns=['Name', 'Ticket', 'PassengerId'])



    # Predicting values from testing set

    preds = model.predict(test_data)



    pred_df = test[['PassengerId']].copy()

    pred_df['Survived'] = preds

    pred_df.to_csv(f'submission_{name}.csv', index=False)
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",

          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

          "Naive Bayes", "QDA"]



classifiers = [

KNeighborsClassifier(3),

SVC(kernel="linear", C=0.025),

SVC(gamma=2, C=1),

GaussianProcessClassifier(1.0 * RBF(1.0)),

DecisionTreeClassifier(max_depth=5),

RandomForestClassifier(max_depth=5, n_estimators=800),

MLPClassifier(alpha=1, max_iter=500),

AdaBoostClassifier(),

GaussianNB(),

QuadraticDiscriminantAnalysis()

]



for i, classifier in enumerate(classifiers):

    classifier.fit(train_x, train_y)

    preds = classifier.predict(test_x)

    print(f"Predicting with {names[i]}, accuracy: {accuracy_score(preds, test_y)}")

    # Predict test and write csv file

    predict_test(classifier, names[i])
# # Importing libraries for building the neural network

# import tensorflow as tf

# from keras.models import Sequential

# from keras.layers import Dense

# from keras.wrappers.scikit_learn import KerasClassifier

# from sklearn.model_selection import StratifiedKFold

# from sklearn.model_selection import cross_val_score

# from sklearn.preprocessing import StandardScaler

# from sklearn.pipeline import Pipeline

# from sklearn.exceptions import DataConversionWarning

# import warnings

# warnings.filterwarnings(action='ignore', category=DataConversionWarning)



# print(tf.test.gpu_device_name())

# config = tf.ConfigProto()

# config.gpu_options.allow_growth = True
# def create_baseline(optimizer='adam', init='uniform'):

#     # create model

#     if verbose: print("Create model with optimizer: %s; init: %s" % (optimizer, init) )

#     model = Sequential()

#     model.add(Dense(16, input_dim=train_x.shape[1], kernel_initializer=init, activation='relu'))

#     model.add(Dense(8, kernel_initializer=init, activation='relu'))

#     model.add(Dense(4, kernel_initializer=init, activation='tanh'))

#     model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))

#     # Compile model

#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#     return model
# # Training model

# verbose = 1

# gridsearch = False

# with tf.device('/GPU:0'):

#     if gridsearch:

#         parameters = {

#         'optimizer' : ['rmsprop', 'adam'],

#         'init' : ['normal', 'uniform'],

#         'epochs' : [100, 200],

#         'batch_size' : [5, 10, 15],

#             }



#         model = KerasClassifier(build_fn=create_baseline, verbose=0)



#         grid_model = GridSearchCV(estimator=model, param_grid=parameters)

#         grid_result = grid_model.fit(train_x, train_y)



#         # summarize results

#         print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#         means = grid_result.cv_results_['mean_test_score']

#         stds = grid_result.cv_results_['std_test_score']

#         params = grid_result.cv_results_['params']

#         if verbose: 

#             for mean, stdev, param in zip(means, stds, params):

#                 print("%f (%f) with: %r" % (mean, stdev, param))

#             elapsed_time = time.time() - start_time  

#             print ("Time elapsed: ",timedelta(seconds=elapsed_time))



#         best_epochs = grid_result.best_params_['epochs']

#         best_batch_size = grid_result.best_params_['batch_size']

#         best_init = grid_result.best_params_['init']

#         best_optimizer = grid_result.best_params_['optimizer']

#     else:

#         best_epochs = 200

#         best_batch_size = 5

#         best_init = 'uniform'

#         best_optimizer = 'adam'
# # Create a classifier with best parameters for whole dataset

# with tf.device('/GPU:0'):

#     model = KerasClassifier(build_fn=create_baseline, optimizer=best_optimizer, init=best_init, epochs=best_epochs, batch_size=best_batch_size, verbose=0)

#     model.fit(x, y)
# # Predicting values

# with tf.device('/GPU:0'):

#     preds = model.predict(test_x)

#     print(f"Keras Classifiers accuracy: {accuracy_score(preds, test_y)}")

    