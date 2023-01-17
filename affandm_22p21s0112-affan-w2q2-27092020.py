import numpy as np

import pandas as pd

import re

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

# Loading the data

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
# Showing overview of the train dataset

train.head(5)
test.head(5)
# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
train['Has_Cabin']
test['Has_Cabin']
# Remove all NULLS in the Embarked column

for dataset in test:

    test['Embarked'] = test['Embarked'].fillna('S')

# Remove all NULLS in the Fare column

for dataset in test:

    test['Fare'] = test['Fare'].fillna(train['Fare'].median())
test.head(5)
# Replace all NULLS in the Age column into 0

train['Age'] = train["Age"].apply(lambda x : 0 if type(x)== float and np.isnan(x) else int(x) )

test['Age'] = test["Age"].apply(lambda x : 0 if type(x)== float and np.isnan(x) else int(x) )

test['Age']
train['Age']
#Replace all sex in data

sex = {'female': 0, 'male': 1}

test['Sex'] = test['Sex'].replace(sex)

train['Sex'] = train['Sex'].replace(sex)
# Replace all Age column in each level of test

test['Age'] = test["Age"].apply(lambda x : 0 if int(x)<= 18 else int(x))

test['Age'] = test["Age"].apply(lambda x : 1 if int(x)> 18 and int(x)<=30 else int(x) )

test['Age'] = test["Age"].apply(lambda x : 2 if int(x)> 30 and int(x)<=45 else int(x) )

test['Age'] = test["Age"].apply(lambda x : 3 if int(x)> 45 and int(x)<=55 else int(x) )

test['Age'] = test["Age"].apply(lambda x : 4 if int(x)> 55 else int(x) )



test['Age'].head(5)
# Replace all Age column in each level of train

train['Age'] = train["Age"].apply(lambda x : 0 if int(x)<= 18 else int(x))

train['Age'] = train["Age"].apply(lambda x : 1 if int(x)> 18 and int(x)<=30 else int(x) )

train['Age'] = train["Age"].apply(lambda x : 2 if int(x)> 30 and int(x)<=45 else int(x) )

train['Age'] = train["Age"].apply(lambda x : 3 if int(x)> 45 and int(x)<=55 else int(x) )

train['Age'] = train["Age"].apply(lambda x : 4 if int(x)> 55 else int(x) )



train['Age'].head(5)
# Replace all Fare column in each level of test

test['Fare'] = test["Fare"].apply(lambda x : 0 if float(x)<= 8.0 else float(x))

test['Fare'] = test["Fare"].apply(lambda x : 1 if float(x)> 8.0 and float(x)<=16.0 else float(x) )

test['Fare'] = test["Fare"].apply(lambda x : 2 if float(x)> 16.0 and float(x)<=32.0 else float(x) )

test['Fare'] = test["Fare"].apply(lambda x : 3 if float(x)> 32.0 else float(x) )



test['Fare'].head(5)
# Replace all Fare column in each level of train

train['Fare'] = train["Fare"].apply(lambda x : 0 if float(x)<= 8.0 else float(x))

train['Fare'] = train["Fare"].apply(lambda x : 1 if float(x)> 8.0 and float(x)<=16.0 else float(x) )

train['Fare'] = train["Fare"].apply(lambda x : 2 if float(x)> 16.0 and float(x)<=32.0 else float(x) )

train['Fare'] = train["Fare"].apply(lambda x : 3 if float(x)> 32.0 else float(x) )



train['Fare'].head(5)
# Feature selection: remove variables no longer containing relevant information

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Embarked']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)
train.head(3)
test.head(3)
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).agg(['mean', 'count', 'sum'])

# Since "Survived" is a binary class (0 or 1), these metrics grouped by the Pclass feature represent:

    # MEAN: survival rate

    # COUNT: total observations

    # SUM: people survived
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean', 'count', 'sum'])

# Since Survived is a binary feature, this metrics grouped by the Sex feature represent:
# Define function to calculate Gini Impurity

def get_gini_impurity(survived_count, total_count):

    survival_prob = survived_count/total_count

    not_survival_prob = (1 - survival_prob)

    random_observation_survived_prob = survival_prob

    random_observation_not_survived_prob = (1 - random_observation_survived_prob)

    mislabelling_survided_prob = not_survival_prob * random_observation_survived_prob

    mislabelling_not_survided_prob = survival_prob * random_observation_not_survived_prob

    gini_impurity = mislabelling_survided_prob + mislabelling_not_survided_prob

    return gini_impurity
# Gini Impurity of starting node

gini_impurity_starting_node = get_gini_impurity(342, 891)

gini_impurity_starting_node
# Gini Impurity decrease of node for 'male' observations

gini_impurity_men = get_gini_impurity(109, 577)

gini_impurity_men
# Gini Impurity decrease if node splited for 'female' observations

gini_impurity_women = get_gini_impurity(233, 314)

gini_impurity_women
# Gini Impurity decrease if node splited by Sex

men_weight = 577/891

women_weight = 314/891

weighted_gini_impurity_sex_split = (gini_impurity_men * men_weight) + (gini_impurity_women * women_weight)



sex_gini_decrease = weighted_gini_impurity_sex_split - gini_impurity_starting_node

sex_gini_decrease
# Gini Impurity decrease of node for observations with Pclass == 1

gini_impurity_Pclass_1 = get_gini_impurity(81, 517)

gini_impurity_Pclass_1
# Gini Impurity decrease if node splited for observations with Pclass != 1

gini_impurity_Pclass_others = get_gini_impurity(261, 374)

gini_impurity_Pclass_others
# Gini Impurity decrease if node splited for observations with Pclass == 1 

Pclass_1_weight = 517/891

Pclass_others_weight = 374/891

weighted_gini_impurity_Pclass_split = (gini_impurity_Pclass_1 * Pclass_1_weight) + (gini_impurity_Pclass_others * Pclass_others_weight)



Pclass_gini_decrease = weighted_gini_impurity_Pclass_split - gini_impurity_starting_node

Pclass_gini_decrease
# Desired number of Cross Validation folds

cv = KFold(n_splits=5)            

accuracies = list()

max_attributes = len(list(test))

depth_range = range(1, max_attributes + 1)



# Testing max_depths from 1 to max attributes

# Uncomment prints for details about each Cross Validation pass

for depth in depth_range:

    fold_accuracy = []

    tree_model = tree.DecisionTreeClassifier(max_depth = depth)

    # print("Current max depth: ", depth, "\n")

    for train_fold, valid_fold in cv.split(train):

        f_train = train.loc[train_fold] # Extract train data with cv indices

        f_valid = train.loc[valid_fold] # Extract valid data with cv indices



        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), 

                               y = f_train["Survived"]) # We fit the model with the fold train data

        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), 

                                y = f_valid["Survived"])# We calculate accuracy with the fold validation data

        fold_accuracy.append(valid_acc)



    avg = sum(fold_accuracy)/len(fold_accuracy)

    accuracies.append(avg)

    # print("Accuracy per fold: ", fold_accuracy, "\n")

    # print("Average accuracy: ", avg)

    # print("\n")

    

# Just to show results conveniently

df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})

df = df[["Max Depth", "Average Accuracy"]]

print(df.to_string(index=False))
y_train = train['Survived']

x_train = train.drop(['Survived'], axis=1).values 

x_test = test.values



# Create Decision Tree with max_depth = 6

decision_tree = tree.DecisionTreeClassifier(max_depth = 6)

decision_tree.fit(x_train, y_train)



# Predicting results for test dataset

y_pred = decision_tree.predict(x_test)

submission = pd.DataFrame({"Survived": y_pred})

submission.to_csv('submission.csv', index=False)



# Export our trained model as a .dot file

with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(decision_tree,

                              out_file=f,

                              max_depth = 6,

                              impurity = True,

                              feature_names = list(train.drop(['Survived'], axis=1)),

                              class_names = ['Died', 'Survived'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)

draw.text((10, 0), # Drawing offset (position)

          '"Pclass corresponds to Sex', # Text to draw

          (0,0,255), # RGB desired color

          font=font) # ImageFont object with desired font

img.save('sample-out.png')

PImage("sample-out.png")



# Code to check available fonts and respective paths

# import matplotlib.font_manager

# matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

acc_decision_tree
train.head()
X = train.drop(['Survived'],axis=1)

y = train.Survived
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# train a Gaussian Naive Bayes classifier on the training set

from sklearn.naive_bayes import GaussianNB





# instantiate the model

gnb = GaussianNB()





# fit the model

gnb.fit(X_train, y_train)
y_predNB = gnb.predict(X_test)

y_predNB
from sklearn.metrics import accuracy_score



print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_predNB)))
# print the scores on training and test set



print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))



print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))
# Print the Confusion Matrix and slice it into four pieces



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_predNB)



print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
# visualize confusion matrix with seaborn heatmap



cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 

                                 index=['Predict Positive:1', 'Predict Negative:0'])



sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
from sklearn.metrics import classification_report



print(classification_report(y_test, y_predNB))
#Classification accuracy

TP = cm[0,0]

TN = cm[1,1]

FP = cm[0,1]

FN = cm[1,0]
# print classification accuracy



classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)



print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
#Classification error



classification_error = (FP + FN) / float(TP + TN + FP + FN)



print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score



precision = TP / float(TP + FP)





print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)



print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)



print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)



print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)



print('Specificity : {0:0.4f}'.format(specificity))
# Calculate class probabilities

# print the first 10 predicted probabilities of two classes- 0 and 1



y_pred_prob = gnb.predict_proba(X_test)[0:10]



y_pred_prob
# store the probabilities in dataframe



y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Dead', 'Survived'])



y_pred_prob_df
gnb.predict_proba(X_test)[0:10, 1]

y_pred1 = gnb.predict_proba(X_test)[:, 1]
# plot histogram of predicted probabilities



# adjust the font size 

plt.rcParams['font.size'] = 12



# plot histogram with 10 bins

plt.hist(y_pred1, bins = 10)



# set the title of predicted probabilities

plt.title('Histogram of predicted probabilities of Survived')





# set the x-axis limit

plt.xlim(0,1)





# set the title

plt.xlabel('Predicted probabilities of Survived')

plt.ylabel('Frequency')
# plot ROC Curve



from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_test, y_pred1)



plt.figure(figsize=(6,4))



plt.plot(fpr, tpr, linewidth=2)



plt.plot([0,1], [0,1], 'k--' )



plt.rcParams['font.size'] = 12



plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Survived')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')



plt.show()
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)



print('ROC AUC : {:.4f}'.format(ROC_AUC))
# calculate cross-validated ROC AUC 



from sklearn.model_selection import cross_val_score



Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, y_train, cv=5, scoring='roc_auc').mean()



print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
# Applying 5-Fold Cross Validation



from sklearn.model_selection import cross_val_score

scores = cross_val_score(gnb, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))
# Requirements

import keras

from keras import Sequential



# Turn off complaints

import warnings

warnings.filterwarnings("ignore")
train.head()
# Feature selection: remove variables no longer containing relevant information

drop_elements = ['Parch', 'Has_Cabin']

train_x = train.drop(drop_elements, axis = 1)

test_x  = test.drop(drop_elements, axis = 1)
# Data Prep



# Manip

target = train_x["Pclass"]

train_x.drop(["Pclass"], axis = 1, inplace=True)





# Normalize

from sklearn import preprocessing



x = train_x.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

train_x = pd.DataFrame(x_scaled)
train
sns.countplot(y=train.Survived ,data=train)

plt.xlabel("Count of each Target class")

plt.ylabel("Target classes")

plt.show()
train.hist(figsize=(15,12),bins = 15)

plt.title("Features Distribution")

plt.show()
plt.figure(figsize=(15,15))

p=sns.heatmap(train.corr(), annot=True,cmap='RdYlGn',center=0) 
import os

import numpy as np

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam



# Set the seed for hash based operations in python

os.environ['PYTHONHASHSEED'] = '0'



# Set the numpy seed

np.random.seed(111)



# Disable multi-threading in tensorflow ops

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)



# Set the random seed in tensorflow at graph level

tf.compat.v1.set_random_seed(111)



# Define a tensorflow session with above session configs

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)



# Set the session in keras

tf.compat.v1.keras.backend.set_session(sess)
A = train.drop(['Survived'],axis=1)

b = train.Survived
A = np.array(A)

b = np.array(b).reshape(-1,1)
encoder = OneHotEncoder()

targets = encoder.fit_transform(b)
train_features, test_features, train_targets, test_targets = train_test_split(A,b, test_size=0.2, random_state=5)
train_features
model = Sequential()

# first parameter is output dimension

model.add(Dense(6, input_dim=6, activation='relu'))

model.add(Dense(6, input_dim=6, activation='relu'))

model.add(Dense(6, input_dim=6, activation='relu'))

model.add(Dense(6, input_dim=6, activation='relu'))

model.add(Dense(6, input_dim=6, activation='relu'))

model.add(Dense(6, input_dim=6, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



#we can define the loss function MSE or negative log lokelihood

#optimizer will find the right adjustements for the weights: SGD, Adagrad, ADAM ...

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.summary()

model.fit(train_features, train_targets, epochs=10, batch_size=20, verbose=0)

loss, accuracy = model.evaluate(test_features, test_targets)
print("Loss on the test dataset: %.10f" %loss)

print("Accuracy on the test dataset: %.2f" %accuracy)

# Part 3 - Making predictions and evaluating the model



# Predicting the Test set results

y_pred = model.predict(test_features)

y_pred = (y_pred > 0.5)



score, acc = model.evaluate(test_features,test_targets,batch_size=10)

print('Test score: ',score)

print('Test accuracy: ',acc)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_targets, y_pred)

p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.metrics import classification_report

print(classification_report(test_targets, y_pred))
#import classification_report

from sklearn.metrics import classification_report

print(classification_report(test_targets,y_pred))
#Classification accuracy

TP = cm[0,0]

TN = cm[1,1]

FP = cm[0,1]

FN = cm[1,0]
# print precision score

precision = TP / float(TP + FP)



print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)



print('Recall or Sensitivity : {0:0.4f}'.format(recall))
from sklearn.metrics import roc_curve

y_pred_proba = model.predict_proba(test_features)

fpr, tpr, thresholds = roc_curve(test_targets, y_pred_proba)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC curve')

plt.show()
#Area under ROC curve

from sklearn.metrics import roc_auc_score

roc_auc_score(test_targets,y_pred_proba)
# Tuning the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential

from keras.layers import Dense

def build_classifier(optimizer):

    classifier = Sequential()

    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))

    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],

              'epochs': [100, 200],

              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10)

grid_search = grid_search.fit(test_features,test_targets,verbose = 0)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_

print('Best Parameters after tuning: {}'.format(best_parameters))

print('Best Accuracy after tuning: {}'.format(best_accuracy))
# Applying 5-Fold Cross Validation



from sklearn.model_selection import cross_val_score

scores = cross_val_score(gnb,train_features, train_targets,cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))