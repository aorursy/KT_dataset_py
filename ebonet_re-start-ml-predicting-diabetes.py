# import the pandas library for working with dataframes

import pandas as pd
# read the csv-formatted data file into a pandas dataframe

df=pd.read_csv('../input/diabetes.csv')

# get shape of data frame

print('Shape (n_rows,n_columns) of dataframe:',df.shape)

# print top 5 rows of data frame

df.head()
df[['Outcome','Pregnancies','Insulin']].head()
df[['Age']].head()
print(df[df.BMI>30].shape) # the first element is the number of rows, the second element is the number of columns

print('The number of rows where BMI>30 = ',df[df.BMI>30].shape[0]) # the first element is labeled 0, the second element is labeled 1
df[df.BMI<10].head()
df.BMI>30
df[df.BMI>30][['Outcome','BMI','Age']].head()
df[(df.Outcome==1)&(df.Pregnancies>0)][['Glucose','BloodPressure']].head(5)
df[(df.Outcome==1)&(df.BloodPressure>70)].shape
df.isnull().sum()
df.notnull().sum()
df.columns
df.dtypes
df.describe()
df.Outcome.value_counts()
df.SkinThickness.value_counts()
df[df.Outcome==1].SkinThickness.mean()
# use the function .max()

df[df.Outcome==0].BMI.max()
max(df[df.Outcome==0].BMI)
# get a plotting library

import matplotlib.pyplot as plt

# make it interactive in the notebook

%matplotlib inline
# plot Glucose vs BloodPressure and color points according to Outcome

plt.figure()

plt.scatter(df[df.Outcome==1].Glucose,df[df.Outcome==1].BloodPressure,label='Diabetes',color='r',s=2)

plt.scatter(df[df.Outcome==0].Glucose,df[df.Outcome==0].BloodPressure,label='No Diabetes',color='b',s=2)

plt.legend()

plt.xlabel('Glucose')

plt.ylabel('BloodPressure')
df.columns
c='Pregnancies'

df[df[c]==0][c].count()
for c in df.columns:

    print('For column',c,' there are',df[df[c]==0][c].count(),'zero values.')

for c in df.columns:

    plt.figure()

    plt.hist(df[c],bins=15)

    plt.xlabel(c)

    plt.ylabel('frequency')

    plt.show()
# example: plot histograms of Age for Outcome=1 and Outcome=0.

plt.figure()

plt.hist(df[df.Outcome==1]['Age'],bins=15,label='Diabetes',color='r',alpha=0.2)

plt.hist(df[df.Outcome==0]['Age'],bins=15,label='No Diabetes',color='b',alpha=0.2)

plt.xlabel('Age')

plt.ylabel('frequency')

plt.legend()

plt.show()
# choose a feature column, plot the histogram, and decide on a split value

# example

# create a new column in the data frame with the predicted outcome based on your split (here, Age<30 means outcome=0, otherwise outcome=1)

df['PredictedOutcome']=np.where(df.Age<30,0,1) # np.where(condition, value if true, value if false)

# calculate accuracy

N_correct=df[df.PredictedOutcome==df.Outcome].shape[0]

N_total=df.shape[0]

accuracy=N_correct/N_total

print('number of correct examples =',N_correct)

print('number of examples in total =',N_total)

print('accuracy =',accuracy)
# now check the accuracy of your column and split

# create a new column in the data frame with the predicted outcome based on your split

# replace "NONE" with your code

df['PredictedOutcome']=np.where(df.NONE<NONE,0,1) # np.where(condition, value if true, value if false)

# calculate accuracy

N_correct=df[df.PredictedOutcome==df.Outcome].shape[0]

N_total=df.shape[0]

accuracy=N_correct/N_total

print('number of correct examples =',N_correct)

print('number of examples in total =',N_total)

print('accuracy =',accuracy)
import numpy as np

import sklearn

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.3, random_state = 0)



train.describe()
train.drop('Insulin',axis=1,inplace=True)

test.drop('Insulin',axis=1,inplace=True) # axis=1 means drop the column, not the row

# check that Insulin is no longer in the list of columns

train.columns
# numpy provides many useful functions, including allowing us to create new columns in our dataframe based on a condition

import numpy as np



def imputeColumns(dataset):

    # create a list of columns that we will impute with the average non-zero value in each column

    columnsToImpute=['Glucose', 'BloodPressure', 'SkinThickness','BMI']



    for c in columnsToImpute:

        avgOfCol=dataset[dataset[c]>0][[c]].mean()

        dataset[c+'_imputed']=np.where(dataset[[c]]!=0,dataset[[c]],avgOfCol)



imputeColumns(train)

imputeColumns(test)

# check that we've imputed the 0 values  

train[train.Glucose==0][['Glucose','Glucose_imputed']].head()
X_train = train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','BMI', 'DiabetesPedigreeFunction', 'Age']]

Y_train = train[['Outcome']]

X_test = test[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

Y_test = test[['Outcome']]
Y_train.describe()
Y_test.describe()
from sklearn.tree import DecisionTreeClassifier



# Create the classifier

decision_tree_classifier = DecisionTreeClassifier(random_state = 0)



# Train the classifier on the training set

decision_tree_classifier.fit(X_train, Y_train)



# Evaluate the classifier on the testing set using classification accuracy

decision_tree_classifier.score(X_test, Y_test)
from sklearn import tree



dot_file = tree.export_graphviz(decision_tree_classifier, out_file='tree.dot', 

                                feature_names = list(X_train),

                                class_names = ['healthy', 'ill']) 

print("Accuracy on training set: {:.3f}".format(decision_tree_classifier.score(X_train, Y_train)))

print("Accuracy on test set: {:.3f}".format(decision_tree_classifier.score(X_test, Y_test)))
import graphviz

with open("tree.dot") as f:

    dot_graph = f.read()

graphviz.Source(dot_graph)
decision_tree_pruned = DecisionTreeClassifier(random_state = 0, max_depth = 4)



decision_tree_pruned.fit(X_train, Y_train)

decision_tree_pruned.score(X_test, Y_test)
print("Accuracy on training set: {:.3f}".format(decision_tree_pruned.score(X_train, Y_train)))

print("Accuracy on test set: {:.3f}".format(decision_tree_pruned.score(X_test, Y_test)))
pre_pruned_dot_file = tree.export_graphviz(decision_tree_pruned, out_file='pruned_tree.dot', 

                                feature_names = list(X_test),

                                class_names = ['healthy', 'ill'])

with open("pruned_tree.dot") as f:

    dot_graph = f.read()

graphviz.Source(dot_graph)