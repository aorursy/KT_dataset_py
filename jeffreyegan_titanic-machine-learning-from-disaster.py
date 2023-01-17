import os

import warnings 

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('../input/train.csv')  # data frame of training data supplied by kaggle

df.head()  # first five rows of data
df.describe()
df.isna().any()  # Review Columns (Features) Available and Discover if any information is missing
sex_mapping = {'male': 0, 'female': 1}  # Codify and map Sex strings to integer values

df['Sex'] = df['Sex'].map(sex_mapping)
df["Age"].fillna(28.0, inplace=True)  # Replace null data in Age column with the median Age = 28.0
# Add a Z-Score column for Fare data because it has outliers that skew some classification models. 

df['FareZ'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()

df.head()
features = ["Pclass","Sex","Age","FareZ", "SibSp", "Parch"]

target = ["Survived"]

df[features+target].isna().any()  # Make sure no nulls remain in data
# Set some styling variables for plots

sns.set_style('whitegrid')

sns.set_palette(sns.diverging_palette(220, 10, sep=80, n=2), 2)



# Define a function to make a bar plot of survivors for each of the categories in the column.

def plot_categorical(x_column, hue_column, df):

    '''Plot a bar plot for the average survivor rate for different groups.

    x_column          (str): The column name of a categorical field.

    hue_column        (str): The column name of a second categorical field.

    df   (pandas.DataFrame): The pandas DataFrame (just use df here!)

    '''

    fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')

    sns.barplot(x=x_column, y='Survived', hue=hue_column, data=df, errwidth=0)

    plt.title('Survivors separated by {} and {}'.format(x_column, hue_column))

    plt.show()





# Define a function to plot the distribution for survivors and non-survivors for a continuous variable.

def plot_distribution(column, df):

    '''Plot a bar plot for the average survivor rate for different groups.

    column            (str): The column name of a continuous (numeric) field.

    df   (pandas.DataFrame): The pandas DataFrame (just use df here!)

    '''

    fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')

    sns.distplot(df[df['Survived'] == 1][column].dropna(), label='Survived')

    sns.distplot(df[df['Survived'] == 0][column].dropna(), label='Did not survive')

    plt.legend()

    plt.title('{} distribution'.format(column))

    plt.show()
plot_categorical('Pclass', 'Sex', df)  # recall sex 0 = male, 1 = female

plot_distribution('Fare', df[df['Fare'] < 100]) # df[df['Fare'] < 100] simply removes some outliers!

plot_distribution('Age', df)

plot_distribution('SibSp', df)

plot_distribution('Parch', df)
features = ["Pclass","Sex","Age","FareZ", "SibSp", "Parch"]

target = ["Survived"]



cor = df[features+target].corr()

f, ax = plt.subplots(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')

sns.set_palette("husl")

sns.heatmap(cor, vmin=-1.0, vmax=1.0, cmap=sns.diverging_palette(10, 220, sep=80, n=11)) #, cmap=blues)
sns.pairplot(df[features+target], hue="Survived", palette=sns.diverging_palette(10, 220, sep=80, n=2), height=3, diag_kind="kde", diag_kws=dict(shade=True, bw=.05, vertical=False) )
features = ["Pclass","Sex","Age","FareZ", "SibSp", "Parch"]

target = ["Survived"]
np.random.seed(0) # Set a fixed random seed so we can reproduce results.

scoring_method = "accuracy"  # Our competition is using the evaluation metric of the F1-Score

cv_folds = 3  # Number of folds used in cross validation
# import sklearn packages that will get used in many model prototypes

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import normalize

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
# Prototype the Model

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1.0)  # Instantiate model object

cv_score = cross_val_score(lr, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

print("Mean cross validation score: "+str(round(np.mean(cv_score),4)))
# Optimization of the Model

from sklearn.linear_model import LogisticRegression

C_range = np.linspace(0.01,1.5,1000)

scores = []



for c_val in C_range:

    lr = LogisticRegression(C=c_val)  # Instantiate model object

    cv_score = cross_val_score(lr, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

    scores.append(np.mean(cv_score))

    

fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')

plot_labels = ['Accuracy', 'Recall', 'Precision', 'F1 Score']

blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "#0E1E2B"]

#plt.plot(C_range, m_accuracy, '-', label=plot_labels[0], color=blues[0])

#plt.plot(C_range, m_recall, '--', label=plot_labels[1], color=blues[1])

#plt.plot(C_range, m_precision, '-.', label=plot_labels[2], color=blues[2])

#plt.plot(C_range, m_f1, ':', label=plot_labels[3], color=blues[3])

plt.plot(C_range, scores, '-', label=plot_labels[3], color=blues[1])

plt.legend(loc='lower right')

plt.title('Logistic Regression Classification - Performance vs C Value')

plt.xlabel('C Value')

plt.ylabel('Accuracy Score')

plt.show() 

print("The Logistic Regression Classifier performance appears to reasonably plateau around "+str(round(np.max(scores),4))+" when the inverse \nof regularization parameter C is greater than 0.1.")
# Prototype the Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)  # Instantiate model object

cv_score = cross_val_score(knn, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

print("Mean cross validation score: "+str(round(np.mean(cv_score),4)))
# Optimization of the Model

from sklearn.neighbors import KNeighborsClassifier

k_range = list(range(1,29,2))

scores = []

   

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)  # Instantiate model object

    cv_score = cross_val_score(knn, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

    scores.append(np.mean(cv_score))

    #m_accuracy.append(accuracy_score(y_test, y_predicted))

    #m_recall.append(recall_score(y_test, y_predicted))

    #m_precision.append(precision_score(y_test, y_predicted))

    #m_f1.append(f1_score(y_test, y_predicted))



fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')

plot_labels = ['Accuracy', 'Recall', 'Precision', 'F1 Score']

blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "#0E1E2B"]

#plt.plot(k_range, m_accuracy, '-', label=plot_labels[0], color=blues[0])

#plt.plot(k_range, m_recall, '--', label=plot_labels[1], color=blues[1])

#plt.plot(k_range, m_precision, '-.', label=plot_labels[2], color=blues[2])

#plt.plot(k_range, m_f1, ':', label=plot_labels[3], color=blues[3])

plt.plot(k_range, scores, '-', label=plot_labels[3], color=blues[1])

plt.legend(loc='lower right')

plt.title('K Nearest Neighbors Classification - Performance vs k Value')

plt.xlabel('k Value')

plt.ylabel('Accuracy Score')

plt.show()

print("The K-Nearest Neighbors Classification Model reaches a peak accuracy score of "+str(round(np.max(scores),4))+" when k=5.")
# Prototype the Model

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()  # Instantiate model object

cv_score = cross_val_score(gnb, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

print("Mean cross validation score: "+str(round(np.mean(cv_score),4)))
# Prototype the Model

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=None, max_leaf_nodes=None)  # Instantiate model object

cv_score = cross_val_score(dt, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

print("Mean cross validation score: "+str(round(np.mean(cv_score),4)))
# Optimization of the Model

from sklearn.tree import DecisionTreeClassifier



depth_range = list(range(1,15))

leaf_range = list(range(2,20))

scores = np.zeros((len(depth_range), len(leaf_range)))



for d_idx in range(len(depth_range)):

    for l_idx in range(len(leaf_range)):

        dt = DecisionTreeClassifier(max_depth=depth_range[d_idx], max_leaf_nodes=leaf_range[l_idx])  # Instantiate model object

        cv_score = cross_val_score(dt, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

        scores[d_idx, l_idx] = np.mean(cv_score)



X, Y = np.meshgrid(leaf_range, depth_range)

Z = scores

fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap="RdYlGn")

ax.view_init(azim=110, elev=15)

ax.set_zlabel('Accuracy Score')

ax.set_ylabel('Max Depth')

ax.set_xlabel('Max Leaf Nodes')

plt.title('Decision Tree Performance')

plt.show()

X, Y = np.meshgrid(leaf_range, depth_range)

Z = scores

fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')  # plot a second angle

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap="RdYlGn")

ax.view_init(azim=350, elev=15)

ax.set_zlabel('Accuracy Score')

ax.set_ylabel('Max Depth')

ax.set_xlabel('Max Leaf Nodes')

plt.title('Decision Tree Performance')

plt.show()

max_score = str(round(np.max(scores),4))

indices = np.where(scores == scores.max())

print("Decision Tree Accuracy Performance peaks at "+max_score+" where Max Leaf Nodes = "+str(leaf_range[indices[1][0]])+" and Max Depth >= "+str(depth_range[indices[0][0]])+".")
# Prototype the Model

from sklearn.svm import SVC

svc = SVC(kernel='linear', C=1.0)  # Instantiate model object

cv_score = cross_val_score(svc, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

print("Mean cross validation score: "+str(round(np.mean(cv_score),4)))



svc = SVC(kernel='rbf', C=1.0)  # Instantiate model object

cv_score = cross_val_score(svc, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

print("Mean cross validation score: "+str(round(np.mean(cv_score),4)))
# Optimization of the Model

from sklearn.svm import SVC



# Linear Kernel

gamma_range = np.linspace(0.05, 3.0, num=10)  # num=30 originally, reduced to 3 for processing speed, no effect on curve

C_range  = np.linspace(0.1,1.5,15)

lin_scores = np.zeros((len(gamma_range), len(C_range)))

kernel_type = 'linear'

for g_idx in range(len(gamma_range)):

    for c_idx in range(len(C_range)):

        svc = SVC(kernel=kernel_type, C=C_range[c_idx], gamma=gamma_range[g_idx])  # Instantiate model object

        cv_score = cross_val_score(svc, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

        lin_scores[g_idx, c_idx] = np.mean(cv_score)

X, Y = np.meshgrid(C_range, gamma_range)

Z = lin_scores

fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap="RdYlGn")

ax.view_init(azim=100, elev=15)

ax.set_zlabel('Accuracy Score')

ax.set_ylabel('gamma')

ax.set_xlabel('C parameter')

plt.title('Support Vector Machine Performance - Kernel: Linear')

plt.show()

print("SVM Accuracy plateaus at an expected value of "+str(round(lin_scores[3,14],4))+" when using the linear kernel and the \ninverse of regularization strength parameter C is greater than "+str(C_range[2])+".")
# Optimization of the Model

from sklearn.svm import SVC

warnings.filterwarnings("ignore")  # suppress warnings for cross_val_scores where F-1 score = 0



# Radial Basis Function Kernel

gamma_range = np.linspace(0.05, 3.0, num=10)

C_range  = np.linspace(0.1,1.5,15)

rbf_scores = np.zeros((len(gamma_range), len(C_range)))

kernel_type = 'rbf'

for g_idx in range(len(gamma_range)):

    for c_idx in range(len(C_range)):

        svc = SVC(kernel=kernel_type, C=C_range[c_idx], gamma=gamma_range[g_idx])  # Instantiate model object

        cv_score = cross_val_score(svc, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

        rbf_scores[g_idx, c_idx] = np.mean(cv_score)

X, Y = np.meshgrid(C_range, gamma_range)

Z = rbf_scores

fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap="RdYlGn")

ax.view_init(azim=110, elev=15)

ax.set_zlabel('Accuracy Score')

ax.set_ylabel('gamma')

ax.set_xlabel('C parameter')

plt.title('Support Vector Machine Performance - Kernel: Radial Basis Function')

plt.show()

print("SVM Accuracy increases for small gamma values and an increasing regularization strength parameter C, \nreaching an an expected maxiumum value of "+str(round(rbf_scores[0,13],4))+" when using the radial basis function kernel \nwith C parameter = "+str(C_range[13])+" and gamma = "+str(gamma_range[0])+".")
features = ["Pclass","Sex","Age","FareZ", "SibSp", "Parch"]

target = ["Survived"]



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=0.175)  # Instantiate model object using optimal parameters

cv_score = cross_val_score(lr, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

print("Mean cross validation score for optimized k-Nearest Neighbors model: "+str(round(np.mean(cv_score),4)))



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)  # Instantiate model object

cv_score = cross_val_score(knn, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

print("Mean cross validation score for optimized k-Nearest Neighbors model: "+str(round(np.mean(cv_score),4)))



from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=7)  # Instantiate model object using optimal parameters

cv_score = cross_val_score(dt, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

print("Mean cross validation score for optimized Decision Tree model: "+str(round(np.mean(cv_score),4)))



from sklearn.svm import SVC

svc = SVC(kernel='rbf', C=1.4, gamma=0.05)  # Instantiate model object using optimal parameters

cv_score = cross_val_score(svc, df[features], df[target].values.ravel(), cv=cv_folds, scoring=scoring_method)

print("Mean cross validation score for optimized SVC model: "+str(round(np.mean(cv_score),4)))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=0.175)  # Instantiate model object using optimal parameters

lr.fit(df[features], df[target].values.ravel())  # Fit model using all available training data
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)  # Instantiate model object using optimal parameters

knn.fit(df[features], df[target].values.ravel())  # Fit model using all available training data
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=7)  # Instantiate model object using optimal parameters

dt.fit(df[features], df[target].values.ravel())  # Fit model using all available training data
from sklearn.svm import SVC

svc = SVC(kernel='rbf', C=1.4, gamma=0.05)  # Instantiate model object using optimal parameters

svc.fit(df[features], df[target].values.ravel())  # Fit model using all available training data
test_df = pd.read_csv("../input/test.csv")  # Read in the Test data supplied by kaggle

test_df.head()  # same as train.csv except for the missing Survived column
# Data Cleansing and Feature Engineering

features = ["Pclass","Sex","Age","FareZ", "SibSp", "Parch"]

target = ["Survived"]



sex_mapping = {'male': 0, 'female': 1}  # Codify and map Sex strings to integer values

test_df['Sex'] = test_df['Sex'].map(sex_mapping)

test_df["Age"].fillna(28.0, inplace=True)  # Replace null data in Age column with the median Age = 28.0

test_df["Fare"].fillna(14.5, inplace=True)  # Replace null data in Fare column with the median Fare = 14.5

test_df['FareZ'] = (test_df['Fare'] - test_df['Fare'].mean()) / test_df['Fare'].std()
lr_test_df = test_df

knn_test_df = test_df

dt_test_df = test_df

svm_test_df = test_df



lr_test_predictions = lr.predict(lr_test_df[features])

lr_test_df['Survived'] = lr_test_predictions

lr_test_df[['PassengerId', 'Survived']].to_csv('lr_submission.csv', index=False)



knn_test_predictions = knn.predict(knn_test_df[features])

knn_test_df['Survived'] = knn_test_predictions

knn_test_df[['PassengerId', 'Survived']].to_csv('knn_submission.csv', index=False)



dt_test_predictions = dt.predict(dt_test_df[features])

dt_test_df['Survived'] = dt_test_predictions

dt_test_df[['PassengerId', 'Survived']].to_csv('dt_submission.csv', index=False)



svm_test_predictions = svc.predict(svm_test_df[features])

svm_test_df['Survived'] = svm_test_predictions

svm_test_df[['PassengerId', 'Survived']].to_csv('svm_submission.csv', index=False)