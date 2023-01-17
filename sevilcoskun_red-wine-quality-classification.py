import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting
import seaborn as sns #good visualizing
import os
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))
data = pd.read_csv('../input/winequality-red.csv')
data.columns = data.columns.str.replace(' ','_')
data.info()
data.describe()
#correlation map view
data.corr() 
f, ax = plt.subplots(figsize = (10,10))
sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt = ".2f", ax=ax)
plt.show()
fig, axes = plt.subplots(11,11, figsize=(50,50))
for i in range(11):
    for j in range(11):
        axes[i, j].scatter(data.iloc[:,i], data.iloc[:,j], c = data.quality)
        axes[i,j].set_xlabel(data.columns[i])
        axes[i,j].set_ylabel(data.columns[j])
        axes[i,j].legend(data.quality)
plt.show()
g = sns.pairplot(data, hue="quality")
#How many wine quality number is realted with how many unique wines
#print(data['quality'].value_counts())
sns.barplot(data['quality'].unique(),data['quality'].value_counts())
plt.xlabel("Quality Rankings")
plt.ylabel("Number of Red Wine")
plt.title("Distribution of Red Wine Quality Ratings")
plt.show()
print(data['quality'].value_counts())
#Check the outliers for each feature with respect to output value
fig, ax1 = plt.subplots(4,3, figsize=(22,16))
k = 0
for i in range(4):
    for j in range(3):
        if k != 11:
            sns.boxplot('quality',data.iloc[:,k], data=data, ax = ax1[i][j])
            k += 1
plt.show()
#Check the outliers for each feature with respect to output value
fig, ax1 = plt.subplots(4,3, figsize=(22,16))
k = 0
for i in range(4):
    for j in range(3):
        if k != 11:
            sns.barplot('quality',data.iloc[:,k], data=data, ax = ax1[i][j])
            k += 1
plt.show()
#Fucntion Part
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import graphviz  
from sklearn.externals.six import StringIO
from IPython.display import Image 

#Normalization ==> x_norm = (x - mean)/std 
#it gives for each value the same value intervals means between 0-1
def normalization(X):
    mean = np.mean(X)
    std = np.std(X)
    X_t = (X - mean)/std
    return X_t

#Train and Test splitting of data     
def train_test(X_t, y):
    x_train, x_test, y_train, y_test = train_test_split(X_t, y, test_size = 0.3, random_state = 42)
    print("Train:",len(x_train), " - Test:", len(x_test))
    return x_train, x_test, y_train, y_test

def grid_search(name_clf, clf, x_train, x_test, y_train, y_test):
    if name_clf == 'Logistic_Regression':
        # Logistic Regression 
        log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
        grid_log_reg.fit(x_train, y_train)
        # We automatically get the logistic regression with the best parameters.
        log_reg = grid_log_reg.best_estimator_
        print("Best Parameters for Logistic Regression: ", grid_log_reg.best_estimator_)
        print("Best Score for Logistic Regression: ", grid_log_reg.best_score_)
        print("------------------------------------------")
        return log_reg
    
    elif name_clf == 'SVM':
        # Support Vector Classifier
        svc_params = {'C':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                      'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
        grid_svc = GridSearchCV(SVC(), svc_params)
        grid_svc.fit(x_train, y_train)
        # SVC best estimator
        svc = grid_svc.best_estimator_
        print("Best Parameters for SVM: ", grid_svc.best_estimator_)
        print("Best Score for SVM: ", grid_svc.best_score_)
        print("------------------------------------------")
        return svc
    
    elif name_clf == 'Decision_Tree':
        # DecisionTree Classifier
        tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,30,1)), 
                  "min_samples_leaf": list(range(5,20,1))}
        grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
        grid_tree.fit(x_train, y_train)
        # tree best estimator
        tree_clf = grid_tree.best_estimator_
        print("Best Parameters for Decision Tree: ", grid_tree.best_estimator_)
        print("Best Score for Decision Tree: ", grid_tree.best_score_)
        print("------------------------------------------")
        
        #FEATURE IMPORTANCE FOR DECISION TREE
        importnce = tree_clf.feature_importances_
        plt.figure(figsize=(10,10))
        plt.title("Feature Importances of Decision Tree")
        plt.barh(X_t.columns, importnce, align="center")
        
        return tree_clf
    
    elif name_clf == 'Random_Forest':
        forest_params = {"bootstrap":[True, False], "max_depth": list(range(2,10,1)), 
                  "min_samples_leaf": list(range(5,20,1))}
        grid_forest = GridSearchCV(RandomForestClassifier(), forest_params)
        grid_forest.fit(x_train, y_train)
        # forest best estimator
        forest_clf = grid_forest.best_estimator_
        print("Best Parameters for Random Forest: ", grid_forest.best_estimator_)
        print("Best Score for Random Forest: ", grid_forest.best_score_)
        print("------------------------------------------")
        
        #FEATURE IMPORTANCE FOR DECISION TREE
        importnce = forest_clf.feature_importances_
        plt.figure(figsize=(10,10))
        plt.title("Feature Importances of Random Forest")
        plt.barh(X_t.columns, importnce, align="center")
        
        return forest_clf
    
def plot_learning_curve(estimator,title, X, y, ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, 
                                                            n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#Create applying classification funciton
def apply_classification(name_clf, clf, x_train, x_test, y_train, y_test):
    #Find the best parameters and get the classification with the best parameters as return valu of grid search
    grid_clf = grid_search(name_clf, clf, x_train, x_test, y_train, y_test)
    
    #Plotting the learning curve
    # score curves, each time with 30% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    plot_learning_curve(grid_clf, name_clf, x_train, y_train, 
                    ylim=(0.1, 1.01), cv=cv, n_jobs=4)
    
    #Apply cross validation to estimate the skills of models with 10 split with using best parameters
    scores = cross_val_score(grid_clf, x_train, y_train, cv=10)
    print("Mean Accuracy of Cross Validation: %", round(scores.mean()*100,2))
    print("Std of Accuracy of Cross Validation: %", round(scores.std()*100))
    print("------------------------------------------")
    
    #Predict the test data as selected classifier
    clf_prediction = grid_clf.predict(x_test)
    clf1_accuracy = sum(y_test == clf_prediction)/len(y_test)
    print("Accuracy of",name_clf,":",clf1_accuracy*100)
    
    #print confusion matrix and accuracy score before best parameters
    clf1_conf_matrix = confusion_matrix(y_test, clf_prediction)
    print("Confusion matrix of",name_clf,":\n", clf1_conf_matrix)
    print("==========================================")
    return grid_clf
#Now seperate the dataset as response variable and feature variabes
X = data.drop(['quality'], axis = 1)
#y = pd.DataFrame(data['value'])
y = data['quality']
#Normalization
X_t = normalization(X)
print("X_t:", X_t.shape)

#Train and Test splitting of data 
x_train, x_test, y_train, y_test = train_test(X_t, y)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
apply_classification('Logistic_Regression', lr, x_train, x_test, y_train, y_test)
from sklearn.svm import SVC

svm = SVC()
apply_classification('SVM', svm, x_train, x_test, y_train, y_test)
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree

dt = DecisionTreeClassifier()
dt_clf = apply_classification('Decision_Tree', dt, x_train, x_test, y_train, y_test)
#Plot the decision tree 
dot_data = export_graphviz(dt_clf, out_file=None, filled=True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
apply_classification('Random_Forest', rf, x_train, x_test, y_train, y_test)
#Add a new feature according to mean of the quality
#Good wine represented by 1, bad wine represented by 0
data['value'] = ""
data['value'] = [1 if each > 5 else 0 for each in data['quality']]

print("Good Wine Class:",data[data['value'] == 1].shape)
print("Bad Wine Class:",data[data['value'] == 0].shape)
#Check the outliers for each feature with respect to output value
fig, ax1 = plt.subplots(4,3, figsize=(22,16))
k = 0
for i in range(4):
    for j in range(3):
        if k != 11:
            sns.boxplot('value',data.iloc[:,k], data=data, ax = ax1[i][j])
            k += 1
plt.show()
#Categorical distribution plots:
fig, ax1 = plt.subplots(4,3, figsize=(22,16))
k = 0
for i in range(4):
    for j in range(3):
        if k != 11:
            sns.barplot(x="value",y=data.iloc[:,k],hue = 'value', data=data, ax = ax1[i][j])
            k += 1
plt.show()
fig, axes = plt.subplots(11,11, figsize=(50,50))
for i in range(11):
    for j in range(11):
        axes[i, j].scatter(data.iloc[:,i], data.iloc[:,j], c = data.value)
        axes[i,j].set_xlabel(data.columns[i])
        axes[i,j].set_ylabel(data.columns[j])
        axes[i,j].legend(data.value)
plt.show()
#Now seperate the dataset as response variable and feature variabes
Xb = data.drop(['quality','value'], axis = 1)
#y = pd.DataFrame(data['value'])
yb = data['value']
#Normalization
Xb_t = normalization(Xb)
print("X_t:", Xb_t.shape)

#Train and Test splitting of data 
xb_train, xb_test, yb_train, yb_test = train_test(Xb_t, yb)
lrb = LogisticRegression()
apply_classification('Logistic_Regression', lrb, xb_train, xb_test, yb_train, yb_test)

svmb = SVC()
apply_classification('SVM', svmb, xb_train, xb_test, yb_train, yb_test)

dtb = DecisionTreeClassifier()
dtb_clf = apply_classification('Decision_Tree', dtb, xb_train, xb_test, yb_train, yb_test)

rfb = RandomForestClassifier(n_estimators=100)
apply_classification('Random_Forest', rfb, xb_train, xb_test, yb_train, yb_test)
#Plot the decision tree 
dot_data = export_graphviz(dtb_clf, out_file=None, filled=True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph
#Add a new feature according to mean of the quality
#Good wine represented by 2, average 1, and bad wine represented by 0
data['value'] = ""
data['value'] = [2 if each > 6 else 1 if ((each > 4) and (each < 7)) else 0 for each in data['quality']]

print("Good Wine Class:",data[data['value'] == 2].shape)
print("Average Wine Class:",data[data['value'] == 1].shape)
print("Bad Wine Class:",data[data['value'] == 0].shape)
#Check the outliers for each feature with respect to output value
fig, ax1 = plt.subplots(4,3, figsize=(22,16))
k = 0
for i in range(4):
    for j in range(3):
        if k != 11:
            sns.boxplot('value',data.iloc[:,k], data=data, ax = ax1[i][j])
            k += 1
plt.show()
#Categorical distribution plots:
fig, ax1 = plt.subplots(4,3, figsize=(22,16))
k = 0
for i in range(4):
    for j in range(3):
        if k != 11:
            sns.barplot(x="value",y=data.iloc[:,k],hue = 'value', data=data, ax = ax1[i][j])
            k += 1
plt.show()
fig, axes = plt.subplots(11,11, figsize=(50,50))
for i in range(11):
    for j in range(11):
        axes[i, j].scatter(data.iloc[:,i], data.iloc[:,j], c = data.value)
        axes[i,j].set_xlabel(data.columns[i])
        axes[i,j].set_ylabel(data.columns[j])
        axes[i,j].legend(data.value)
plt.show()
#Now seperate the dataset as response variable and feature variabes
X3 = data.drop(['quality','value'], axis = 1)
#y = pd.DataFrame(data['value'])
y3 = data['value']
#Normalization
X3_t = normalization(X3)
print("X_t:", X3_t.shape)

#Train and Test splitting of data 
x3_train, x3_test, y3_train, y3_test = train_test(X3_t, y3)
lr3 = LogisticRegression()
apply_classification('Logistic_Regression', lr3, x3_train, x3_test, y3_train, y3_test)

svm3 = SVC()
apply_classification('SVM', svm3, x3_train, x3_test, y3_train, y3_test)

dt3 = DecisionTreeClassifier()
dt3_clf = apply_classification('Decision_Tree', dt3, x3_train, x3_test, y3_train, y3_test)

rf3 = RandomForestClassifier(n_estimators=100)
apply_classification('Random_Forest', rf3, x3_train, x3_test, y3_train, y3_test)
#Plot the decision tree 
dot_data = export_graphviz(dt3_clf, out_file=None, filled=True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph