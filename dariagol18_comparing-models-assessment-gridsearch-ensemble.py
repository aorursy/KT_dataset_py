# Main libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#Processing libraries

from sklearn.preprocessing import StandardScaler



# Model libraries

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB



# Testing libraries

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, learning_curve

from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report, roc_curve, auc





# Other useful things

# View all columns of dataframes

pd.options.display.max_columns = None



# View all cell outputs

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)

# Load the data

d = pd.read_csv('../input/train.csv')

survived = d['Survived']



# Must do - checklist:

#    - create new variables (enhansment of some variables: child, or women on the first class)

#    - drop irrelevant variables ('Name', 'Ticket', 'Cabin')

#    - fill nulls (mean, median, 0, depending on the data)

#    - deal with categorical - posibly create dummy variables ('Sex', 'Embarked')

#    - normilise/standadise (int and floats)

#    - balance the data set (by the dependant variable)

#    - check for multicolinearity (posibly drop variables)
d = d.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
# Dealing with null values, for numerical:

d.isnull().sum(axis = 0)

for col in d.columns:

    if d[col].dtype != object:

        d[col]=d[col].fillna(d[col].mean())

        

#could think of a better way such as use the different mens for male and female
# Create dummies - for better model outcomes (also deals with nulls in categorical)

d = pd.get_dummies(d)
# Data Normalisation

names = d.columns

scaler = StandardScaler()

scaled_d = scaler.fit_transform(d)

d = pd.DataFrame(scaled_d, columns=names)
d['Survived'] = pd.Series(survived)



# Class balance

ld = d[d.Survived == 1]

no_ld = d[d.Survived == 0]



balanced_d = pd.concat([ld.sample(int(len(no_ld)*2/3), replace = True), no_ld] )

x = balanced_d.iloc[:,:-1]

y = balanced_d.iloc[:,-1:]
# Check for multicollinearity

#d.corr() #less representative way

plt.figure(figsize=(30,15))

sns.heatmap(d[d.columns].corr(),cmap="BrBG",annot=True)

plt.show();
# Drop values to avoid multicolinearity

x = x.drop(['Sex_male', 'Embarked_C'], axis=1)
# Random state

rs = 2



# Split the data to check which algorithms learn better (later on we can check )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=rs)



# look at the shape of the data (many problems can arise from wrong shape)

x_train.shape

y_test.shape
# List of classifiers:

classifiers = [

LogisticRegression(random_state = rs),

DecisionTreeClassifier(random_state=rs),

GaussianNB(),

LinearDiscriminantAnalysis(),

QuadraticDiscriminantAnalysis(),

AdaBoostClassifier((DecisionTreeClassifier(random_state=rs)), random_state=rs, learning_rate=0.1),

MLPClassifier(random_state=rs),

RandomForestClassifier(n_estimators = 10, random_state=rs),

ExtraTreesClassifier(random_state=rs),

GradientBoostingClassifier(random_state=rs),

]



# List of results that will occure:

clf_name = [] # names of the clasifiers

model_results = pd.DataFrame.copy(y_test) #resulting of prediction from the models



kfold = StratifiedKFold(n_splits=10) #cross-validation

cv_results = [] # scores from cross validation

cv_acc = [] # mean accuracy from cross validation, need to maximize

cv_std = [] # standard deviation from cross validation, need to minimise



cnfm = [] #confusion matrix

clr = [] #clasification report

roc_auc = [] #roc curve:

roc_tpr = []

roc_fpr = []
# Training the algorithms and results

for clf in classifiers:

    name = clf.__class__.__name__

    clf_name.append(name)

    

    #fitting and predictions

    model = clf.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    model_results[name] = y_pred

    

    #accuracy and log loss

    cv_results.append(cross_val_score(clf, x_train, y_train, scoring = "accuracy",cv = kfold))

    acc = round(accuracy_score(y_test, y_pred), 2) #need to maximise

    train_pred = clf.predict_proba(x_test)

    ll = round(log_loss(y_test, train_pred), 2) #need to minimise

    print(f'Accuracy: {acc} \t Log Loss: {ll} \t ---> {name} ')

    

    #confusion matrix, clasification report, roc curve

    cnfm.append(confusion_matrix(y_test, y_pred))

    clr.append(classification_report(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_pred, y_test)

    roc_auc.append(auc(fpr, tpr))

    roc_tpr.append(tpr)

    roc_fpr.append(fpr)

    



for i in cv_results:

    cv_acc.append(i.mean())

    cv_std.append(i.std())
# Cross validation accuracy results graph

cv_res = pd.DataFrame({"CrossValMeans":cv_acc, "CrossValerrors": cv_std,"Algorithm":clf_name})



plt.figure(figsize=(15,8));

sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set2",orient = "h",**{'xerr':cv_std});

plt.xlabel("Mean Accuracy");

plt.title("Cross validation scores");

plt.grid();

plt.show;
# Confusion matrixes (not-normalized confusion matrix)

plt.figure(figsize=(10,10))

for i in range(len(classifiers)):

    plt.subplot(5,2,i+1) #adjust this acourding to the number of algorithms

    sns.heatmap(cnfm[i], annot=True, fmt="d",cmap="Blues");

    plt.xlabel('Predicted');

    plt.ylabel('Actual');

    plt.title(clf_name[i]);

plt.show;
#Clasification reports

for i in range(len(classifiers)):

    print (f"{clf_name[i]} Clasification Report:" );

    print (clr[i]);
# ROC Curve

plt.figure(figsize=(15,8))

for i in range(len(classifiers)):

    cm = ['red', 'blue', 'orange', 'green', 'pink', 'yellow', 'lightgreen', 'black', 'purple', 'lightblue'] #add more colours for more algorithms

    plt.plot(roc_fpr[i], roc_tpr[i], c=cm[i], lw=1, label=f'{clf_name[i]}: area = {roc_auc[i]}0.2f)')

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

    plt.xlabel('False Positive Rate');

    plt.ylabel('True Positive Rate');

    plt.title('ROC curve: Receiver Operating Characteristic');

    plt.legend(loc="lower right");

plt.show;
# Search grid for optimal parameters

MLP_param = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100)],

            'activation': ['tanh', 'relu'],

            'solver': ['sgd', 'adam'],

            'alpha': [0.0001, 0.05],

            'learning_rate': ['constant','adaptive']}



RF_param = {"max_depth": [None],

            "max_features": [0.3, 0.7, 1],

            "min_samples_split": [2, 3, 10],

            "min_samples_leaf": [1, 3, 10],

            "bootstrap": [False],

            "n_estimators" :[100,300],

            "criterion": ["gini"]}



ET_param = {"max_depth": [None],

            "max_features": [0.3, 0.7, 1],

            "min_samples_split": [2, 3, 10],

            "min_samples_leaf": [1, 3, 10],

            "bootstrap": [False],

            "n_estimators" :[100,300],

            "criterion": ["gini"]}



GB_param = {'loss' : ["deviance"],

            'n_estimators' : [100,200,300],

            'learning_rate': [0.1, 0.05, 0.01],

            'max_depth': [4, 8],

            'min_samples_leaf': [100,150],

            'max_features': [0.3, 0.1]}



#using 4 n_jobs for faster processing

MLPgs = GridSearchCV(MLPClassifier(), param_grid=MLP_param, cv=kfold, n_jobs=4, scoring="accuracy", verbose = 1)

RFgs = GridSearchCV(RandomForestClassifier(), param_grid=RF_param, cv=kfold, n_jobs=4, scoring="accuracy", verbose = 1)

ETgs = GridSearchCV(ExtraTreesClassifier(), param_grid=ET_param, cv=kfold, n_jobs=4, scoring="accuracy", verbose = 1)

GBgs = GridSearchCV(GradientBoostingClassifier(), param_grid=GB_param, cv=kfold, n_jobs=4, scoring="accuracy", verbose = 1)



models = [MLPgs, RFgs, ETgs, GBgs]



gs_model = []

score = []



for model in models:

    model.fit(x_train, y_train)

    gs_model.append(model.best_estimator_)

    score.append(model.best_score_)
gs_model #models with best estimators

cv_acc[-4:] #scores without gridsearch

score #scores with gridsearch
# Check for correlation between models: Make a correlation graph

plt.figure(figsize=(15,8))

sns.heatmap(model_results.corr(),annot=True)

plt.title("Correlation between models")
# Do model ensembling

best_model = VotingClassifier(estimators=[('rf', gs_model[2]), ('ext', gs_model[3]),

('mpl', gs_model[0]), ('gb',gs_model[1])], voting='soft', n_jobs=4)
# Load the test data

d2 = pd.read_csv('../input/test.csv')

PassengerId = d2['PassengerId'] #save this variable for correct format of the submision



# Drop values

d2 = d2.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)



# Dealing with null values, for numerical:

for col in d2.columns:

    if d2[col].dtype != object:

        d2[col]=d2[col].fillna(d[col].mean())



# Create dummies - for better model outcomes (also deals with nulls in categorical)

d2 = pd.get_dummies(d2)



# Data Normalisation

names = d2.columns

scaler = StandardScaler()

scaled_d2 = scaler.fit_transform(d2)

d2 = pd.DataFrame(scaled_d2, columns=names)



# Drop values to avoid multicolinearity

d2 = d2.drop(['Sex_male', 'Embarked_C'], axis=1)
#train the model on the whole dataset and produce results:

model = best_model.fit(x, y)

survived_pred = pd.Series(model.predict(d2), name='Survived')
# Results

d3 = pd.concat([PassengerId, survived_pred], axis=1)

d3.to_csv('submision.csv', index=False)