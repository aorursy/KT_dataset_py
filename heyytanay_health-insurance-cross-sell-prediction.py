! pip install -q dabl
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import dabl





from IPython import display



from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_auc_score



import warnings



warnings.simplefilter("ignore")
train = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")

test = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")

sub = pd.read_csv("../input/health-insurance-cross-sell-prediction/sample_submission.csv")



train = train.drop(['id'], axis=1)

test = test.drop(['id'], axis=1)
train.head()
dabl.plot(train, target_col='Response')
train['Gender'] = train['Gender'].map({'Male':1, 'Female':0})

train['Vehicle_Age'] = train['Vehicle_Age'].map({'> 2 Years':0, '1-2 Year':1, '< 1 Year':2})

train['Vehicle_Damage'] = train['Vehicle_Damage'].map({'Yes':1, 'No':0})



train.head()
# Split the data

split_pcent = 0.20

split = int(len(train) * split_pcent)



data = train.sample(frac=1).reset_index(drop=True)



valid = data[:split]

train = data[split:]



tX, tY = train.drop(['Response'], axis=1).values, train['Response'].values

vX, vY = valid.drop(['Response'], axis=1).values, valid['Response'].values



print(tX.shape[0], vX.shape[0])
names = ["Logistic Regression", "Nearest Neighbors", 

         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

         "Naive Bayes", "QDA"]



classifiers = [

    LogisticRegression(),

    KNeighborsClassifier(3),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    MLPClassifier(alpha=1, max_iter=1000),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()]
# Let's do the classification and store the name of the classifier and it's test score into a dictionary



roc_results = {}

acc_results = {}



for name, clf in zip(names, classifiers):

    # Fit on the traning data

    clf.fit(tX, tY)

    

    # Store the validation accuracy

    val_acc = clf.score(vX, vY)

    acc_results[name] = val_acc

    

    # Get the test time prediction

    preds = clf.predict(vX)

    

    # Calculate Test ROC_AUC

    roc_score = roc_auc_score(vY, preds)

    

    # Store the results in a dictionary

    roc_results[name] = roc_score

    

    print(f"Classifier: {name} | val_acc: {val_acc:.4f} | roc_auc: {roc_score:.4f}")
# Sort the Model Accuracies based on the test score

sort_clf = dict(sorted(acc_results.items(), key=lambda x: x[1], reverse=True))



# Get the names and the corresponding scores

clf_names = list(sort_clf.keys())[::-1]

clf_scores = list(sort_clf.values())[::-1]



# Plot the per-model performance

fig = px.bar(

    x=clf_scores,

    y=clf_names,

    color=clf_names,

    labels={'x':'Validation Accuracy Score', 'y':'Models'},

    title=f"Model Performance [ Best Model: {clf_names[-1]} | Accuracy: {clf_scores[-1]:.2f} ]"

)



fig.show()