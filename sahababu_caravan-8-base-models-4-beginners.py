import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cross_validation import StratifiedShuffleSplit

%matplotlib inline



from sklearn.preprocessing import StandardScaler
input_data = pd.read_csv('../input/caravan-insurance-challenge.csv')

input_data.head()
target = input_data['CARAVAN']

features = input_data.drop(['ORIGIN','CARAVAN'], axis=1)
sss = StratifiedShuffleSplit(target, 10, test_size=0.3, random_state=23)



for train_index, test_index in sss:

    X_train, X_test = features.values[train_index], features.values[test_index]

    y_train, y_test = target[train_index], target[test_index]
features.describe()
target.value_counts()
fig = plt.figure(figsize=(10,10))



# Tells the total count of different values in CARAVAN

plt.subplot(3,1,1)

target.value_counts().plot(kind='bar', title='Classifying CARAVAN', color='steelblue', grid=True)



# Tells the total count of different values in customer subtype

plt.subplot(3,1,2)

features['MOSTYPE'].value_counts().plot(kind='bar', align='center', title='Classifying customer subtypes', color='steelblue', grid=True)
categorysubtype_caravan = pd.crosstab(features['MOSTYPE'], target)

categorysubtype_caravan_pct = categorysubtype_caravan.div(categorysubtype_caravan.sum(1).astype(float), axis=0)

categorysubtype_caravan_pct.plot(figsize= (8,5), kind='bar', stacked=True, color=['steelblue', 'springgreen'], title='category type vs Caravan', grid=True)

plt.xlabel('Category subtype')

plt.ylabel('Caravan or not')
features['MGEMLEEF'].hist(figsize=(5,3), fc='steelblue', grid=True)

plt.xlabel('age')

plt.ylabel('count')
age_caravan = pd.crosstab(features['MGEMLEEF'], target)

age_caravan_pct = age_caravan.div(age_caravan.sum(1).astype(float),axis=0)

age_caravan_pct.plot(figsize=(5,3), kind='bar', stacked=True, color=['steelblue', 'springgreen'], title='dependency of caravan on age groups', grid=True)

plt.xlabel('age groups')

plt.ylabel('Caravan')
features['MOSHOOFD'].value_counts().plot(kind='bar', color='steelblue', grid=True)

plt.xlabel('Customer Main Types')

plt.ylabel('count')
cust_type_caravan = pd.crosstab(features['MOSHOOFD'], target)

cust_type_caravan_pct = cust_type_caravan.div(cust_type_caravan.sum(1).astype(float), axis=0)

cust_type_caravan_pct.plot(kind='bar', stacked=True, color = ['steelblue', 'springgreen'])

plt.xlabel('customer types')

plt.ylabel('caravan')
from sklearn.ensemble import RandomForestClassifier

clf1 = RandomForestClassifier()

clf1.fit(X_train, y_train)

test_predictions1 = clf1.predict(X_test)
from sklearn.metrics import confusion_matrix





def draw_confusion_matrices(confusion_matricies,class_names):

    class_names = class_names.tolist()

    for cm in confusion_matrices:

        classifier, cm = cm[0], cm[1]

        print(cm)

        

        fig = plt.figure()

        ax = fig.add_subplot(111)

        cax = ax.matshow(cm)

        plt.title('Confusion matrix for %s' % classifier)

        fig.colorbar(cax)

        ax.set_xticklabels([''] + class_names)

        ax.set_yticklabels([''] + class_names)

        plt.xlabel('Predicted')

        plt.ylabel('True')

        plt.show()
class_names = np.unique(np.array(y_test))

confusion_matrices = [

    #( "Support Vector Machines", confusion_matrix(y,run_cv(X,y,SVC)) ),

    ( "Random Forest", confusion_matrix(y_test, test_predictions1)),

    #( "K-Nearest-Neighbors", confusion_matrix(y,run_cv(X,y,KNN)) ),

    #( "Gradient Boosting Classifier", confusion_matrix(y,run_cv(X,y,GBC)) ),

    #( "Logisitic Regression", confusion_matrix(y,run_cv(X,y,LR)) )

]

draw_confusion_matrices(confusion_matrices,class_names)
importances = clf1.feature_importances_[:10]

std = np.std([tree.feature_importances_ for tree in clf1.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



features = features.columns



for f in range(10):

    print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))



# Plot the feature importances of the forest

#import pylab as pl

plt.figure()

plt.title("Feature importances")

plt.bar(range(10), importances[indices], yerr=std[indices], color="r", align="center")

plt.xticks(range(10), indices)

plt.xlim([-1, 10])

plt.show()
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import f1_score





classifiers = [

    KNeighborsClassifier(3),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]



# Logging for Visual Comparison

log_cols=["Classifier", "F-score", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    test_predictions = clf.predict(X_test)

    acc = f1_score(y_test, test_predictions)

    print("F-score: {:.4%}".format(acc))

    

    test_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, test_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
sns.set_color_codes("muted")

sns.barplot(x='F-score', y='Classifier', data=log, color="b")



plt.xlabel('F-score %')

plt.title('Classifier F-Score')

plt.show()



sns.set_color_codes("muted")

sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")



plt.xlabel('Log Loss')

plt.title('Classifier Log Loss')

plt.show()
from sklearn.metrics import classification_report

clf1 = RandomForestClassifier()

clf1.fit(X_train, y_train)

test_predictions1 = clf1.predict(X_test)

report = classification_report(y_test, test_predictions1)

print(report)
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, test_predictions1)
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc



import seaborn as sns

sns.set('talk', 'whitegrid', 'dark', font_scale=1.0, font='Ricty',

        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})





fpr, tpr, _ = roc_curve(y_test, test_predictions1)

roc_auc = auc(fpr, tpr)



lw = 2

plt.figure()

plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.legend(loc="lower right")

plt.show()