import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import StandardScaler
input_data = pd.read_csv('../input/caravan-insurance-challenge.csv')

input_data.head()
data_no_origin_train = input_data[input_data['ORIGIN']=='train'].drop(['ORIGIN'], axis=1)

data_no_origin_train.head()
data_no_origin_test = input_data[input_data['ORIGIN']=='test'].drop(['ORIGIN'], axis=1)
input_train, input_cv = train_test_split(data_no_origin_train, test_size=0.30)

input_test = data_no_origin_test
input_train.describe()
fig = plt.figure(figsize=(10,10))



# Tells the total count of different values in CARAVAN

plt.subplot(3,1,1)

input_train['CARAVAN'].value_counts().plot(kind='bar', title='Classifying CARAVAN', color='steelblue', grid=True)



# Tells the total count of different values in customer subtype

plt.subplot(3,1,2)

input_train['MOSTYPE'].value_counts().plot(kind='bar', align='center', title='Classifying customer subtypes', color='steelblue', grid=True)
categorysubtype_caravan = pd.crosstab(input_train['MOSTYPE'], input_train['CARAVAN'])

categorysubtype_caravan_pct = categorysubtype_caravan.div(categorysubtype_caravan.sum(1).astype(float), axis=0)

categorysubtype_caravan_pct.plot(figsize= (8,5), kind='bar', stacked=True, color=['steelblue', 'springgreen'], title='category type vs Caravan', grid=True)

plt.xlabel('Category subtype')

plt.ylabel('Caravan or not')
input_train['MGEMLEEF'].hist(figsize=(5,3), fc='steelblue', grid=True)

plt.xlabel('age')

plt.ylabel('count')
age_caravan = pd.crosstab(input_train['MGEMLEEF'], input_train['CARAVAN'])

age_caravan_pct = age_caravan.div(age_caravan.sum(1).astype(float),axis=0)

age_caravan_pct.plot(figsize=(5,3), kind='bar', stacked=True, color=['steelblue', 'springgreen'], title='dependency of caravan on age groups', grid=True)

plt.xlabel('age groups')

plt.ylabel('Caravan')
input_train['MOSHOOFD'].value_counts().plot(kind='bar', color='steelblue', grid=True)

plt.xlabel('Customer Main Types')

plt.ylabel('count')
cust_type_caravan = pd.crosstab(input_train['MOSHOOFD'], input_train['CARAVAN'])

cust_type_caravan_pct = cust_type_caravan.div(cust_type_caravan.sum(1).astype(float), axis=0)

cust_type_caravan_pct.plot(kind='bar', stacked=True, color = ['steelblue', 'springgreen'])

plt.xlabel('customer types')

plt.ylabel('caravan')
train_data = input_train.values

train_data
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=100)
# Training data features, skip the first column 'Survived'

train_features = train_data[:, :-1]



# 'Survived' column values

train_target = train_data[:, -1]



# Fit the model to our training data

clf = clf.fit(train_features, train_target)

score = clf.score(train_features, train_target)

"Mean accuracy of Random Forest: {0}".format(score)
cv_data = input_cv.values

cv_data
# Training data features, skip the last column 'CARAVAN'

cv_features = cv_data[:, :-1]



# 'caravan' column values

cv_target = cv_data[:, -1]
cv_predictions = clf.predict(cv_features)
from sklearn.metrics import accuracy_score

print ("Accuracy = %.2f" % (accuracy_score(cv_target, cv_predictions)))
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score



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
class_names = np.unique(np.array(cv_target))

confusion_matrices = [

    #( "Support Vector Machines", confusion_matrix(y,run_cv(X,y,SVC)) ),

    ( "Random Forest", confusion_matrix(cv_target, cv_predictions)),

    #( "K-Nearest-Neighbors", confusion_matrix(y,run_cv(X,y,KNN)) ),

    #( "Gradient Boosting Classifier", confusion_matrix(y,run_cv(X,y,GBC)) ),

    #( "Logisitic Regression", confusion_matrix(y,run_cv(X,y,LR)) )

]

draw_confusion_matrices(confusion_matrices,class_names)
forest = RandomForestClassifier()

forest_fit = forest.fit(train_features, train_target)

forest_predictions = forest_fit.predict(cv_features)



importances = forest_fit.feature_importances_[:10]

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



features = input_train.columns



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
test_data = input_test.values

test_data
# Training data features, skip the last column 'CARAVAN'

test_features = test_data[:, :-1]



# 'caravan' column values

test_target = test_data[:, -1]
test_predictions = clf.predict(test_features)
from sklearn.metrics import accuracy_score

print ("Accuracy = %.3f" % (accuracy_score(test_target, test_predictions)))