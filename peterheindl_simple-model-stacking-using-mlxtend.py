import pandas as pd

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB 

from sklearn import model_selection

import numpy as np

from mlxtend.classifier import StackingCVClassifier



# Load data for training 

mydata = pd.read_csv('../input/train.csv', sep=',')

# Select only 5000 obs. to let the kernel run here

mydata = mydata.head(5000)

mydata = mydata.drop('ID_code', 1)



# Load prediction data

preddata = pd.read_csv('../input/test.csv', sep=',')

predids = preddata[['ID_code']] 

iddf = preddata[['ID_code']] 

preddata = preddata.drop('ID_code', 1)



# Format train data

y_train = mydata['target']

x_train = mydata.drop('target', 1)



# Scale data

scaler = preprocessing.StandardScaler()

scaled_df = scaler.fit_transform(x_train)

x_train = pd.DataFrame(scaled_df)

scaled_df = scaler.fit_transform(preddata)

preddata = pd.DataFrame(scaled_df)



# x,y to np (needed for scipy CV)

x_train = x_train.values

y_train = y_train.values
# Set up models

clf1 = KNeighborsClassifier(n_neighbors=600)

clf2 = RandomForestClassifier(random_state=1, n_estimators=300)

clf3 = GaussianNB()

# Logit will be used for stacking

lr = LogisticRegression(solver='lbfgs')

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr, use_probas=True, cv=3)



# Do CV

for clf, label in zip([clf1, clf2, clf3, sclf], 

                      ['KNN', 

                       'Random Forest', 

                       'Naive Bayes',

                       'StackingClassifier']):



    scores = model_selection.cross_val_score(clf, x_train, y_train, cv=3, scoring='roc_auc')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



# Fit on train data / predict on test data

sclf_fit = sclf.fit(x_train, y_train)

mypreds = sclf_fit.predict_proba(preddata)

# "predict" delivers classes, "predict_proba" delivers probabilities



# Probabilities for classes (1,0)

zeros = [i[0] for i in mypreds]

ones  = [i[1] for i in mypreds]



# Get IDs and predictions

y_id = predids.values.tolist()

preddf = pd.DataFrame({'ID_code': y_id,'target': ones})

preddf['ID_code'] = preddf['ID_code'].map(lambda x: str(x)[:-2])

preddf['ID_code'] = preddf['ID_code'].map(lambda x: str(x)[2:])



# Look at predictions

print(preddf.head())



# Save DF

preddf.to_csv('submission.csv', index=False)