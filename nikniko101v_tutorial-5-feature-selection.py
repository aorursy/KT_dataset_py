import pandas as pd

import numpy as np



from sklearn import preprocessing #needed for scaling attributes to the nterval [0,1]



from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFE



from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/seeds-dataset-binary/seeds_dataset_binary.csv')

df.describe()
# target attribute

target_attribute_name = 'type'

target = df[target_attribute_name]



# predictor attributes

predictors = df.drop(target_attribute_name, axis=1).values
# pepare independent stratified data sets for training and test of the final model

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size=0.20, shuffle=True, stratify=target)
min_max_scaler = preprocessing.MinMaxScaler()

predictors_train = min_max_scaler.fit_transform(predictors_train)

predictors_test = min_max_scaler.fit_transform(predictors_test)
# create a base classifier used to evaluate a subset of attributes

estimatorSVM = svm.SVR(kernel="linear")

selectorSVM = RFE(estimatorSVM, 3)

selectorSVM = selectorSVM.fit(predictors_train, target_train)

# summarize the selection of the attributes

print(selectorSVM.support_)

print(selectorSVM.ranking_)
# create a base classifier used to evaluate a subset of attributes

estimatorLR = LogisticRegression(solver='lbfgs')

# create the RFE model and select 3 attributes

selectorLR = RFE(estimatorLR, 3)

selectorLR = selectorLR.fit(predictors_train, target_train)

# summarize the selection of the attributes

print(selectorLR.support_)

print(selectorLR.ranking_)
predictors_train_SVMselected = selectorSVM.transform(predictors_train)

predictors_test_SVMselected = selectorSVM.transform(predictors_test)
predictors_train_LRselected = selectorLR.transform(predictors_train)

predictors_test_LRselected = selectorLR.transform(predictors_test)
classifier = svm.SVC(gamma='auto')
model1 = classifier.fit(predictors_train_SVMselected, target_train)

model1.score(predictors_test_SVMselected, target_test)
model2 = classifier.fit(predictors_train_LRselected, target_train)

model2.score(predictors_test_LRselected, target_test)
model3 = classifier.fit(predictors_train, target_train)

model3.score(predictors_test, target_test)