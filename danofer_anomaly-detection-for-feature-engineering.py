import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier, IsolationForest



from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold , train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve, auc , confusion_matrix

import numpy as np

np.random.seed(42)

from sklearn.metrics import roc_curve, auc
credit_cards=pd.read_csv('../input/creditcard.csv')

columns=credit_cards.columns

# The labels are in the last column ('Class').
import numpy as np

credit_cards['Amount'] = np.log(credit_cards['Amount'] + 1)

credit_cards['Time'] = np.log(credit_cards['Time'] + 1)
features_train = credit_cards
model = IsolationForest(random_state=42,  max_samples=0.96, bootstrap=True, n_estimators=90,contamination= 0.002)

model.fit(features_train)
print(model.decision_function(credit_cards[credit_cards['Class'] == 0].values).mean())
# print(model.decision_function(credit_cards[credit_cards['Class'] == 0].drop('Class', axis=1).values).mean())

# print(model.decision_function(credit_cards[credit_cards['Class'] == 1].drop('Class', axis=1).values).mean())
credit_cards["isolation_score"] = model.decision_function(credit_cards.drop('Class', axis=1).values)
features=credit_cards.drop(['Class'],axis=1)

labels=credit_cards['Class']



features_train, features_test, labels_train, labels_test = train_test_split(features, 

                                                                            labels, 

                                                                            test_size=0.2, 

                                                                            random_state=0)
labels_train.shape
# oversampler=SMOTE(random_state=0)

sampler = RandomUnderSampler(random_state=0,ratio=0.006)

os_features,os_labels=sampler.fit_sample(features_train,labels_train)
# verify new data set is balanced

len(os_labels[os_labels==1])
os_labels.shape
clf=RandomForestClassifier(random_state=0,class_weight="balanced",n_estimators=300,criterion='entropy')

clf.fit(os_features.drop("isolation_score",axis=1),os_labels)
# perform predictions on test set

actual=labels_test

predictions=clf.predict(features_test)
confusion_matrix(actual,predictions)
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)