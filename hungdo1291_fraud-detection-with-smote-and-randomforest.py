import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
credit_cards=pd.read_csv('../input/creditcard.csv')



columns=credit_cards.columns

# The labels are in the last column ('Class'). Simply remove it to obtain features columns

features_columns=columns.delete(len(columns)-1)



features=credit_cards[features_columns]

labels=credit_cards['Class']
features_train, features_test, labels_train, labels_test = train_test_split(features, 

                                                                            labels, 

                                                                            test_size=0.2, 

                                                                            random_state=0)
oversampler=SMOTE(random_state=0)

os_features,os_labels=oversampler.fit_sample(features_train,labels_train)
# verify new data set is balanced

len(os_labels[os_labels==1])
clf=RandomForestClassifier(random_state=0)

clf.fit(os_features,os_labels)
# perform predictions on test set

actual=labels_test

predictions=clf.predict(features_test)
confusion_matrix(actual,predictions)
from sklearn.metrics import roc_curve, auc



false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)
import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')