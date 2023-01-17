import pandas  as pd

import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data['V1'].describe()
data['V2'].describe()
# Undersample the majority - Non Fraud 

# Oversample the minority calss - Fraud



# Balance the dataset
pd.value_counts(data['Class']).plot.bar()

plt.title('Fraud class histogram')

plt.xlabel('Class')

plt.ylabel('Frequency')

data['Class'].value_counts()
# Standard and Robust Scaler

# Min Max scaler
# Preprocessing data in scikit

# Trained and evaluated the model performance
from sklearn.preprocessing import StandardScaler



data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

data = data.drop(['Time', 'Amount'], axis=1)

data.head()
X = np.array(data.iloc[:, data.columns != 'Class'])

y = np.array(data.iloc[:, data.columns == 'Class'])

print('Shape of X: {}'.format(X.shape))

print('Shape of y: {}'.format(y.shape))
from imblearn.over_sampling import SMOTE



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))



sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())



print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report



parameters = {

    'C': np.linspace(1, 10, 10)

             }

lr = LogisticRegression()

clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)

clf.fit(X_train_res, y_train_res.ravel())
clf.best_params_
lr1 = LogisticRegression(C=9,penalty='l2', verbose=5)

lr1.fit(X_train_res, y_train_res.ravel())
import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
y_train_pre = lr1.predict(X_train)



cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)



print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))





class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')

plt.show()
cnf_matrix_tra
y_pre = lr1.predict(X_test)



cnf_matrix = confusion_matrix(y_test, y_pre)



print("Recall metric in the testing dataset: {}%".format(100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))

#print("Precision metric in the testing dataset: {}%".format(100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])))

# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix , classes=class_names, title='Confusion matrix')

plt.show()
cnf_matrix
tmp = lr1.fit(X_train_res, y_train_res.ravel())
y_pred_sample_score = tmp.decision_function(X_test)





fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score)



roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
roc_auc