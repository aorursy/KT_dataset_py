import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' 
    thresh = cm.max() 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
np.random.seed(7)
df = pd.read_csv('../input/creditcard.csv', engine = 'c', error_bad_lines = False)
print(df.isnull().any())
# Resampling data
# Undersampling
fraud = df[df.Class == 1]
not_fraud = df[df.Class == 0]
not_fraud = not_fraud.sample(len(fraud), random_state = 42)
data = pd.concat([fraud, not_fraud])
X = data.loc[:, data.columns != 'Class']
Y = data.loc[:, data.columns == 'Class']
sc = StandardScaler()
X = sc.fit_transform(X)
Y = np.asarray(Y)
Y = np.reshape(Y, Y.shape[0])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size = .33,
                                                        random_state = 42)
rfc = RandomForestClassifier(n_estimators = 5)
rfc.fit(X_train, Y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(Y_test, pred_rfc))
cnf = confusion_matrix(Y_test, pred_rfc)
plot_confusion_matrix(cnf, classes = ['Normal', 'Fraud'], normalize = False)
plt.show(block=True)
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = Y_train, cv = 10)
print(rfc_eval.mean())
print('For skewed dataset.................')
X = df.loc[:, df.columns != 'Class']
Y = df.loc[:, df.columns == 'Class']
sc = StandardScaler()
X = sc.fit_transform(X)
Y = np.asarray(Y)
Y = np.reshape(Y, Y.shape[0])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size = .33,
                                                        random_state = 42)
rfc_skew = RandomForestClassifier(n_estimators = 20)
rfc_skew.fit(X_train, Y_train)
rfc_pred_skew = rfc_skew.predict(X_test)
print(classification_report(Y_test, rfc_pred_skew,
                            target_names = ['Normal', 'Fraud']))
cnf = confusion_matrix(Y_test, rfc_pred_skew)
plot_confusion_matrix(cnf, classes = ['Normal', 'Fraud'], normalize = False)
plt.show(block=False)