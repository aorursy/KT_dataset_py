import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, f1_score
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
df.Class.hist()
print(f'Доля подозрительных транзакций {np.sum(df.Class==1)/len(df.Class):.4f}')
X_train, X_test, y_train, y_test = train_test_split(df.drop('Class', axis=1), df.Class)
logReg = LogisticRegression(max_iter=1000)
svm = SVC()
def process_model(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print(f'Accuracy на тренировочных данных {logReg.score(X_train, y_train):.4f}')
    y_pred = clf.predict(X_test)
    print(f'Accuracy на тестовых данных {accuracy_score(y_pred, y_test):.4f}')
    print(f'F1-score на тестовых данных {f1_score(y_pred, y_test):.4f}')
    plot_confusion_matrix(clf, X_test, y_test) 
process_model(logReg, X_train, X_test, y_train, y_test)
process_model(svm, X_train, X_test, y_train, y_test)
ax = plt.gca()
plot_roc_curve(logReg, X_test, y_test, ax=ax)
plot_roc_curve(svm, X_test, y_test, ax=ax);
df_test = pd.concat( (X_test, y_test), axis=1 )
from imblearn.over_sampling import RandomOverSampler
logRegOver = LogisticRegression(max_iter=1000)
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
print('Доля подозрительных транзакций после oversampling', np.sum(y_resampled==0)/len(y_resampled))
process_model(logRegOver, X_resampled, X_test, y_resampled, y_test)
svmOver = SVC(verbose=True, max_iter=1000) #можно и так балансировать классы: class_weight='balanced'
process_model(svmOver, X_resampled, X_test, y_resampled, y_test)
ax = plt.gca()
plot_roc_curve(logReg, X_test, y_test, ax=ax, name='LogReg')
plot_roc_curve(logRegOver, X_test, y_test, ax=ax, name='LogReg Oversampling');
plot_roc_curve(svm, X_test, y_test, ax=ax, name='SVM')
plot_roc_curve(svmOver, X_test, y_test, ax=ax, name='SVM Oversampling');
from imblearn.under_sampling import RandomUnderSampler
logRegUnder = LogisticRegression(max_iter=1000)
svmUnder = SVC(verbose=True, max_iter=1000)
ros = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
process_model(logRegUnder, X_resampled, X_test, y_resampled, y_test)
process_model(svmUnder, X_resampled, X_test, y_resampled, y_test)
ax = plt.gca()
plot_roc_curve(logReg, X_test, y_test, ax=ax, name='LogReg')
plot_roc_curve(logRegOver, X_test, y_test, ax=ax, name='LogReg Oversampling');
plot_roc_curve(logRegUnder, X_test, y_test, ax=ax, name='LogReg Undersampling');
ax = plt.gca()
plot_roc_curve(svm, X_test, y_test, ax=ax, name='SVM')
plot_roc_curve(svmOver, X_test, y_test, ax=ax, name='SVM Oversampling');
plot_roc_curve(svmUnder, X_test, y_test, ax=ax, name='SVM Undersampling');