%pylab inline

import pandas as pd

from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTEENN 

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve,auc

from sklearn.metrics import confusion_matrix
data = pd.read_csv('../input/creditcard.csv')

data.head()
X_train, X_test, y_train, y_test = train_test_split(data.drop('Class',axis=1), data['Class'], test_size=0.33)
sme = SMOTEENN()

X_train, y_train = sme.fit_sample(X_train, y_train)

unique(y_train, return_counts=True)
clf = RandomForestClassifier(random_state=42)

clf = clf.fit(X_train,y_train)



y_test_hat = clf.predict(X_test)
accuracy_score(y_test,y_test_hat)
print (classification_report(y_test, y_test_hat))
y_score = clf.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_score)



title('Random Forest ROC curve: CC Fraud')

xlabel('FPR (Precision)')

ylabel('TPR (Recall)')



plot(fpr,tpr)

plot((0,1), ls='dashed',color='black')

plt.show()

print ('Area under curve (AUC): ', auc(fpr,tpr))
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap) 

    plt.title(title)

    class_labels = ['Valid','Fraudulent']

    plt.colorbar()

    

    tick_marks = np.arange(len(class_labels)) 

    plt.xticks(tick_marks, class_labels, rotation=90) 

    plt.yticks(tick_marks, class_labels) 

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cm = confusion_matrix(y_test, y_test_hat)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 

plt.figure(figsize=(5,5))

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')