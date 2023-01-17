import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/Breast_cancer_data.csv')
df.head()
df.info()
df.describe()
df.hist()
df.corr()
cmap = sns.diverging_palette(260, 10, as_cmap=True)
sns.heatmap(df.corr(),cmap=cmap)
sns.boxplot(df.mean_radius,df.diagnosis, orient='h')
sns.boxplot(df.mean_texture,df.diagnosis, orient='h')
sns.boxplot(df.mean_perimeter,df.diagnosis, orient='h')
sns.boxplot(df.mean_area,df.diagnosis, orient='h')
sns.boxplot(df.mean_smoothness,df.diagnosis, orient='h')
X = df.drop(['diagnosis'], axis=1)
y = df.diagnosis
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2)
from sklearn import tree, ensemble, svm
from sklearn.metrics import accuracy_score
classifiers = [
                tree.DecisionTreeClassifier(),
                ensemble.AdaBoostClassifier(), 
                ensemble.BaggingClassifier(),
                ensemble.GradientBoostingClassifier(),
                ensemble.RandomForestClassifier(n_estimators=2),
                svm.SVC(kernel='linear', gamma='scale', probability=True)
              ]
classifiers_names = ['decision_tree', 'ada_boost', 'bagging', 'gradient_boosting', 'random_forest', 'svc']

trained_models = {}
for i, clf in enumerate(classifiers):
    trained_models[classifiers_names[i]] = clf.fit(X_train, y_train)
accuracy = {}
for name, classifier in trained_models.items():
    accuracy[name] = accuracy_score(classifier.predict(X_test), y_test)

df_classifiers = pd.DataFrame.from_dict(accuracy,orient='index', columns=['accuracy'])
df_classifiers.sort_values('accuracy', ascending=0)
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from itertools import cycle

def make_roc_curve(classifier, X, y):
    print("Analyzing classifier: " + str(classifier))
    n_samples, n_features = X.shape

    cv = StratifiedKFold(n_splits=12)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.loc[train], y.loc[train]).predict_proba(X.loc[test])
        fpr, tpr, thresholds = roc_curve(y.loc[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return mean_auc
for classifier in classifiers:
    make_roc_curve(classifier, X, y)
from sklearn.ensemble import AdaBoostClassifier
final_classifier = AdaBoostClassifier()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=11)
final_classifier.fit(X_train, y_train)
predictions = final_classifier.predict(X_test)
accuracy_score(predictions, y_test)
from sklearn.metrics import confusion_matrix
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

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def show_confusion_matrix(y_test, y_pred):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    class_names = list(np.unique(y_pred))
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
show_confusion_matrix(y_test, predictions)
