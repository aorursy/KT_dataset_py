import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler, SMOTE

from collections import Counter

from scipy import interp

import itertools

%matplotlib inline



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
table = pd.read_csv('../input/creditcardfraud/creditcard.csv')
table.isnull().sum().all()
table.head()
table.tail()
table.info()
table.describe()
plt.bar(['Non-Fraud','Fraud'], table['Class'].value_counts(), color=['b','r'])

plt.xlabel('Class')

plt.ylabel('Number of transactions')

plt.annotate('{}\n({:.4}%)'.format(table['Class'].value_counts()[0], 

                                         table['Class'].value_counts()[0]/table['Class'].count()*100),

             (0.20, 0.45), xycoords='axes fraction')

plt.annotate('{}\n({:.4}%)'.format(table['Class'].value_counts()[1], 

                                         table['Class'].value_counts()[1]/table['Class'].count()*100),

             (0.70, 0.45), xycoords='axes fraction')

plt.tight_layout()

plt.show()
plt.scatter(table['Time']/(60*60), table['Class'])

plt.xlabel('Time of transaction (in hours)')

plt.ylabel('Class')



plt.tight_layout()

plt.show()
plt.boxplot(table['Amount'], labels = ['Boxplot'])

plt.ylabel('Transaction amount')

plt.plot()



amount = table[['Amount']].sort_values(by='Amount')

q1, q3 = np.percentile(amount,[25,75])

iqr = q3 - q1

lower_bound = q1 -(1.5 * iqr) 

upper_bound = q3 +(1.5 * iqr)



print('Number of outliers below the lower bound: ', amount[amount['Amount'] < lower_bound].count()[0],

     ' ({:.4}%)'.format(amount[amount['Amount'] < lower_bound].count()[0] / amount['Amount'].count() * 100))

print('Number of outliers above the upper bound: ', amount[amount['Amount'] > upper_bound].count()[0],

      ' ({:.4}%)'.format(amount[amount['Amount'] > upper_bound].count()[0] / amount['Amount'].count() * 100))
table[table['Class']==1].where(table['Amount']>upper_bound).count()['Amount']
plt.scatter(table['Amount'], table['Class'])

plt.xlabel('Amount')

plt.ylabel('Class')

plt.show()
target_0 = table.loc[table['Class'] == 0]

target_1 = table.loc[table['Class'] == 1]

ax1=sns.distplot(target_0[['Amount']], hist=False, color='b', label='Non-fraud')

ax2=sns.distplot(target_1[['Amount']], hist=False, color='r', label='Fraud')

ax1.set_xlim(0, max(table[table['Class']==1]['Amount']))

ax2.set_xlim(0, max(table[table['Class']==1]['Amount']))

plt.legend()

plt.xlabel('Amount')

plt.ylabel('Density of probability')

plt.show()
table.loc[table['Class'] == 1]['Amount'].describe()
heatmap = sns.heatmap(table.corr(method='spearman'))
table.corrwith(table.Class).plot.bar(figsize = (20, 10), title = "Correlation with class", fontsize = 15, 

                                     rot = 45, grid = True, color=['blue'])

plt.show()
y = table['Class']

X = table.drop(columns=['Class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, random_state=42)
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42, replacement=False)

X_rus, y_rus = rus.fit_resample(X_train, y_train)
plt.bar(['Non-Fraud','Fraud'], [Counter(y_rus)[0], Counter(y_rus)[1]], color=['b','r'])

plt.xlabel('Class')

plt.ylabel('Number of transactions')

plt.annotate('{}'.format(Counter(y_rus)[0]), (0.25, 0.45), xycoords='axes fraction')

plt.annotate('{}'.format(Counter(y_rus)[1]), (0.75, 0.45), xycoords='axes fraction')



plt.tight_layout()

plt.show()
assert Counter(y_rus)[1] == Counter(y_train)[1] #Checking if they have the same number of fraud cases
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)

X_ros, y_ros = ros.fit_resample(X_train, y_train)
plt.bar(['Non-Fraud','Fraud'], [Counter(y_ros)[0], Counter(y_ros)[1]], color=['b','r'])

plt.xlabel('Class')

plt.ylabel('Number of transactions')

plt.annotate('{}'.format(Counter(y_ros)[0]), (0.20, 0.45), xycoords='axes fraction')

plt.annotate('{}'.format(Counter(y_ros)[1]), (0.70, 0.45), xycoords='axes fraction')



plt.tight_layout()

plt.show()
assert Counter(y_ros)[0] == Counter(y_train)[0] #Checking if they have the same number of non-fraud cases
smote = SMOTE(sampling_strategy='auto', random_state=42)

X_smote, y_smote = smote.fit_resample(X_train, y_train)
plt.bar(['Non-Fraud','Fraud'], [Counter(y_smote)[0], Counter(y_smote)[1]], color=['b','r'])

plt.xlabel('Class')

plt.ylabel('Number of transactions')

plt.annotate('{}'.format(Counter(y_smote)[0]), (0.20, 0.45), xycoords='axes fraction')

plt.annotate('{}'.format(Counter(y_smote)[1]), (0.70, 0.45), xycoords='axes fraction')



plt.tight_layout()

plt.show()
assert Counter(y_smote)[0] == Counter(y_train)[0] #Checking if they have the same number of non-fraud cases
def feature_scaling(X, X_test=X_test):

    std_scale = StandardScaler().fit(X)

    X_std = std_scale.transform(X)

    X_test_std = std_scale.transform(X_test)

    return X_std, X_test_std
X_rus_std, X_test_rus_std = feature_scaling(X_rus)

X_ros_std, X_test_ros_std = feature_scaling(X_ros)

X_smote_std, X_test_smote_std = feature_scaling(X_smote)
pca = PCA(n_components=2)

X_ros_pca = pca.fit_transform(X_ros_std)

X_smote_pca = pca.fit_transform(X_smote_std)
def plot_2d_space(X, y, label='Classes'):

    '''Plots the data points in a 2D scatterplot.'''

    colors = ['blue', 'red']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(X[y==l, 0], X[y==l, 1], c=c, label=l, marker=m)

    plt.title(label)

    plt.legend(loc='best')

    plt.show()
plot_2d_space(X_ros_pca, y_ros, 'Balanced dataset (2 PCA components) using random oversampling')

plot_2d_space(X_smote_pca, y_smote, 'Balanced dataset (2 PCA components) using SMOTE')
classifiers = []



classifiers.append(('Logistic Regression', LogisticRegression(random_state=42)))

classifiers.append(('Naive Bayes', GaussianNB()))

classifiers.append(('KNN', KNeighborsClassifier()))

#classifiers.append(('SVM', SVC(random_state=42, probability=True))) #This one takes a very long time to run!

classifiers.append(('Decision Tree', DecisionTreeClassifier(random_state=42)))

classifiers.append(('Random Forest', RandomForestClassifier(random_state=42)))



#Ensemble classifier - All classifiers have the same weight

eclf = VotingClassifier(estimators=classifiers, voting='soft', weights=np.ones(len(classifiers)))
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

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    #if normalize:

    #    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #    print("Normalized confusion matrix")

    #else:

    #    print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn import svm

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import StratifiedKFold

from scipy import interp



def plot_CM_and_ROC_curve(classifier, X_train, y_train, X_test, y_test):

    '''Plots the ROC curve and the confusion matrix, and calculates AUC, recall and precision.'''

    

    name = classifier[0]

    classifier = classifier[1]



    mean_fpr = np.linspace(0, 1, 100)

    class_names = ['Non-Fraud', 'Fraud']

    confusion_matrix_total = [[0, 0], [0, 0]]

    

    #Obtain probabilities for each class

    probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)

    

    # Compute ROC curve and area the curve

    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=1, alpha=1, color='b', label='ROC (AUC = %0.7f)' % (roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',

             label='Chance', alpha=.8)

    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC curve - model: ' + name)

    plt.legend(loc="lower right")

    plt.show()

    

    #Store the confusion matrix result to plot a table later

    y_pred=classifier.predict(X_test)

    cnf_matrix = confusion_matrix(y_test, y_pred)

    confusion_matrix_total += cnf_matrix

    

    #Print precision and recall

    tn, fp = confusion_matrix_total.tolist()[0]

    fn, tp = confusion_matrix_total.tolist()[1]

    accuracy = (tp+tn)/(tp+tn+fp+fn)

    precision = tp/(tp+fp)

    recall = tp/(tp+fn)

    print('Accuracy = {:2.2f}%'.format(accuracy*100))

    print('Precision = {:2.2f}%'.format(precision*100))

    print('Recall = {:2.2f}%'.format(recall*100))

    

    # Plot confusion matrix

    plt.figure()

    plot_confusion_matrix(confusion_matrix_total, classes=class_names, title='Confusion matrix - model: ' + name)

    plt.show()
for clf in classifiers:

    plot_CM_and_ROC_curve(clf, X_rus_std, y_rus, X_test_rus_std, y_test)
plot_CM_and_ROC_curve(('Ensemble model', eclf), X_rus_std, y_rus, X_test_rus_std, y_test)
for clf in classifiers:

    plot_CM_and_ROC_curve(clf, X_ros_std, y_ros, X_test_ros_std, y_test)
plot_CM_and_ROC_curve(('Ensemble model', eclf), X_ros_std, y_ros, X_test_ros_std, y_test)
for clf in classifiers:

    plot_CM_and_ROC_curve(clf, X_smote_std, y_smote, X_test_smote_std, y_test)
plot_CM_and_ROC_curve(('Ensemble model', eclf), X_smote_std, y_smote, X_test_smote_std, y_test)