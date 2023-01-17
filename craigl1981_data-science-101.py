import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

from sklearn import datasets, preprocessing

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc

from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout, Histogram

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
train, test = pd.read_csv('../input/train.csv') , pd.read_csv('../input/test.csv')


train.head(10)
# One-hot encode sex, class and derive a married variable
train['female'] = train['Sex'].apply(lambda x: 1 if x == 'female' else 0)
train['class_1'] = train['Pclass'].apply(lambda x: 1 if x == 1 else 0)
train['class_2'] = train['Pclass'].apply(lambda x: 1 if x == 2 else 0)
train['class_3'] = train['Pclass'].apply(lambda x: 1 if x == 3 else 0)

train['married'] = train['Name'].apply(lambda x: 1 if 'MRS' in x.upper() else 0)
# Let's engineer a feature
train['n_names'] = train['Name'].apply(lambda x: len(x.split(' '))-1)

train['embarked_C'] = train['Embarked'].apply(lambda x: 1 if x == 'C' else 0)
train['embarked_S'] = train['Embarked'].apply(lambda x: 1 if x == 'S' else 0)
train['embarked_Q'] = train['Embarked'].apply(lambda x: 1 if x == 'Q' else 0)



# We need to replicate this for the Test data too
test['female'] = test['Sex'].apply(lambda x: 1 if x == 'female' else 0)
test['class_1'] = test['Pclass'].apply(lambda x: 1 if x == 1 else 0)
test['class_2'] = test['Pclass'].apply(lambda x: 1 if x == 2 else 0)
test['class_3'] = test['Pclass'].apply(lambda x: 1 if x == 3 else 0)
test['married'] = test['Name'].apply(lambda x: 1 if 'MRS' in x.upper() else 0)
test['n_names'] = test['Name'].apply(lambda x: len(x.split(' '))-1)
test['embarked_C'] = test['Embarked'].apply(lambda x: 1 if x == 'C' else 0)
test['embarked_S'] = test['Embarked'].apply(lambda x: 1 if x == 'S' else 0)
test['embarked_Q'] = test['Embarked'].apply(lambda x: 1 if x == 'Q' else 0)


#We need to remove any NaN values
features = ['female', 'Age', 'Fare', 'class_1', 'class_2','class_3',
            'married','n_names', 'embarked_C', 'embarked_Q', 'embarked_S']
output = 'Survived'

train_all = train[features + [output]]
train_all[train_all==np.inf]=np.nan
train_all.dropna(inplace=True)

# Let's also scale the age and fare while we are at it, this should minimize training time
scaler = preprocessing.MinMaxScaler()
scaler.fit(train_all[['Age', 'Fare']])
train_all['Age_t'] = scaler.transform(train_all[['Age', 'Fare']])[:,0]
train_all['Fare_t'] = scaler.transform(train_all[['Age', 'Fare']])[:,1]



train.groupby('Survived').describe().T
train.boxplot(column='Age', by='Survived', figsize=(15,7))
train.boxplot(column='Fare', by='Survived', figsize=(15,7))

print(sp.stats.ttest_ind(train.loc[train['Survived']==1, 'Age'].dropna(),
                         train.loc[train['Survived']!=1, 'Age'].dropna()))
print(sp.stats.ttest_ind(train.loc[train['Survived']==1, 'Fare'].dropna(),
                         train.loc[train['Survived']!=1, 'Fare'].dropna()))
# If we want to test categorical variables univariately we could use chi-squared test
cont = pd.crosstab(train['Embarked'], train['Survived'])
chsq = sp.stats.chi2_contingency(cont)
print('Chi^2: ' +  str(chsq[0]))
print('P-value: ' +  str(chsq[1]))
cont

#list of features I want to include
features = ['female', 'Age_t', 'Fare', 'class_1',
            'married','n_names']
output = 'Survived'

#Split the data set so we can validate our model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_all[features], train_all[output], test_size=0.33, random_state=42)




import seaborn as sns
corr = train_all.iloc[:,:20].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True)
# First using 
logit_mod = sm.Logit(y_train, X_train)
logit_res = logit_mod.fit()
print(logit_res.summary2())
# Take the exponential of the parameters and we get the odds 
logit_res.params.apply(np.exp)
X_test.head()
# Logit function, this should work regardless of the number of variables, 
# as long as the params match the number of columns in test data (an they are the correct order)

1/(1 + np.exp(-sum((X_test.values * logit_res.params.values.T).T)))
# just to show we get the same
logit_res.predict(X_test)
from sklearn.metrics import confusion_matrix,roc_curve, auc
import itertools
import numpy as np

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
    
# Create classifiers, istantiate the objects
lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0, random_state=42)
rfc = RandomForestClassifier(n_estimators=100, random_state=42)


# #############################################################################
# Plot calibration plots

plt.figure(figsize=(10, 10))
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fpr, tpr, _ = roc_curve(y_test, prob_pos)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s (auc = %0.2f)' % (name, roc_auc))

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


plt.show()
    
    
for clf, name in [(lr, 'Logistic'),
              (gnb, 'Naive Bayes'),
              (svc, 'Support Vector Classification'),
              (rfc, 'Random Forest')]:
        # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, clf.predict(X_test))
    np.set_printoptions(precision=2)

    class_names = ['Dead', 'Survived']
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix for %s' % name)



