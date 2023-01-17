import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt



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

    plt.show()
def evaluate_model(predictions, probs, train_predictions, train_probs):

    """Compare machine learning model to baseline performance.

    Computes statistics and shows ROC curve."""

    

    baseline = {}

    

    baseline['recall'] = recall_score(y_test, [1 for _ in range(len(y_test))])

    baseline['precision'] = precision_score(y_test, [1 for _ in range(len(y_test))])

    baseline['roc'] = 0.5

    

    results = {}

    

    results['recall'] = recall_score(y_test, predictions)

    results['precision'] = precision_score(y_test, predictions)

    results['roc'] = roc_auc_score(y_test, probs)

    

    train_results = {}

    train_results['recall'] = recall_score(y_train, train_predictions)

    train_results['precision'] = precision_score(y_train, train_predictions)

    train_results['roc'] = roc_auc_score(y_train, train_probs)

    

#     for metric in ['recall', 'precision', 'roc']:

#         print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

    

    # Calculate false positive rates and true positive rates

    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])

    model_fpr, model_tpr, _ = roc_curve(y_test, probs)



    plt.figure(figsize = (8, 6))

    plt.rcParams['font.size'] = 16

    

    # Plot both curves

    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')

    plt.plot(model_fpr, model_tpr, 'r', label = 'model')

    plt.legend();

    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
def predictAndCreateConfusionMatrix(model,X_predict,y_true):

    yhat = model.predict(X_predict)

    

    cnf_matrix = confusion_matrix(y_true, yhat, labels=[1,0])

    np.set_printoptions(precision=2)





    # Plot non-normalized confusion matrix

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=['1','0'],normalize= False,  title='Confusion matrix')
import pandas as pd

cs_test = pd.read_csv("../input/give-me-some-credit-dataset/cs-test.csv")

cs_training = pd.read_csv("../input/give-me-some-credit-dataset/cs-training.csv")

sampleEntry = pd.read_csv("../input/give-me-some-credit-dataset/sampleEntry.csv")

print(len(cs_training.index))
dfCredit_NoBaby = cs_training.loc[cs_training['age'] > 0]

dfCredit_NoBaby_NoNulls = dfCredit_NoBaby.loc[dfCredit_NoBaby['NumberOfDependents'].notnull() & dfCredit_NoBaby['MonthlyIncome'].notnull()]

dfCredit_Removed  = dfCredit_NoBaby_NoNulls.loc[:, dfCredit_NoBaby_NoNulls.columns != 'Id']

print(len(dfCredit_Removed.index))
#filt_df.describe()

low = .02

high = .98

quant_df = dfCredit_Removed.quantile([low, high])

#print(quant_df)

dfCredit_RemoveOutliers = dfCredit_Removed.apply(lambda x: x[(x >= quant_df.loc[low,x.name]) & 

                                    (x <= quant_df.loc[high,x.name])], axis=0)

dfCredit_98Quantile = dfCredit_RemoveOutliers.dropna()

print(len(dfCredit_98Quantile.index))

dfCredit_98Quantile.describe()
dfOutlierMissing = cs_training.dropna()

dfOutlierMissing.describe()
dfCredit_Use = dfOutlierMissing#dfCredit_NoBaby_NoNulls_NoID.dropna()#dfCredit_98Quantile

#lsIxDelete = dfCredit_Use[dfCredit_Use['DebtRatio'] >= 1].index

#lsIxDelete += dfCredit_Use[dfCredit_Use['MonthlyIncome'] > 13000].index

#dfCredit_Use = dfCredit_Use.drop(lsIxDelete)

desc = dfCredit_Use.describe()

desc
dfCredit_Use['NumberOfNonRealEstateCreditLines'] = dfCredit_Use['NumberOfOpenCreditLinesAndLoans']  - dfCredit_Use['NumberRealEstateLoansOrLines']

dfCredit_Use['TotalDebtIncome'] = dfCredit_Use['DebtRatio'] * dfCredit_Use['MonthlyIncome']

dfCredit_Use['NumberOfTimesPastDue'] = dfCredit_Use['NumberOfTime30-59DaysPastDueNotWorse'] + dfCredit_Use['NumberOfTime60-89DaysPastDueNotWorse'] + dfCredit_Use['NumberOfTimes90DaysLate']
dfCredit_Use.loc[:,'disc_age'] = '<29'



dfCredit_Use.loc[:,'age_21_29'] = 0

lsIx = dfCredit_Use.loc[(dfCredit_Use['age'] <= 29)].index

dfCredit_Use.ix[lsIx,'age_21_29'] = 1



dfCredit_Use.loc[:,'age_30_69'] = 0

lsIx = dfCredit_Use.loc[(dfCredit_Use['age'] >= 30) & (dfCredit_Use['age']<= 69)].index

dfCredit_Use.ix[lsIx,'age_30_69'] = 1

dfCredit_Use.ix[lsIx,'disc_age'] = '30_69'



dfCredit_Use.loc[:,'age_70_'] = 0

lsIx = dfCredit_Use.loc[(dfCredit_Use['age'] >= 70)].index

dfCredit_Use.ix[lsIx,'age_70_'] = 1

dfCredit_Use.ix[lsIx,'disc_age'] = '>70'
dfCredit_Use.loc[:,'disc_numcreditlines'] = '<4'



lsIx = dfCredit_Use.loc[(dfCredit_Use['NumberOfOpenCreditLinesAndLoans'] >=4)].index

dfCredit_Use.ix[lsIx,'disc_numcreditlines'] = '4+'
dfCredit_Use.loc[:,'disc_debtratio'] = '<1'



lsIx = dfCredit_Use.loc[(dfCredit_Use['DebtRatio'] >=1)].index

dfCredit_Use.ix[lsIx,'disc_debtratio'] = '1+'
dfCredit_Use.loc[:,'disc_income'] = '<1000'



dfCredit_Use.loc[:,'income_0_1999'] = 0

lsIx = dfCredit_Use.loc[(dfCredit_Use['MonthlyIncome'] <= 1999)].index

dfCredit_Use.ix[lsIx,'income_0_1999'] = 1



dfCredit_Use.loc[:,'income_1999_13000'] = 0

lsIx = dfCredit_Use.loc[(dfCredit_Use['MonthlyIncome'] > 1999) & (dfCredit_Use['MonthlyIncome']<= 13000)].index

dfCredit_Use.ix[lsIx,'income_1999_13000'] = 1

dfCredit_Use.ix[lsIx,'disc_income'] = '1999_13000'



dfCredit_Use.loc[:,'income_13000_'] = 0

lsIx = dfCredit_Use.loc[(dfCredit_Use['MonthlyIncome'] > 13000)].index

dfCredit_Use.ix[lsIx,'income_13000_'] = 1

dfCredit_Use.ix[lsIx,'disc_income'] = '>13000'
dfCredit_Use['irc_age_21_29'] = dfCredit_Use['age_21_29'] * dfCredit_Use['age']

dfCredit_Use['irc_age_30_69'] = dfCredit_Use['age_30_69'] * dfCredit_Use['age']

dfCredit_Use['irc_age_70_'] = dfCredit_Use['age_70_'] * dfCredit_Use['age']



dfCredit_Use['irc_income_0_1999'] = dfCredit_Use['income_0_1999'] * dfCredit_Use['MonthlyIncome']

dfCredit_Use['irc_income_1999_13000'] = dfCredit_Use['income_1999_13000'] * dfCredit_Use['MonthlyIncome']
dfCredit_Use.describe()
lsContinuousRegressors = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',

                         'DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',

                         'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',

                         'NumberOfDependents']
lsContinuousRegressors = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTimesPastDue',

                            'NumberOfNonRealEstateCreditLines'] #,'NumberOfDependents','TotalDebtIncome',
import numpy as np

from matplotlib.ticker import NullFormatter

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import itertools
lsContinuousRegressors = ['age','TotalDebtIncome','MonthlyIncome','DebtRatio',

                         'NumberRealEstateLoansOrLines', 

                         'NumberOfDependents',

                         'RevolvingUtilizationOfUnsecuredLines',

                         'NumberOfTime30-59DaysPastDueNotWorse','NumberOfTime60-89DaysPastDueNotWorse','NumberOfTimes90DaysLate',

                         #'NumberOfOpenCreditLinesAndLoans',

                         'NumberOfNonRealEstateCreditLines',

                         'NumberOfDependents']

lsInteractionRegressors = ['irc_age_21_29','irc_age_30_69','irc_age_70_','irc_income_0_1999','irc_income_1999_13000']

lsDiscreteRegressors = ['age_21_29','age_70_','income_0_1999','income_13000_']



lsRegressors = lsContinuousRegressors + lsInteractionRegressors + lsDiscreteRegressors

lsRegressors
y = np.asarray(dfCredit_Use['SeriousDlqin2yrs'])

X_cont = np.asarray(dfCredit_Use[lsContinuousRegressors])#

X_cont_norm = preprocessing.StandardScaler().fit(X_cont).transform(X_cont)

x_irc = np.asarray(dfCredit_Use[lsInteractionRegressors])

X_disc = np.asarray(dfCredit_Use[lsDiscreteRegressors])

X_full = np.concatenate((X_cont_norm,X_disc),axis=1)#x_irc,

print(len(y))

print(len(X_full))
X_train, X_test, y_train, y_test = train_test_split( X_cont, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
LR = LogisticRegression(C=.01,solver='liblinear').fit(X_train,y_train)

print(LR)
from collections import OrderedDict

dictCoeff = OrderedDict()

inc = 0

for coef in lsRegressors:

    dictCoeff[coef] = LR.coef_[0][inc]/(1-LR.coef_[0][inc])

    print(coef + ': ' + str(dictCoeff[coef]))

    inc+=1
predictAndCreateConfusionMatrix(LR,X_test,y_test)
predictAndCreateConfusionMatrix(LR,X_train,y_train)
clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)
predictAndCreateConfusionMatrix(clf,X_test,y_test)
predictAndCreateConfusionMatrix(clf,X_train,y_train)
from collections import Counter

dictCounter = Counter(y_test)



print("Increase in true positive: " + str((537-465)/dictCounter[1]))

print("Increase in false positive: " + str((430-277)/dictCounter[0]))
# RANDOM FOREST



from sklearn.ensemble import RandomForestClassifier



# Create the model with 100 trees

model = RandomForestClassifier(n_estimators=100, 

                               bootstrap = True,

                               max_features = 'sqrt')

# Fit on training data

model.fit(X_train,y_train)
# Actual class predictions

rf_predictions = model.predict(X_test)

# Probabilities for each class

rf_probs = model.predict_proba(X_test)[:, 1]



# train set

rf_train_predictions = model.predict(X_train)

# Probabilities for each class

rf_train_probs = model.predict_proba(X_train)[:, 1]
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve



# Calculate roc auc

AUC_rf = roc_auc_score(y_test, rf_probs)

AUC_rf
from collections import Counter

print(Counter(rf_predictions))

print(Counter(y_test))
print (classification_report(y_test, rf_predictions))
# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, rf_predictions, labels=[1,0])

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['1','0'],normalize= False,  title='Confusion matrix')
evaluate_model(rf_predictions, rf_probs, rf_train_predictions, rf_train_probs)
from sklearn.model_selection import RandomizedSearchCV



# Hyperparameter grid

param_grid = {

    'n_estimators': np.linspace(10, 200).astype(int),

    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),

    'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),

    'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),

    'min_samples_split': [2, 5, 10],

    'bootstrap': [True, False]

}



# Estimator for use in random search

estimator = RandomForestClassifier(random_state = 5)



# Create the random search model

rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1, 

                        scoring = 'roc_auc', cv = 3, 

                        n_iter = 10, verbose = 1, random_state=5)



# Fit 

rs.fit(X_train, y_train)
rs.best_params_
best_model = rs.best_estimator_
train_rfb_predictions = best_model.predict(X_train)

train_rfb_probs = best_model.predict_proba(X_train)[:, 1]



rfb_predictions = best_model.predict(X_test)

rfb_probs = best_model.predict_proba(X_test)[:, 1]
n_nodes = []

max_depths = []



for ind_tree in best_model.estimators_:

    n_nodes.append(ind_tree.tree_.node_count)

    max_depths.append(ind_tree.tree_.max_depth)

    

print(f'Average number of nodes {int(np.mean(n_nodes))}')

print(f'Average maximum depth {int(np.mean(max_depths))}')
evaluate_model(rfb_predictions, rfb_probs, train_rfb_predictions, train_rfb_probs)
# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, rfb_predictions, labels=[1,0])

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['1','0'],normalize= False,  title='Confusion matrix')
# Calculate roc auc

AUC_rfb = roc_auc_score(y_testset, rfb_probs)

AUC_rfb