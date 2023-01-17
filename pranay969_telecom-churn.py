import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

%config IPCompleter.greedy=True

import statistics



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/telecom_churn_data.csv')
df.info()
df.head()
df.isnull().sum()
avg_rech = (df.total_rech_amt_6 + df.total_rech_amt_7) / 2

df = df.loc[avg_rech > avg_rech.quantile(.7)]

df.info()
df = df.drop(df.loc[:,list((100*(df.isnull().sum()/len(df.index)) > 60))].columns, 1)
df.info()
df = df.loc[df.isnull().sum(axis=1) < 5]
df.isnull().sum()
df.info()
df = df.drop(['last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8', 'last_date_of_month_9'], axis=1)
df = df.drop(['std_og_t2c_mou_6', 'std_og_t2c_mou_7', 'std_og_t2c_mou_8', 'std_og_t2c_mou_9'], axis=1)
df = df.drop(['loc_og_t2o_mou', 'std_og_t2o_mou', 'loc_ic_t2o_mou'], axis=1)
df = df.drop(['mobile_number', 'circle_id'], axis=1)
df = df.drop(['std_ic_t2o_mou_6', 'std_ic_t2o_mou_7', 'std_ic_t2o_mou_8', 'std_ic_t2o_mou_9'], axis=1)
df = df.drop(['date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8', 'date_of_last_rech_9'], axis=1)
df = df.drop(['last_day_rch_amt_6', 'last_day_rch_amt_7', 'last_day_rch_amt_8', 'last_day_rch_amt_9'], axis=1)
df['churn'] = df['total_ic_mou_9'] + df['total_og_mou_9'] + df['vol_2g_mb_9'] + df['vol_3g_mb_9']
df['churn'] = df['churn'].apply(lambda x: 0 if x > 0 else 1)
df['churn'].value_counts()
churns = df.loc[df['churn'] == 1]

nonchurns = df.loc[df['churn'] == 0]



import matplotlib.pyplot as plt

import seaborn as sns



def plotFig(feature, df, str):

    f, ax = plt.subplots(figsize=(10, 3))

    sns.distplot(df[feature + '_6'], hist=False, label=feature + '_6')

    sns.distplot(df[feature + '_7'], hist=False, label=feature + '_7')

    sns.distplot(df[feature + '_8'], hist=False, label=feature + '_8')

    sns.distplot(df[feature + '_9'], hist=False, label=feature + '_9')

    ax.set(xlabel=(feature + '_' + str))

    plt.legend();
# ic mou plots for churns

plotFig('loc_ic_mou', churns, 'churn')

plotFig('std_ic_mou', churns, 'churn')

plotFig('isd_ic_mou', churns, 'churn')

plotFig('total_ic_mou', churns, 'churn')



# og plots mou for churns

plotFig('loc_og_mou', churns, 'churn')

plotFig('std_og_mou', churns, 'churn')

plotFig('isd_og_mou', churns, 'churn')

plotFig('total_og_mou', churns, 'churn')
# ic plots for non-churns

plotFig('loc_ic_mou', nonchurns, 'non_churn')

plotFig('std_ic_mou', nonchurns, 'non_churn')

plotFig('isd_ic_mou', nonchurns, 'non_churn')

plotFig('total_ic_mou', nonchurns, 'non_churn')



# og plots for non-churns

plotFig('loc_og_mou', nonchurns, 'non_churn')

plotFig('std_og_mou', nonchurns, 'non_churn')

plotFig('isd_og_mou', nonchurns, 'non_churn')

plotFig('total_og_mou', nonchurns, 'non_churn')
plotFig('vol_2g_mb', churns, 'churn')

plotFig('vol_3g_mb', churns, 'churn')

plotFig('monthly_2g', churns, 'churn')

plotFig('monthly_3g', churns, 'churn')

plotFig('sachet_2g', churns, 'churn')

plotFig('sachet_3g', churns, 'churn')
plotFig('vol_2g_mb', nonchurns, 'non_churn')

plotFig('vol_3g_mb', nonchurns, 'non_churn')

plotFig('monthly_2g', nonchurns, 'non_churn')

plotFig('monthly_3g', nonchurns, 'non_churn')

plotFig('sachet_2g', nonchurns, 'non_churn')

plotFig('sachet_3g', nonchurns, 'non_churn')
plotFig('total_rech_num', churns, 'churn')

plotFig('total_rech_amt', churns, 'churn')

plotFig('max_rech_amt', churns, 'churn')
plotFig('total_rech_num', nonchurns, 'non_churn')

plotFig('total_rech_amt', nonchurns, 'non_churn')

plotFig('max_rech_amt', nonchurns, 'non_churn')
plotFig('arpu', churns, 'churn')

plotFig('arpu', nonchurns, 'non_churn')
def createDerivedFeature(df, feature):

    newFeature = 'delta_' + feature

    

    feature6 = feature + '_6'

    feature7 = feature + '_7'

    feature8 = feature + '_8'

    

    feature_avg = df[[feature6, feature7, feature8]].mean(axis=1)

    df[newFeature] = ((df[feature8] - ((df[feature6] + df[feature7]) / 2)) / feature_avg)

                   

    df.drop([feature6, feature7, feature8], axis=1, inplace=True)
# Incoming

createDerivedFeature(df, 'loc_ic_t2t_mou')

createDerivedFeature(df, 'loc_ic_t2m_mou')

createDerivedFeature(df, 'loc_ic_t2f_mou')

createDerivedFeature(df, 'loc_ic_mou')



createDerivedFeature(df, 'std_ic_t2t_mou')

createDerivedFeature(df, 'std_ic_t2m_mou')

createDerivedFeature(df, 'std_ic_t2f_mou')

createDerivedFeature(df, 'std_ic_mou')



createDerivedFeature(df, 'spl_ic_mou')

createDerivedFeature(df, 'isd_ic_mou')

createDerivedFeature(df, 'ic_others')

createDerivedFeature(df, 'roam_ic_mou')



createDerivedFeature(df, 'total_ic_mou')



# Outgoing

createDerivedFeature(df, 'loc_og_t2t_mou')

createDerivedFeature(df, 'loc_og_t2m_mou')

createDerivedFeature(df, 'loc_og_t2f_mou')

createDerivedFeature(df, 'loc_og_t2c_mou')

createDerivedFeature(df, 'loc_og_mou')



createDerivedFeature(df, 'std_og_t2t_mou')

createDerivedFeature(df, 'std_og_t2m_mou')

createDerivedFeature(df, 'std_og_t2f_mou')

createDerivedFeature(df, 'std_og_mou')



createDerivedFeature(df, 'spl_og_mou')

createDerivedFeature(df, 'isd_og_mou')

createDerivedFeature(df, 'og_others')

createDerivedFeature(df, 'roam_og_mou')



createDerivedFeature(df, 'total_og_mou')



createDerivedFeature(df, 'onnet_mou')

createDerivedFeature(df, 'offnet_mou')



# 2G and 3G

createDerivedFeature(df, 'vol_2g_mb')

createDerivedFeature(df, 'vol_3g_mb')

createDerivedFeature(df, 'monthly_2g')

createDerivedFeature(df, 'monthly_3g')

createDerivedFeature(df, 'sachet_2g')

createDerivedFeature(df, 'sachet_3g')



# Recharge and revenure

createDerivedFeature(df, 'total_rech_num')

createDerivedFeature(df, 'max_rech_amt')

createDerivedFeature(df, 'total_rech_amt')

createDerivedFeature(df, 'arpu')



df['vbc_3g_6'] = df['jun_vbc_3g']

df['vbc_3g_7'] = df['jul_vbc_3g']

df['vbc_3g_8'] = df['aug_vbc_3g']

createDerivedFeature(df, 'vbc_3g')

df.drop(['jun_vbc_3g', 'jul_vbc_3g', 'aug_vbc_3g'], axis=1, inplace=True)
df = df.fillna(0)
cols = df.columns

cols = [i for i in cols if '_9'  in i] 

cols.append('sep_vbc_3g')

df = df.drop(cols, axis=1)
df.churn.value_counts()
from imblearn.over_sampling import SMOTE

y = df.pop('churn')

X = df
# Keep the ratio of minority : majority = 0.8

smote = SMOTE(ratio=0.8, random_state=42)

X_blncd, y = smote.fit_sample(X, y)



# Check the new ratio. 

X = pd.DataFrame(X_blncd , columns=X.columns)

unique, counts = np.unique(y, return_counts=True)

print(np.asarray((unique, counts)).T)
from sklearn.feature_selection import RFE

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LogisticRegression
# Running RFE with the output number of the variable equal to 20

logreg = LogisticRegression()



from sklearn.feature_selection import RFE

rfe = RFE(logreg, 20)             

rfe = rfe.fit(X, y)
col = X.columns[rfe.support_]

col
plt.figure(figsize = (16,10)) 

sns.heatmap(X[col].corr(),annot = True)
rfe_df = X[col]
rfe_df.drop(['delta_loc_ic_t2m_mou',

             'delta_loc_og_t2m_mou',

             'delta_loc_og_t2t_mou',

             'delta_loc_ic_t2t_mou',

             'delta_roam_og_mou',

             'delta_offnet_mou',

             'delta_onnet_mou',

             'delta_arpu',

             'delta_max_rech_amt',

             'delta_total_rech_amt',

             'delta_total_rech_num'], axis=1, inplace=True)
plt.figure(figsize = (16,10)) 

sns.heatmap(rfe_df.corr(),annot = True)
rfe_df.columns
from sklearn import preprocessing

X_scaler = preprocessing.StandardScaler().fit(X)

X = X_scaler.transform(X) 
from sklearn.decomposition import PCA



pca = PCA(svd_solver='randomized', random_state=101)

pca.fit(X)



fig = plt.figure(figsize = (8,6))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
pca = PCA(n_components=20,random_state=100)



# fit_transform and transform to get the reduced data

X = pca.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=100)
cols=['a', 'b', 'c', 'd', 'e', 'f',

         'g', 'h', 'i', 'j', 'k', 'l',

         'm', 'n', 'o', 'p', 'q', 'r',

         's', 't']

X_train = pd.DataFrame(X_train, columns=cols)

X_test = pd.DataFrame(X_test, columns=cols)
X_train_sm = sm.add_constant(X_train)

logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

print(res.summary())
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

print(vif)
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)
y_train_pred_final = pd.DataFrame({'Churn':y_train, 'Churn_Prob':y_train_pred})

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )

print(confusion)
def calcSensi(confusion):

    TP = confusion[1,1] # true positive 

    TN = confusion[0,0] # true negatives

    FP = confusion[0,1] # false positives

    FN = confusion[1,0] # false negatives



    print(TP / float(TP+FN))

    

def calcAccuracy(confusion):

    TP = confusion[1,1] # true positive 

    TN = confusion[0,0] # true negatives

    FP = confusion[0,1] # false positives

    FN = confusion[1,0] # false negatives



    print((TP + TN) / float(TP + TN + FP + FN))
calcSensi(confusion)
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Churn, y_train_pred_final.Churn_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred = pd.DataFrame(y_test_pred)

y_test_pred.reset_index(drop=True, inplace=True)

y_test_pred = y_test_pred.rename(columns={ 0 : 'Churn_Prob'})

y_test_pred.head()
y_test = pd.DataFrame(y_test)

y_test.reset_index(drop=True, inplace=True)

y_test= y_test.rename(columns={ 0 : 'Churn'})

y_test.head()
y_pred_final = pd.concat([y_test, y_test_pred],axis=1)
y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.350 else 0)
metrics.accuracy_score(y_pred_final.Churn, y_pred_final.final_predicted)
confusion = metrics.confusion_matrix(y_pred_final.Churn, y_pred_final.final_predicted )

confusion
calcSensi(confusion)
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=10, gamma=0.1)

svm.fit(X_train, y_train)
y_test_pred = svm.predict(X_test)
confusion = metrics.confusion_matrix(y_test, y_test_pred )

confusion
calcAccuracy(confusion)
calcSensi(confusion)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(1, 20)}

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)



tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")



tree.fit(X_train, y_train)
scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_max_depth"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

dtree = DecisionTreeClassifier(criterion = "gini", max_depth=3)

dtree.fit(X_train, y_train)

print((dtree.score(X, y))*100)
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(5, 200, 20)}



# instantiate the model

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf

plt.figure()

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_leaf")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# tree with max_depth = 3

tree = DecisionTreeClassifier(criterion = "gini", 

                                  random_state = 100,

                                  max_depth=10, 

                                  min_samples_leaf=50,

                                  min_samples_split=50)

tree.fit(X_train, y_train)
from sklearn.metrics import classification_report,confusion_matrix

y_pred = tree.predict(X_test)

print(classification_report(y_test, y_pred))
# confusion matrix

print(confusion_matrix(y_test,y_pred))
calcSensi(confusion_matrix(y_test,y_pred))
calcAccuracy(confusion_matrix(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier



# number of CV folds

n_folds = 5



# parameters to build the model on

# max_depth - [0-100]

# n_extimators - 20

parameters = {'max_depth': range(1, 15)}

dtree = RandomForestClassifier(criterion = "gini",

                               n_estimators= 20,

                               random_state = 100)



tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_max_depth"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

from sklearn.ensemble import RandomForestClassifier



# number of CV folds

n_folds = 5



# parameters to build the model on

# min_sample_leaf - [5-200]

# n_extimators - 20

parameters = {'min_samples_leaf': range(5, 200, 20)}



dtree = RandomForestClassifier(criterion = "gini",

                               n_estimators= 20,

                               random_state = 100)



tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf

plt.figure()

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_leaf")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
dtree = RandomForestClassifier(criterion = "gini",

                               n_estimators= 200,

                               max_depth=12,

                               min_samples_leaf=50,

                               min_samples_split=50)

dtree.fit(X_train, y_train)
from sklearn.metrics import classification_report,confusion_matrix

y_pred = dtree.predict(X_test)

print(classification_report(y_test, y_pred))
# confusion matrix

print(confusion_matrix(y_test,y_pred))
calcSensi(confusion_matrix(y_test,y_pred))
calcAccuracy(confusion_matrix(y_test,y_pred))