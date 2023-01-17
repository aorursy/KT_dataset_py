import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import time
df=pd.read_excel("../input/default of credit card clientsv1.xlsx")
pd.set_option('display.max_columns',None)

df.head()
df['OUTPUT_LABEL'] = (df.default_payment_next_month == 1).astype('int')
#Feature Engineering of Age Range

age_id={'(20-30)':1,

        '(30-40)':2,

        '(40-50)':3,

        '(50-60)':4,

        '(60-70)':5,

        '(70-80)':6}

df['age_group']=df.AGE_RANGE.replace(age_id)
#Feature Engineering of Limit Range

limit_id={'(0-100000)':1,

          '(100000-200000)':2,

          '(200000-300000)':3,

          '(300000-400000)':4,

          '(400000-500000)':5,

          '(500000-600000)':6,

          '(600000-700000)':7,

          '(700000-800000)':8,

          '(800000-900000)':9,

          '(900000-1000000)':10}

df['limit_group']=df.LIMIT_RANGE.replace(limit_id)
cols_extra = ['age_group','limit_group']

cols_num = ['LIMIT_BAL','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',

            'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2',

            'PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
#Splitting of Dataset to Training, Validation and Test set

df_data = df[cols_num + cols_extra + ['OUTPUT_LABEL']]

df_data = df_data.sample(n = len(df_data), random_state = 42)

df_data = df_data.reset_index(drop = True)

df_valid_test=df_data.sample(frac=0.30,random_state=42)

df_test = df_valid_test.sample(frac = 0.5, random_state = 42)

df_valid = df_valid_test.drop(df_test.index)

df_train_all=df_data.drop(df_valid_test.index)
def calc_prevalence(y_actual):

    return (sum(y_actual)/len(y_actual))

print('Test prevalence(n = %d):%.3f'%(len(df_test),calc_prevalence(df_test.OUTPUT_LABEL.values)))

print('Valid prevalence(n = %d):%.3f'%(len(df_valid),calc_prevalence(df_valid.OUTPUT_LABEL.values)))

print('Train all prevalence(n = %d):%.3f'%(len(df_train_all), calc_prevalence(df_train_all.OUTPUT_LABEL.values)))

print('all samples (n = %d)'%len(df_data))

assert len(df_data) == (len(df_test)+len(df_valid)+len(df_train_all)),'math didnt work'
df_train_all.to_csv('df_train_all.csv',index=False)

df_valid.to_csv('df_valid.csv',index=False)

df_test.to_csv('df_test.csv',index=False)
df_train_all = pd.read_csv('df_train_all.csv')

df_valid= pd.read_csv('df_valid.csv')
print('Valid prevalence(n = %d):%.3f'%(len(df_valid),calc_prevalence(df_valid.OUTPUT_LABEL.values)))

print('Train all prevalence(n = %d):%.3f'%(len(df_train_all), calc_prevalence(df_train_all.OUTPUT_LABEL.values)))
rows_pos = df_train_all.OUTPUT_LABEL == 1

df_train_pos = df_train_all.loc[rows_pos]

df_train_neg = df_train_all.loc[~rows_pos]

df_train = pd.concat([df_train_pos, df_train_neg.sample(n = len(df_train_pos), random_state = 42)],axis = 0)

df_train = df_train.sample(n = len(df_train), random_state = 42).reset_index(drop = True)

print('Train balanced prevalence(n = %d):%.3f'%(len(df_train), calc_prevalence(df_train.OUTPUT_LABEL.values)))
col2use = [c for c in list(df_train_all.columns) if c != 'OUTPUT_LABEL']

print('Number of columns to use:', len(col2use))
X_train = df_train[col2use].values

X_train_all = df_train_all[col2use].values

X_valid = df_valid[col2use].values



y_train = df_train['OUTPUT_LABEL'].values

y_valid = df_valid['OUTPUT_LABEL'].values



print('Training All shapes:',X_train_all.shape)

print('Training shapes:',X_train.shape, y_train.shape)

print('Validation shapes:',X_valid.shape, y_valid.shape)
#Feature Scaling

from sklearn.preprocessing import StandardScaler



scaler  = StandardScaler()

scaler.fit(X_train_all)
import pickle

scalerfile = 'scaler.sav'

pickle.dump(scaler, open(scalerfile, 'wb'))
scaler = pickle.load(open(scalerfile, 'rb'))
X_train_tf = scaler.transform(X_train)

X_valid_tf = scaler.transform(X_valid)
#K Nearest Neighbors First Trial

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors = 100)

knn.fit(X_train_tf, y_train)
#Time Taken

t1 = time.time()

y_train_preds = knn.predict_proba(X_train_tf)[:,1]

y_valid_preds = knn.predict_proba(X_valid_tf)[:,1]

t2 = time.time()

print(t2-t1)
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score



def calc_specificity(y_actual, y_pred, thresh):

    return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)



def print_report(y_actual, y_pred, thresh):

    

    auc = roc_auc_score(y_actual, y_pred)

    accuracy = accuracy_score(y_actual, (y_pred > thresh))

    recall = recall_score(y_actual, (y_pred > thresh))

    precision = precision_score(y_actual, (y_pred > thresh))

    specificity = calc_specificity(y_actual, y_pred, thresh)

    print('AUC:%.3f'%auc)

    print('accuracy:%.3f'%accuracy)

    print('recall:%.3f'%recall)

    print('precision:%.3f'%precision)

    print('specificity:%.3f'%specificity)

    print('prevalence:%.3f'%calc_prevalence(y_actual))

    print(' ')

    return auc, accuracy, recall, precision, specificity 
thresh = 0.5

print('Training:')

print_report(y_train,y_train_preds, thresh)

print('Validation:')

print_report(y_valid,y_valid_preds, thresh);
from sklearn.metrics import roc_curve 



fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_preds)

auc_train = roc_auc_score(y_train, y_train_preds)





fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_preds)

auc_valid = roc_auc_score(y_valid, y_valid_preds)



plt.plot(fpr_train, tpr_train, 'r-',label ='Train AUC:%.3f'%auc_train)

plt.plot(fpr_valid, tpr_valid, 'b-',label ='Valid AUC:%.3f'%auc_valid)

plt.plot([0,1],[0,1],'k--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()
# 7 Machine Learning Algorithms

# k-nearest neighbors

knn=KNeighborsClassifier(n_neighbors = 100)

knn.fit(X_train_tf, y_train)
y_train_preds = knn.predict_proba(X_train_tf)[:,1]

y_valid_preds = knn.predict_proba(X_valid_tf)[:,1]



print('KNN')

print('Training:')

knn_train_auc, knn_train_accuracy, knn_train_recall, knn_train_precision, knn_train_specificity = print_report(y_train,y_train_preds, thresh)

print('Validation:')

knn_valid_auc, knn_valid_accuracy, knn_valid_recall, knn_valid_precision, knn_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
# logistic regression

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(random_state = 42)

lr.fit(X_train_tf, y_train)
y_train_preds = lr.predict_proba(X_train_tf)[:,1]

y_valid_preds = lr.predict_proba(X_valid_tf)[:,1]



print('Logistic Regression')

print('Training:')

lr_train_auc, lr_train_accuracy, lr_train_recall, lr_train_precision, lr_train_specificity = print_report(y_train,y_train_preds, thresh)

print('Validation:')

lr_valid_auc, lr_valid_accuracy, lr_valid_recall, lr_valid_precision, lr_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
#Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier

sgdc=SGDClassifier(loss = 'log',alpha = 0.1,random_state = 42)

sgdc.fit(X_train_tf, y_train)
y_train_preds = sgdc.predict_proba(X_train_tf)[:,1]

y_valid_preds = sgdc.predict_proba(X_valid_tf)[:,1]



print('Stochastic Gradient Descend')

print('Training:')

sgdc_train_auc, sgdc_train_accuracy, sgdc_train_recall, sgdc_train_precision, sgdc_train_specificity =print_report(y_train,y_train_preds, thresh)

print('Validation:')

sgdc_valid_auc, sgdc_valid_accuracy, sgdc_valid_recall, sgdc_valid_precision, sgdc_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
# Naive Bayes

from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(X_train_tf, y_train)
y_train_preds = nb.predict_proba(X_train_tf)[:,1]

y_valid_preds = nb.predict_proba(X_valid_tf)[:,1]



print('Naive Bayes')

print('Training:')

nb_train_auc, nb_train_accuracy, nb_train_recall, nb_train_precision, nb_train_specificity =print_report(y_train,y_train_preds, thresh)

print('Validation:')

nb_valid_auc, nb_valid_accuracy, nb_valid_recall, nb_valid_precision, nb_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(max_depth = 10, random_state = 42)

tree.fit(X_train_tf, y_train)
y_train_preds = tree.predict_proba(X_train_tf)[:,1]

y_valid_preds = tree.predict_proba(X_valid_tf)[:,1]



print('Decision Tree')

print('Training:')

tree_train_auc, tree_train_accuracy, tree_train_recall, tree_train_precision, tree_train_specificity =print_report(y_train,y_train_preds, thresh)

print('Validation:')

tree_valid_auc, tree_valid_accuracy, tree_valid_recall, tree_valid_precision, tree_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
# Randomforest

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(max_depth = 6, random_state = 42)

rf.fit(X_train_tf, y_train)
y_train_preds = rf.predict_proba(X_train_tf)[:,1]

y_valid_preds = rf.predict_proba(X_valid_tf)[:,1]



print('Random Forest')

print('Training:')

rf_train_auc, rf_train_accuracy, rf_train_recall, rf_train_precision, rf_train_specificity =print_report(y_train,y_train_preds, thresh)

print('Validation:')

rf_valid_auc, rf_valid_accuracy, rf_valid_recall, rf_valid_precision, rf_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
# Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

gbc =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,

     max_depth=3, random_state=42)

gbc.fit(X_train_tf, y_train)
y_train_preds = gbc.predict_proba(X_train_tf)[:,1]

y_valid_preds = gbc.predict_proba(X_valid_tf)[:,1]



print('Gradient Boosting Classifier')

print('Training:')

gbc_train_auc, gbc_train_accuracy, gbc_train_recall, gbc_train_precision, gbc_train_specificity = print_report(y_train,y_train_preds, thresh)

print('Validation:')

gbc_valid_auc, gbc_valid_accuracy, gbc_valid_recall, gbc_valid_precision, gbc_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
# Machine Learning Algo Analysis

df_results = pd.DataFrame({'classifier':['KNN','KNN','LR','LR','SGD','SGD','NB','NB','DT','DT','RF','RF','GB','GB'],

                           'data_set':['train','valid']*7,

                          'auc':[knn_train_auc, knn_valid_auc,lr_train_auc,lr_valid_auc,sgdc_train_auc,sgdc_valid_auc,nb_train_auc,nb_valid_auc,tree_train_auc,tree_valid_auc,rf_train_auc,rf_valid_auc,gbc_train_auc,gbc_valid_auc,],

                          'accuracy':[knn_train_accuracy, knn_valid_accuracy,lr_train_accuracy,lr_valid_accuracy,sgdc_train_accuracy,sgdc_valid_accuracy,nb_train_accuracy,nb_valid_accuracy,tree_train_accuracy,tree_valid_accuracy,rf_train_accuracy,rf_valid_accuracy,gbc_train_accuracy,gbc_valid_accuracy,],

                          'recall':[knn_train_recall, knn_valid_recall,lr_train_recall,lr_valid_recall,sgdc_train_recall,sgdc_valid_recall,nb_train_recall,nb_valid_recall,tree_train_recall,tree_valid_recall,rf_train_recall,rf_valid_recall,gbc_train_recall,gbc_valid_recall,],

                          'precision':[knn_train_precision, knn_valid_precision,lr_train_precision,lr_valid_precision,sgdc_train_precision,sgdc_valid_precision,nb_train_precision,nb_valid_precision,tree_train_precision,tree_valid_precision,rf_train_precision,rf_valid_precision,gbc_train_auc,gbc_valid_precision,],

                          'specificity':[knn_train_specificity, knn_valid_specificity,lr_train_specificity,lr_valid_specificity,sgdc_train_specificity,sgdc_valid_specificity,nb_train_specificity,nb_valid_specificity,tree_train_specificity,tree_valid_specificity,rf_train_specificity,rf_valid_specificity,gbc_train_specificity,gbc_valid_specificity,]})
#Plotting Results

import seaborn as sns

sns.set(style="darkgrid")

ax = sns.barplot(x="classifier", y="auc", hue="data_set", data=df_results)

ax.set_xlabel('Classifier',fontsize = 15)

ax.set_ylabel('AUC', fontsize = 15)

ax.tick_params(labelsize=15)



plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 15)

plt.show()
ax = sns.barplot(x="classifier", y="recall", hue="data_set", data=df_results)

ax.set_xlabel('Classifier',fontsize = 15)

ax.set_ylabel('recall', fontsize = 15)

ax.tick_params(labelsize=15)



plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 15)



plt.show()
ax = sns.barplot(x="classifier", y="specificity", hue="data_set", data=df_results)

ax.set_xlabel('Classifier',fontsize = 15)

ax.set_ylabel('specificity', fontsize = 15)

ax.tick_params(labelsize=15)



plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 15)



plt.show()
ax = sns.barplot(x="classifier", y="precision", hue="data_set", data=df_results)

ax.set_xlabel('Classifier',fontsize = 15)

ax.set_ylabel('precision', fontsize = 15)

ax.tick_params(labelsize=15)



plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 15)



plt.show()
ax = sns.barplot(x="classifier", y="accuracy", hue="data_set", data=df_results)

ax.set_xlabel('Classifier',fontsize = 15)

ax.set_ylabel('accuracy', fontsize = 15)

ax.tick_params(labelsize=15)



plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 15)



plt.show()
#Learning Curves

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("AUC")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring = 'roc_auc')

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="b")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
title = "Learning Curves (Random Forest)"

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = RandomForestClassifier(max_depth = 20, random_state = 42)

plot_learning_curve(estimator, title, X_train_tf, y_train, ylim=(0.2, 1.01), cv=cv, n_jobs=4)



plt.show()
title = "Learning Curves (Random Forest)"

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = RandomForestClassifier(max_depth = 3, random_state = 42)

plot_learning_curve(estimator, title, X_train_tf, y_train, ylim=(0.2, 1.01), cv=cv, n_jobs=4)



plt.show()
title = "Learning Curves (Gradient Boosting)"

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,

     max_depth=20, random_state=42)

plot_learning_curve(estimator, title, X_train_tf, y_train, ylim=(0.2, 1.01), cv=cv, n_jobs=4)



plt.show()
title = "Learning Curves (Gradient Boosting)"

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,

     max_depth=3, random_state=42)

plot_learning_curve(estimator, title, X_train_tf, y_train, ylim=(0.2, 1.01), cv=cv, n_jobs=4)



plt.show()
#Logistic Regression Feature Importance

lr=LogisticRegression(random_state = 42)

lr.fit(X_train_tf, y_train)
feature_importances = pd.DataFrame(lr.coef_[0],

                                   index = col2use,

                                    columns=['importance']).sort_values('importance',

                                                                        ascending=False)
num = 5

ylocs = np.arange(num)

values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]

feature_labels = list(feature_importances.iloc[:num].index)[::-1]



plt.figure(num=None, figsize=(5, 7), dpi=80, facecolor='w', edgecolor='k');

plt.barh(ylocs, values_to_plot, align = 'center')

plt.ylabel('Features')

plt.xlabel('Importance Score')

plt.title('Positive Feature Importance Score - Logistic Regression')

plt.yticks(ylocs, feature_labels)

plt.show()
values_to_plot = feature_importances.iloc[-num:].values.ravel()

feature_labels = list(feature_importances.iloc[-num:].index)



plt.figure(num=None, figsize=(5, 7), dpi=80, facecolor='w', edgecolor='k');

plt.barh(ylocs, values_to_plot, align = 'center')

plt.ylabel('Features')

plt.xlabel('Importance Score')

plt.title('Negative Feature Importance Score - Logistic Regression')

plt.yticks(ylocs, feature_labels)

plt.show()
#RandomForest Feature Importance

rf=RandomForestClassifier(max_depth = 6, random_state = 42)

rf.fit(X_train_tf, y_train)
feature_importances = pd.DataFrame(rf.feature_importances_,

                                   index = col2use,

                                    columns=['importance']).sort_values('importance',

                                                                        ascending=False)
feature_importances.head()
num = 5

ylocs = np.arange(num)

values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]

feature_labels = list(feature_importances.iloc[:num].index)[::-1]



plt.figure(num=None, figsize=(5, 7), dpi=80, facecolor='w', edgecolor='k');

plt.barh(ylocs, values_to_plot, align = 'center')

plt.ylabel('Features')

plt.xlabel('Importance Score')

plt.title('Feature Importance Score - Random Forest')

plt.yticks(ylocs, feature_labels)

plt.show()
max_depths = np.arange(2,20,2)



train_aucs = np.zeros(len(max_depths))

valid_aucs = np.zeros(len(max_depths))



for jj in range(len(max_depths)):

    max_depth = max_depths[jj]



    # fit model

    rf=RandomForestClassifier(n_estimators = 100, max_depth = max_depth, random_state = 42)

    rf.fit(X_train_tf, y_train)        

    # get predictions

    y_train_preds = rf.predict_proba(X_train_tf)[:,1]

    y_valid_preds = rf.predict_proba(X_valid_tf)[:,1]



    # calculate auc

    auc_train = roc_auc_score(y_train, y_train_preds)

    auc_valid = roc_auc_score(y_valid, y_valid_preds)



    # save aucs

    train_aucs[jj] = auc_train

    valid_aucs[jj] = auc_valid
#Plotting Max Depth Tuning Curve



plt.plot(max_depths, train_aucs,'o-',label = 'train')

plt.plot(max_depths, valid_aucs,'o-',label = 'valid')



plt.xlabel('max_depth')

plt.ylabel('AUC')

plt.legend()

plt.show()
rf.get_params()
#Gradient Boosting Feature Importance

gbc =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,

     max_depth=3, random_state=42)

gbc.fit(X_train_tf, y_train)
feature_importances = pd.DataFrame(gbc.feature_importances_,

                                   index = col2use,

                                    columns=['importance']).sort_values('importance',

                                                                        ascending=False)
feature_importances.head()
num = 5

ylocs = np.arange(num)

values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]

feature_labels = list(feature_importances.iloc[:num].index)[::-1]



plt.figure(num=None, figsize=(5, 7), dpi=80, facecolor='w', edgecolor='k');

plt.barh(ylocs, values_to_plot, align = 'center')

plt.ylabel('Features')

plt.xlabel('Importance Score')

plt.title('Feature Importance Score - Gradient Boosting')

plt.yticks(ylocs, feature_labels)

plt.show()
max_depths = np.arange(1,20,1)



train_aucs = np.zeros(len(max_depths))

valid_aucs = np.zeros(len(max_depths))



for jj in range(len(max_depths)):

    max_depth = max_depths[jj]



    # fit model

    gbc =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=max_depth, random_state=42)

    gbc.fit(X_train_tf, y_train)      

    # get predictions

    y_train_preds = gbc.predict_proba(X_train_tf)[:,1]

    y_valid_preds = gbc.predict_proba(X_valid_tf)[:,1]



    # calculate auc

    auc_train = roc_auc_score(y_train, y_train_preds)

    auc_valid = roc_auc_score(y_valid, y_valid_preds)



    # save aucs

    train_aucs[jj] = auc_train

    valid_aucs[jj] = auc_valid
plt.plot(max_depths, train_aucs,'o-',label = 'train')

plt.plot(max_depths, valid_aucs,'o-',label = 'valid')



plt.xlabel('max_depth')

plt.ylabel('AUC')

plt.legend()

plt.show()
#RandomForest Optimization

from sklearn.model_selection import RandomizedSearchCV



# number of trees

n_estimators = range(200,1000,200)

# maximum number of features to use at each split

max_features = ['auto','sqrt']

# maximum depth of the tree

max_depth = range(2,20,2)

# minimum number of samples to split a node

min_samples_split = range(2,10,2)

# criterion for evaluating a split

criterion = ['gini','entropy']



# random grid



random_grid = {'n_estimators':n_estimators,

              'max_features':max_features,

              'max_depth':max_depth,

              'min_samples_split':min_samples_split,

              'criterion':criterion}



print(random_grid)
from sklearn.metrics import make_scorer, roc_auc_score

auc_scoring = make_scorer(roc_auc_score)
# create a baseline model

rf = RandomForestClassifier()



# create the randomized search cross-validation

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,

                               n_iter = 20, cv = 10, scoring=auc_scoring,verbose = 1, random_state = 42)
t1 = time.time()

rf_random.fit(X_train_tf, y_train)

t2 = time.time()

print(t2-t1)
rf_random.best_params_
rf=RandomForestClassifier(max_depth = 6, random_state = 42)

rf.fit(X_train_tf, y_train)



y_train_preds = rf.predict_proba(X_train_tf)[:,1]

y_valid_preds = rf.predict_proba(X_valid_tf)[:,1]



thresh = 0.5



print('Baseline Random Forest')

print('Training AUC:%.3f'%(roc_auc_score(y_train, y_train_preds)))

print('Validation AUC:%.3f'%(roc_auc_score(y_valid, y_valid_preds)))



print('Optimized Random Forest')

y_train_preds_random = rf_random.best_estimator_.predict_proba(X_train_tf)[:,1]

y_valid_preds_random = rf_random.best_estimator_.predict_proba(X_valid_tf)[:,1]



rf_train_auc = roc_auc_score(y_train, y_train_preds_random)

rf_valid_auc = roc_auc_score(y_valid, y_valid_preds_random)



print('Training AUC:%.3f'%(rf_train_auc))

print('Validation AUC:%.3f'%(rf_valid_auc))
#Stochastic Gradient Descent Optimization

sgdc=SGDClassifier(loss = 'log',alpha = 0.1,random_state = 42)

sgdc.fit(X_train_tf, y_train)
penalty = ['none','l2','l1']

max_iter = range(200,1000,200)

alpha = [0.001,0.003,0.01,0.03,0.1,0.3]

random_grid_sgdc = {'penalty':penalty,

              'max_iter':max_iter,

              'alpha':alpha}

# create the randomized search cross-validation

sgdc_random = RandomizedSearchCV(estimator = sgdc, param_distributions = random_grid_sgdc,

                                 n_iter = 20, cv = 2, scoring=auc_scoring,verbose = 0, random_state = 42)



t1 = time.time()

sgdc_random.fit(X_train_tf, y_train)

t2 = time.time()

print(t2-t1)
sgdc_random.best_params_
y_train_preds = sgdc.predict_proba(X_train_tf)[:,1]

y_valid_preds = sgdc.predict_proba(X_valid_tf)[:,1]



thresh = 0.5



print('Baseline sgdc')

print('Training AUC:%.3f'%(roc_auc_score(y_train, y_train_preds)))

print('Validation AUC:%.3f'%(roc_auc_score(y_valid, y_valid_preds)))



print('Optimized sgdc')

y_train_preds_random = sgdc_random.best_estimator_.predict_proba(X_train_tf)[:,1]

y_valid_preds_random = sgdc_random.best_estimator_.predict_proba(X_valid_tf)[:,1]

sgdc_train_auc = roc_auc_score(y_train, y_train_preds_random)

sgdc_valid_auc = roc_auc_score(y_valid, y_valid_preds_random)



print('Training AUC:%.3f'%(sgdc_train_auc))

print('Validation AUC:%.3f'%(sgdc_valid_auc))
#Gradient Boosting Optimization

gbc =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,

     max_depth=3, random_state=42)

gbc.fit(X_train_tf, y_train)
# number of stages

n_estimators = range(100,500,100)



# maximum depth of the stage

max_depth = range(1,10,1)



# learning rate

learning_rate = [0.001,0.003,0.01,0.03,0.1,0.3]



# random grid



random_grid_gbc = {'n_estimators':n_estimators,

              'max_depth':max_depth,

              'learning_rate':learning_rate}



# create the randomized search cross-validation

gbc_random = RandomizedSearchCV(estimator = gbc, param_distributions = random_grid_gbc, 

                                n_iter = 20, cv = 10, scoring=auc_scoring,verbose = 0, random_state = 42)



t1 = time.time()

gbc_random.fit(X_train_tf, y_train)

t2 = time.time()

print(t2-t1)
gbc_random.best_params_
y_train_preds = gbc.predict_proba(X_train_tf)[:,1]

y_valid_preds = gbc.predict_proba(X_valid_tf)[:,1]



thresh = 0.5



print('Baseline gbc')

print('Training AUC:%.3f'%(roc_auc_score(y_train, y_train_preds)))

print('Validation AUC:%.3f'%(roc_auc_score(y_valid, y_valid_preds)))



print('Optimized gbc')

y_train_preds_random = gbc_random.best_estimator_.predict_proba(X_train_tf)[:,1]

y_valid_preds_random = gbc_random.best_estimator_.predict_proba(X_valid_tf)[:,1]

gbc_train_auc = roc_auc_score(y_train, y_train_preds_random)

gbc_valid_auc = roc_auc_score(y_valid, y_valid_preds_random)



print('Training AUC:%.3f'%(gbc_train_auc))

print('Validation AUC:%.3f'%(gbc_valid_auc))
# K Nearest Neighbors Optimization

knn = KNeighborsClassifier()



n_neighbors=range(20,500,20)

leaf_size=range(10,100,5)



random_grid_knn = {'n_neighbors':n_neighbors,

                   'leaf_size':leaf_size

              }



# create the randomized search cross-validation

knn_random = RandomizedSearchCV(estimator = knn, param_distributions = random_grid_knn, 

                                n_iter = 20, cv = 5, scoring=auc_scoring,verbose = 1, random_state = 42)
t1 = time.time()

knn_random.fit(X_train_tf, y_train)

t2 = time.time()

print(t2-t1)
knn_random.best_params_
knn=KNeighborsClassifier(n_neighbors = 100)

knn.fit(X_train_tf, y_train)



y_train_preds = rf.predict_proba(X_train_tf)[:,1]

y_valid_preds = rf.predict_proba(X_valid_tf)[:,1]



thresh = 0.5



print('Baseline KNN')

print('Training AUC:%.3f'%(roc_auc_score(y_train, y_train_preds)))

print('Validation AUC:%.3f'%(roc_auc_score(y_valid, y_valid_preds)))



print('Optimized KNN')

y_train_preds_random = knn_random.best_estimator_.predict_proba(X_train_tf)[:,1]

y_valid_preds_random = knn_random.best_estimator_.predict_proba(X_valid_tf)[:,1]



knn_train_auc = roc_auc_score(y_train, y_train_preds_random)

knn_valid_auc = roc_auc_score(y_valid, y_valid_preds_random)



print('Training AUC:%.3f'%(knn_train_auc))

print('Validation AUC:%.3f'%(knn_valid_auc))
df_results = pd.DataFrame({'classifier':['SGD','SGD','RF','RF','GB','GB','KNN','KNN'],

                           'data_set':['train','valid']*4,

                          'auc':[sgdc_train_auc,sgdc_valid_auc,rf_train_auc,rf_valid_auc,gbc_train_auc,gbc_valid_auc,knn_train_auc,knn_valid_auc],

                          })
df_results
ax = sns.barplot(x="classifier", y="auc", hue="data_set", data=df_results)

ax.set_xlabel('Classifier',fontsize = 15)

ax.set_ylabel('AUC', fontsize = 15)

ax.tick_params(labelsize=15)

# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize = 15)



plt.show()
pickle.dump(gbc_random.best_estimator_, open('best_classifier.pkl', 'wb'),protocol = 4)
# Results of Selected Model

y_train_preds_random = gbc_random.best_estimator_.predict_proba(X_train_tf)[:,1]

y_valid_preds_random = gbc_random.best_estimator_.predict_proba(X_valid_tf)[:,1]

gbc1 =GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,

     max_depth=3, random_state=42)

gbc1.fit(X_train_tf, y_train)

print('Gradient Boosting Classifier Optimized')

print('Training:')

gbc1_train_auc, gbc1_train_accuracy, gbc1_train_recall, gbc1_train_precision, gbc1_train_specificity = print_report(y_train,y_train_preds_random, thresh)

print('Validation:')

gbc1_valid_auc, gbc1_valid_accuracy, gbc1_valid_recall, gbc1_valid_precision, gbc1_valid_specificity = print_report(y_valid,y_valid_preds_random, thresh)
best_model = pickle.load(open('best_classifier.pkl','rb'))

df_test= pd.read_csv('df_test.csv')
X_train = df_train[col2use].values

X_valid = df_valid[col2use].values

X_test = df_test[col2use].values



y_train = df_train['OUTPUT_LABEL'].values

y_valid = df_valid['OUTPUT_LABEL'].values

y_test = df_test['OUTPUT_LABEL'].values



scaler = pickle.load(open('scaler.sav', 'rb'))



X_train_tf = scaler.transform(X_train)

X_valid_tf = scaler.transform(X_valid)

X_test_tf = scaler.transform(X_test)
y_train_preds = best_model.predict_proba(X_train_tf)[:,1]

y_valid_preds = best_model.predict_proba(X_valid_tf)[:,1]

y_test_preds = best_model.predict_proba(X_test_tf)[:,1]



thresh = 0.5



print('Training:')

train_auc, train_accuracy, train_recall, train_precision, train_specificity = print_report(y_train,y_train_preds, thresh)

print('Validation:')

valid_auc, valid_accuracy, valid_recall, valid_precision, valid_specificity = print_report(y_valid,y_valid_preds, thresh)

print('Test:')

test_auc, test_accuracy, test_recall, test_precision, test_specificity = print_report(y_test,y_test_preds, thresh)
from sklearn.metrics import roc_curve 



fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_preds)

auc_train = roc_auc_score(y_train, y_train_preds)



fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_preds)

auc_valid = roc_auc_score(y_valid, y_valid_preds)



fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_preds)

auc_test = roc_auc_score(y_test, y_test_preds)



plt.plot(fpr_train, tpr_train, 'r-',label ='Train AUC:%.3f'%auc_train)

plt.plot(fpr_valid, tpr_valid, 'b-',label ='Valid AUC:%.3f'%auc_valid)

plt.plot(fpr_test, tpr_test, 'g-',label ='Test AUC:%.3f'%auc_test)

plt.plot([0,1],[0,1],'k--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()