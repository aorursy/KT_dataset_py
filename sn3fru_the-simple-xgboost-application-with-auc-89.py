import pandas as pd

import numpy as np

import sys

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, average_precision_score

import xgboost as xgb

from xgboost import plot_importance, to_graphviz

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

import statsmodels.formula.api as smf

import graphviz



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline
gs = pd.read_csv('../input/gender_submission.csv')

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
def proportion(df,column,plot=False):

    t=df[column].count()

    c=df[column].sum()

    print(f'Total rows: {t}, with flag: {c}, proportion: {c/t}')

    if plot:

        sns.countplot(x=column, data=df)



proportion(df_train,'Survived')        
variables_to_drop = ['Name','Ticket','Cabin']

for variable in variables_to_drop:

    del df_train[variable]

    del df_test[variable]
def plot_roc(fpr, tpr, roc_auc):

    plt.plot(fpr, tpr)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC curve (area = %0.6f)' % roc_auc)

    plt.legend(loc="lower right")

    plt.show()





def plot_pr(recall,precision,average_precision):

    plt.step(recall, precision, color='b', alpha=0.2, where='post')

    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.ylim([0.0, 1.05])

    plt.xlim([0.0, 1.0])

    plt.title('Precision-Recall: {0:0.6f}'.format(average_precision))

    plt.show()



    

def plot_learning_curve(results,epochs):

    x_axis = range(0, epochs)

    fig, ax = plt.subplots()

    ax.plot(x_axis, results['validation_0']['logloss'], label='logloss-Train')

    ax.plot(x_axis, results['validation_1']['logloss'], label='logloss-Test')

    ax.plot(x_axis, results['validation_0']['auc'], label='auc-Train')

    ax.plot(x_axis, results['validation_1']['auc'], label='auc-Test')

    ax.legend()

    plt.ylabel('Log Loss/AUC')

    plt.title('XGBoost Log Loss and AUC evolution')

    plt.show()





def run_xgboost_and_calculate_auc(df,

                                  target='target',

                                  drop='variable_to_find_and_exclude',

                                  w=1,

                                  plot_learning=False,

                                  plot_variables=False,

                                  plot_ROC_PR=False,

                                  plot_confusion=False,

                                  plot_graph_tree=False,

                                  learning_rate=0.05,

                                  max_depth=10,

                                  esr=10,

                                  CV=True,

                                  title='',

                                  plot_all=False,

                                  ensembler=False,

                                  test_size=.2):

    '''

    Generic function to run xgboost to test the added changes and plot roc, learning and others and save the model.

    '''

    # features

    X = df.drop(target, axis=1)

    X = pd.get_dummies(X)

    

    # targets

    Y = df[target]



    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,

                                                        test_size=test_size,

                                                        random_state=12345,

                                                        stratify=Y)



    print(f'variables: {len(list(X))}, rows:{len(X)}, flags:{sum(list(df[target]))}')

    

        

    clf = xgb.XGBClassifier(

        learning_rate=learning_rate,

        n_estimators=1000,

        max_depth=4,

        min_child_weight=4,

        gamma=0.6,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=5e-05,

        objective='binary:logistic',

        nthread=20,

        scale_pos_weight=w,

        seed=27)



    eval_set = [(X_train, Y_train), (X_test, Y_test)]



    if plot_all:

        plot_learning=True

        plot_variables=True

        plot_ROC_PR=True

        plot_confusion=True

        plot_graph_tree=True

        

    if CV:

        X = pd.concat([X_train,X_test])

        y = pd.concat([Y_train,Y_test])

        xgb_param = clf.get_xgb_params()

        xgtrain = xgb.DMatrix(X.values, y.values)

        cvresult = xgb.cv(xgb_param,

                          xgtrain,

                          num_boost_round=clf.get_params()['n_estimators'],

                          nfold=5,

                          metrics='auc',

                          early_stopping_rounds=esr)

        print(cvresult.tail(1))

        clf.set_params(n_estimators=cvresult.shape[0])



    clf.fit(X_train,

            Y_train,

            early_stopping_rounds=25,

            eval_metric=['auc','error','logloss'],

            eval_set=eval_set,

            verbose=False)



    Y_pred = clf.predict_proba(X_test)

    y_true = np.array(Y_test)

    y_scores = Y_pred[:, 1]



    fpr, tpr, _ = roc_curve(Y_test, y_scores)

    roc_auc = auc(fpr, tpr)

    average_precision = average_precision_score(Y_test, y_scores)

    precision, recall, _ = precision_recall_curve(Y_test, y_scores)

    

    if plot_ROC_PR:

        plot_roc(fpr, tpr, roc_auc)

        plot_pr(recall,precision,average_precision)

    else:

        print('Area under ROC: %0.6f' % roc_auc)



    if plot_graph_tree:

        xgb.plot_tree(clf, rankdir='LR')

        fig = plt.gcf()

        fig.set_size_inches(150, 100)



    if plot_learning:

        results = clf.evals_result()

        epochs = len(results['validation_0']['error'])



    if plot_variables:

        xgb.plot_importance(clf,max_num_features=20,importance_type='gain',xlabel='gain')



    if plot_confusion:

        print('\n', classification_report(y_true, y_scores.round()))



    if title:

        timestr = time.strftime("%Y%m%d-%H%M%S")

        path = 'E:\\data\\data-fraud\\models\\'

        save_model(model=clf,path=path,title=str(title))

        

    if ensembler:

        return Y_test, y_scores



    print('-------------END EXECUTION-------------')
Y_test, y_scores = run_xgboost_and_calculate_auc(df_train,target='Survived',plot_all=True,ensembler=True)