# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.tree import DecisionTreeClassifier, export_graphviz

from graphviz import Source



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_excel('/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')

df.head()
cols_to_drop=['AGE_PERCENTIL', 'WINDOW']

df=df.drop(cols_to_drop,axis=1)

df.columns
plt.style.use('dark_background')

df['ICU'].value_counts().plot.barh()
import missingno as msno

# checking null values

msno.bar(df, figsize=(16, 4))
(df.isnull().sum() / df.shape[0]).sort_values(ascending=False).head()
contain_null_series = (df.isnull().sum() != 0).index
#target = 'ICU'

#just_one_target = []



#for col in contain_null_series:

#    i = df[df[col].notnull()][target].nunique()

#    if i == 1:

#        just_one_target.append(col)    



# columns that only are present when ICU is needed        

#print(just_one_target)
#for col in just_one_target:

 #   print(df[df[col].notnull()][target].unique())
#df.drop(just_one_target, axis=1, inplace=True)
not_null_series = (df.isnull().sum() == 0)

not_null_columns = not_null_series[not_null_series == True].index

not_null_columns = not_null_columns[1:]
def plot_histograms(df, cols, subplots_rows, subplots_cols, figsize=(16, 8), target='ICU'):

    df_neg = df[df[target] == 'negative']

    df_pos = df[df[target] == 'positive']

    

    cols = cols.tolist()

    cols.remove(target)

    

    plt.figure()

    fig, ax = plt.subplots(subplots_rows, subplots_cols, figsize=figsize)

    

    i = 0    

    for col in cols:

        i += 1

        plt.subplot(subplots_rows, subplots_cols, i)

        sns.distplot(df_neg[col], label="Negative", bins=15, kde=False)

        sns.distplot(df_pos[col], label="Positive", bins=15, kde=False)

        plt.legend()

    plt.show()

    

plot_histograms(df, not_null_columns, 2, 2)
x = df.drop(['PATIENT_VISIT_IDENTIFIER', 'ICU'], axis=1)

x.fillna(999999, inplace=True)

y = df['ICU']
dt = DecisionTreeClassifier(max_depth=3)
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(x, y)
dt_feat = pd.DataFrame(dt.feature_importances_, index=x.columns, columns=['feat_importance'])

dt_feat.sort_values('feat_importance').tail(8).plot.barh()

plt.show()
from IPython.display import SVG

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



graph = Source(export_graphviz(dt, out_file=None, feature_names=x.columns, filled = True))

display(SVG(graph.pipe(format='svg')))
sns.distplot(df[df['ICU'] == 1]['LEUKOCYTES_MAX'], label="ICU")

sns.distplot(df[df['ICU'] == 0]['LEUKOCYTES_MAX'], label="NO ICU")

plt.legend()
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.svm import SVC

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler
classifiers = {'Logistic Regression' : LogisticRegression(),

               'KNN': KNeighborsClassifier(),

               'Decision Tree': DecisionTreeClassifier(),

               'Random Forest': RandomForestClassifier(),

               'AdaBoost': AdaBoostClassifier(),

               'SVM': SVC()}



samplers = {'Random_under_sampler': RandomUnderSampler(),

            'Random_over_sampler': RandomOverSampler()}



drop_cols = ['PATIENT_VISIT_IDENTIFIER']
def df_split(df, target='ICU', drop_cols=drop_cols):

    df = df.drop(drop_cols, axis=1)

    df = df.fillna(999)

    x = df.drop(target, axis=1)

    y = df[target]    

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)                          

    return x_train, x_test, y_train, y_test



def train_clfs(df, classifiers, samplers):

    

    x_train, x_test, y_train, y_test = df_split(df)

    

    names_samplers = []

    names_clfs = []

    results_train_cv_roc_auc = []

    results_train_cv_recall = []

    results_train_cv_accuracy = []

    results_test_roc_auc = []

    results_test_recall = []

    results_test_accuracy = []

    

    for name_sampler, sampler in samplers.items():

        print(f'Sampler: {name_sampler}\n')

        for name_clf, clf in classifiers.items():

            print(f'Classifier: {name_clf}\n')

            

            pipeline = Pipeline([('sampler', sampler),

                                 ('clf', clf)])

            

            cv_auc = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='roc_auc') 

            cv_rec = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='recall')                                

            cv_acc = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='accuracy')        



            pipeline.fit(x_train, y_train)        

            y_pred = pipeline.predict(x_test)

            

            names_samplers.append(name_sampler)

            names_clfs.append(name_clf)

            results_train_cv_roc_auc.append(cv_auc)

            results_train_cv_recall.append(cv_rec)

            results_train_cv_accuracy.append(cv_acc)

            results_test_roc_auc.append(roc_auc_score(y_test, y_pred))

            results_test_recall.append(recall_score(y_test, y_pred))

            results_test_accuracy.append(accuracy_score(y_test, y_pred))



            print(f'CV\t-\troc_auc:\t{round(cv_auc.mean(), 3)}')

            print(f'CV\t-\trecall:\t\t{round(cv_rec.mean(), 3)}')

            print(f'CV\t-\taccuracy:\t{round(cv_acc.mean(), 3)}')



            print(f'Test\t-\troc_auc:\t{round(roc_auc_score(y_test, y_pred), 3)}')         

            print(f'Test\t-\trecall:\t\t{round(recall_score(y_test, y_pred), 3)}')          

            print(f'Test\t-\taccuracy:\t{round(accuracy_score(y_test, y_pred), 3)}')      

            print('\n<-------------------------->\n')



    df_results_test = pd.DataFrame(index=[names_clfs, names_samplers], columns=['ROC_AUC', 'RECALL', 'ACCURACY'])

    df_results_test['ROC_AUC'] = results_test_roc_auc

    df_results_test['RECALL'] = results_test_recall

    df_results_test['ACCURACY'] = results_test_accuracy



    return df_results_test
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

from imblearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix

import xgboost as xgb

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
df_results_test = train_clfs(df, classifiers, samplers)
def train_xgb(df, clf):

    

    x_train, x_test, y_train, y_test = df_split(df)



    scale_pos_weight = len(df[df['ICU'] == 0]) / len(df[df['ICU'] == 1])



    param_grid = {'xgb__max_depth': [3, 4, 5, 6, 7, 8],

                  'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],

                  'xgb__colsample_bytree': [0.6, 0.7, 0.8],

                  'xgb__min_child_weight': [0.4, 0.5, 0.6],

                  'xgb__gamma': [0, 0.01, 0.1],

                  'xgb__reg_lambda': [6, 7, 8, 9, 10],

                  'xgb__n_estimators': [150, 200, 300],

                  'xgb__scale_pos_weight': [scale_pos_weight]}



    rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=100,

                                n_jobs=-1, verbose=2, cv=5,                            

                                scoring='roc_auc', random_state=42)



    rs_clf.fit(x_train, y_train)

    

    print(f'XGBOOST BEST PARAMS: {rs_clf.best_params_}')

    

    y_pred = rs_clf.predict(x_test)



    df_results_xgb = pd.DataFrame(index=[['XGBoost'], ['No_sampler']], columns=['ROC_AUC', 'RECALL', 'ACCURACY'])



    df_results_xgb['ROC_AUC'] = roc_auc_score(y_test, y_pred)

    df_results_xgb['RECALL'] = recall_score(y_test, y_pred)

    df_results_xgb['ACCURACY'] = accuracy_score(y_test, y_pred)

    

    return df_results_xgb
df_results_xgb = train_xgb(df, xgb.XGBClassifier())
df_results = pd.concat([df_results_test, df_results_xgb])
df_plot = pd.concat([df_results.sort_values('ROC_AUC', ascending=False).head(3),

                     df_results.sort_values('RECALL', ascending=False).head(3),

                     df_results.sort_values('ACCURACY', ascending=False).head(3)])
def plot_test(df, xlim_min, xlim_max):



    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,12))

    color = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'navy', 'turquoise', 'darkorange']



    df['ROC_AUC'].plot(kind='barh', ax=ax1, xlim=(xlim_min, xlim_max), title='ROC_AUC', color=color)

    df['RECALL'].plot(kind='barh', ax=ax2, xlim=(xlim_min, xlim_max), title='RECALL', color=color)

    df['ACCURACY'].plot(kind='barh', ax=ax3, xlim=(xlim_min, xlim_max), title='ACCURACY', color=color)

    plt.show()
plot_test(df_plot, 0.4, 1)
def plot_confusion_matrix(y_test, y_pred, title='Confusion matrix'):

    



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

def train_clf_threshold(df, clf, sampler=None):

    thresholds = np.arange(0.1, 1, 0.1)

    

    x_train, x_test, y_train, y_test = df_split(df)

    

    if sampler:

        clf_train = Pipeline([('sampler', sampler),

                              ('clf', clf)])

        

    else:        

        clf_train = clf

            

    clf_train.fit(x_train, y_train)

    y_proba = clf_train.predict_proba(x_test)

    

    plt.figure(figsize=(10,10))



    j = 1

    for i in thresholds:

        y_pred = y_proba[:,1] > i



        plt.subplot(4, 3, j)

        j += 1



        # Compute confusion matrix

        cnf_matrix = confusion_matrix(y_test,y_pred)

        np.set_printoptions(precision=2)



        print(f"Threshold: {round(i, 1)} | Test Accuracy: {round(accuracy_score(y_test, y_pred), 2)}| Test Recall: {round(recall_score(y_test, y_pred), 2)} | Test Roc Auc: {round(roc_auc_score(y_test, y_pred), 2)}")



        # Plot non-normalized confusion matrix

        plot_confusion_matrix(y_test, y_pred, title=f'Threshold >= {round(i, 1)}')
train_clf_threshold(df, RandomForestClassifier(), sampler=RandomUnderSampler())