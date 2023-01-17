import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import ListedColormap

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from pandas_profiling import ProfileReport

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

import math

from sklearn.impute import KNNImputer



pd.set_option('display.max_columns', 50)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()
df.info()
profile = ProfileReport(df, title="Pandas Profiling Report")

# profile = ProfileReport(df, minimal=True)
profile.to_widgets()
profile.to_notebook_iframe()
# Unique Value. Remove it



df = df.drop('PassengerId', axis = 1)
df.head()
df['Survived'].value_counts()
sns.countplot(df['Survived'])
sns.countplot(df['Pclass'])
sns.countplot(df['Pclass'], hue = df['Sex'])
sns.boxplot(x = df['Pclass'], y = df['Age'])
sns.boxplot(x = df['Pclass'], y = df['Age'], hue = df['Sex'])
# df = pd.get_dummies(df,columns=['Pclass'],drop_first=True)
df.head()
# Drop because it's unique



df = df.drop('Name', axis = 1)
sns.countplot(df['Sex'])
# df = pd.get_dummies(df,columns=['Sex'],drop_first=True)
figure = plt.figure(figsize = [20,8])



sns.distplot(df['Age'])
figure = plt.figure(figsize = [10,8])



sns.boxplot(data=df, x='Pclass', y="Age")
figure = plt.figure(figsize = [15,8])



sns.boxplot(data=df, x='SibSp', y="Age")
figure = plt.figure(figsize = [15,8])



sns.boxplot(data=df, x='Parch', y="Age")
figure = plt.figure(figsize = [10,8])



sns.violinplot(data=df, x='Pclass', y="Age",hue="Survived",

               split=True, inner="quart", linewidth=1)
figure = plt.figure(figsize = [15,8])



sns.violinplot(data=df, x='SibSp', y="Age",hue="Survived",

               split=True, inner="quart", linewidth=1)
figure = plt.figure(figsize = [15,8])



sns.violinplot(data=df, x='Parch', y="Age",hue="Survived",

               split=True, inner="quart", linewidth=1)
sum(df['Age'].isna())
col_for_knn = 'Age SibSp Pclass Parch'.split()



imputer = KNNImputer(n_neighbors=3, weights='distance')



df[col_for_knn] = imputer.fit_transform(df[col_for_knn])
sum(df['Age'].isna())
figure = plt.figure(figsize = [20,8])



sns.distplot(df['Age'])
df['SibSp'].value_counts()
df['Parch'].value_counts()
df = df.drop('Ticket', axis = 1)
df.head()
figure = plt.figure(figsize = [20,8])



sns.distplot(df['Fare'])
figure = plt.figure(figsize = [15,8])



sns.scatterplot(data=df, x='Fare', y="Age", hue = 'Survived')
figure = plt.figure(figsize = [15,8])



sns.boxplot(data=df, x='Pclass', y="Fare")
figure = plt.figure(figsize = [15,8])



sns.boxplot(data=df, x='SibSp', y="Fare")
df['Fare'].describe()
sum(df['Fare'] == 0)
sum(df['Fare'] == 512.3292)
df.sort_values(by=['Fare'], ascending=False).head(10)
df
df['Cabin'].value_counts()
len(set(df['Cabin']))
sum(df['Cabin'].isna())
# df['Cabin'].fillna('Z', inplace = True)
# df['Cabin_alp'] = df['Cabin'].apply(lambda x: x[0])
# df['Cabin_alp'].value_counts()
# Most of it is missing value. Drop it.



df = df.drop('Cabin', axis = 1)
# df = pd.get_dummies(df,columns=['Cabin_alp'],drop_first=True)
df.head()
sum(df['Embarked'].isna())
df['Embarked'].value_counts()
sns.boxplot(x = df['Embarked'], y = df['Age'], hue = df['Sex'])
sns.countplot(df['Embarked'], hue = df['Pclass'])
sns.boxplot(x = df['Embarked'], y = df['Fare'], hue = df['Sex'])
df[df['Embarked'].isna()]
# Null Embarked related to Fare (80) and Age (Female)



df['Embarked'].fillna('C', inplace = True)
df[df['Embarked'].isna()]
df = pd.get_dummies(df,columns=['Embarked'],drop_first=True)
df.info()
df = pd.get_dummies(df,columns=['Sex'],drop_first=True)

df = pd.get_dummies(df,columns=['Pclass'],drop_first=True)
def pipeline(df):

    

    df = df.drop('PassengerId', axis = 1)

    

    col_for_knn = 'Age SibSp Pclass Parch'.split()

    df[col_for_knn] = imputer.transform(df[col_for_knn])

    

    df = pd.get_dummies(df,columns=['Pclass'],drop_first=True)

    df = pd.get_dummies(df,columns=['Sex'],drop_first=True)



    df = df.drop('Name', axis = 1)

    df = df.drop('Ticket', axis = 1)

    

#     df['Cabin'].fillna('Z', inplace = True)

#     df['Cabin_alp'] = df['Cabin'].apply(lambda x: x[0])

    df = df.drop('Cabin', axis = 1)

#     df = pd.get_dummies(df,columns=['Cabin_alp'],drop_first=True)

    

    df['Embarked'].fillna('C', inplace = True)

    df = pd.get_dummies(df,columns=['Embarked'],drop_first=True)

    

    return df
# df = pipeline(df)



# df.head()
X = df.drop('Survived', axis = 1)

y = df['Survived']
X.head()
# !pip install Boruta
from boruta import BorutaPy

from sklearn.ensemble import RandomForestClassifier
X_fs = X.values

y_fs = y.ravel()
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)



# define Boruta feature selection method

feat_selector = BorutaPy(rf, \

                         n_estimators='auto', \

                         verbose=1, random_state=1, \

                         alpha = 0.05, \

                         perc = 50, \

                         max_iter = 30)



# find all relevant features - 5 features should be selected

feat_selector.fit(X_fs, y_fs)
# check selected features - first 5 features are selected

feat_selector.support_
# check ranking of features

feat_selector.ranking_
X_filtered = pd.DataFrame(feat_selector.transform(X_fs), columns = X.columns[feat_selector.support_])



X_filtered.head()
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer



def evaluation_metric_from_result(y_test, y_pred):

    

    ##########

    # Making the Confusion Matrix

    cm = confusion_matrix(y_test, y_pred)

    

    ##########

    # Precision, Recall, F1-Score

    

    # Survived = 1

    tn_1, fp_1, fn_1, tp_1 = cm.ravel()

    precision_1 = tp_1/(tp_1 + fp_1)

    recall_1 = tp_1/(tp_1 + fn_1)

    f1_1 = 2*(precision_1 * recall_1)/(precision_1 + recall_1)

    

    # Survived = 0

    tp_2, fn_2, fp_2, tn_2 = cm.ravel()

    precision_2 = tp_2/(tp_2 + fp_2)

    recall_2 = tp_2/(tp_2 + fn_2)

    f1_2 = 2*(precision_2 * recall_2)/(precision_2 + recall_2)

    

    # Both

    avg_f1 = (f1_1 + f1_2)/2

    

    ##########

    # Print    

    #print('{}\n'.format(cm))

    

    print('Actual 0 Predict 0:\t{}'.format(tn_1))

    print('Actual 0 Predict 1:\t{}'.format(fp_1))

    print('Actual 1 Predict 0:\t{}'.format(fn_1))

    print('Actual 1 Predict 1:\t{}\n'.format(tp_1))

    

    print('[ Class 1 ]\n')

    print('Precision:\t{}\nRecall:\t\t{}\nF1-Score:\t{}\n'.format(precision_1, recall_1, f1_1))

    

    print('[ Class 0 ]\n')

    print('Precision:\t{}\nRecall:\t\t{}\nF1-Score:\t{}\n'.format(precision_2, recall_2, f1_2))

    

    print('[ Average ]\n')

    print('Average F1-Score:\t{}'.format(avg_f1))



#     # Also can use this one in 'macro', will be the same result.

#     precision_recall_fscore_support(y_test, y_pred_dt, average='macro')

    

#     # 'micro':

#     # Calculate metrics globally by counting the total true positives, false negatives and false positives.



#     # 'macro':

#     # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.



#     # 'weighted':

#     # Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). 

#     # This alters ‘macro’ to account for label imbalance; 

#     # it can result in an F-score that is not between precision and recall.
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]

def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]



scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),

            'fp': make_scorer(fp), 'fn': make_scorer(fn)}
def evaluation_metric_from_cv(cv_result):

    

    precision_1_list = []

    recall_1_list = []

    f1_1_list = []

    

    precision_2_list = []

    recall_2_list = []

    f1_2_list = []

    

    avg_f1_list = []

    

    for i in range(len(cv_result['fit_time'])):

        

        ##########

        # Precision, Recall, F1-Score



        # Survived = 1

        tn_1 = cv_result['test_tn'][i]

        fp_1 = cv_result['test_fp'][i]

        fn_1 = cv_result['test_fn'][i]

        tp_1 = cv_result['test_tp'][i]

        precision_1 = tp_1/(tp_1 + fp_1)

        recall_1 = tp_1/(tp_1 + fn_1)

        f1_1 = 2*(precision_1 * recall_1)/(precision_1 + recall_1)

        

        precision_1_list.append(np.round(precision_1, 4))

        recall_1_list.append(np.round(recall_1, 4))

        f1_1_list.append(np.round(f1_1, 4))

        

        # Survived = 0

        tp_2 = cv_result['test_tn'][i]

        fn_2 = cv_result['test_fp'][i]

        fp_2 = cv_result['test_fn'][i]

        tn_2 = cv_result['test_tp'][i]

        precision_2 = tp_2/(tp_2 + fp_2)

        recall_2 = tp_2/(tp_2 + fn_2)

        f1_2 = 2*(precision_2 * recall_2)/(precision_2 + recall_2)



        precision_2_list.append(np.round(precision_2, 4))

        recall_2_list.append(np.round(recall_2, 4))

        f1_2_list.append(np.round(f1_2, 4))

        

        # Both

        avg_f1 = (f1_1 + f1_2)/2

        

        avg_f1_list.append(np.round(avg_f1, 4))

    

    

    eval_df = pd.DataFrame({'Precision Class 1': precision_1_list, 'Recall Class 1': recall_1_list, 'F1-Score Class 1': f1_1_list, \

                        'Precision Class 0': precision_2_list, 'Recall Class 0': recall_2_list, 'F1-Score Class 0': f1_2_list, \

                        'Average F1-Score': avg_f1_list})

    

    return eval_df
# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier_dt.fit(X_train, y_train)



# Predicting the Test set results

y_pred_dt = classifier_dt.predict(X_test)
evaluation_metric_from_result(y_test, y_pred_dt)
cv_results_dt = cross_validate(classifier_dt, X, y, cv=5, scoring=scoring)



evaluation_metric_from_cv(cv_results_dt)
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB
# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifier_nb = GaussianNB()

classifier_nb.fit(X_train, y_train)



# Predicting the Test set results

y_pred_nb = classifier_nb.predict(X_test)
evaluation_metric_from_result(y_test, y_pred_nb)
cv_results_nb = cross_validate(classifier_nb, X, y, cv=5, scoring=scoring)



evaluation_metric_from_cv(cv_results_nb)
from sklearn.neural_network import MLPClassifier



# alpha for regularization (L2 regularization)

classifier_mlp = MLPClassifier(solver='sgd', alpha=1e-5, activation = 'logistic',

                    hidden_layer_sizes=(30,), max_iter=500, learning_rate = 'adaptive',

                    random_state=42)



classifier_mlp.fit(X_train, y_train)



y_pred_mlp = classifier_mlp.predict(X_test)
evaluation_metric_from_result(y_test, y_pred_mlp)
cv_results_mlp = cross_validate(classifier_mlp, X, y, cv=5, scoring=scoring)



evaluation_metric_from_cv(cv_results_mlp)
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



df_test.head()
df_test_fil = pipeline(df_test)



df_test_fil.head()
df_test_fil['Fare'] = df_test_fil['Fare'].fillna(df_test_fil['Fare'].mean())
df_test_fil.info()
# df_test_fil['Cabin_alp_T'] = pd.DataFrame(np.zeros(len(df_test_fil)))
df_test_fil = pd.DataFrame(feat_selector.transform(df_test_fil.values), columns = X.columns[feat_selector.support_])



df_test_fil.head()                       
# Predict



y_pred = classifier_nb.predict(df_test_fil)



y_pred
df_test['Label'] = y_pred



df_test.head()
df_result = df_test[['PassengerId','Label']]
df_result = df_result.rename({'Label': 'Survived'}, axis='columns')



df_result.head()
df_result.to_csv('./titanic_results.csv', index = False)
!pip install pycaret
import pycaret

from pycaret.classification import *
df.head()
# init setup

clf = setup(df, target = 'Survived',session_id = 888, log_experiment = False, \

            experiment_name = 'titanic', train_size = 0.75, \

#             numeric_features = ['open_count_last_10_days'], 

            categorical_features = ['Sex_male', 'Embarked_Q', 'Embarked_S', 'Pclass_2.0', 'Pclass_3.0'],\

            feature_selection_method = 'boruta',

            feature_selection = True, \

            fix_imbalance = False, normalize = True, \

            remove_multicollinearity = True, multicollinearity_threshold = 0.8,

            ignore_low_variance = True, \

           )
models()
pycaret_classifier = compare_models(sort = 'Accuracy', n_select = 5)



# selected_model = ['lightgbm','xgboost','et','gbc', 'ada']

# top5 = compare_models(n_select = 5, sort = 'MCC', whitelist = selected_model)
pycaret_classifier
interpret_model(

    estimator=pycaret_classifier[0],

    plot = 'summary',

    feature = None,

    observation = None

)
plot_model(pycaret_classifier[1], plot = 'feature')
plot_model(pycaret_classifier[1], plot = 'confusion_matrix')
best_pycaret = tune_model(pycaret_classifier[0], optimize = 'Accuracy', n_iter = 50, fold = 5)
interpret_model(

    estimator=best_pycaret,

    plot = 'summary',

    feature = None,

    observation = None

)
interpret_model(

    estimator=best_pycaret,

    plot = 'reason',

    feature = None,

    observation = None

)
second_pycaret = tune_model(pycaret_classifier[1], optimize = 'Accuracy', n_iter = 50, fold = 5)
plot_model(second_pycaret, plot = 'feature')
plot_model(second_pycaret, plot = 'confusion_matrix')
plot_model(estimator = second_pycaret, plot = 'learning')
best_pycaret = finalize_model(best_pycaret)
interpret_model(

    estimator=best_pycaret,

    plot = 'summary',

    feature = None,

    observation = None

)
second_pycaret = finalize_model(second_pycaret)
plot_model(estimator = second_pycaret, plot = 'learning')
df_test_pycaret = pd.read_csv('/kaggle/input/titanic/test.csv')



df_test_pycaret.head()
df_test_pycaret_fil = pipeline(df_test_pycaret)
y_pred_pycaret_best = predict_model(best_pycaret, data = df_test_pycaret_fil)



y_pred_pycaret_second = predict_model(second_pycaret, data = df_test_pycaret_fil)
y_pred_pycaret_best['PassengerId'] = df_test_pycaret['PassengerId']



y_pred_pycaret_second['PassengerId'] = df_test_pycaret['PassengerId']
df_result_pycaret_best = y_pred_pycaret_best[['PassengerId','Label']]



df_result_pycaret_second = y_pred_pycaret_second[['PassengerId','Label']]
df_result_pycaret_best.head()
df_result_pycaret_second.head()
df_result_pycaret_best = df_result_pycaret_best.rename({'Label': 'Survived'}, axis='columns')



df_result_pycaret_best.head()
df_result_pycaret_second = df_result_pycaret_second.rename({'Label': 'Survived'}, axis='columns')



df_result_pycaret_second.head()
df_result_pycaret_best.to_csv('./titanic_result_pycaret1.csv', index = False)
df_result_pycaret_second.to_csv('./titanic_result_pycaret2.csv', index = False)