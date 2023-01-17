import os

import math

import numpy as np 

import pandas as pd 



from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.compose import make_column_transformer

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn import base





from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split, learning_curve, validation_curve

from sklearn.metrics import roc_auc_score, make_scorer

from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.model_selection import cross_validate

from sklearn.compose import ColumnTransformer

from sklearn import metrics



from collections import Counter

from sklearn.datasets import make_classification

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report

from category_encoders import TargetEncoder







kf = StratifiedKFold(5, random_state=0, shuffle=True)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col='id')

test =  pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col='id')

submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv", index_col='id')
def chi_2_and_cramers_v(contingency_df):

    """

    perform chi-square and cramer's V

        H0: feature and target are independent

        H1: feature and target are not independent

    """

    from scipy import stats

    chi2, p, dof, expected = stats.chi2_contingency(contingency_df)

    n = contingency_df.sum().sum()

    r, k = contingency_df.shape

    cramers_v = np.sqrt(chi2 / (n * min((r-1), (k-1))))

    return chi2, p, cramers_v





def describe_data(train_df, test_df, confidence_value=0.95):

    # train_df = train_df.drop("id", axis=1)

    # test_df = test_df.drop("id", axis=1)

    feat_property_list = []

    columns = [col for col in train_df.columns if not col == 'target']

    for col in columns:

        feat_description = {}

        contingency_df = pd.crosstab(train_df[col], train_df['target'], margins=False)

        chi2, p, cramers_v = chi_2_and_cramers_v(contingency_df)

                

        feat_description['feature'] = col

        #feat_description['chi_2'] = chi2

        feat_description['chi_2_p_value'] = p

        if p < (1- confidence_value):

            feat_description['feat-dependence'] = "dependent"

        else:

            feat_description['feat-dependence'] = "independent"

            

        



        feat_description['cramers_v'] = cramers_v

                

        feat_description['train_cardinality'] = train_df[col].nunique()

        feat_description['test_cardinality'] = test_df[col].nunique()

        feat_description['in_train_not_in_test'] = len(set(train_df[col]) - set(test_df[col]))

        feat_description['in_test_not_in_train'] = len(set(test_df[col]) - set(train_df[col]))

        # feat_description['cardinality_diff(test-train)'] = feat_description['test_cardinality'] - feat_description['train_cardinality']

        feat_description['biggest-cat(train)%'] = 100 * train[col].value_counts(normalize = True, dropna = False).values[0]

        feat_description['dtype'] = train_df[col].dtype



        feat_property_list.append(feat_description)

        

    return pd.DataFrame(feat_property_list).round(3)



        

data_description = describe_data(train, test)

data_description.sort_values("cramers_v")      
def plt_count_and_target(train_df, features, sortby='count', h=20, w=20, pad=8, tm_max_range=100, max_features=15):

    import seaborn as sns

    feat_num = len(features)

    m = np.ceil((feat_num)/2.)

    fig, axes = plt.subplots(int(m), 2, figsize=(w, h+(feat_num//2)*pad))

    fig.tight_layout(pad=pad)

    try:

        axes = axes.ravel()

    except AttributeError:

        axes = [axes]

    for feat in features:

        if train_df[feat].dtype != 'object':

            train_df[feat] = train_df[feat].astype(str)

    for idx, feat in enumerate(features):

        if idx < feat_num:

            ax1 = axes[idx]

            ax2 = ax1.twinx()

            dt1 = train.groupby(feat).agg({'target':'count'})

            dt2 = pd.crosstab(train[feat], train['target'], margins=True).sort_values('All', ascending=False).drop(index=['All'])

            dt2['target_mean'] = (dt2[1]*100)/dt2['All']

            if sortby == 'count':

                master_data = pd.merge(dt1, dt2, left_index=True, right_index=True).reset_index().sort_values('All', ascending=False)

            if sortby == 'target_mean':

                master_data = pd.merge(dt1, dt2, left_index=True, right_index=True).reset_index().sort_values('target_mean', ascending=False)



            master_data[feat] = master_data[feat].astype(str)

            if master_data.shape[0] > (max_features + 1):

                master_data = master_data.iloc[:max_features]

            sns.barplot(master_data[feat].tolist(), master_data['target'].tolist(), ax=ax1)

            ax2.plot(master_data[feat].tolist(), master_data['target_mean'].tolist(), '-ro')

            ax2.axhline((train.target.sum()*100)/train.shape[0], color='r', label='overall_target_mean')

            ax1.set_xlabel('feature')

            ax1.set_ylabel('count', )

            ax2.set_ylabel('target mean(%)',)

            ax1.set_title(f"{features[idx]}")

            ax1.xaxis.set_tick_params(rotation=90)

            # step_ = (int(master_data.target_mean.max() + 5) - int(master_data.target_mean.min()-5))//10

            #ax2.yaxis.set_ticks(np.arange(int(master_data.target_mean.min()-5), int(master_data.target_mean.max() + 5), step_))

            ax2.set_ylim(max(0, int(master_data.target_mean.min()-10)), int(master_data.target_mean.max() + 3),)



            plt.legend()

    plt.show()
plt_count_and_target(train, features=[col for col in train.columns if col.startswith("bin")], sortby='target_mean', w=18, h=10, pad=4, tm_max_range=55)
plt_count_and_target(train, features=[col for col in train.columns if col.startswith("nom")], sortby='target_mean', w=15, h=5, pad=3, tm_max_range=105, max_features=100)
plt_count_and_target(train, features=[col for col in train.columns if col.startswith("ord")], sortby='target_mean', w=18, h=10, pad=4, tm_max_range=55, max_features=256)
plt_count_and_target(train, features=['day', 'month'], sortby='target_mean', w=15, h=0, pad=4.5, tm_max_range=105)
X = train.drop("target", axis=1)

y = train['target']
def pipeline_baseline(**kwargs):

    """This function return data encoding pipeline"""

    # Logistic Regression parameters

    ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore"),

            [f for f in X.columns])



    pipe = make_pipeline(make_column_transformer(ohe1,remainder='drop'), LogisticRegression(**kwargs))

    return pipe



pipe = pipeline_baseline()



lr_params = {'logisticregression__penalty': ['l2'], 

             'logisticregression__solver': ['lbfgs'], 

             'logisticregression__C': np.random.uniform(0.1, 0.13, 20), 

             'logisticregression__max_iter':[1000],

             'logisticregression__fit_intercept': [True, False],

              'logisticregression__class_weight': [None, 'balanced'],

            }

gs = RandomizedSearchCV(pipe, param_distributions=lr_params, cv=kf, 

                        verbose=10, scoring='roc_auc', 

                        n_jobs=-1, n_iter=3, random_state=0)

model = gs.fit(X, y)



print(f"bast score={model.best_score_} and best params={model.best_params_}")
def plot_roc(model, X, y, label="", title=''):

    y_pred_proba = model.predict_proba(X)[::,1]

    fpr, tpr, _ = metrics.roc_curve(y,  y_pred_proba)

    auc = metrics.roc_auc_score(y, y_pred_proba)

    plt.plot(fpr,tpr,label=label +str(round(auc, 5)))

    plt.legend(loc=4)

    plt.title(title)

    plt.show()
plot_roc(model, X, y, label="Baseline ROC_AUC=", title="Baseline Model Training ROC")
def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')

    plt.fill_between(train_sizes, train_mean + train_std,

                     train_mean - train_std, color='blue', alpha=alpha)

    plt.plot(train_sizes, test_mean, label='val score', color='red', marker='o')



    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)

    plt.title(title)

    plt.xlabel('Number of training points')

    plt.ylabel('roc_auc')

    plt.grid(ls='--')

    plt.legend(loc='best')

    plt.ylim(0.5, 1)

    plt.show()





def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.1):

    param_range = [x for x in param_range] 

    sort_idx = np.argsort(param_range)

    param_range=np.array(param_range)[sort_idx]

    train_mean = np.mean(train_scores, axis=1)[sort_idx]

    train_std = np.std(train_scores, axis=1)[sort_idx]

    test_mean = np.mean(test_scores, axis=1)[sort_idx]

    test_std = np.std(test_scores, axis=1)[sort_idx]

    plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')

    plt.fill_between(param_range, train_mean + train_std,

                 train_mean - train_std, color='blue', alpha=alpha)

    plt.plot(param_range, test_mean, label='val score', color='red', marker='o')

    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)

    plt.xscale("log")

    plt.title(title)

    plt.grid(ls='--')

    plt.xlabel('Weight of class 2')

    plt.ylabel('Average values and standard deviation for ROC_AUC')

    plt.legend(loc='best')

    plt.show()

plt.figure(figsize=(9, 6))

train_sizes1, train_scores1, test_scores1 = learning_curve(

    estimator=model.best_estimator_, X=X, y=y,

    train_sizes=np.arange(0.1, 1.1, 0.1), cv=kf, scoring='roc_auc', n_jobs=- 1)



plot_learning_curve(train_sizes1, train_scores1, test_scores1, title='Learning curve for Baseline Model')
param_range1 = [math.pow(10, i) for i in range(-4, 4)]

train_scores, test_scores = validation_curve(

    estimator=model.best_estimator_, X=X, y=y, param_name="logisticregression__C", param_range=param_range1,

    cv=kf, scoring="roc_auc", n_jobs=-1, verbose=True)



plot_validation_curve(param_range1, train_scores, test_scores, title="Validation Basline Model:Param=C", alpha=0.1)
def pipeline_target_encode_nom(**kwargs):

    """This function return data encoding pipeline"""

    # Logistic Regression parameters

    ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore"),

            ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 

             # 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4',

            ])

    

    te_nom_5 = (TargetEncoder(), ['nom_5',])

    te_nom_6 = (TargetEncoder(),['nom_6',])

    te_nom_7 = (TargetEncoder(), ['nom_7',])

    te_nom_8 = (TargetEncoder(),['nom_8',])

    te_nom_9 = (TargetEncoder(),['nom_9',])

    

    #std = (StandardScaler(), ['nom_5', 'nom_6','nom_7','nom_8','nom_9','ord_5',])

    te_ord_0 = (TargetEncoder(), ['ord_0',])

    te_ord_1 = (TargetEncoder(), ['ord_1',])

    te_ord_2 = (TargetEncoder(), ['ord_2',])

    te_ord_3 = (TargetEncoder(), ['ord_3',])

    te_ord_4 = (TargetEncoder(), ['ord_4',])

    te_ord_5 = (TargetEncoder(), ['ord_5',])

   

    pipe = make_pipeline(make_column_transformer(ohe1, te_nom_5, te_nom_6, te_nom_7, te_nom_8, te_nom_9, 

                                                 te_ord_0, te_ord_1, te_ord_2, te_ord_3, te_ord_4, te_ord_5, remainder='drop'), StandardScaler(with_mean=False), 

                         LogisticRegression(**kwargs))

    return pipe



params = {'columntransformer__targetencoder-1__smoothing': [i/20. for i in range(20)],

          'columntransformer__targetencoder-2__smoothing': [i/20. for i in range(20)],

          'columntransformer__targetencoder-3__smoothing': [i/20. for i in range(20)],

          'columntransformer__targetencoder-4__smoothing': [i/20. for i in range(20)],

          'columntransformer__targetencoder-5__smoothing': [i/20. for i in range(20)],

          'columntransformer__targetencoder-1__min_samples_leaf': [5, 10, 15],

          'columntransformer__targetencoder-6__smoothing': [i/20. for i in range(20)],

    'logisticregression__penalty': ['l2'], 

     'logisticregression__solver': ['lbfgs'], 

     'logisticregression__C': np.random.uniform(0.1, 0.13, 20), 

     'logisticregression__max_iter':[1000],

     'logisticregression__fit_intercept': [True],

      'logisticregression__class_weight': [None, 'balanced'],

            }

pipe = pipeline_target_encode_nom()



gs = RandomizedSearchCV(pipe, param_distributions=params, cv=kf, 

                        verbose=10, scoring='roc_auc', 

                        n_jobs=-1, n_iter=5, random_state=0)

model = gs.fit(X, y)



print(f"bast score={model.best_score_} and best params={model.best_params_}")
pipe.get_params().keys()
plt.figure(figsize=(9, 6))

train_sizes1, train_scores1, test_scores1 = learning_curve(

    estimator=model.best_estimator_, X=X, y=y,

    train_sizes=np.arange(0.1, 1.1, 0.1), cv=kf, scoring='roc_auc', n_jobs=- 1)



plot_learning_curve(train_sizes1, train_scores1, test_scores1, title='Learning curve for Target Encode Model')
param_range1 = [i/20. for i in range(20)]

train_scores, test_scores = validation_curve(

    estimator=model.best_estimator_, X=X, y=y, param_name="columntransformer__targetencoder-5__smoothing", param_range=param_range1,

    cv=kf, scoring="roc_auc", n_jobs=-1, verbose=True)



plot_validation_curve(param_range1, train_scores, test_scores, title="Validation Basline Model:param=nom_9_smoothing", alpha=0.1)
param_range1 = [i/5. for i in range(5)]

train_scores, test_scores = validation_curve(

    estimator=model.best_estimator_, X=X, y=y, param_name="columntransformer__targetencoder-4__smoothing", param_range=param_range1,

    cv=kf, scoring="roc_auc", n_jobs=-1, verbose=True)



plot_validation_curve(param_range1, train_scores, test_scores, title="Validation Basline Model:param=nom_8_smoothing", alpha=0.1)