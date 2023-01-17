import sys

import pandas as pd

import numpy as np



import seaborn as sns



#from scipy.stats import entropy

from matplotlib import pyplot as plt

%matplotlib inline



from scipy.stats import skew

from sklearn.model_selection import  KFold , GridSearchCV, train_test_split

from sklearn.ensemble import  RandomForestClassifier

import json

from sklearn.feature_selection import SelectKBest, chi2, f_classif

from sklearn.metrics import confusion_matrix, log_loss, make_scorer, accuracy_score, f1_score



from sklearn.preprocessing import scale



from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.feature_selection import VarianceThreshold



from sklearn.preprocessing import LabelEncoder
def missing_perc(df):

     """

     Return a dataframe with percentage of missing values

     in each column in a sorted order

     Args:

     df: dataframe

     Returns:

     dataframe with percentage of missing values in each column

     """



     missing = df.isnull().sum()

     missing = missing[missing > 0] * 100 / df.shape[0]

     missing.sort_values(inplace=True)

     return pd.DataFrame(missing, columns=['missing_perc'])





def unique_val(df, outputcol='n_unique_vals'):

     """

     Count the number of distinct values in each column in a sorted order

     Args:

     df: dataframe

     col: string, column name of the output dataframe

     Returns:

     dataframe with the number of distinct values in each column

     """



     columns = df.columns

     undict = {}

     for col in columns:

         undict[col] = df[col].astype(str).nunique()

     undf = pd.DataFrame.from_dict(undict,

         'index',

         columns=[outputcol])

     undf.sort_values(by=[outputcol], inplace=True)

     return undf





def getCategoricalVariablesRanking(df, target, limit=50):

    """

    Return sorted chi square statistics for categorical features

    Source: https://scikit-learn.org/stable/modules/generated/sklearn.

    feature_selection.chi2.html#sklearn.feature_selection.chi2

    Args:

         df: pandas dataframe

         target: string corresponding to the categorical target column

    Returns:

         list object with sorted chi square statistics

    """

    categorical_variables = [i for i in list(df.dtypes[df.dtypes == 'object'].index) if i != target]

    chi2_selector = SelectKBest(chi2, k='all')

    df_chi_final = pd.DataFrame(columns=["scaled_importance", "value", "column"])

    for col in categorical_variables:

        dummy = pd.get_dummies(df[col])

        chi2_selector.fit_transform(dummy, df[target])

        df_chi = pd.DataFrame(chi2_selector.scores_,

                              columns=['scaled_importance'])

        df_chi["value"] = dummy.columns

        df_chi["column"] = col

        df_chi_final = pd.concat([df_chi_final, df_chi], axis=0)



    df_chi_final["scaled_importance"] -= df_chi_final["scaled_importance"].min()

    df_chi_final["scaled_importance"] /= df_chi_final["scaled_importance"].max()

    df_chi_final = df_chi_final.sort_values(by='scaled_importance', ascending=False).head(limit)



    return df_chi_final



def getContinuousVariablesRanking(df, target):

    """

    Return sorted F value statistics for continuous features

    Source: https://scikit-learn.org/stable/modules/generated/sklearn.

    feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest

    Args:

         df: pandas dataframe

         target: string corresponding to the categorical target column

    Returns:

         dataframe with sorted F value statistics

    """

    cont_vars = [i for i in list(df.dtypes[df.dtypes != 'object'].index) if i != target]



    Fvalue_selector = SelectKBest(f_classif, k=len(cont_vars))

    Fvalue_selector.fit_transform(df[cont_vars].fillna(-1), df[target])

    df_Fvalue = pd.DataFrame(Fvalue_selector.scores_,

                             columns=['scaled_importance'])

    # scaling the statistics

    df_Fvalue -= df_Fvalue.min()

    df_Fvalue /= df_Fvalue.max()

    df_Fvalue['columns'] = cont_vars

    df_Fvalue.sort_values(by='scaled_importance', ascending=False, inplace=True)

    

    return df_Fvalue
pwd


sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')




train_features.head()
train_features.shape
train_targets_scored.shape
plt.hist(train_targets_scored.mean())
test_features.head()
train_targets_scored.describe()
sub.head()
#missing values

missing_perc(train_features)
#missing values

missing_perc(test_features)
#missing values

missing_perc(train_targets_scored)
unique_val(train_features)
unique_val(test_features)
catList = ['cp_type', 'cp_dose']



countList =  list (set(train_features.columns) - set(catList))



countList.remove('sig_id')
print(train_features.shape,  train_features.drop_duplicates().shape)
fig, axes = plt.subplots(2, 1, figsize=(6, 4))



for i, ax in enumerate(fig.axes):

    if i < len(catList):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)

        sns.countplot(x=catList[i], alpha=0.7, data=train_features, ax=ax)



fig.tight_layout()



fig, axes = plt.subplots(1, 1, figsize=(6, 4))



for i, ax in enumerate(fig.axes):



    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)



    sns.countplot(x='cp_time', alpha=0.7, data=train_features, ax=ax)
fig, axes = plt.subplots(2, 1, figsize=(6, 4))



for i, ax in enumerate(fig.axes):

    if i < len(catList):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)

        sns.countplot(x=catList[i], alpha=0.7, data=test_features, ax=ax)



fig.tight_layout()



fig, axes = plt.subplots(1, 1, figsize=(6, 4))



for i, ax in enumerate(fig.axes):



    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)



    sns.countplot(x='cp_time', alpha=0.7, data=test_features, ax=ax)
train_targets_scored.mean()[train_targets_scored.mean() == train_targets_scored.mean().max()]


train_features_toptarget_count = pd.concat([train_features[countList], train_targets_scored['nfkb_inhibitor'].astype('str')], axis = 1)



df_Fvalue_s = getContinuousVariablesRanking(train_features_toptarget_count, 'nfkb_inhibitor')



df_Fvalue_s
n = 10

plt.figure(figsize=(10,5))

plt.title("F-value scaled importance for continuous features (top 10, target = nfkb_inhibitor)",fontsize=15)

plt.xlabel("Continuous Features",fontsize=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.bar(range(10),df_Fvalue_s.head(10)['scaled_importance'],align='edge',color='rgbkymc')

plt.xticks(range(10),df_Fvalue_s.head(10)['columns'],rotation=90,color='g')

plt.show()

plt.close()
n = 10

plt.figure(figsize=(10,5))

plt.title("F-value scaled importance for continuous features (bottom 10, target = nfkb_inhibitor)",fontsize=15)

plt.xlabel("Continuous Features",fontsize=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.bar(range(10),df_Fvalue_s.tail(10)['scaled_importance'],align='edge',color='rgbkymc')

plt.xticks(range(10),df_Fvalue_s.tail(10)['columns'],rotation=90,color='g')

plt.show()

plt.close()
colist = df_Fvalue_s.head(10)['columns']

label = 'nfkb_inhibitor'



for col in colist: 

    

    g = sns.FacetGrid(train_features_toptarget_count[[col, label]],  hue =label, height = 4, aspect = 1.5) 

    g.map(sns.distplot, col, hist = False, kde_kws = {'shade': True, 'linewidth': 3}).set_axis_labels(col,"density").add_legend()



    


train_features_toptarget_cat = pd.concat([train_features[catList], train_targets_scored['nfkb_inhibitor'].astype('str')], axis = 1)





getCategoricalVariablesRanking(train_features_toptarget_cat, 'nfkb_inhibitor')
train_features_toptarget_cat[train_features_toptarget_cat.cp_type == 'ctl_vehicle'][label].value_counts()
test_features[test_features.cp_type == 'ctl_vehicle'].shape
lb=LabelEncoder()



for f in catList: 



    train_features[f]=lb.fit_transform(train_features[f])

    test_features[f]=lb.transform(test_features[f])
colList = list(train_targets_scored.columns[1:])

train_features2 = train_features[countList+ catList]

#mask = test_features.cp_type == test_features.cp_type.value_counts().index[-1]



for label in colList: 

    

    y_train = train_targets_scored[label]

    rf = RandomForestClassifier(class_weight='balanced', max_depth=15,

                            n_estimators=500, #500

                            n_jobs=-1, 

                            random_state=1234)

    rf.fit(train_features2, y_train)

    test_features[label] =  rf.predict_proba(test_features[countList + catList])[:,1] 

    #test_features.loc[mask][label] = 0

    #print('label:', label)
test_features[

['sig_id'] + colList].head()



test_features[

['sig_id'] + colList ].to_csv('submission.csv', index=False)