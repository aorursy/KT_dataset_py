import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%%time

train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')
target = train['target']

train_id = train['id']

test_id = test['id']



train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)
df = pd.concat([train, test], axis=0, sort=False )
bin_dict = {'T':1, 'F':0, 'Y':1, 'N':0}

df['bin_3'] = df['bin_3'].map(bin_dict)

df['bin_4'] = df['bin_4'].map(bin_dict)
print(f'Shape before dummy transformation: {df.shape}')

df = pd.get_dummies(df, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],

                    prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], 

                    drop_first=True)

print(f'Shape after dummy transformation: {df.shape}')
from pandas.api.types import CategoricalDtype 



ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

                                     'Master', 'Grandmaster'], ordered=True)

ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',

                                     'Boiling Hot', 'Lava Hot'], ordered=True)

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',

                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',

                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)



df.ord_1 = df.ord_1.astype(ord_1)

df.ord_2 = df.ord_2.astype(ord_2)

df.ord_3 = df.ord_3.astype(ord_3)

df.ord_4 = df.ord_4.astype(ord_4)



df.ord_1 = df.ord_1.cat.codes

df.ord_2 = df.ord_2.cat.codes

df.ord_3 = df.ord_3.cat.codes

df.ord_4 = df.ord_4.cat.codes
def date_cyc_enc(df, col, max_vals):

    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    return df



df = date_cyc_enc(df, 'day', 7)

df = date_cyc_enc(df, 'month', 12)
%%time

from sklearn.preprocessing import LabelEncoder



# Label Encoding

for f in ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5']:

    lbl = LabelEncoder()

    lbl.fit(df[f])

    df[f'le_{f}'] = lbl.transform(df[f])
df.drop(['nom_5','nom_6','nom_7','nom_8','nom_9', 'ord_5'] , axis=1, inplace=True)
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
df = reduce_mem_usage(df)
#https://slundberg.github.io/shap/notebooks/plots/decision_plot.html (Good !!)
from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

import eli5

from eli5.sklearn import PermutationImportance

from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots

import shap
train = df[:train.shape[0]]

test = df[train.shape[0]:]



train.shape
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state = 0)
#  RandomForest

Model=LGBMClassifier(max_depth=10, n_estimators=1000, n_jobs=-1, num_leaves=45,  learning_rate=0.01)

Model.fit(X_train,y_train)

y_pred=Model.predict(X_val)

print(classification_report(y_pred,y_val))

#print(confusion_matrix(y_pred,y_val))

#Accuracy Score

print('Roc_Auc is ',roc_auc_score(y_pred,y_val))
perm = PermutationImportance(Model, random_state=1).fit(X_val, y_val)

eli5.show_weights(perm, feature_names = X_val.columns.tolist(),top=42)
explainer = shap.TreeExplainer(Model)

expected_value = explainer.expected_value

if isinstance(expected_value, list):

    expected_value = expected_value[1]

print(f"Explainer expected value: {expected_value}")



select = range(20)

features = X_val.iloc[select]

#features_display = features.columns



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    shap_values = explainer.shap_values(features)[1]

    shap_interaction_values = explainer.shap_interaction_values(features)

if isinstance(shap_interaction_values, list):

    shap_interaction_values = shap_interaction_values[1]
shap.decision_plot(expected_value, shap_values, features)
shap.decision_plot(expected_value, shap_values, features,link='logit')
# Our naive cutoff point is zero log odds (probability 0.5).

y_pred = (shap_values.sum(1) + expected_value) > 0

misclassified = y_pred != y_val.iloc[select]

shap.decision_plot(expected_value, shap_values, features, highlight=misclassified, link='logit')
shap.decision_plot(expected_value, shap_values[misclassified], features[misclassified],

                    link='logit', highlight=1)
shap.initjs()

shap.force_plot(expected_value, shap_values[misclassified], features[misclassified])
# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=Model, dataset=X_val, model_features=features.columns, feature='le_ord_5')



# plot it

pdp.pdp_plot(pdp_goals, 'le_ord_5')

plt.show()
# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=Model, dataset=X_val, model_features=features.columns, feature='ord_4')



# plot it

pdp.pdp_plot(pdp_goals, 'ord_4')

plt.show()
# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot

features_to_plot = ['le_ord_5', 'ord_4']

inter1  =  pdp.pdp_interact(model=Model, dataset=X_val, model_features=features.columns, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour', plot_pdp=True)

plt.show()