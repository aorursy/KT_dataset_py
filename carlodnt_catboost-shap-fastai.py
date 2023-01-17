# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.tabular import * 

from pathlib import Path



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path=Path('/kaggle/input/cat-in-the-dat-ii/train.csv')

df = pd.read_csv(path)

df.set_index('id',drop=True,inplace=True)

for column in df.columns:

    df[column].fillna(df[column].mode()[0], inplace=True)

df.head()
dfprocessed= df.copy()

dfprocessed['bin_3'] = dfprocessed['bin_3'].apply(lambda x: 0 if x == 'F' else 1)

dfprocessed['bin_4'] = dfprocessed['bin_4'].apply(lambda x: 0 if x == 'N' else 1)



dfprocessed.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],

                         value = [0, 1, 2, 3, 4], inplace = True)



dfprocessed.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],

                         value = [0, 1, 2, 3, 4, 5], inplace = True)



dfprocessed.ord_3.replace(to_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],

                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inplace = True)



dfprocessed.ord_4.replace(to_replace = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 

                                     'P', 'Q', 'R','S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],

                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 

                                  22, 23, 24, 25], inplace = True)



high_card = ['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9','ord_5']

for col in high_card:

    enc_nom = (dfprocessed.groupby(col).size()) / len(dfprocessed)

    dfprocessed[f'{col}'] = dfprocessed[col].apply( lambda x: hash(str(x)) % 5000 )





df=dfprocessed.copy()

df.head()
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split



X = df.drop('target', axis=1)

y = df.target



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=111)



categorical_features_indices = []



model=CatBoostClassifier(iterations=600,

                              learning_rate=0.1,

                              depth=5,

                              bootstrap_type='Bernoulli',

                              loss_function='Logloss',

                              subsample=0.9,

                              eval_metric='AUC',

                              metric_period=20,

                              allow_writing_files=False)



model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_val, y_val))
import catboost

from catboost import *

import shap

shap.initjs()



explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(Pool(X_train, y_train, cat_features=categorical_features_indices))
shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.force_plot(explainer.expected_value, shap_values[:1000,:], X_train.iloc[:1000,:])
shap.summary_plot(shap_values, X_train)
for i in range(10):

    inds = shap.approximate_interactions(f'nom_{i}', shap_values, X_train)

    shap.dependence_plot(f'nom_{i}', shap_values, X_train, interaction_index=inds[0])
for i in range(6):

    inds = shap.approximate_interactions(f'ord_{i}', shap_values, X_train)

    shap.dependence_plot(f'ord_{i}', shap_values, X_train, interaction_index=inds[0])
expected_value = explainer.expected_value



select = range(100)

features = X_train.iloc[select]



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    shap_values = explainer.shap_values(features)

    shap_interaction_values = explainer.shap_interaction_values(features)

    

#shap.decision_plot(explainer.expected_value, explainer.shap_interaction_values(features), features, feature_display_range=slice(None, None, -1),ignore_warnings=True)
y_pred = (np.sum(shap_values,axis=1) + expected_value) > 0

misclassified = y_pred != y_train.iloc[select]

shap.decision_plot(expected_value, shap_values[misclassified], features[misclassified],

                   link='logit')
from category_encoders import  LeaveOneOutEncoder

leaveOneOut_encoder = LeaveOneOutEncoder()



path=Path('/kaggle/input/cat-in-the-dat-ii/train.csv')

df = pd.read_csv(path)

df.set_index('id',drop=True,inplace=True)



df['bin_3'] = df['bin_3'].apply(lambda x: 0.0 if x == 'F' else 1.0)

df['bin_4'] = df['bin_4'].apply(lambda x: 0.0 if x == 'N' else 1.0)



df.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],

                         value = [0, 1, 2, 3, 4], inplace = True)



df.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],

                         value = [0, 1, 2, 3, 4, 5], inplace = True)



df.ord_3.replace(to_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],

                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inplace = True)



df.ord_4.replace(to_replace = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 

                                     'P', 'Q', 'R','S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],

                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 

                                  22, 23, 24, 25], inplace = True)

#______________________



pathsub=Path('/kaggle/input/cat-in-the-dat-ii/test.csv')

dfsub = pd.read_csv(pathsub)

dfsub.set_index('id',drop=True,inplace=True)



dfsub['bin_3'] = dfsub['bin_3'].apply(lambda x: 0.0 if x == 'F' else 1.0)

dfsub['bin_4'] = dfsub['bin_4'].apply(lambda x: 0.0 if x == 'N' else 1.0)



dfsub.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],

                         value = [0, 1, 2, 3, 4], inplace = True)



dfsub.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],

                         value = [0, 1, 2, 3, 4, 5], inplace = True)



dfsub.ord_3.replace(to_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],

                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inplace = True)



dfsub.ord_4.replace(to_replace = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 

                                     'P', 'Q', 'R','S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],

                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 

                                  22, 23, 24, 25], inplace = True)



high_card = ['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9','ord_5']



for nom in high_card:

    df[f'{nom}_lOO'] = leaveOneOut_encoder.fit_transform(df[nom], df["target"])

    dfsub[f'{nom}_lOO'] = leaveOneOut_encoder.transform(dfsub[nom])

    

#After running FastAi Class Confusion

df['nom_9_lOO_bool']=df['nom_9_lOO']

df['nom_9_lOO_bool'].fillna(df['nom_9_lOO_bool'].mode()[0], inplace=True)

df['nom_9_lOO_bool']=df['nom_9_lOO_bool'].apply(lambda x: 0 if (x>0.35 and x<0.7) else 1)

dfsub['nom_9_lOO_bool']=dfsub['nom_9_lOO']

dfsub['nom_9_lOO_bool'].fillna(dfsub['nom_9_lOO_bool'].mode()[0], inplace=True)

dfsub['nom_9_lOO_bool']=dfsub['nom_9_lOO_bool'].apply(lambda x: 0 if (x>0.35 and x<0.7) else 1)

#______________________



procs = [Normalize,FillMissing,Categorify]

dep_var = 'target'

cat_names=['nom_9_lOO_bool','bin_0','bin_1','bin_2','bin_3','bin_4','ord_0','ord_1','ord_2','ord_3','ord_4','day','month','nom_0','nom_1','nom_2','nom_3','nom_4','nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9','ord_5']

cont_names = ['nom_0_lOO','nom_1_lOO','nom_2_lOO','nom_3_lOO','nom_4_lOO','nom_5_lOO', 'nom_6_lOO', 'nom_7_lOO', 'nom_8_lOO','nom_9_lOO','ord_5_lOO']

valid_idx=np.random.random_integers(0,600000,100000)

test = TabularList.from_df(dfsub, path=pathsub, cat_names=cat_names,cont_names=cont_names, procs=procs)

data = (TabularList.from_df(df, path=path, cat_names=cat_names,cont_names=cont_names, procs=procs)

                           .split_by_idx(valid_idx=valid_idx)

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch(bs=1024))

data.show_batch()
weights = [0.3, 0.7]

class_weights=torch.FloatTensor(weights)
learn = tabular_learner(data, layers=[1500,500,250], metrics=AUROC(),callback_fns=ShowGraph,path='.',emb_drop=0.01,use_bn=True).to_fp32()

learn.loss_func = nn.CrossEntropyLoss(weight=class_weights)
learn.fit_one_cycle(4, 1e-2,wd = 0.25)
learn.fit_one_cycle(3, 1e-3,wd = 0.25)
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
from fastai.widgets import ClassConfusion

ClassConfusion(interp,[0, 1],varlist=cont_names,figsize=(12,12))
learn.save('deepmodel')
test_preds = learn.get_preds(ds_type=DatasetType.Test)
test_preds = test_preds[0][:,1]

test_preds
pathsub=Path('/kaggle/input/cat-in-the-dat-ii/test.csv')

dfsub = pd.read_csv(pathsub)

dfsub['target'] = test_preds

dfsub.to_csv('submission.csv', columns=['id', 'target'], index=False)