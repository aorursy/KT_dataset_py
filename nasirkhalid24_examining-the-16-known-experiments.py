import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.express as px # plotting

from sklearn.decomposition import PCA # Principal Component Analysis
train = pd.read_csv('../input/lish-moa/train_features.csv')

targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
targets[targets.columns[1:]].sum().sort_values()[:20]
known_experiments = ['diuretic',

'autotaxin_inhibitor',                           

'protein_phosphatase_inhibitor',                 

'antiarrhythmic',                                

'retinoid_receptor_antagonist',                  

'nicotinic_receptor_agonist',                    

'atm_kinase_inhibitor',                          

'calcineurin_inhibitor',                         

'lxr_agonist',                                   

'elastase_inhibitor',                            

'steroid',                                       

'leukotriene_inhibitor',                         

'coagulation_factor_inhibitor',                  

'ubiquitin_specific_protease_inhibitor',         

'tropomyosin_receptor_kinase_inhibitor',         

'laxative']
len(known_experiments)
"""

Code by @namanj27 from CatBoost MoA [EDA | Starter] 



https://www.kaggle.com/namanj27/catboost-moa-eda-starter

"""



train_n = pd.merge(train, targets, on='sig_id')



X_train = []

X_train_columns = train_n.columns



for v in train_n.values:

    info = v[:876]

    binary = v[876:]

    index = [k for k, i in enumerate(binary) if i==1]

    

    for i in index:

        for k in range(len(binary)):

            if k==i:

                X_train.append(list(info) + [X_train_columns[876+k]])



X_train = pd.DataFrame(X_train, columns=train.columns.tolist() + ['pred'])
X_train
X_train['Known_Experiment'] = 'No'
# If its in known experiment then add the target to 'experiment'

for i, row in X_train.iterrows():

    if row['pred'] in known_experiments:

        X_train.loc[i, 'Known_Experiment'] = row['pred']
only_known_experiments = X_train[X_train['pred'].isin(known_experiments)]

only_known_experiments
# Each of the targets have 6 unique ids that are the product of the cp_time, cp_dose and cp_type

only_known_experiments.groupby('pred').nunique()[['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
X_train
pca = PCA(n_components=50)

results = pca.fit_transform(X_train[X_train.columns[4:-2]])
fig = px.scatter_3d(x=results[:, 0],

                    y=results[:, 1],

                    z=results[:, 2],

                    opacity=0.4,

                    title="PCA Plot of Known Experiment Targets and All",

                    color=X_train['Known_Experiment'])

fig.show()
only_exp_idx = X_train.index[X_train['Known_Experiment'] != "No"]
fig = px.scatter_3d(x=results[only_exp_idx, 0],

                    y=results[only_exp_idx, 1],

                    z=results[only_exp_idx, 2],

                    opacity=0.8,

                    title="PCA Plot of Known Experiment Targets Only",

                    color=X_train.loc[only_exp_idx, 'Known_Experiment'])

fig.show()
sig_id_testing = only_known_experiments[['sig_id', 'cp_type', 'cp_time', 'cp_dose', 'pred']].reset_index()

del sig_id_testing['index']

sig_id_testing
sig_id_testing['sig_id'] = sig_id_testing['sig_id'].apply(lambda x: x[3:]) # Removing id_
letter_cols = ["Letter "+str(i+1) for i in range(9)]

letter_cols
individual_letters = sig_id_testing.sig_id.str.split("",expand=True)

del individual_letters[0], individual_letters[10] # Remove spaces

individual_letters.columns = letter_cols

individual_letters
# Combination of 2 letters (forward pass)

for i in range(8):

    individual_letters['Letter '+str(i+1)+'+'+'Letter '+str(i+2)] = individual_letters['Letter '+str(i+1)] + individual_letters['Letter '+str(i+2)] 
# Combination of 3 letters (forward pass)

for i in range(7):

    individual_letters['Letter '+str(i+1)+'+'+'Letter '+str(i+2)+'+'+'Letter '+str(i+3)] = individual_letters['Letter '+str(i+1)] + individual_letters['Letter '+str(i+2)] + individual_letters['Letter '+str(i+3)] 
# Combination of 4 letters (forward pass)

for i in range(6):

    individual_letters['Letter '+str(i+1)+'+'+'Letter '+str(i+2)+'+'+'Letter '+str(i+3)+'+'+'Letter '+str(i+4)] = individual_letters['Letter '+str(i+1)] + individual_letters['Letter '+str(i+2)] + individual_letters['Letter '+str(i+3)] + individual_letters['Letter '+str(i+4)] 
individual_letters
sig_id_testing = pd.concat([sig_id_testing, individual_letters], axis=1).reset_index()
del sig_id_testing['index']

sig_id_testing
categorical_ = sig_id_testing.copy()



for column in categorical_.columns[1:]:

    categorical_[column] = categorical_[column].astype('category').cat.codes
categorical_
corr = categorical_[categorical_.columns[2:]].corr()

corr[['cp_time', 'cp_dose', 'pred']].style.background_gradient(cmap='coolwarm').set_precision(2)
# There are 4/5/6 unique letters for each prediction so this doesnt really help isolate anything

# Using the sig_id to cluster 6 at a time may not help either



sig_id_testing[['Letter 4', 'pred']].groupby('pred').agg(['nunique'])