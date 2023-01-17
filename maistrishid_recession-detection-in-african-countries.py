import random

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import eli5

from eli5.sklearn import PermutationImportance
pd.set_option('display.max_columns', 60)
data = pd.read_csv('/kaggle/input/african-country-recession-dataset-2000-to-2017/africa_recession.csv')

data.head(10)
data.info()
'Dataset contains nulls: {}'.format(data.isnull().any().any())
sns.distplot(a=data.growthbucket, kde=False)
plt.figure(figsize=(15, 13))

sns.heatmap(data=data.corr())
X_usampled, y_usampled = RandomUnderSampler(random_state=0).fit_resample(

    data.drop(columns=['growthbucket']), 

    data.growthbucket

)



under_X = pd.DataFrame(

    X_usampled, 

    columns=data.drop(columns=['growthbucket']).columns

)

under_y = pd.DataFrame(y_usampled, columns=['growthbucket'])

under_data = under_X.join(under_y)
plt.figure(figsize=(15, 13))



plt.subplot(3, 2, 1)

sns.swarmplot(x=under_data.growthbucket, y=under_data.rdana)



plt.subplot(3, 2, 2)

sns.swarmplot(x=under_data.growthbucket, y=under_data.emp)



plt.subplot(3, 2, 3)

sns.swarmplot(x=under_data.growthbucket, y=under_data.csh_g)



plt.subplot(3, 2, 4)

sns.swarmplot(x=under_data.growthbucket, y=under_data.pl_c)



plt.subplot(3, 2, 5)

sns.swarmplot(x=under_data.growthbucket, y=under_data.total_change)
def test_with_tts(model, data, smpl_strat='rusamp', random_state=0):

    

    X_train, X_test, y_train, y_test = train_test_split(

        data.drop('growthbucket', axis=1), 

        data.growthbucket, 

        random_state=random_state

    )

    if smpl_strat == 'rusamp':

        rus = RandomUnderSampler(random_state=0)

        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    elif smpl_strat == 'rosamp':

        ros = RandomOverSampler(random_state=0)

        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    elif smpl_strat == 'smote':

        smote = SMOTE(random_state=0)

        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)        

    X_resampled = pd.DataFrame(X_resampled, columns=X_test.columns)

    

    model.fit(X_resampled, y_resampled)

    preds = model.predict(X_test)

    y_test.name = 'expected'

    results = pd.concat(

        [y_test, pd.Series(preds, index=y_test.index, name='got')], 

        axis=1

    )



    recessed = results[results.expected == 1]

    non_recessed = results[results.expected == 0]

    acc_rec = accuracy_score(recessed.expected, recessed.got)

    acc_nrec = accuracy_score(non_recessed.expected, non_recessed.got)

    

    return (acc_rec, acc_nrec)



def test_with_mc(n_it, model, data, smpl_strat='rusamp'):

    random.seed(0)

    

    results_rec = []

    results_nrec = []

    for i in range(n_it):

        r_r, r_n = test_with_tts(

            model, 

            data, 

            smpl_strat, 

            i*random.randint(0,1e2) # randomises train_test_split

        )

        results_rec.append(r_r)

        results_nrec.append(r_n)

    acc_rec = sum(results_rec)/len(results_rec)

    acc_nrec = sum(results_nrec)/len(results_nrec)

    return (acc_rec, acc_nrec)
def print_acc_res(a_r):

    print('Recessed classification accuracy: {}%'.format(round(a_r[0], 2)))

    print('None-recessed classification accuracy: {}%'.format(round(a_r[1], 2)))
xgb_us_full = test_with_mc(

    20, 

    XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=0), 

    data, 

    smpl_strat='rusamp'

)



print_acc_res(xgb_us_full)
xgb_os_full = test_with_mc(

    20, 

    XGBClassifier(n_estimators=500, learning_rate=0.001, random_state=0), 

    data, 

    smpl_strat='rosamp'

)



print_acc_res(xgb_os_full)
xgb_smote_full = test_with_mc(

    20, 

    XGBClassifier(n_estimators=500, learning_rate=0.001, random_state=0), 

    data, 

    smpl_strat='smote'

)



print_acc_res(xgb_smote_full)
rf_us_full = test_with_mc(

    20, 

    RandomForestClassifier(random_state=0), 

    data, 

    smpl_strat='rusamp'

)



print_acc_res(rf_us_full)
rf_os_full = test_with_mc(

    20, 

    RandomForestClassifier(max_depth=3, random_state=0), 

    data, 

    smpl_strat='rosamp'

)



print_acc_res(rf_os_full)
rf_smote_full = test_with_mc(

    20, 

    RandomForestClassifier(max_depth=3, random_state=0), 

    data, 

    smpl_strat='smote'

)



print_acc_res(rf_smote_full)
rid_us_full = test_with_mc(

    20, 

    RidgeClassifier(alpha=1.0, random_state=0), 

    data, 

    smpl_strat='rusamp'

)



print_acc_res(rid_us_full)
rid_os_full = test_with_mc(

    20, 

    RidgeClassifier(alpha=1.0, random_state=0), 

    data, 

    smpl_strat='rosamp'

)



print_acc_res(rid_os_full)
rid_smote_full = test_with_mc(

    20, 

    RidgeClassifier(alpha=10.0, random_state=0), 

    data, 

    smpl_strat='smote'

)



print_acc_res(rid_smote_full)
X = data.drop(columns=['growthbucket'])

y = data.growthbucket



X_resampled, y_resampled = SMOTE(random_state=0).fit_resample(X, y)



model = XGBClassifier(n_estimators=500, learning_rate=0.001, random_state=0)

model.fit(X_resampled,y_resampled)

permi = PermutationImportance(model, random_state=0).fit(

    X_resampled,

    y_resampled

)

eli5.show_weights(permi, feature_names=X.columns.tolist(), top=100)
important_columns = [

    'rdana',

    'cwtfp',

    'pl_x',

    'pl_c',

    'pl_n',

    'rwtfpna',

    'energy',

    'csh_r',

    'metals_minerals_change',

    'metals_minerals',

    'agriculture_change',

    'xr',

    'cn',

]

important_columns += ['growthbucket']



important_data = data[important_columns]



xgb_smote_imp = test_with_mc(

    20, 

    XGBClassifier(n_estimators=500, learning_rate=0.001, random_state=0),

    important_data, 

    smpl_strat='smote'

)



print_acc_res(xgb_smote_imp)
sns.swarmplot(x=under_data.growthbucket, y=under_data.rdana)
plt.figure(figsize=(15, 4))



plt.subplot(1, 2, 1)

sns.swarmplot(x=under_data.growthbucket, y=under_data.cwtfp)



plt.subplot(1, 2, 2)

sns.swarmplot(x=under_data.growthbucket, y=under_data.rwtfpna)
plt.figure(figsize=(15, 13))



plt.subplot(3, 2, 1)

sns.swarmplot(x=under_data.growthbucket, y=under_data.pl_x)



plt.subplot(3, 2, 2)

sns.swarmplot(x=under_data.growthbucket, y=under_data.pl_c)



plt.subplot(3, 2, 3)

sns.swarmplot(x=under_data.growthbucket, y=under_data.pl_n)



plt.subplot(3, 2, 4)

sns.swarmplot(x=under_data.growthbucket, y=under_data.energy)



plt.subplot(3, 2, 5)

sns.swarmplot(x=under_data.growthbucket, y=under_data.metals_minerals)
plt.figure(figsize=(15, 4))



plt.subplot(1, 2, 1)

sns.swarmplot(x=under_data.growthbucket, y=under_data.metals_minerals_change)



plt.subplot(1, 2, 2)

sns.swarmplot(x=under_data.growthbucket, y=under_data.agriculture_change)
sns.swarmplot(x=under_data.growthbucket, y=under_data.csh_r)
sns.swarmplot(x=under_data.growthbucket, y=under_data.xr)
sns.swarmplot(x=under_data.growthbucket, y=under_data.cn)
f_data = important_data.copy()



f_data['rdana_to_pl_x'] = f_data['rdana']/f_data['pl_x']

f_data['energy_to_rdana'] = f_data['energy']/f_data['rdana'] 

f_data['xr_to_cwtfp'] = f_data['xr']/f_data['cwtfp'] 



xgb_smote_eng = test_with_mc(

    20, 

    XGBClassifier(n_estimators=500, learning_rate=0.001, random_state=0),

    f_data, 

    smpl_strat='smote'

)



print_acc_res(xgb_smote_eng)