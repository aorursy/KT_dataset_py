import warnings, sys

warnings.filterwarnings("ignore")



# Chris's RAPIDS dataset

!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

import scipy

# from sklearn.linear_model import LogisticRegression

# cuml uses GPU so it is much faster than sklearn 

import cuml



import optuna



from sklearn.metrics import log_loss, make_scorer

ftwo_scorer = make_scorer(log_loss)

import pickle
# Read the Dataset



data_train=pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

data_test = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

target_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

target_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

data_train = data_train.drop("sig_id", axis=1)

data_test = data_test.drop("sig_id", axis=1)
# Ytrain=target_scored['5-alpha_reductase_inhibitor']

data_train=data_train[list(data_test)]

all_data=pd.concat((data_train, data_test))

print(data_train.shape, data_test.shape, all_data.shape)



all_data = pd.concat([all_data, pd.get_dummies(all_data['cp_dose'], prefix='cp_dose', dtype=float)],axis=1)

all_data = pd.concat([all_data, pd.get_dummies(all_data['cp_time'], prefix='cp_time', dtype=float)],axis=1)

all_data = pd.concat([all_data, pd.get_dummies(all_data['cp_type'], prefix='cp_type', dtype=float)],axis=1)

all_data = all_data.drop(['cp_dose', 'cp_time', 'cp_type'], axis=1)







all_data.head()
Xtrain=all_data[:len(data_train)]

Xtest=all_data[len(data_train):]





kf=StratifiedKFold(n_splits=5)





# def objective(trial):

#     C=trial.suggest_loguniform('C', 10e-10, 10)

#     model=LogisticRegression(C=C, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)

#     score=-cross_val_score(model, Xtrain, Ytrain, cv=kf, scoring=ftwo_scorer).mean()

#     return score

# study=optuna.create_study()

# study.optimize(objective, n_trials=20)



# print(study.best_params)



# #print(-study.best_value)

# params=study.best_params
# params['C']

# model=LogisticRegression(C=params['C'], class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)

# model.fit(Xtrain, Ytrain)

# predictions=model.predict_proba(Xtest)[:,1]
target_scored.head()
# import pickle



# pick_file_name = "model.pkl"

# with open(pick_file_name, 'wb') as file:

#     pickle.dump(model, file)
# Select First two columns from target(not id)

select = target_scored.iloc[:,1:3]



for i in select:

    Ytrain = select[i]

#     print(column.values)

    

    def objective(trial):

        C=trial.suggest_loguniform('C', 10e-10, 10)

#         model=LogisticRegression(C=C, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)

        model = cuml.linear_model.LogisticRegression(C=C)

        score=-cross_val_score(model, Xtrain, Ytrain, cv=kf, scoring=ftwo_scorer).mean()

        return score

    study=optuna.create_study()

    study.optimize(objective, n_trials=20)

    print(study.best_params)

    params=study.best_params

#     model=LogisticRegression(C=params['C'], class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)

    model = cuml.linear_model.LogisticRegression(C=params['C'])

    model.fit(Xtrain, Ytrain)

    

    pick_file_name = "model"+str(i)+".pkl"

    with open(pick_file_name, 'wb') as file:

        pickle.dump(model, file)

    

    