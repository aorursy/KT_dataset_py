# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
CATEGORICAL_COLUMNS = ['line_stat', 'serv_type', 'serv_code',

                       'bandwidth', 'term_reas_code', 'term_reas_desc',

                       'with_phone_service', 'current_mth_churn']

NUMERIC_COLUMNS = ['contract_month', 'ce_expiry', 'secured_revenue', 'complaint_cnt']



df = pd.read_csv('/kaggle/input/broadband-customers-base-churn-analysis/bbs_cust_base_scfy_20200210.csv').assign(complaint_cnt = lambda df: pd.to_numeric(df.complaint_cnt, 'coerce'))

df.loc[:, NUMERIC_COLUMNS] = df.loc[:, NUMERIC_COLUMNS].astype(np.float32).pipe(lambda df: df.fillna(df.mean())).pipe(lambda df: (df - df.mean())/df.std())

df.loc[:, CATEGORICAL_COLUMNS] = df.loc[:, CATEGORICAL_COLUMNS].astype(str).applymap(str).fillna('')

df = df.groupby('churn').apply(lambda df: df.sample(df.churn.value_counts().min()))

df.head()
from sklearn.model_selection import train_test_split



def get_labels(x: pd.Series) -> pd.Series:

    """

    Converts strings to unqiue ints for use in Pytorch Embedding

    """

    labels, levels = pd.factorize(x)

    return pd.Series(labels, name=x.name, index=x.index)



X, E, y = (df

           .loc[:, NUMERIC_COLUMNS]

           .astype('float32')

           .join(pd.get_dummies(df.loc[:, CATEGORICAL_COLUMNS])),

           df

           .loc[:, NUMERIC_COLUMNS]

           .astype('float32')

           .join(df.loc[:, CATEGORICAL_COLUMNS].apply(get_labels)),

           df.churn == 'Y')



X_train, X_valid, E_train, E_valid, y_train, y_valid = train_test_split(X.to_numpy(), E.to_numpy(), y.to_numpy())
! pip install --quiet pytorch-tabnet
from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.metrics import accuracy_score



cat_idxs = [E.columns.get_loc(c) for c in E.select_dtypes(exclude='float32').columns]



clf = TabNetClassifier(cat_idxs=cat_idxs, cat_emb_dim=1)

clf.fit(E_train, y_train, E_valid, y_valid, max_epochs=15)

preds = clf.predict(E_valid)



tabnet_accuracy = accuracy_score(y_valid, preds)

tabnet_accuracy
pd.Series(clf.feature_importances_, index=E.columns).plot.bar(title=f'TabNet Global Feature Importances ({round(tabnet_accuracy*100, 2)}% Accuracy)')
import matplotlib.pyplot as plt

explain_matrix, masks = clf.explain(E_valid)



fig, axs = plt.subplots(1, 3, figsize=(20,20))



for i in range(3):

    axs[i].imshow(masks[i][:50])

    axs[i].set_title(f"mask {i}")
print(clf.network)
tabnet_parameters = sum([param.nelement() for param in clf.network.parameters()])

print(f'# parameters: {tabnet_parameters}')

from catboost import CatBoostClassifier



C_train = pd.DataFrame(E_train, columns=E.columns)

C_train.loc[:, CATEGORICAL_COLUMNS] = C_train.loc[:, CATEGORICAL_COLUMNS].astype(str)
%time

cat = CatBoostClassifier(verbose=0, 

                         task_type="GPU")

cat.fit(E_train, y_train)
C_valid = pd.DataFrame(E_valid, columns=E.columns)

C_valid.loc[:, CATEGORICAL_COLUMNS] = C_valid.loc[:, CATEGORICAL_COLUMNS].astype(str)
cat_pred = cat.predict(C_valid) == 'True'

catboost_accuracy = accuracy_score(y_valid, cat_pred)

catboost_accuracy
pd.Series(cat.feature_importances_, index=E.columns).plot.bar(title=f'Catboost Global Feature Importances ({round(catboost_accuracy*100, 2)}% Accuracy)')
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
# lazy line search

for h in range(25):

    equivalent_parameters = (X.shape[1] * h  + h) + (h + 1)

    if  equivalent_parameters >= tabnet_parameters:

        break

h
pipeline = Pipeline([('scale', StandardScaler()), 

                     ('mlp', MLPClassifier(hidden_layer_sizes=(h, )))])

pipeline.fit(X_train, y_train)
sum([np.sum(l**0) for l in pipeline.named_steps['mlp'].coefs_ ] + [np.sum(l**0) for l in pipeline.named_steps['mlp'].intercepts_ ] )
mlp_y_pred = pipeline.predict(X_valid)

accuracy_score(y_true=y_valid, y_pred=mlp_y_pred)
# lazy line search

for j in range(25):

    equivalent_parameters = (X.shape[1] * j  + j) + (j*j + j) + (j*j + j) + (j + 1)

    if  equivalent_parameters >= tabnet_parameters:

        break

j
pipeline = Pipeline([('scale', StandardScaler()), 

                     ('mlp', MLPClassifier(hidden_layer_sizes=(j, j, j, )))])

pipeline.fit(X_train, y_train)
accuracy_score(y_true=y_valid, y_pred=mlp_y_pred)
sum([np.sum(l**0) for l in pipeline.named_steps['mlp'].coefs_ ] + [np.sum(l**0) for l in pipeline.named_steps['mlp'].intercepts_ ] )