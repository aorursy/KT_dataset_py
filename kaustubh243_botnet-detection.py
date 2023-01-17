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
#dependenies

import seaborn as sns

import pandas as pd

import networkx as nx

import  matplotlib.pyplot as plt

import keras

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RepeatedKFold

from category_encoders.ordinal import OrdinalEncoder

import shap

import tensorflow as tf

from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, average_precision_score, log_loss,f1_score

from skopt import forest_minimize

from sklearn.utils import parallel_backend
#importing the dataset

path = '/kaggle/input/dataset/capture20110810.binetflow.xz'

df = pd.read_csv(path)

df.info()

df.head()
#visualizing network topology of the infected node

df_purb = df[["SrcAddr","DstAddr"]]

df_purb = df_purb[df_purb['SrcAddr'] == '147.32.84.165']

df_purb=df_purb.drop_duplicates()

df_purb.head()

df_new = df_purb 

g = nx.from_pandas_edgelist(df_purb[:1000], source='SrcAddr', target='DstAddr')

pos = nx.spring_layout(g)

nx.draw_networkx(g, pos = pos ,node_size=5, \

    node_color='blue', linewidths=0.01, dpi = 10000, with_labels = False, edge_color = 'grey')

nx.draw_networkx(g.subgraph('147.32.84.165'), pos=pos, node_color='red', with_labels = False, node_size = 10)

#nx.draw_networkx(g.subgraph('147.32.84.59'), pos=pos, node_color='red', with_labels = False, node_size = 10)
def preprocessing(X, train =True):

    X = X.drop(['StartTime', 'SrcAddr', 'Sport', 'DstAddr', 'Dport'], axis = 1)

    X['dTos'] = X['dTos'].fillna(-1)

    X['sTos'] = X['sTos'].fillna(-1)

    X['State'] = X['State'].fillna('None')

    X['Bytes_per_Pkts'] = X['TotBytes'] / X['TotPkts']

    X['Src_Bytes_per_Pkts'] = X['SrcBytes'] / X['TotPkts']

    X['TotPkts_over_Dur'] = X['TotPkts'] / X['Dur']

    X['TotBytes_over_Dur'] = X['TotBytes'] / X['Dur']

    X['SrcBytes_over_Dur'] = X['SrcBytes'] / X['Dur']

    def label_trans(string):

        if 'Botnet' in string:

            return 1

        else:

            return 0

    X['Label'] = X['Label'].apply(label_trans)

    y = X['Label']

    X = X.drop(['Label'], axis = 1)

    encoder = OrdinalEncoder(cols = ['Proto', 'Dir', 'State'])

    if train:

        X = encoder.fit_transform(X)

    else:

        X = encoder.transform(X)

    return X,y

    

    
train_data, test_data = train_test_split(df, test_size = 0.2)
X_training, y_training = preprocessing(train_data)

X_training.head(2)

X_training.info()
print('Percent Botnet: {:.4%} '.format(y_training.mean()))

print('Percent of Normal Nodes: {:.4%}'.format(1 - y_training.mean()))
X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size = 0.5, random_state = 2020)
# def model_opti(params):

#     lr = params[0]

#     leaves = params[1]

#     min_child = params[2]

#     sub = params[3]

#     bytree = params[4]

#     model = LGBMClassifier(learning_rate=lr,num_leaves=leaves, min_child_samples=min_child, subsample= sub,colsample_bytree=bytree,n_estimators=1000,class_weight='balanced',subsample_freq=1,random_state=0)

#     model.fit(X_train, y_train)

#     p = model.predict_proba(X_test)[:,1]

#     print(roc_auc_score(y_test,p))

#     return log_loss(y_test,p)



# space = [(1e-3, 1e-1, 'log-uniform'), (2, 128),(1, 100),(0.05, 1.0),(0.1, 1.0)]

# results_forest = forest_minimize(model_opti, space, verbose=1, n_calls=10, n_random_starts=10)
# from skopt.plots import plot_convergence

# plot_convergence(results_forest)


with parallel_backend('threading'):

    model = RepeatedKFold(n_splits = 10, n_repeats=2, random_state=0)

    result = {}

    counter = 1

    for train_index, test_index in model.split(X_training, y_training):

        X_train, X_test = X_training.iloc[train_index], X_training.iloc[test_index]

        y_train, y_test = y_training.iloc[train_index], y_training.iloc[test_index]

        Classifier = LGBMClassifier(learning_rate=(0.012521721941931553)

                                    ,num_leaves=(126)

                                    , min_child_samples=(4)

                                    , subsample= (0.17881314374565294)

                                    ,colsample_bytree=(0.6606728298036824)

                                    ,n_estimators=1000,class_weight='balanced',subsample_freq=1,random_state=0)

        Classifier.fit(X_test, y_test)

        y_pred = Classifier.predict(X_test)

        probas = Classifier.predict_proba(X_test)[:,1]

        

        result[counter] = {'Average Precision Score': average_precision_score(y_test, probas),

                            'AUC': roc_auc_score(y_test, probas),

                            'Precision': precision_score(y_test, y_pred),

                            'Recall': recall_score(y_test, y_pred)}



        print('Round: ', counter)

        print('Average Precision Score: ', round(average_precision_score(y_test, probas),3))

        print('AUC Score: ', round(roc_auc_score(y_test, probas),3))

        print('Precision Score: ', round(precision_score(y_test, y_pred),3))

        print('Recall Score: ', round(recall_score(y_test, y_pred),3))

        print('\n')

        counter += 1

        

        

        

    
result = pd.DataFrame(result)

result['Mean'] = result.mean(axis=1)

result['STD'] = result.std(axis=1)

result
shap.initjs()
sample_X_train = X_training.sample(frac=0.01, random_state = 1)

sample_y_final = y_training[sample_X_train.index]
explainer = shap.TreeExplainer(Classifier)

shap_values = explainer.shap_values(sample_X_train)

shap.force_plot(explainer.expected_value[1], shap_values[1][95,:], sample_X_train.iloc[95,:])

shap.summary_plot(shap_values[1], sample_X_train, plot_size=(10,10), alpha = .2)
X_f, y_f = preprocessing(test_data)

# encoder = OrdinalEncoder(cols = ['Proto', 'Dir', 'State'])

# X_enc = encoder.transform(X_f)

pred = Classifier.predict(X_f)

print('F1 Score: ', round(f1_score(y_f, pred),3))

print('AUC Score: ', round(roc_auc_score(y_f, pred),3))

print('Precision Score: ', round(precision_score(y_f, pred),3))

print('Recall Score: ', round(recall_score(y_f, pred),3))

print(classification_report(y_f, pred))
print(confusion_matrix(y_f, pred))