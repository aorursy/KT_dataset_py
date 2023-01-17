# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installedimport numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import networkx as nx

import graph_utils

import custom_lstm_cell 

import evolve_graph_network as egcn

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics



id_time=["txId", "time_step"]

feature_names = ['feature_'+str(i) for i in range(1,166)]

column_names = id_time + feature_names

elliptic_classes = pd.read_csv('/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')

elliptic_classes.columns = ['txId', 'class_label']

elliptic_edgelist = pd.read_csv('/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')

elliptic_features = pd.read_csv('/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_bitcoin_dataset/elliptic_txs_features.csv', names=column_names)
import shutil

shutil.os.mkdir('/kaggle/working/base_model')

shutil.os.mkdir('/kaggle/working/retask_model')

shutil.os.mkdir('/kaggle/working/finetune_model')
time_steps = list(range(1,17))

graphs = []

for time_step in time_steps:

    extract_nodes = list(set(elliptic_features[elliptic_features['time_step']==time_step]['txId'].values.tolist()))

    edgelist_extract = elliptic_edgelist[elliptic_edgelist['txId1'].isin(extract_nodes) & elliptic_edgelist['txId2'].isin(extract_nodes)].values.tolist()

    edgelist = [tuple(row) for row in edgelist_extract]

    G = nx.DiGraph()

    G.add_edges_from(edgelist)

    graphs.append(G)

extract_nodes = list(set(elliptic_features[elliptic_features['time_step'].isin(time_steps)]['txId'].values.tolist()))
elliptic_classes_ext = elliptic_classes[elliptic_classes['txId'].isin(extract_nodes) & elliptic_classes['class_label'].isin(['1','2'])].reset_index().drop(columns=['index'])

node_list = [node[0] for node in elliptic_classes_ext.values.tolist()]



np.random.seed(1234)

nodes_num = len(node_list)

permutation_indices = np.random.permutation(nodes_num).tolist()

permutation_nodes = [node_list[x] for x in permutation_indices]



train_num = int(nodes_num*0.8)

test_num = int(nodes_num*0.2)



data_predict ={'features': {'dataframe': elliptic_features,

                            'drop_columns':['time_step']}, 

               'train_nodes':  permutation_nodes[:train_num],

               'test_nodes':   permutation_nodes[train_num:(train_num+test_num)],

               'node_column':  'txId'}



print("train num:", train_num, "| test num:",test_num)
elliptic_classes_ext = elliptic_classes[elliptic_classes['txId'].isin(extract_nodes) & elliptic_classes['class_label'].isin(['1','2'])].reset_index().drop(columns=['index'])

label_licit = elliptic_classes_ext[elliptic_classes_ext['class_label']=='2']

label_illicit = elliptic_classes_ext[elliptic_classes_ext['class_label']=='1']

num_illicit = len(label_illicit)

label_licit_sample = label_licit.sample(n=num_illicit, random_state=1)

label_df_balance = pd.concat([label_licit_sample, label_illicit], ignore_index=True)



node_list = [node[0] for node in label_df_balance.values.tolist()]



np.random.seed(1234)

nodes_num = len(node_list)

permutation_indices = np.random.permutation(nodes_num).tolist()

permutation_nodes = [node_list[x] for x in permutation_indices]

elliptic_classes_ext_reindex = label_df_balance.reindex(permutation_indices)

permutation_labels = [(float(node[1])-1) for node in elliptic_classes_ext_reindex.values.tolist()]

permutation_labels = np.expand_dims(permutation_labels,1)



train_num = int(nodes_num*0.8)

test_num = int(nodes_num*0.2)



data_classify ={'features': {'dataframe': elliptic_features,

                             'drop_columns':['time_step']}, 

                'train_nodes':  permutation_nodes[:train_num],

                'train_labels': permutation_labels[:train_num],

                'test_nodes':   permutation_nodes[train_num:(train_num+test_num)],

                'test_labels':  permutation_labels[train_num:(train_num+test_num)],

                'node_column':  'txId'}



print("num train:", train_num, "| num test:", test_num)
data_classify_rus ={'features': {'dataframe': elliptic_features,

                                 'drop_columns':['time_step']}, 

                    'data_label': {'dataframe': elliptic_classes,

                                   'time_steps':time_steps},

                    'node_column': 'txId'}
tf.compat.v1.disable_eager_execution()

GCN = egcn.evolve_graph_conv_nn(batch_size=70, 

                                neighbor_samples=[20,10], 

                                num_hiddens=[20,20], 

                                num_features=165, 

                                num_labels=1,

                                seq_len = len(time_steps)-1)
#losses_base = GCN.train_evolve_gcn_predict(data_predict, 

#              graphs, 100, verbose=False,

#              save='/kaggle/working/base_model/base_evolve_gcn.ckpt')
#training_mse = losses_base['training_losses']

#testing_mse = losses_base['testing_losses']

training_mse = np.load("/kaggle/input/evolve-gcn-base-model/gcn_training_mse.npy").tolist()

testing_mse = np.load("/kaggle/input/evolve-gcn-base-model/gcn_testing_mse.npy").tolist()



fig, ax_mse = plt.subplots(1, 1, figsize=(8, 8))

ax_mse.set_ylabel("Mean Squred Error (MSE.)", fontsize=16)

ax_mse.set_xlabel("Epoches", fontsize=16)

ax_mse.plot(training_mse, 'C1', label="Train MSE.")

ax_mse.plot(testing_mse, 'C0', label="Test MES.")

ax_mse.grid()

ax_mse.set_title('Mean Squared Error over Epoches', fontsize=16)

ax_mse.legend(loc="upper right", prop=dict(size=14))

fig.show()
losses_retask = GCN.train_evolve_gcn_classify(data_classify_rus, 

                graphs, 50, verbose=False,

                pretrain_model = '/kaggle/input/evolve-gcn-base-model/base_evolve_gcn.ckpt',

                #'/kaggle/working/base_model/base_evolve_gcn.ckpt',

                save='/kaggle/working/retask_model/retask_evolve_gcn.ckpt',

                retask = True)
training_losses = losses_retask['training_losses']

training_accuracy = losses_retask['training_acces']



testing_losses = losses_retask['testing_losses']

testing_accuracy = losses_retask['testing_acces']



fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(18, 8))



ax_loss.set_ylabel("Cross Entropy", fontsize=16)

ax_loss.set_xlabel("Epoches", fontsize=16)

ax_loss.set_title("Cross Entropy over Epoches in Re-Task Phase", fontsize=16)

ax_loss.plot(training_losses, 'C1', label="Train Cross Entropy")

ax_loss.plot(testing_losses, 'C0', label="Test Cross Entropy")

ax_loss.legend(loc="upper right", prop=dict(size=14))

ax_loss.grid()



ax_acc.set_ylabel("Accuracy", fontsize=16)

ax_acc.set_xlabel("Epoches", fontsize=16)

ax_acc.set_title("Accuracy over Epoches in Re-Task Phase", fontsize=16)

ax_acc.plot(training_accuracy, 'C1', label="Train Accuracy")

ax_acc.plot(testing_accuracy, 'C0', label="Test Accuracy")

ax_acc.legend(loc="lower right", prop=dict(size=14))

ax_acc.grid()



fig.show()
train_nodes_set = losses_retask['training_nodes']

test_nodes_set = losses_retask['testing_nodes']



train_node_label = elliptic_classes[elliptic_classes['txId'].isin(train_nodes_set)].reset_index().drop(columns=['index'])

test_node_label = elliptic_classes[elliptic_classes['txId'].isin(test_nodes_set)].reset_index().drop(columns=['index'])



train_nodes =  [node[0] for node in train_node_label.values.tolist()]

train_labels = [(float(node[1])-1) for node in train_node_label.values.tolist()]



test_nodes =  [node[0] for node in test_node_label.values.tolist()]

test_labels = [(float(node[1])-1) for node in test_node_label.values.tolist()]



train_data = {'features': {'dataframe': elliptic_features,

                           'drop_columns':['time_step']}, 

              'nodes':  train_nodes,

              'node_column':  'txId'}



test_data = {'features': {'dataframe': elliptic_features,

                          'drop_columns':['time_step']}, 

             'nodes':  test_nodes,

             'node_column':  'txId'}



rt_train_class_socres = GCN.feedforward(train_data, graphs, 

                        model_path='/kaggle/working/retask_model/retask_evolve_gcn.ckpt')



rt_test_class_socres  = GCN.feedforward(test_data, graphs, 

                        model_path='/kaggle/working/retask_model/retask_evolve_gcn.ckpt')
from random import sample 

train_scores = [x[0] for x in rt_train_class_socres]

test_scores = [x[0] for x in rt_test_class_socres]



train_scores_illicit = [train_scores[i] for i, label in enumerate(train_labels) if label == 0]

train_scores_licit = [train_scores[i] for i, label in enumerate(train_labels) if label == 1]

train_scores_licit_sample = sample(train_scores_licit, len(train_scores_illicit))



test_scores_illicit = [test_scores[i] for i, label in enumerate(test_labels) if label == 0]

test_scores_licit = [test_scores[i] for i, label in enumerate(test_labels) if label == 1]

test_scores_licit_sample = sample(test_scores_licit, len(test_scores_illicit))



y_train_illicit = np.zeros(len(train_scores_illicit))

y_train_licit_sample = np.ones(len(train_scores_licit_sample))

y_test_illicit = np.zeros(len(test_scores_illicit))

y_test_licit_sample = np.ones(len(test_scores_licit_sample))



scores_train_illicit = np.array(train_scores_illicit)

scores_train_licit_sample = np.array(train_scores_licit_sample)

scores_test_illicit = np.array(test_scores_illicit)

scores_test_licit_sample = np.array(test_scores_licit_sample)



scores_train = np.concatenate([scores_train_illicit, scores_train_licit_sample])

scores_test  = np.concatenate([scores_test_illicit,  scores_test_licit_sample])



y_train = np.concatenate([y_train_illicit, y_train_licit_sample])

y_test  = np.concatenate([y_test_illicit,  y_test_licit_sample])
fpr_train, tpr_train, thresholds_train_roc = metrics.roc_curve(y_train+1, scores_train, pos_label=2)

fpr_test, tpr_test, thresholds_test_roc = metrics.roc_curve(y_test+1, scores_test, pos_label=2)

roc_auc_train = metrics.auc(fpr_train, tpr_train)

roc_auc_test = metrics.auc(fpr_test, tpr_test)



precision_train, recall_train, thresholds_train_pr = metrics.precision_recall_curve(y_train, scores_train)

precision_test, recall_test, thresholds_test_pr = metrics.precision_recall_curve(y_test, scores_test)

f1_train = max(2 * (precision_train * recall_train) / (precision_train + recall_train))

f1_test = max(2 * (precision_test * recall_test) / (precision_test + recall_test))



fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(18, 8))

ax_roc.set_xlim([0.0, 1.0])

ax_roc.set_ylim([0.0, 1.05])

ax_roc.set_xlabel('False Positive Rate', fontsize=16)

ax_roc.set_ylabel('True Positive Rate', fontsize=16)

ax_roc.plot(fpr_train, tpr_train, color='C1',

            lw=2, label='train ROC curve (area = %0.2f)' % roc_auc_train)

ax_roc.plot(fpr_test, tpr_test, color='C0',

            lw=2, label='test ROC curve (area = %0.2f)' % roc_auc_test)

ax_roc.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', alpha=0.2)

ax_roc.set_title('Receiver operating characteristic', fontsize=16)

ax_roc.legend(loc="lower right", prop=dict(size=14))

ax_roc.grid()



lines = []

labels = []

f_scores = np.linspace(0.2, 0.8, num=5)

for f_score in f_scores:

    x = np.linspace(0.001, 1)

    y = f_score * x / (2 * x - f_score)

    l, = ax_pr.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2, lw=1)

    ax_pr.annotate('F1={0:0.1f}'.format(f_score), xy=(0.88, y[45] - 0.07), color='gray', size=12)

lines.append(l)

labels.append('iso-F1 curves')



l, = ax_pr.plot(recall_train, precision_train, color='C1',lw=2)

lines.append(l)

labels.append('train Precision-Recall curve (F1 = %0.2f)' % f1_train)



l, = ax_pr.plot(recall_test, precision_test, color='C0', lw=2)

lines.append(l)

labels.append('test Precision-Recall curve (F1 = %0.2f)' % f1_test)



ax_pr.set_xlim([0.0, 1.0])

ax_pr.set_ylim([0.0, 1.05])

ax_pr.set_xlabel('recall', fontsize=16)

ax_pr.set_ylabel('Precision', fontsize=16)

ax_pr.set_title('Precision-Recall', fontsize=16)

ax_pr.legend(lines, labels, loc="lower left", prop=dict(size=14))

ax_pr.grid()



fig.show()
if ('train_nodes' in data_classify_rus.keys()) or ('train_labels' in data_classify_rus.keys()):

    del data_classify_rus['train_nodes']

    del data_classify_rus['train_labels']

    del data_classify_rus['test_nodes']

    del data_classify_rus['test_labels']

losses_finetune = GCN.train_evolve_gcn_classify(data_classify_rus, 

                  graphs, 50, verbose=False,

                  pretrain_model = '/kaggle/working/retask_model/retask_evolve_gcn.ckpt',

                  save='/kaggle/working/finetune_model/finetune_evolve_gcn.ckpt')
training_losses = losses_finetune['training_losses']

training_accuracy = losses_finetune['training_acces']



testing_losses = losses_finetune['testing_losses']

testing_accuracy = losses_finetune['testing_acces']



fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(18, 8))



ax_loss.set_ylabel("Cross Entropy", fontsize=16)

ax_loss.set_xlabel("Epoches", fontsize=16)

ax_loss.set_title("Cross Entropy over Epoches in Fine-Tune Phase", fontsize=16)

ax_loss.plot(training_losses, 'C1', label="Train Cross Entropy")

ax_loss.plot(testing_losses, 'C0', label="Test Cross Entropy")

ax_loss.legend(loc="upper right", prop=dict(size=14))

ax_loss.grid()



ax_acc.set_ylabel("Accuracy", fontsize=16)

ax_acc.set_xlabel("Epoches", fontsize=16)

ax_acc.set_title("Accuracy over Epoches in Fine-Tune Phase", fontsize=16)

ax_acc.plot(training_accuracy, 'C1', label="Train Accuracy")

ax_acc.plot(testing_accuracy, 'C0', label="Test Accuracy")

ax_acc.legend(loc="lower right", prop=dict(size=14))

ax_acc.grid()



fig.show()
ft_train_class_socres = GCN.feedforward(train_data, graphs, 

                        model_path='/kaggle/working/finetune_model/finetune_evolve_gcn.ckpt')



ft_test_class_socres  = GCN.feedforward(test_data, graphs, 

                        model_path='/kaggle/working/finetune_model/finetune_evolve_gcn.ckpt')
from random import sample 

train_scores = [x[0] for x in ft_train_class_socres]

test_scores = [x[0] for x in ft_test_class_socres]



train_scores_illicit = [train_scores[i] for i, label in enumerate(train_labels) if label == 0]

train_scores_licit = [train_scores[i] for i, label in enumerate(train_labels) if label == 1]

train_scores_licit_sample = sample(train_scores_licit, len(train_scores_illicit))



test_scores_illicit = [test_scores[i] for i, label in enumerate(test_labels) if label == 0]

test_scores_licit = [test_scores[i] for i, label in enumerate(test_labels) if label == 1]

test_scores_licit_sample = sample(test_scores_licit, len(test_scores_illicit))



y_train_illicit = np.zeros(len(train_scores_illicit))

y_train_licit_sample = np.ones(len(train_scores_licit_sample))

y_test_illicit = np.zeros(len(test_scores_illicit))

y_test_licit_sample = np.ones(len(test_scores_licit_sample))



scores_train_illicit = np.array(train_scores_illicit)

scores_train_licit_sample = np.array(train_scores_licit_sample)

scores_test_illicit = np.array(test_scores_illicit)

scores_test_licit_sample = np.array(test_scores_licit_sample)



scores_train = np.concatenate([scores_train_illicit, scores_train_licit_sample])

scores_test  = np.concatenate([scores_test_illicit,  scores_test_licit_sample])



y_train = np.concatenate([y_train_illicit, y_train_licit_sample])

y_test  = np.concatenate([y_test_illicit,  y_test_licit_sample])
fpr_train, tpr_train, thresholds_train_roc = metrics.roc_curve(y_train+1, scores_train, pos_label=2)

fpr_test, tpr_test, thresholds_test_roc = metrics.roc_curve(y_test+1, scores_test, pos_label=2)

roc_auc_train = metrics.auc(fpr_train, tpr_train)

roc_auc_test = metrics.auc(fpr_test, tpr_test)



precision_train, recall_train, thresholds_train_pr = metrics.precision_recall_curve(y_train, scores_train)

precision_test, recall_test, thresholds_test_pr = metrics.precision_recall_curve(y_test, scores_test)

f1_train = max(2 * (precision_train * recall_train) / (precision_train + recall_train))

f1_test = max(2 * (precision_test * recall_test) / (precision_test + recall_test))



fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(18, 8))

ax_roc.set_xlim([0.0, 1.0])

ax_roc.set_ylim([0.0, 1.05])

ax_roc.set_xlabel('False Positive Rate', fontsize=16)

ax_roc.set_ylabel('True Positive Rate', fontsize=16)

ax_roc.plot(fpr_train, tpr_train, color='C1',

            lw=2, label='train ROC curve (area = %0.2f)' % roc_auc_train)

ax_roc.plot(fpr_test, tpr_test, color='C0',

            lw=2, label='test ROC curve (area = %0.2f)' % roc_auc_test)

ax_roc.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', alpha=0.2)

ax_roc.set_title('Receiver operating characteristic', fontsize=16)

ax_roc.legend(loc="lower right", prop=dict(size=14))

ax_roc.grid()



lines = []

labels = []

f_scores = np.linspace(0.2, 0.8, num=5)

for f_score in f_scores:

    x = np.linspace(0.001, 1)

    y = f_score * x / (2 * x - f_score)

    l, = ax_pr.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2, lw=1)

    ax_pr.annotate('F1={0:0.1f}'.format(f_score), xy=(0.88, y[45] - 0.07), color='gray', size=12)

lines.append(l)

labels.append('iso-F1 curves')



l, = ax_pr.plot(recall_train, precision_train, color='C1',lw=2)

lines.append(l)

labels.append('train Precision-Recall curve (F1 = %0.2f)' % f1_train)



l, = ax_pr.plot(recall_test, precision_test, color='C0', lw=2)

lines.append(l)

labels.append('test Precision-Recall curve (F1 = %0.2f)' % f1_test)



ax_pr.set_xlim([0.0, 1.0])

ax_pr.set_ylim([0.0, 1.05])

ax_pr.set_xlabel('recall', fontsize=16)

ax_pr.set_ylabel('Precision', fontsize=16)

ax_pr.set_title('Precision-Recall', fontsize=16)

ax_pr.legend(lines, labels, loc="lower left", prop=dict(size=14))

ax_pr.grid()



fig.show()