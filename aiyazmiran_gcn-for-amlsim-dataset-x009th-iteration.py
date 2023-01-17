!pip install -q stellargraph[demos]==1.0.0rc1
import pandas as pd

import numpy as np

import stellargraph as sg

from stellargraph.mapper import PaddedGraphGenerator

from stellargraph.layer import GCNSupervisedGraphClassification

from stellargraph import StellarGraph

from stellargraph import StellarDiGraph

from stellargraph import datasets

from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, LabelEncoder

from sklearn import model_selection

from IPython.display import display, HTML

from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense

from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from stellargraph.mapper import FullBatchNodeGenerator

from stellargraph.layer import GCN



from tensorflow.keras import layers, optimizers, losses, metrics, Model

from sklearn import preprocessing, model_selection

from IPython.display import display, HTML

import matplotlib.pyplot as plt
transactions = pd.read_csv("../input/aml-sim-curated-1k/transactions.csv")

accounts = pd.read_csv("../input/aml-sim-curated-1k/accounts.csv")
def processing_dataframes():

    print("Dropping NaN columns")

    accounts.dropna(axis=1, how='all', inplace=True)

    print("Renaming columns wrt to Stellar Config")

    transactions.columns  = ['tran_id', 'source', 'target', 'tx_type', 'weight',

       'tran_timestamp', 'is_sar', 'alert_id']

    #label encoder performing encoding for all object

    print("Label encoding categorical features")

    le = LabelEncoder()

    for col in transactions.columns:

        if transactions[col].dtype == "O":

            transactions[col] = le.fit_transform(transactions[col].astype(str))   

    le = LabelEncoder()

    for col in accounts.columns:

        if accounts[col].dtype == "O":

            accounts[col] = le.fit_transform(accounts[col].astype(str))   

    print('\n')

    print("--> Account df done!")

    display(accounts.head())

    print("--> Transaction df done!")

    display(transactions.head())    
processing_dataframes()
G = StellarDiGraph(accounts, transactions[['source','target', 'weight']])



node_subjects = accounts['prior_sar_count'].astype(int)



print(G.info())
node_subjects.value_counts().to_frame()
train_subjects, test_subjects = model_selection.train_test_split(

    node_subjects, test_size=None, stratify=node_subjects

)

val_subjects, test_subjects = model_selection.train_test_split(

    test_subjects,test_size=None, stratify=test_subjects

)
# target_encoding = preprocessing.LabelBinarizer()

train_targets = train_subjects

val_targets = val_subjects

test_targets = test_subjects
generator = FullBatchNodeGenerator(G, method="gcn",k=1, sparse=True)

train_gen = generator.flow(train_subjects.index, train_targets)

gcn = GCN(

    layer_sizes=[512, 512], activations=["relu", "relu"], generator=generator, dropout=0.1

)

x_inp, x_out = gcn.in_out_tensors()

predictions = layers.Dense(units=1, activation="softmax")(x_out)

model = Model(inputs=x_inp, outputs=predictions)

model.compile(

    optimizer=optimizers.Adam(lr=0.01),

    loss=losses.categorical_crossentropy,

    metrics=["acc"],

)

val_gen = generator.flow(val_subjects.index, val_targets)

from tensorflow.keras.callbacks import EarlyStopping

es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

history = model.fit(

    train_gen,

    epochs=5,

    validation_data=val_gen,

    verbose=1,

    shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph

    callbacks=[es_callback],

)
sg.utils.plot_history(history)
test_gen = generator.flow(test_subjects.index, test_targets)

test_metrics = model.evaluate(test_gen)

print("\nTest Set Metrics:")

for name, val in zip(model.metrics_names, test_metrics):

    print("\t{}: {:0.4f}".format(name, val))
all_nodes = node_subjects.index

all_gen = generator.flow(all_nodes)

#all_predictions = model.predict(all_gen)
embedding_model = Model(inputs=x_inp, outputs=x_out)

emb = embedding_model.predict(all_gen)

emb.shape
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

transform = TSNE
X = emb.squeeze(0)

X.shape
trans = transform(n_components=2)

X_reduced = trans.fit_transform(X)

X_reduced.shape
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(

    X_reduced[:, 0],

    X_reduced[:, 1],

    c=node_subjects.astype("category").cat.codes,

    cmap="jet",

    alpha=0.7,

)

ax.set(

    aspect="equal",

    xlabel="$X_1$",

    ylabel="$X_2$",

    title=f"{transform.__name__} visualization of GCN embeddings for AMLSim 1k v1",

)
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from xgboost import plot_importance

from sklearn.model_selection import train_test_split
X_trans = transactions.drop(["is_sar",'alert_id','tran_id'], axis=1)

y_trans = transactions['is_sar']

X_train, X_test, y_train, y_test = train_test_split(

    X_trans, y_trans, train_size=0.8, test_size=None, stratify=y_trans

)

clf = XGBClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

plot_importance(clf)

roc_auc_score(y_test, y_pred)
node_embeddings = emb[0]

graph_vectors = pd.DataFrame(data=node_embeddings, index=accounts.index).reset_index()

graph_vectors.rename(columns={'index':'source'}, inplace=True)

X_trans = transactions.drop(["is_sar",'alert_id','tran_id','tran_timestamp'], axis=1)

X_trans = pd.merge(X_trans, graph_vectors, on="source", how='inner')

# graph_vectors.rename(columns={'source':'target'}, inplace=True)

# X_trans = pd.merge(X_trans, graph_vectors, on="target", how='inner')

y_trans = transactions['is_sar']
X_trans
X_train, X_test, y_train, y_test = train_test_split(X_trans, y_trans, train_size=0.8, test_size=None, stratify=y_trans)

clf = XGBClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

roc_auc_score(y_test, y_pred)
plot_importance(clf)