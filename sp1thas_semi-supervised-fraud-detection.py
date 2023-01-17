import numpy as np

import pandas as pd



from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, precision_score

from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from sklearn.decomposition import PCA, IncrementalPCA, LatentDirichletAllocation

from sklearn.manifold import TSNE



from tqdm.notebook import tqdm, trange

from typing import NoReturn, Union, List

from mlxtend.classifier import EnsembleVoteClassifier

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from umap import UMAP
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.head()
# use a dataset sample for development purposes

dev = False
df.groupby(['Class']).Class.count().plot(kind='pie', title='Fraudulent VS Genuine Transactions')
df_s = df.sample(2000)

X = df_s[[_ for _ in df.columns if _ != 'Class']]



pca = PCA(n_components=2)

tsne = TSNE(n_components=2)

ump = UMAP(n_components=2)

ipca = IncrementalPCA(n_components=2)



x_pca = pca.fit_transform(X)

x_tsne = tsne.fit_transform(X)

x_umap = ump.fit_transform(X)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))



sizes = pd.Series(df_s['Class']+1).pow(5) # represent fraud with bigger point



axes[0].scatter(x_pca[:, 0], x_pca[:, 1], s=sizes, c=df_s['Class'].values)

axes[1].scatter(x_tsne[:, 0], x_tsne[:, 1], s=sizes, c=df_s['Class'].values)

axes[2].scatter(x_umap[:, 0], x_umap[:, 1], s=sizes, c=df_s['Class'].values)



axes[0].set_title('PCA')

axes[1].set_title('t-SNE')

axes[2].set_title('UMAP')



fig.tight_layout()

plt.show()
# time and amount scaling

df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))

df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))



df_anom = df[df['Class'] == 1]

df_norm = df[df['Class'] == 0]



if dev:

    df_norm = df_norm.sample(5000, random_state=42)

df_test_norm = df_norm.sample(df_anom.shape[0])

df_test = pd.concat([

    df_anom,

    df_test_norm

])

df_train = df_norm.drop(df_test_norm.index)



feature_cols = [_ for _ in df.columns if _ != 'Class']
X_train = df_train[feature_cols]

y_train = df_train['Class'] # will not be used

X_test = df_test[feature_cols]

y_test = df_test['Class'] # for evaluation

print('''

train: [{:>8} x {:<5}]

 test: [{:>8} x {:<5}]

'''.format(*X_train.shape, *X_test.shape))
def sensitivity_keras(y_true, y_pred):

    """credits: https://datascience.stackexchange.com/a/40746/28592

    

    param:

    y_pred - Predicted labels

    y_true - True labels 

    Returns:

    Specificity score

    """

    neg_y_true = 1 - y_true

    neg_y_pred = 1 - y_pred

    fp = tf.keras.backend.sum(neg_y_true * y_pred)

    tn = tf.keras.backend.sum(neg_y_true * neg_y_pred)

    specificity = tn / (tn + fp + tf.keras.backend.epsilon())

    return specificity
class Scaled_IsolationForest(IsolationForest):

    """The purpose of this class is to transform prediction values from {-1, 1} to {1,0}

    """

    

    def predict(self, X):

        pred = super().predict(X)

        scale_func = np.vectorize(lambda x: 1 if x == -1 else 0)

        return scale_func(pred)



class Scaled_OneClass_SVM(OneClassSVM):

    """Transform OneClassSVM prediction into 0s and 1s

    """

    def predict(self, X):

        return np.array([y==-1 for y in super().predict(X)])

    

class NoveltyDetection_Sequential(tf.keras.models.Sequential):

    """This custom `tf.keras.models.Sequential` based class transforms autoencoder's output into {1,0}.

    Output value is determined based on reproduction (decode) loss. If reproduction loss is more than a threashold then, the inpu sample is considered as anomaly (outlier).

    Based on few experiments, 1.5 is a dissent threashold. Future work: determine threashold using a more sophisticated method.

    """

    def predict(self, x, *args, **kwargs):

        pred = super().predict(x, *args, **kwargs)

        mse = np.mean(np.power(x - pred, 2), axis=1)

        scale_func = np.vectorize(lambda x: 1 if x > 1.5 else 0)

        return scale_func(mse)
# define early stop in order to prevent overfitting and useless training

early_stop = tf.keras.callbacks.EarlyStopping(

    monitor='mse',

    patience=10,

    verbose=1, 

    mode='min',

    restore_best_weights=True,

)



# it's a common practice to store the best model

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    filepath='autoenc.hdf5',

    save_best_only=True,

    monitor='val_loss',

    mode='min',

    verbose=0

)



def get_autoencoder() -> tf.keras.models.Sequential:

    """Builds an autoencoder

    """

    model = NoveltyDetection_Sequential([

        tf.keras.layers.Dense(X_train.shape[1], activation='relu', input_shape=(X_train.shape[1], )),

        # add some noise to prevent overfitting

        tf.keras.layers.GaussianNoise(0.05),

        tf.keras.layers.Dense(2, activation='relu'),

        tf.keras.layers.Dense(X_train.shape[1], activation='relu')

    ])

    model.compile(optimizer='adam', 

                        loss='mse',

                        metrics=['acc', sensitivity_keras])

    return model
clfs = {

    'isolation_forest': {

        'label': 'Isolation Forest',

        'clb': Scaled_IsolationForest,

        'params': {

            'contamination': 'auto',

            'n_estimators': 300

        },

        'predictions': None,

        'model': None

    },

    'ocsvm': {

        'label': 'OneClass SVM',

        'clb': Scaled_OneClass_SVM,

        'params': {

            'kernel': 'rbf',

            'gamma': 0.3,

            'nu': 0.01,

        },

        'prediction': None,

        'model': None

    },

    'auto-encoder': {

        'label': 'Autoncoder',

        'clb': get_autoencoder,

        'params': {},

        'fit_params': {

            'x': X_train, 'y': X_train,

            'validation_split': 0.2,

            'callbacks': [early_stop, checkpoint],

            'epochs': 64,

            'batch_size': 256,

            'verbose': 0

        },

        'predictions': None,

        'model': None

    }

}
%%time



t = trange(len(clfs))

for name in clfs:

    t.set_description(clfs[name]['label'])

    clfs[name]['model'] = clfs[name]['clb'](**clfs[name]['params'])

    if 'fit_params' in clfs[name]:

        clfs[name]['model'].fit(**clfs[name].get('fit_params', {}))

    else:

        clfs[name]['model'].fit(X_train)

    clfs[name]['predictions'] = clfs[name]['model'].predict(X_test)

    t.update()

t.close()
def print_eval_metrics(y_true, y_pred, name='', header=True):

    """Function for printing purposes

    """

    if header:

        print('{:>20}\t{:>10}\t{:>10}\t{:>8}\t{:>5}'.format('Algorith', 'Accuracy', 'Recall', 'Precision', 'f1'))

    acc = accuracy_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    prec = precision_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    print('{:>20}\t{:>1.8f}\t{:>1.8f}\t{:>1.6f}\t{:>1.3f}'.format(

        name, acc, recall, prec, f1

    ))
y_preds = np.column_stack([clfs[_]['predictions'] for _ in clfs])

enseble_preds = []
hard_vot = EnsembleVoteClassifier([clfs[_]['model'] for _ in clfs], refit=False)

hard_vot.fit(X_test, y_test)

enseble_preds.append((hard_vot.predict(X_test), 'Hard Voting'))
wei_hard_vot = EnsembleVoteClassifier([clfs[_]['model'] for _ in clfs], weights=[

        0.4,

        0.1,

        0.8

    ], refit=False)

wei_hard_vot.fit(X_test, y_test)

enseble_preds.append((wei_hard_vot.predict(X_test), 'Weighted Hard Voting'))
rf = RandomForestClassifier()



x_tr_ens, x_ts_ens, y_tr_ens, y_ts_ens = train_test_split(y_preds, y_test, test_size=.5)

rf.fit(x_tr_ens, y_tr_ens)
print_header = True

for k, v in clfs.items():

    print_eval_metrics(y_test, v['predictions'], v['label'], print_header)

    print_header = False



print('\n')



for prds, l in enseble_preds:

    print_eval_metrics(y_test, prds, l, print_header)

    print_header = False



print('\n')

    

print_eval_metrics(

    y_ts_ens,

    rf.predict(x_ts_ens),

    'Bleding using RF', False

)