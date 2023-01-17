%matplotlib inline



import gc



from category_encoders.ordinal import OrdinalEncoder

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.metrics import confusion_matrix, precision_recall_curve

from sklearn.metrics import recall_score, classification_report, auc, roc_curve

from sklearn.metrics import precision_recall_fscore_support, f1_score, roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import *

from tqdm import tqdm
input_dir = '../input/ieee-fraud-detection/'

train_id_file = input_dir + 'train_identity.csv'

train_trans_file = input_dir + 'train_transaction.csv'



test_id_file = input_dir + 'test_identity.csv'

test_trans_file = input_dir + 'test_transaction.csv'
df_id = pd.read_csv(train_id_file, index_col='TransactionID')

df_trans = pd.read_csv(train_trans_file, index_col='TransactionID')

train = df_trans.merge(df_id, how='left', left_index=True, right_index=True)

del df_id, df_trans

train.shape
gc.collect()
X = train.drop('isFraud', axis=1)

y = train['isFraud'].copy()

del train

gc.collect()
cat_fea = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',

           'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',

           'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',

           'DeviceType', 'DeviceInfo'] + ['id_' + str(i) for i in range(12, 39)]

num_fea = []

drop_fea = []
cat_fea = list(set(cat_fea) - set(drop_fea))
num_fea += list(X.loc[:, ~X.columns.isin(cat_fea + drop_fea)])
X = X.drop(drop_fea, axis=1)
%%time



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='mean')),

    ('std_scaler', StandardScaler()),

], verbose=True)



X[num_fea] = num_pipeline.fit_transform(X[num_fea], y)
X = X.fillna(-999)
for col in tqdm(num_fea):

    if X[col].dtype == 'float64':

        X[col] = X[col].astype(np.float32)

    if (X[col].dtype == 'int64') or (col in cat_fea):

        X[col] = X[col].astype(np.int32)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
def rounded_accuracy(y_true, y_pred):

    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))



K = keras.backend

def kl_divergence(p, q):

    return p * K.log(p / q) + (1 - p) * K.log((1 - p) / (1 - q))



class KLDivergenceRegularizer(keras.regularizers.Regularizer):

    def __init__(self, weight, target=0.1):

        self.weight = weight

        self.target = target        

        

    def __call__(self, inputs):

        mean_activities = K.mean(inputs)

        return self.weight * (

            kl_divergence(self.target, mean_activities) +

            kl_divergence(1. - self.target, 1. - mean_activities))

    

    def get_config(self):

        return {"weight": self.weight, 'target': self.target}
def sparse_autoencoder(n_input):   

    tf.random.set_random_seed(42)

    np.random.seed(42)



    kld_reg = KLDivergenceRegularizer(weight=0.05, target=0.1)

    sparse_kl_encoder = keras.models.Sequential([

        keras.layers.Dense(100, activation="selu", input_shape=(n_input,), 

                           kernel_initializer='lecun_normal'),

        keras.layers.Dense(300, activation="sigmoid", activity_regularizer=kld_reg)

    ])

    sparse_kl_decoder = keras.models.Sequential([

        keras.layers.Dense(100, activation="selu", input_shape=[300], 

                           kernel_initializer='lecun_normal'),

        keras.layers.Dense(n_input, activation=None),        

    ])

    

    sparse_kl_ae = keras.models.Sequential([sparse_kl_encoder, sparse_kl_decoder])

    

    sparse_kl_ae.compile(loss="mean_squared_error", 

                         optimizer='nadam',

                         metrics=['acc', rounded_accuracy])

    

    return sparse_kl_ae
train = X_train[num_fea]

val = X_val[num_fea]



model = sparse_autoencoder(train.shape[1])

lre = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 

                                        patience=3, 

                                        verbose=1, 

                                        factor=0.5, 

                                        min_lr=0.00001)



es = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=6)



model.fit(train, train,

          callbacks=[lre, es],     

          validation_data=[val, val],

          batch_size=32, 

          epochs=50, 

          verbose=2)
X_val_pred = model.predict(val)

mse = np.mean(np.power(val - X_val_pred, 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': y_val})

error_df.describe()
false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)

roc_auc = auc(false_pos_rate, true_pos_rate,)



plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)

plt.plot([0,1],[0,1], linewidth=5)



plt.xlim([-0.01, 1])

plt.ylim([0, 1.01])

plt.legend(loc='lower right')

plt.title('Receiver operating characteristic curve (ROC)')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)

plt.plot(recall_rt, precision_rt, linewidth=5, label='Precision-Recall curve')

plt.title('Recall vs Precision')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.show()
plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)

plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)

plt.title('Precision and recall for different threshold values')

plt.xlabel('Threshold')

plt.ylabel('Precision/Recall')

plt.legend()

plt.show()
threshold_fixed = 2

groups = error_df.groupby('True_class')

fig, ax = plt.subplots()



for name, group in groups:

    ax.plot(group.index.values, group.Reconstruction_error.values, marker='o', ms=3.5, linestyle='',

            label= "Fraud" if name == 1 else "Normal")    

    

ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')

ax.legend()

plt.title("Reconstruction error for different classes")

plt.ylabel("Reconstruction error")

plt.xlabel("Data point index")

plt.gcf().set_size_inches(15, 10)

plt.show();
y_val_pred = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.True_class, y_val_pred)



plt.figure(figsize=(12, 12))

sns.heatmap(conf_matrix, xticklabels=["Normal","Fraud"], yticklabels=["Normal","Fraud"], annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()
print(classification_report(y_val, y_val_pred))
def min_max_normalization(x):

    x_min = x.min()

    x_max = x.max()

    x_norm = (x-x_min) / (x_max-x_min)

    return x_norm
roc_auc_score(y_val, min_max_normalization(error_df.Reconstruction_error.values))
confusion_matrix(y_val, y_val_pred)