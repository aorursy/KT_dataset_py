import numpy as np
import pandas as pd

train = pd.read_csv('../input/train_transaction.csv')
test = pd.read_csv('../input/test_transaction.csv')
sub = pd.read_csv('../input/sample_submission.csv')
useful_features = list(train.iloc[:, 3:55].columns)

y = train.sort_values('TransactionDT')['isFraud']
X = train.sort_values('TransactionDT')[useful_features]
X_test = test[useful_features]
del train, test
categorical_features = [
    'ProductCD',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2',
    'P_emaildomain',
    'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]

continuous_features = list(filter(lambda x: x not in categorical_features, X))
class ContinuousFeatureConverter:
    def __init__(self, name, feature, log_transform):
        self.name = name
        self.skew = feature.skew()
        self.log_transform = log_transform
        
    def transform(self, feature):
        if self.skew > 1:
            feature = self.log_transform(feature)
        
        mean = feature.mean()
        std = feature.std()
        return (feature - mean)/(std + 1e-6)        
from tqdm.autonotebook import tqdm

feature_converters = {}
continuous_features_processed = []
continuous_features_processed_test = []

for f in tqdm(continuous_features):
    feature = X[f]
    feature_test = X_test[f]
    log = lambda x: np.log10(x + 1 - min(0, x.min()))
    converter = ContinuousFeatureConverter(f, feature, log)
    feature_converters[f] = converter
    continuous_features_processed.append(converter.transform(feature))
    continuous_features_processed_test.append(converter.transform(feature_test))
    
continuous_train = pd.DataFrame({s.name: s for s in continuous_features_processed}).astype(np.float32)
continuous_test = pd.DataFrame({s.name: s for s in continuous_features_processed_test}).astype(np.float32)
continuous_train['isna_sum'] = continuous_train.isna().sum(axis=1)
continuous_test['isna_sum'] = continuous_test.isna().sum(axis=1)

continuous_train['isna_sum'] = (continuous_train['isna_sum'] - continuous_train['isna_sum'].mean())/continuous_train['isna_sum'].std()
continuous_test['isna_sum'] = (continuous_test['isna_sum'] - continuous_test['isna_sum'].mean())/continuous_test['isna_sum'].std()
isna_columns = []
for column in tqdm(continuous_features):
    isna = continuous_train[column].isna()
    if isna.mean() > 0.:
        continuous_train[column + '_isna'] = isna.astype(int)
        continuous_test[column + '_isna'] = continuous_test[column].isna().astype(int)
        isna_columns.append(column)
        
continuous_train = continuous_train.fillna(0.)
continuous_test = continuous_test.fillna(0.)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm.autonotebook import tqdm

def categorical_encode(df_train, df_test, categorical_features, n_values=50):
    df_train = df_train[categorical_features].astype(str)
    df_test = df_test[categorical_features].astype(str)
    
    categories = []
    for column in tqdm(categorical_features):
        categories.append(list(df_train[column].value_counts().iloc[: n_values - 1].index) + ['Other'])
        values2use = categories[-1]
        df_train[column] = df_train[column].apply(lambda x: x if x in values2use else 'Other')
        df_test[column] = df_test[column].apply(lambda x: x if x in values2use else 'Other')
        
    
    ohe = OneHotEncoder(categories=categories)
    ohe.fit(pd.concat([df_train, df_test]))
    df_train = pd.DataFrame(ohe.transform(df_train).toarray()).astype(np.float16)
    df_test = pd.DataFrame(ohe.transform(df_test).toarray()).astype(np.float16)
    return df_train, df_test
train_categorical, test_categorical = categorical_encode(X, X_test, categorical_features)
X = pd.concat([continuous_train, train_categorical], axis=1)
del continuous_train, train_categorical
X_test = pd.concat([continuous_test, test_categorical], axis=1)
del continuous_test, test_categorical

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA()
pca_fit_train_v = pca.fit_transform(X)
pca_fit_test_v = pca.fit_transform(X_test)
X = pd.DataFrame(data = pca_fit_train_v)
X_test = pd.DataFrame(data = pca_fit_test_v)


X.shape
split_ind = int(X.shape[0]*0.8)

X_tr = X.iloc[:split_ind]
X_val = X.iloc[split_ind:]

y_tr = y.iloc[:split_ind]
y_val = y.iloc[split_ind:]

del X
X_tr.shape
import keras
import random
import tensorflow as tf
import keras.backend as K

from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from keras.utils.generic_utils import get_custom_objects
from keras.optimizers import Adam, Nadam
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

np.random.seed(42) # NumPy
random.seed(42) # Python
tf.set_random_seed(42) # Tensorflow
# Compatible with tensorflow backend
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_val: %s' % (str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.mean((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})
get_custom_objects().update({'focal_loss_fn': focal_loss()})
def create_model(loss_fn):
    inps = Input(shape=(X_tr.shape[1],))
    x = keras.layers.Reshape((X_tr.shape[1],1))(inps)
    x = keras.layers.Conv1D(32, 5, activation='elu')(x)
    x = BatchNormalization()(x)
    x = keras.layers.Conv1D(24,1, activation='elu')(x)
    x = BatchNormalization()(x)
    x = keras.layers.Conv1D(16,1, activation='elu')(x)
    x = BatchNormalization()(x)
    x = keras.layers.Conv1D(4,1, activation='elu')(x)
    x = keras.layers.Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inps, outputs=x)
    model.compile(
        optimizer='adam',
        loss=[loss_fn],
        metrics=['accuracy']
        
    )
    #model.summary()
    return model

  
model_bce = create_model('binary_crossentropy')
model_bce.fit(
    X_tr, y_tr, epochs=100, batch_size=2048, validation_data=(X_val, y_val), verbose=True, 
    callbacks=[roc_callback(training_data=(X_val, y_tr), validation_data=(X_val, y_val))]
)
val_preds_bce = model_bce.predict(X_val).flatten()
from scipy.stats import rankdata, spearmanr

print('BCE preds: ', roc_auc_score(y_val, val_preds_bce))
model_bce.fit(X_val, y_val, epochs=2, batch_size=2048, verbose=True)
val_preds_bce = model_bce.predict(X_val).flatten()
print('BCE preds: ', roc_auc_score(y_val, val_preds_bce))
sub.isFraud = rankdata(model_bce.predict(X_test).flatten(), method='dense' )
sub.isFraud = sub.isFraud/sub.isFraud.max()
sub.to_csv('submission.csv', index=False)