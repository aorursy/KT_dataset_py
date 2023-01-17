from typing import Optional, Union, Tuple



import numpy as np

import tensorflow as tf

import tensorflow_probability as tfp

import tensorflow_addons as tfa

import pandas as pd

from sklearn.metrics import accuracy_score





@tf.function

def identity(x):

    return x
class GLUBlock(tf.keras.layers.Layer):

    def __init__(self, units: Optional[int] = None,

                 virtual_batch_size: Optional[int] = 128, 

                 momentum: Optional[float] = 0.02):

        super(GLUBlock, self).__init__()

        self.units = units

        self.virtual_batch_size = virtual_batch_size

        self.momentum = momentum

        

    def build(self, input_shape: tf.TensorShape):

        if self.units is None:

            self.units = input_shape[-1]

            

        self.fc_outout = tf.keras.layers.Dense(self.units, 

                                               use_bias=False)

        self.bn_outout = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, 

                                                            momentum=self.momentum)

        

        self.fc_gate = tf.keras.layers.Dense(self.units, 

                                             use_bias=False)

        self.bn_gate = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, 

                                                          momentum=self.momentum)

        

    def call(self, inputs: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None):

        output = self.bn_outout(self.fc_outout(inputs), 

                                training=training)

        gate = self.bn_gate(self.fc_gate(inputs), 

                            training=training)

    

        return output * tf.keras.activations.sigmoid(gate) # GLU
class FeatureTransformerBlock(tf.keras.layers.Layer):

    def __init__(self, units: Optional[int] = None, virtual_batch_size: Optional[int]=128, 

                 momentum: Optional[float] = 0.02, skip=False):

        super(FeatureTransformerBlock, self).__init__()

        self.units = units

        self.virtual_batch_size = virtual_batch_size

        self.momentum = momentum

        self.skip = skip

        

    def build(self, input_shape: tf.TensorShape):

        if self.units is None:

            self.units = input_shape[-1]

        

        self.initial = GLUBlock(units = self.units, 

                                virtual_batch_size=self.virtual_batch_size, 

                                momentum=self.momentum)

        self.residual =  GLUBlock(units = self.units, 

                                  virtual_batch_size=self.virtual_batch_size, 

                                  momentum=self.momentum)

        

    def call(self, inputs: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None):

        initial = self.initial(inputs, training=training)

        

        if self.skip == True:

            initial += inputs



        residual = self.residual(initial, training=training) # skip

        

        return (initial + residual) * np.sqrt(0.5)
class AttentiveTransformer(tf.keras.layers.Layer):

    def __init__(self, units: Optional[int] = None, virtual_batch_size: Optional[int] = 128, 

                 momentum: Optional[float] = 0.02):

        super(AttentiveTransformer, self).__init__()

        self.units = units

        self.virtual_batch_size = virtual_batch_size

        self.momentum = momentum

        

    def build(self, input_shape: tf.TensorShape):

        if self.units is None:

            self.units = input_shape[-1]

            

        self.fc = tf.keras.layers.Dense(self.units, 

                                        use_bias=False)

        self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, 

                                                     momentum=self.momentum)

        

    def call(self, inputs: Union[tf.Tensor, np.ndarray], priors: Optional[Union[tf.Tensor, np.ndarray]] = None, training: Optional[bool] = None) -> tf.Tensor:

        feature = self.bn(self.fc(inputs), 

                          training=training)

        if priors is None:

            output = feature

        else:

            output = feature * priors

        

        return tfa.activations.sparsemax(output)
class TabNetStep(tf.keras.layers.Layer):

    def __init__(self, units: Optional[int] = None, virtual_batch_size: Optional[int]=128, 

                 momentum: Optional[float] =0.02):

        super(TabNetStep, self).__init__()

        self.units = units

        self.virtual_batch_size = virtual_batch_size

        self.momentum = momentum

        

    def build(self, input_shape: tf.TensorShape):

        if self.units is None:

            self.units = input_shape[-1]

        

        self.unique = FeatureTransformerBlock(units = self.units, 

                                              virtual_batch_size=self.virtual_batch_size, 

                                              momentum=self.momentum,

                                              skip=True)

        self.attention = AttentiveTransformer(units = input_shape[-1], 

                                              virtual_batch_size=self.virtual_batch_size, 

                                              momentum=self.momentum)

        

    def call(self, inputs, shared, priors, training=None) -> Tuple[tf.Tensor]:  

        split = self.unique(shared, training=training)

        keys = self.attention(split, priors, training=training)

        masked = keys * inputs

        

        return split, masked, keys
class TabNetEncoder(tf.keras.layers.Layer):

    def __init__(self, units: int =1, 

                 n_steps: int = 3, 

                 n_features: int = 8,

                 outputs: int = 1, 

                 gamma: float = 1.3,

                 epsilon: float = 1e-8, 

                 sparsity: float = 1e-5, 

                 virtual_batch_size: Optional[int]=128, 

                 momentum: Optional[float] =0.02):

        super(TabNetEncoder, self).__init__()

        

        self.units = units

        self.n_steps = n_steps

        self.n_features = n_features

        self.virtual_batch_size = virtual_batch_size

        self.gamma = gamma

        self.epsilon = epsilon

        self.momentum = momentum

        self.sparsity = sparsity

        

    def build(self, input_shape: tf.TensorShape):            

        self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, 

                                                     momentum=self.momentum)

        self.shared_block = FeatureTransformerBlock(units = self.n_features, 

                                                    virtual_batch_size=self.virtual_batch_size, 

                                                    momentum=self.momentum)        

        self.initial_step = TabNetStep(units = self.n_features, 

                                       virtual_batch_size=self.virtual_batch_size, 

                                       momentum=self.momentum)

        self.steps = [TabNetStep(units = self.n_features, 

                                 virtual_batch_size=self.virtual_batch_size, 

                                 momentum=self.momentum) for _ in range(self.n_steps)]

        self.final = tf.keras.layers.Dense(units = self.units, 

                                           use_bias=False)

    



    def call(self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None) -> Tuple[tf.Tensor]:        

        entropy_loss = 0.

        encoded = 0.

        output = 0.

        importance = 0.

        prior = tf.reduce_mean(tf.ones_like(X), axis=0)

        

        B = prior * self.bn(X, training=training)

        shared = self.shared_block(B, training=training)

        _, masked, keys = self.initial_step(B, shared, prior, training=training)



        for step in self.steps:

            entropy_loss += tf.reduce_mean(tf.reduce_sum(-keys * tf.math.log(keys + self.epsilon), axis=-1)) / tf.cast(self.n_steps, tf.float32)

            prior *= (self.gamma - tf.reduce_mean(keys, axis=0))

            importance += keys

            

            shared = self.shared_block(masked, training=training)

            split, masked, keys = step(B, shared, prior, training=training)

            features = tf.keras.activations.relu(split)

            

            output += features

            encoded += split

            

        self.add_loss(self.sparsity * entropy_loss)

          

        prediction = self.final(output)

        return prediction, encoded, importance
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

           .join(df.loc[:, CATEGORICAL_COLUMNS].apply(get_labels).add(1).astype('int32')),

           df.churn == 'Y')



X_train, X_valid, E_train, E_valid, y_train, y_valid = train_test_split(X.to_numpy(), E, y.to_numpy(), train_size=250000, test_size=250000)
def get_feature(x: pd.DataFrame, dimension=1) -> Union[tf.python.feature_column.NumericColumn, tf.python.feature_column.EmbeddingColumn]:

    if x.dtype == np.float32:

        return tf.feature_column.numeric_column(x.name)

    else:

        return tf.feature_column.embedding_column(

        tf.feature_column.categorical_column_with_identity(x.name, num_buckets=x.max() + 1, default_value=0),

        dimension=dimension)

    

def df_to_dataset(X: pd.DataFrame, y: pd.Series, shuffle=False, batch_size=50000) -> tf.python.data.ops.dataset_ops.TensorSliceDataset:

    ds = tf.data.Dataset.from_tensor_slices((dict(X.copy()), y.copy()))

    if shuffle:

        ds = ds.shuffle(buffer_size=len(X))

    ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return ds



columns = [get_feature(f) for k, f in E_train.iteritems()]

feature_column = tf.keras.layers.DenseFeatures(columns, trainable=True)



train, valid = df_to_dataset(E_train, y_train), df_to_dataset(E_valid, y_valid)
class TabNetClassifier(tf.keras.Model):

    def __init__(self, outputs: int = 1, 

                 n_steps: int = 3, 

                 n_features: int = 8,

                 gamma: float = 1.3, 

                 epsilon: float = 1e-8, 

                 sparsity: float = 1e-5, 

                 feature_column: Optional[tf.keras.layers.DenseFeatures] = None, 

                 pretrained_encoder: Optional[tf.keras.layers.Layer] = None,

                 virtual_batch_size: Optional[int] = 128, 

                 momentum: Optional[float] = 0.02):

        super(TabNetClassifier, self).__init__()

        

        self.outputs = outputs

        self.n_steps = n_steps

        self.n_features = n_features

        self.feature_column = feature_column

        self.pretrained_encoder = pretrained_encoder

        self.virtual_batch_size = virtual_batch_size

        self.gamma = gamma

        self.epsilon = epsilon

        self.momentum = momentum

        self.sparsity = sparsity

        

        if feature_column is None:

            self.feature = tf.keras.layers.Lambda(identity)

        else:

            self.feature = feature_column

            

        if pretrained_encoder is None:

            self.encoder = TabNetEncoder(units=outputs, 

                                        n_steps=n_steps, 

                                        n_features = n_features,

                                        outputs=outputs, 

                                        gamma=gamma, 

                                        epsilon=epsilon, 

                                        sparsity=sparsity,

                                        virtual_batch_size=self.virtual_batch_size, 

                                        momentum=momentum)

        else:

            self.encoder = pretrained_encoder



    def forward(self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None) -> Tuple[tf.Tensor]:

        X = self.feature(X)

        output, encoded, importance = self.encoder(X)

          

        prediction = tf.keras.activations.sigmoid(output)

        return prediction, encoded, importance

    

    def call(self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None) -> tf.Tensor:

        prediction, _, _ = self.forward(X)

        return prediction

    

    def transform(self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None) -> tf.Tensor:

        _, encoded, _ = self.forward(X)

        return encoded

    

    def explain(self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None) -> tf.Tensor:

        _, _, importance = self.forward(X)

        return importance    
m = TabNetClassifier(outputs=1, n_steps=3, n_features = 2, feature_column=feature_column, virtual_batch_size=250)

m.compile(tf.keras.optimizers.Adam(learning_rate=0.025), tf.keras.losses.binary_crossentropy)

m.fit(train, epochs=100)
m.summary()
tf_tabnet_y_pred = m.predict(train)



accuracy_score(y_train, tf_tabnet_y_pred > 0.5)
tf_tabnet_y_pred = m.predict(valid)



accuracy_score(y_valid, tf_tabnet_y_pred > 0.5)
import holoviews as hv

hv.extension('bokeh')



Z_train = m.transform(dict(E_train)).numpy()



hv.Scatter(pd.DataFrame(Z_train, columns=['Component 1', 'Component 2'])

 .assign(label=y_train.astype(str))

 .sample(1000),

  kdims='Component 1', vdims=['Component 2', 'label']).opts(color='label', cmap="Category10", title='Latent feature space')
A_train = m.explain(dict(E_train)).numpy()



pd.Series(A_train.mean(0), index=E.columns).plot.bar(title='Global Importances')
class TabNetDecoder(tf.keras.layers.Layer):

    def __init__(self, units=1, 

                 n_steps = 3, 

                 n_features = 8,

                 outputs = 1, 

                 gamma = 1.3,

                 epsilon = 1e-8, 

                 sparsity = 1e-5, 

                 virtual_batch_size=128, 

                 momentum=0.02):

        super(TabNetDecoder, self).__init__()

        

        self.units = units

        self.n_steps = n_steps

        self.n_features = n_features

        self.virtual_batch_size = virtual_batch_size

        self.momentum = momentum

        

    def build(self, input_shape: tf.TensorShape):

        self.shared_block = FeatureTransformerBlock(units = self.n_features, 

                                                    virtual_batch_size=self.virtual_batch_size, 

                                                    momentum=self.momentum)

        self.steps = [FeatureTransformerBlock(units = self.n_features,

                                              virtual_batch_size=self.virtual_batch_size, 

                                              momentum=self.momentum) for _ in range(self.n_steps)]

        self.fc = [tf.keras.layers.Dense(units = self.units) for _ in range(self.n_steps)]

    



    def call(self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None) -> tf.Tensor:

        decoded = 0.

        

        for ftb, fc in zip(self.steps, self.fc):

            shared = self.shared_block(X, training=training)

            feature = ftb(shared, training=training)

            output = fc(feature)

            

            decoded += output

        return decoded
class TabNetAutoencoder(tf.keras.Model):

    def __init__(self, outputs: int = 1, 

                 inputs: int = 12,

                 n_steps: int  = 3, 

                 n_features: int  = 8,

                 gamma: float = 1.3, 

                 epsilon: float = 1e-8, 

                 sparsity: float = 1e-5, 

                 feature_column: Optional[tf.keras.layers.DenseFeatures] = None, 

                 virtual_batch_size: Optional[int] = 128, 

                 momentum: Optional[float] = 0.02):

        super(TabNetAutoencoder, self).__init__()

        

        self.outputs = outputs

        self.inputs = inputs

        self.n_steps = n_steps

        self.n_features = n_features

        self.feature_column = feature_column

        self.virtual_batch_size = virtual_batch_size

        self.gamma = gamma

        self.epsilon = epsilon

        self.momentum = momentum

        self.sparsity = sparsity

        

        if feature_column is None:

            self.feature = tf.keras.layers.Lambda(identity)

        else:

            self.feature = feature_column

            

        self.encoder = TabNetEncoder(units=outputs, 

                                    n_steps=n_steps, 

                                    n_features = n_features,

                                    outputs=outputs, 

                                    gamma=gamma, 

                                    epsilon=epsilon, 

                                    sparsity=sparsity,

                                    virtual_batch_size=self.virtual_batch_size, 

                                    momentum=momentum)

        

        self.decoder = TabNetDecoder(units=inputs, 

                                     n_steps=n_steps, 

                                     n_features = n_features,

                                     virtual_batch_size=self.virtual_batch_size, 

                                     momentum=momentum)

        

        self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, 

                                                     momentum=momentum)

        

        self.do = tf.keras.layers.Dropout(0.25)



    def forward(self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None) -> Tuple[tf.Tensor]:

        X = self.feature(X)

        X = self.bn(X)

        

        # training mask

        M = self.do(tf.ones_like(X), training=training)

        D = X*M

        

        #encoder

        output, encoded, importance = self.encoder(D)

        prediction = tf.keras.activations.sigmoid(output)        

        

        return prediction, encoded, importance, X, M

    

    def call(self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None) -> tf.Tensor:

        # encode

        prediction, encoded, _, X, M = self.forward(X)

        T = X * (1 - M)



        #decode

        reconstruction = self.decoder(encoded)

        

        #loss

        loss  = tf.reduce_mean(tf.where(M != 0., tf.square(T-reconstruction), tf.zeros_like(reconstruction)))

        

        self.add_loss(loss)

        

        return prediction

    

    def transform(self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None) -> tf.Tensor:

        _, encoded, _, _, _ = self.forward(X)

        return encoded

    

    def explain(self, X: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None) -> tf.Tensor:

        _, _, importance, _, _ = self.forward(X)

        return importance
@tf.function

def dummy_loss(y, t):

    return 0.
ae = TabNetAutoencoder(outputs=1, inputs=12, n_steps=3, n_features = 2, feature_column=feature_column, virtual_batch_size=250)

ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss=dummy_loss)

ae.fit(train, epochs=100)
ae.summary()
import holoviews as hv

hv.extension('bokeh')



Z_train = ae.transform(dict(E_train)).numpy()



hv.Scatter(pd.DataFrame(Z_train, columns=['Component 1', 'Component 2'])

 .assign(label=y_train.astype(str))

 .sample(1000),

  kdims='Component 1', vdims=['Component 2', 'label']).opts(color='label', cmap="Category10", title='Latent feature space')
AE_train = ae.explain(dict(E_train)).numpy()



pd.Series(AE_train.mean(0), index=E.columns).plot.bar(title='Global Importances')
ae.layers[1]
pm = TabNetClassifier(outputs=1, n_steps=3, n_features = 2, feature_column=feature_column, pretrained_encoder=ae.layers[1], virtual_batch_size=250)

pm.compile(tf.keras.optimizers.Adam(learning_rate=0.05), tf.keras.losses.binary_crossentropy)

pm.fit(train, epochs=150) 
tf_tabnet_y_pred = pm.predict(train)



accuracy_score(y_train, tf_tabnet_y_pred > 0.5)
tf_tabnet_y_pred = pm.predict(valid)



accuracy_score(y_valid, tf_tabnet_y_pred > 0.5)
Z_train = pm.transform(dict(E_train)).numpy()



hv.Scatter(pd.DataFrame(Z_train, columns=['Component 1', 'Component 2'])

 .assign(label=y_train.astype(str))

 .sample(1000),

  kdims='Component 1', vdims=['Component 2', 'label']).opts(color='label', cmap="Category10", title='Latent feature space')
AE_train = pm.explain(dict(E_train)).numpy()



pd.Series(AE_train.mean(0), index=E.columns).plot.bar(title='Global Importances')