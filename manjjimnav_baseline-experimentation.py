# Import usual stuff

import matplotlib.pyplot as plt

from functools import partial

import pandas as pd

import random as rn

import numpy as np

import pickle

import copy

import os



# Import preprocessing and metrics functions

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import PowerTransformer

from sklearn.decomposition import PCA

from sklearn.metrics import hamming_loss, accuracy_score, f1_score, log_loss

from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification





# Baseline models

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from skmultilearn.adapt import MLkNN

from catboost import CatBoostClassifier

from xgboost import XGBClassifier



# Import modeling and tuning functions

import tensorflow as tf

import tensorflow_addons as tfa

import kerastuner as kt



from scipy.sparse.csr import csr_matrix

from scipy.sparse.lil import lil_matrix



# Lets seed everything

tf.random.set_seed(123)

os.environ['PYTHONHASHSEED'] = '123'

np.random.seed(123)

rn.seed(123)
train = pd.read_csv('/kaggle/input/lish-moa//train_features.csv')

labels = pd.read_csv('/kaggle/input/lish-moa//train_targets_scored.csv')

test = pd.read_csv('/kaggle/input/lish-moa//test_features.csv')



train = train.set_index('sig_id')

labels = labels.set_index('sig_id')

test = test.set_index('sig_id')



print(f'Train shape {train.shape}; number of labels {labels.shape[1]}; test shape {test.shape} ({train.shape[0]//test.shape[0]}%)')
train.info()
train.describe().T
number_of_genetic_features = len(list(filter(lambda col: 'g-'in col, train.columns)))

number_of_cell_features = len(list(filter(lambda col: 'c-'in col, train.columns)))

print(f'Number of genetic columns: {number_of_genetic_features} Number of cell columns: {number_of_cell_features}')
train[[f'g-{rn.randint(0, 772)}' for i in range(6)]].hist(figsize=(20,10))
train[[f'c-{rn.randint(0, 100)}' for i in range(6)]].hist(figsize=(20,10))
train['cp_time'].hist()
train['cp_type'].hist()
ctl_vahicle_index = train[train['cp_type']=='ctl_vehicle'].index

number_classes_ctl_vahicle = labels.loc[ctl_vahicle_index,:].sum().sum()

print(f'The number of non zero classes with cp_type=ctl_vahicle is {number_classes_ctl_vahicle}')
index_to_remove = train.index[train.cp_type == 'ctl_vehicle']

train = train.drop(index_to_remove).drop('cp_type', axis=1)

labels = labels.drop(index_to_remove)

index_to_remove_test = test.index[test.cp_type == 'ctl_vehicle']

test = test.drop(index_to_remove_test).drop('cp_type', axis=1)
number_of_rows_by_class = labels.sum(axis=0).sort_values()

print('======== Classes with less rows ========')

print(number_of_rows_by_class[:10])

print('======== Classes with more rows ========')

print(number_of_rows_by_class[-10:])
number_of_classes_distribution = labels.sum(axis=1).value_counts()

print('============== Number of classes in every row distribution ==============')

number_of_classes_distribution
labels_7_classes = labels.loc[labels.sum(axis=1)[labels.sum(axis=1)==7].index, :]

rows_with_7_labels = labels_7_classes.sum()[labels_7_classes.sum()>0]

print('============== Number of rows with 7 classes ==============')

rows_with_7_labels
print('============== Number of rows of the classes independently ==============')

number_of_rows_by_class[rows_with_7_labels.index]
def read_data():

    train = pd.read_csv('/kaggle/input/lish-moa//train_features.csv')

    labels = pd.read_csv('/kaggle/input/lish-moa//train_targets_scored.csv')

    test = pd.read_csv('/kaggle/input/lish-moa//test_features.csv')

    

    train = train.set_index('sig_id')

    labels = labels.set_index('sig_id')

    test = test.set_index('sig_id')

    

    # Removed because all classes are always zero when cp_type = 'ctl_vehicle

    index_to_remove = train.index[train.cp_type == 'ctl_vehicle']

    train = train.drop(index_to_remove).drop('cp_type', axis=1)

    labels = labels.drop(index_to_remove)

    index_to_remove_test = test.index[test.cp_type == 'ctl_vehicle']

    test = test.drop(index_to_remove_test).drop('cp_type', axis=1)

    

    # Binarize column cp_dose

    label_bin = LabelBinarizer()

    train['cp_dose'] = label_bin.fit_transform(train['cp_dose']).squeeze()    

    test['cp_dose'] = label_bin.transform(test['cp_dose']).squeeze()

    

    # Encode cp_time in three categories

    train['cp_time'] = train['cp_time'].astype('category')

    test['cp_time'] = test['cp_time'].astype('category')

    train = pd.get_dummies(train)

    test = pd.get_dummies(test)

        

    return train, labels, test
train, labels, test = read_data()
special_rows_index = labels[(labels['atp-sensitive_potassium_channel_antagonist']>0) | (labels['erbb2_inhibitor']>0)].index

special_rows_features = train.loc[special_rows_index].reset_index(drop=True)

special_rows_labels = labels.loc[special_rows_index].reset_index(drop=True)

train = train.drop(special_rows_index)

labels = labels.drop(special_rows_index)
X_train, Y_train, X_valid, Y_valid = iterative_train_test_split(train.values, labels.values, .8)



train_short = pd.DataFrame(X_train, columns=train.columns)

labels_short = pd.DataFrame(Y_train, columns=labels.columns)



# Ensure that the special cases are included

train_short = pd.concat([train_short, special_rows_features])

labels_short = pd.concat([labels_short, special_rows_labels])



train_short.shape
def run_experiment(data, configurations):

  

    results = []

    metrics = None

    

    experiment_config = configurations['experiment_config']

    model_config = configurations['model_config']

    name = experiment_config['name']



    print(f'========================= Runing experiment for {name} =========================')

    

    # Apply the data transformations

    data, pipeline, experiment_config = preprocess(data, experiment_config)

    

    # The data can change the number of features due to PCA

    if 'n_features' in experiment_config:

        model_config['n_features'] = experiment_config['n_features']

    

    # Obtain the model 

    model = get_model(model_config)

    

    # Train the model

    current_metrics, model = train_model(model, data, experiment_config=experiment_config)



    # Update the metrics

    metrics = update_metrics(name, current_metrics, metrics)



    # The model with their transformatios will be returned

    results.append((model, pipeline))

        

    return metrics, results
def get_model(model_config):

    

    model_type = model_config['type']

    model = None

    

    if model_type=='nn-mixed':

        model = mixednn(model_config)

    elif model_type=='nn-dense':

        model = densenn(model_config)

    elif model_type=='rf':

        model = RandomForestClassifier(model_config['max_depth'], random_state=123, n_jobs=-1)

    elif model_type=='lr':

        model = OneVsRestClassifier(LogisticRegression(random_state=123, max_iter=model_config['max_iter'], solver=model_config['solver'], C=model_config['C']))

    elif model_type=='xgb':

        model = OneVsRestClassifier(XGBClassifier(max_depth=model_config['max_depth'], learning_rate=model_config['learning_rate'], n_estimators=model_config['n_estimators']

                                                  , objective='binary:logistic', booster=model_config['booster'],n_jobs=-1))

    elif model_type=='catboost':

        model = OneVsRestClassifier(CatBoostClassifier(iterations=model_config['iterations'],random_state=123, logging_level='Silent'))

    elif model_type=='knn':

        model = MLkNN(k=model_config['k'])

    elif model_type=='stabnet':

        model = stabnet(model_config)

    else:

        raise Exception('No model provided')

    

    return model

    
# Credits to -> https://github.com/titu1994/tf-TabNet

# I just modified the activation to be a sigmoid



def register_keras_custom_object(cls):

    tf.keras.utils.get_custom_objects()[cls.__name__] = cls

    return cls





def glu(x, n_units=None):

    """Generalized linear unit nonlinear activation."""

    if n_units is None:

        n_units = tf.shape(x)[-1] // 2



    return x[..., :n_units] * tf.nn.sigmoid(x[..., n_units:])





"""

Code replicated from https://github.com/tensorflow/addons/blob/master/tensorflow_addons/activations/sparsemax.py

"""





@register_keras_custom_object

@tf.function

def sparsemax(logits, axis):

    """Sparsemax activation function [1].

    For each batch `i` and class `j` we have

      $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$

    [1]: https://arxiv.org/abs/1602.02068

    Args:

        logits: Input tensor.

        axis: Integer, axis along which the sparsemax operation is applied.

    Returns:

        Tensor, output of sparsemax transformation. Has the same type and

        shape as `logits`.

    Raises:

        ValueError: In case `dim(logits) == 1`.

    """

    logits = tf.convert_to_tensor(logits, name="logits")



    # We need its original shape for shape inference.

    shape = logits.get_shape()

    rank = shape.rank

    is_last_axis = (axis == -1) or (axis == rank - 1)



    if is_last_axis:

        output = _compute_2d_sparsemax(logits)

        output.set_shape(shape)

        return output



    # If dim is not the last dimension, we have to do a transpose so that we can

    # still perform softmax on its last dimension.



    # Swap logits' dimension of dim and its last dimension.

    rank_op = tf.rank(logits)

    axis_norm = axis % rank

    logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))



    # Do the actual softmax on its last dimension.

    output = _compute_2d_sparsemax(logits)

    output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))



    # Make shape inference work since transpose may erase its static shape.

    output.set_shape(shape)

    return output





def _swap_axis(logits, dim_index, last_index, **kwargs):

    return tf.transpose(

        logits,

        tf.concat(

            [

                tf.range(dim_index),

                [last_index],

                tf.range(dim_index + 1, last_index),

                [dim_index],

            ],

            0,

        ),

        **kwargs,

    )





def _compute_2d_sparsemax(logits):

    """Performs the sparsemax operation when axis=-1."""

    shape_op = tf.shape(logits)

    obs = tf.math.reduce_prod(shape_op[:-1])

    dims = shape_op[-1]



    # In the paper, they call the logits z.

    # The mean(logits) can be substracted from logits to make the algorithm

    # more numerically stable. the instability in this algorithm comes mostly

    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close

    # to zero. However, in practise the numerical instability issues are very

    # minor and substacting the mean causes extra issues with inf and nan

    # input.

    # Reshape to [obs, dims] as it is almost free and means the remanining

    # code doesn't need to worry about the rank.

    z = tf.reshape(logits, [obs, dims])



    # sort z

    z_sorted, _ = tf.nn.top_k(z, k=dims)



    # calculate k(z)

    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)

    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)

    z_check = 1 + k * z_sorted > z_cumsum

    # because the z_check vector is always [1,1,...1,0,0,...0] finding the

    # (index + 1) of the last `1` is the same as just summing the number of 1.

    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)



    # calculate tau(z)

    # If there are inf values or all values are -inf, the k_z will be zero,

    # this is mathematically invalid and will also cause the gather_nd to fail.

    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then

    # fixed later (see p_safe) by returning p = nan. This results in the same

    # behavior as softmax.

    k_z_safe = tf.math.maximum(k_z, 1)

    indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)

    tau_sum = tf.gather_nd(z_cumsum, indices)

    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)



    # calculate p

    p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))

    # If k_z = 0 or if z = nan, then the input is invalid

    p_safe = tf.where(

        tf.expand_dims(

            tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),

            axis=-1,

        ),

        tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),

        p,

    )



    # Reshape back to original size

    p_safe = tf.reshape(p_safe, shape_op)

    return p_safe





"""

Code replicated from https://github.com/tensorflow/addons/blob/master/tensorflow_addons/layers/normalizations.py

"""





@register_keras_custom_object

class GroupNormalization(tf.keras.layers.Layer):

    """Group normalization layer.

    Group Normalization divides the channels into groups and computes

    within each group the mean and variance for normalization.

    Empirically, its accuracy is more stable than batch norm in a wide

    range of small batch sizes, if learning rate is adjusted linearly

    with batch sizes.

    Relation to Layer Normalization:

    If the number of groups is set to 1, then this operation becomes identical

    to Layer Normalization.

    Relation to Instance Normalization:

    If the number of groups is set to the

    input dimension (number of groups is equal

    to number of channels), then this operation becomes

    identical to Instance Normalization.

    Arguments

        groups: Integer, the number of groups for Group Normalization.

            Can be in the range [1, N] where N is the input dimension.

            The input dimension must be divisible by the number of groups.

        axis: Integer, the axis that should be normalized.

        epsilon: Small float added to variance to avoid dividing by zero.

        center: If True, add offset of `beta` to normalized tensor.

            If False, `beta` is ignored.

        scale: If True, multiply by `gamma`.

            If False, `gamma` is not used.

        beta_initializer: Initializer for the beta weight.

        gamma_initializer: Initializer for the gamma weight.

        beta_regularizer: Optional regularizer for the beta weight.

        gamma_regularizer: Optional regularizer for the gamma weight.

        beta_constraint: Optional constraint for the beta weight.

        gamma_constraint: Optional constraint for the gamma weight.

    Input shape

        Arbitrary. Use the keyword argument `input_shape`

        (tuple of integers, does not include the samples axis)

        when using this layer as the first layer in a model.

    Output shape

        Same shape as input.

    References

        - [Group Normalization](https://arxiv.org/abs/1803.08494)

    """



    def __init__(

            self,

            groups: int = 2,

            axis: int = -1,

            epsilon: float = 1e-3,

            center: bool = True,

            scale: bool = True,

            beta_initializer="zeros",

            gamma_initializer="ones",

            beta_regularizer=None,

            gamma_regularizer=None,

            beta_constraint=None,

            gamma_constraint=None,

            **kwargs

    ):

        super().__init__(**kwargs)

        self.supports_masking = True

        self.groups = groups

        self.axis = axis

        self.epsilon = epsilon

        self.center = center

        self.scale = scale

        self.beta_initializer = tf.keras.initializers.get(beta_initializer)

        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)

        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)

        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)

        self.beta_constraint = tf.keras.constraints.get(beta_constraint)

        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

        self._check_axis()



    def build(self, input_shape):



        self._check_if_input_shape_is_none(input_shape)

        self._set_number_of_groups_for_instance_norm(input_shape)

        self._check_size_of_dimensions(input_shape)

        self._create_input_spec(input_shape)



        self._add_gamma_weight(input_shape)

        self._add_beta_weight(input_shape)

        self.built = True

        super().build(input_shape)



    def call(self, inputs, training=None):

        # Training=none is just for compat with batchnorm signature call

        input_shape = tf.keras.backend.int_shape(inputs)

        tensor_input_shape = tf.shape(inputs)



        reshaped_inputs, group_shape = self._reshape_into_groups(

            inputs, input_shape, tensor_input_shape

        )



        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)



        outputs = tf.reshape(normalized_inputs, tensor_input_shape)



        return outputs



    def get_config(self):

        config = {

            "groups": self.groups,

            "axis": self.axis,

            "epsilon": self.epsilon,

            "center": self.center,

            "scale": self.scale,

            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),

            "gamma_initializer": tf.keras.initializers.serialize(

                self.gamma_initializer

            ),

            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),

            "gamma_regularizer": tf.keras.regularizers.serialize(

                self.gamma_regularizer

            ),

            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),

            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),

        }

        base_config = super().get_config()

        return {**base_config, **config}



    def compute_output_shape(self, input_shape):

        return input_shape



    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):



        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]

        group_shape[self.axis] = input_shape[self.axis] // self.groups

        group_shape.insert(self.axis, self.groups)

        group_shape = tf.stack(group_shape)

        reshaped_inputs = tf.reshape(inputs, group_shape)

        return reshaped_inputs, group_shape



    def _apply_normalization(self, reshaped_inputs, input_shape):



        group_shape = tf.keras.backend.int_shape(reshaped_inputs)

        group_reduction_axes = list(range(1, len(group_shape)))

        axis = -2 if self.axis == -1 else self.axis - 1

        group_reduction_axes.pop(axis)



        mean, variance = tf.nn.moments(

            reshaped_inputs, group_reduction_axes, keepdims=True

        )



        gamma, beta = self._get_reshaped_weights(input_shape)

        normalized_inputs = tf.nn.batch_normalization(

            reshaped_inputs,

            mean=mean,

            variance=variance,

            scale=gamma,

            offset=beta,

            variance_epsilon=self.epsilon,

        )

        return normalized_inputs



    def _get_reshaped_weights(self, input_shape):

        broadcast_shape = self._create_broadcast_shape(input_shape)

        gamma = None

        beta = None

        if self.scale:

            gamma = tf.reshape(self.gamma, broadcast_shape)



        if self.center:

            beta = tf.reshape(self.beta, broadcast_shape)

        return gamma, beta



    def _check_if_input_shape_is_none(self, input_shape):

        dim = input_shape[self.axis]

        if dim is None:

            raise ValueError(

                "Axis " + str(self.axis) + " of "

                                           "input tensor should have a defined dimension "

                                           "but the layer received an input with shape " + str(input_shape) + "."

            )



    def _set_number_of_groups_for_instance_norm(self, input_shape):

        dim = input_shape[self.axis]



        if self.groups == -1:

            self.groups = dim



    def _check_size_of_dimensions(self, input_shape):



        dim = input_shape[self.axis]

        if dim < self.groups:

            raise ValueError(

                "Number of groups (" + str(self.groups) + ") cannot be "

                                                          "more than the number of channels (" + str(dim) + ")."

            )



        if dim % self.groups != 0:

            raise ValueError(

                "Number of groups (" + str(self.groups) + ") must be a "

                                                          "multiple of the number of channels (" + str(dim) + ")."

            )



    def _check_axis(self):



        if self.axis == 0:

            raise ValueError(

                "You are trying to normalize your batch axis. Do you want to "

                "use tf.layer.batch_normalization instead"

            )



    def _create_input_spec(self, input_shape):



        dim = input_shape[self.axis]

        self.input_spec = tf.keras.layers.InputSpec(

            ndim=len(input_shape), axes={self.axis: dim}

        )



    def _add_gamma_weight(self, input_shape):



        dim = input_shape[self.axis]

        shape = (dim,)



        if self.scale:

            self.gamma = self.add_weight(

                shape=shape,

                name="gamma",

                initializer=self.gamma_initializer,

                regularizer=self.gamma_regularizer,

                constraint=self.gamma_constraint,

            )

        else:

            self.gamma = None



    def _add_beta_weight(self, input_shape):



        dim = input_shape[self.axis]

        shape = (dim,)



        if self.center:

            self.beta = self.add_weight(

                shape=shape,

                name="beta",

                initializer=self.beta_initializer,

                regularizer=self.beta_regularizer,

                constraint=self.beta_constraint,

            )

        else:

            self.beta = None



    def _create_broadcast_shape(self, input_shape):

        broadcast_shape = [1] * len(input_shape)

        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups

        broadcast_shape.insert(self.axis, self.groups)

        return broadcast_shape

    

class TransformBlock(tf.keras.Model):



    def __init__(self, features,

                 norm_type,

                 momentum=0.9,

                 virtual_batch_size=None,

                 groups=2,

                 block_name='',

                 **kwargs):

        super(TransformBlock, self).__init__(**kwargs)



        self.features = features

        self.norm_type = norm_type

        self.momentum = momentum

        self.groups = groups

        self.virtual_batch_size = virtual_batch_size



        self.transform = tf.keras.layers.Dense(self.features, use_bias=False, name=f'transformblock_dense_{block_name}')



        if norm_type == 'batch':

            self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum,

                                                         virtual_batch_size=virtual_batch_size,

                                                         name=f'transformblock_bn_{block_name}')



        else:

            self.bn = GroupNormalization(axis=-1, groups=self.groups, name=f'transformblock_gn_{block_name}')



    def call(self, inputs, training=None):

        x = self.transform(inputs)

        x = self.bn(x, training=training)

        return x





class TabNet(tf.keras.Model):



    def __init__(self, feature_columns,

                 feature_dim=64,

                 output_dim=64,

                 num_features=None,

                 num_decision_steps=5,

                 relaxation_factor=1.5,

                 sparsity_coefficient=1e-5,

                 norm_type='group',

                 batch_momentum=0.98,

                 virtual_batch_size=None,

                 num_groups=2,

                 epsilon=1e-5,

                 **kwargs):

        """

        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)

        # Hyper Parameter Tuning (Excerpt from the paper)

        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting

        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter

        selection:

            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and

            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from

            overfitting and yield poor generalization.

            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off

            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A

            very high value of Nd and Na may suffer from overfitting and yield poor generalization.

            - An optimal choice of γ can have a major role on the overall performance. Typically a larger

            Nsteps value favors for a larger γ.

            - A large batch size is beneficial for performance - if the memory constraints permit, as large

            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically

            much smaller than the batch size.

            - Initially large learning rate is important, which should be gradually decayed until convergence.

        Args:

            feature_columns: The Tensorflow feature columns for the dataset.

            feature_dim (N_a): Dimensionality of the hidden representation in feature

                transformation block. Each layer first maps the representation to a

                2*feature_dim-dimensional output and half of it is used to determine the

                nonlinearity of the GLU activation where the other half is used as an

                input to GLU, and eventually feature_dim-dimensional output is

                transferred to the next layer.

            output_dim (N_d): Dimensionality of the outputs of each decision step, which is

                later mapped to the final classification or regression output.

            num_features: The number of input features (i.e the number of columns for

                tabular data assuming each feature is represented with 1 dimension).

            num_decision_steps(N_steps): Number of sequential decision steps.

            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each

                feature at different decision steps. When it is 1, a feature is enforced

                to be used only at one decision step and as it increases, more

                flexibility is provided to use a feature at multiple decision steps.

            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.

                Sparsity may provide a favorable inductive bias for convergence to

                higher accuracy for some datasets where most of the input features are redundant.

            norm_type: Type of normalization to perform for the model. Can be either

                'batch' or 'group'. 'group' is the default.

            batch_momentum: Momentum in ghost batch normalization.

            virtual_batch_size: Virtual batch size in ghost batch normalization. The

                overall batch size should be an integer multiple of virtual_batch_size.

            num_groups: Number of groups used for group normalization.

            epsilon: A small number for numerical stability of the entropy calculations.

        """

        super(TabNet, self).__init__(**kwargs)



        # Input checks

        if feature_columns is not None:

            if type(feature_columns) not in (list, tuple):

                raise ValueError("`feature_columns` must be a list or a tuple.")



            if len(feature_columns) == 0:

                raise ValueError("`feature_columns` must be contain at least 1 tf.feature_column !")



            if num_features is None:

                num_features = len(feature_columns)

            else:

                num_features = int(num_features)



        else:

            if num_features is None:

                raise ValueError("If `feature_columns` is None, then `num_features` cannot be None.")



        if num_decision_steps < 1:

            raise ValueError("Num decision steps must be greater than 0.")



        if feature_dim <= output_dim:

            raise ValueError("To compute `features_for_coef`, feature_dim must be larger than output dim")



        feature_dim = int(feature_dim)

        output_dim = int(output_dim)

        num_decision_steps = int(num_decision_steps)

        relaxation_factor = float(relaxation_factor)

        sparsity_coefficient = float(sparsity_coefficient)

        batch_momentum = float(batch_momentum)

        num_groups = max(1, int(num_groups))

        epsilon = float(epsilon)



        if relaxation_factor < 0.:

            raise ValueError("`relaxation_factor` cannot be negative !")



        if sparsity_coefficient < 0.:

            raise ValueError("`sparsity_coefficient` cannot be negative !")



        if virtual_batch_size is not None:

            virtual_batch_size = int(virtual_batch_size)



        if norm_type not in ['batch', 'group']:

            raise ValueError("`norm_type` must be either `batch` or `group`")



        self.feature_columns = feature_columns

        self.num_features = num_features

        self.feature_dim = feature_dim

        self.output_dim = output_dim



        self.num_decision_steps = num_decision_steps

        self.relaxation_factor = relaxation_factor

        self.sparsity_coefficient = sparsity_coefficient

        self.norm_type = norm_type

        self.batch_momentum = batch_momentum

        self.virtual_batch_size = virtual_batch_size

        self.num_groups = num_groups

        self.epsilon = epsilon



        if num_decision_steps > 1:

            features_for_coeff = feature_dim - output_dim

            print(f"[TabNet]: {features_for_coeff} features will be used for decision steps.")



        if self.feature_columns is not None:

            self.input_features = tf.keras.layers.DenseFeatures(feature_columns, trainable=True)



            if self.norm_type == 'batch':

                self.input_bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_momentum, name='input_bn')

            else:

                self.input_bn = GroupNormalization(axis=-1, groups=self.num_groups, name='input_gn')



        else:

            self.input_features = None

            self.input_bn = None



        self.transform_f1 = TransformBlock(2 * self.feature_dim, self.norm_type,

                                           self.batch_momentum, self.virtual_batch_size, self.num_groups,

                                           block_name='f1')



        self.transform_f2 = TransformBlock(2 * self.feature_dim, self.norm_type,

                                           self.batch_momentum, self.virtual_batch_size, self.num_groups,

                                           block_name='f2')



        self.transform_f3_list = [

            TransformBlock(2 * self.feature_dim, self.norm_type,

                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'f3_{i}')

            for i in range(self.num_decision_steps)

        ]



        self.transform_f4_list = [

            TransformBlock(2 * self.feature_dim, self.norm_type,

                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'f4_{i}')

            for i in range(self.num_decision_steps)

        ]



        self.transform_coef_list = [

            TransformBlock(self.num_features, self.norm_type,

                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'coef_{i}')

            for i in range(self.num_decision_steps - 1)

        ]



        self._step_feature_selection_masks = None

        self._step_aggregate_feature_selection_mask = None



    def call(self, inputs, training=None):

        if self.input_features is not None:

            features = self.input_features(inputs)

            features = self.input_bn(features, training=training)



        else:

            features = inputs



        batch_size = tf.shape(features)[0]

        self._step_feature_selection_masks = []

        self._step_aggregate_feature_selection_mask = None



        # Initializes decision-step dependent variables.

        output_aggregated = tf.zeros([batch_size, self.output_dim])

        masked_features = features

        mask_values = tf.zeros([batch_size, self.num_features])

        aggregated_mask_values = tf.zeros([batch_size, self.num_features])

        complementary_aggregated_mask_values = tf.ones(

            [batch_size, self.num_features])



        total_entropy = 0.0

        entropy_loss = 0.



        for ni in range(self.num_decision_steps):

            # Feature transformer with two shared and two decision step dependent

            # blocks is used below.=

            transform_f1 = self.transform_f1(masked_features, training=training)

            transform_f1 = glu(transform_f1, self.feature_dim)



            transform_f2 = self.transform_f2(transform_f1, training=training)

            transform_f2 = (glu(transform_f2, self.feature_dim) +

                            transform_f1) * tf.math.sqrt(0.5)



            transform_f3 = self.transform_f3_list[ni](transform_f2, training=training)

            transform_f3 = (glu(transform_f3, self.feature_dim) +

                            transform_f2) * tf.math.sqrt(0.5)



            transform_f4 = self.transform_f4_list[ni](transform_f3, training=training)

            transform_f4 = (glu(transform_f4, self.feature_dim) +

                            transform_f3) * tf.math.sqrt(0.5)



            if (ni > 0 or self.num_decision_steps == 1):

                decision_out = tf.nn.relu(transform_f4[:, :self.output_dim])



                # Decision aggregation.

                output_aggregated += decision_out



                # Aggregated masks are used for visualization of the

                # feature importance attributes.

                scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True)



                if self.num_decision_steps > 1:

                    scale_agg = scale_agg / tf.cast(self.num_decision_steps - 1, tf.float32)



                aggregated_mask_values += mask_values * scale_agg



            features_for_coef = transform_f4[:, self.output_dim:]



            if ni < (self.num_decision_steps - 1):

                # Determines the feature masks via linear and nonlinear

                # transformations, taking into account of aggregated feature use.

                mask_values = self.transform_coef_list[ni](features_for_coef, training=training)

                mask_values *= complementary_aggregated_mask_values

                mask_values = sparsemax(mask_values, axis=-1)



                # Relaxation factor controls the amount of reuse of features between

                # different decision blocks and updated with the values of

                # coefficients.

                complementary_aggregated_mask_values *= (

                        self.relaxation_factor - mask_values)



                # Entropy is used to penalize the amount of sparsity in feature

                # selection.

                total_entropy += tf.reduce_mean(

                    tf.reduce_sum(

                        -mask_values * tf.math.log(mask_values + self.epsilon), axis=1)) / (

                                     tf.cast(self.num_decision_steps - 1, tf.float32))



                # Add entropy loss

                entropy_loss = total_entropy



                # Feature selection.

                masked_features = tf.multiply(mask_values, features)



                # Visualization of the feature selection mask at decision step ni

                # tf.summary.image(

                #     "Mask for step" + str(ni),

                #     tf.expand_dims(tf.expand_dims(mask_values, 0), 3),

                #     max_outputs=1)

                mask_at_step_i = tf.expand_dims(tf.expand_dims(mask_values, 0), 3)

                self._step_feature_selection_masks.append(mask_at_step_i)



            else:

                # This branch is needed for correct compilation by tf.autograph

                entropy_loss = 0.



        # Adds the loss automatically

        self.add_loss(self.sparsity_coefficient * entropy_loss)



        # Visualization of the aggregated feature importances

        # tf.summary.image(

        #     "Aggregated mask",

        #     tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),

        #     max_outputs=1)



        agg_mask = tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3)

        self._step_aggregate_feature_selection_mask = agg_mask



        return output_aggregated



    @property

    def feature_selection_masks(self):

        return self._step_feature_selection_masks



    @property

    def aggregate_feature_selection_mask(self):

        return self._step_aggregate_feature_selection_mask

    

class StackedTabNet(tf.keras.Model):



    def __init__(self, feature_columns,

                 num_layers=1,

                 feature_dim=64,

                 output_dim=64,

                 num_features=None,

                 num_decision_steps=5,

                 relaxation_factor=1.5,

                 sparsity_coefficient=1e-5,

                 norm_type='group',

                 batch_momentum=0.98,

                 virtual_batch_size=None,

                 num_groups=2,

                 epsilon=1e-5,

                 **kwargs):

        """

        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)

        Stacked variant of the TabNet model, which stacks multiple TabNets into a singular model.

        # Hyper Parameter Tuning (Excerpt from the paper)

        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting

        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter

        selection:

            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and

            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from

            overfitting and yield poor generalization.

            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off

            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A

            very high value of Nd and Na may suffer from overfitting and yield poor generalization.

            - An optimal choice of γ can have a major role on the overall performance. Typically a larger

            Nsteps value favors for a larger γ.

            - A large batch size is beneficial for performance - if the memory constraints permit, as large

            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically

            much smaller than the batch size.

            - Initially large learning rate is important, which should be gradually decayed until convergence.

        Args:

            feature_columns: The Tensorflow feature columns for the dataset.

            num_layers: Number of TabNets to stack together.

            feature_dim (N_a): Dimensionality of the hidden representation in feature

                transformation block. Each layer first maps the representation to a

                2*feature_dim-dimensional output and half of it is used to determine the

                nonlinearity of the GLU activation where the other half is used as an

                input to GLU, and eventually feature_dim-dimensional output is

                transferred to the next layer. Can be either a single int, or a list of

                integers. If a list, must be of same length as the number of layers.

            output_dim (N_d): Dimensionality of the outputs of each decision step, which is

                later mapped to the final classification or regression output.

                Can be either a single int, or a list of

                integers. If a list, must be of same length as the number of layers.

            num_features: The number of input features (i.e the number of columns for

                tabular data assuming each feature is represented with 1 dimension).

            num_decision_steps(N_steps): Number of sequential decision steps.

            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each

                feature at different decision steps. When it is 1, a feature is enforced

                to be used only at one decision step and as it increases, more

                flexibility is provided to use a feature at multiple decision steps.

            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.

                Sparsity may provide a favorable inductive bias for convergence to

                higher accuracy for some datasets where most of the input features are redundant.

            norm_type: Type of normalization to perform for the model. Can be either

                'batch' or 'group'. 'group' is the default.

            batch_momentum: Momentum in ghost batch normalization.

            virtual_batch_size: Virtual batch size in ghost batch normalization. The

                overall batch size should be an integer multiple of virtual_batch_size.

            num_groups: Number of groups used for group normalization.

            epsilon: A small number for numerical stability of the entropy calculations.

        """

        super(StackedTabNet, self).__init__(**kwargs)



        if num_layers < 1:

            raise ValueError("`num_layers` cannot be less than 1")



        if type(feature_dim) not in [list, tuple]:

            feature_dim = [feature_dim] * num_layers



        if type(output_dim) not in [list, tuple]:

            output_dim = [output_dim] * num_layers



        if len(feature_dim) != num_layers:

            raise ValueError("`feature_dim` must be a list of length `num_layers`")



        if len(output_dim) != num_layers:

            raise ValueError("`output_dim` must be a list of length `num_layers`")



        self.num_layers = num_layers



        layers = []

        layers.append(TabNet(feature_columns=feature_columns,

                             num_features=num_features,

                             feature_dim=feature_dim[0],

                             output_dim=output_dim[0],

                             num_decision_steps=num_decision_steps,

                             relaxation_factor=relaxation_factor,

                             sparsity_coefficient=sparsity_coefficient,

                             norm_type=norm_type,

                             batch_momentum=batch_momentum,

                             virtual_batch_size=virtual_batch_size,

                             num_groups=num_groups,

                             epsilon=epsilon))



        for layer_idx in range(1, num_layers):

            layers.append(TabNet(feature_columns=None,

                                 num_features=output_dim[layer_idx - 1],

                                 feature_dim=feature_dim[layer_idx],

                                 output_dim=output_dim[layer_idx],

                                 num_decision_steps=num_decision_steps,

                                 relaxation_factor=relaxation_factor,

                                 sparsity_coefficient=sparsity_coefficient,

                                 norm_type=norm_type,

                                 batch_momentum=batch_momentum,

                                 virtual_batch_size=virtual_batch_size,

                                 num_groups=num_groups,

                                 epsilon=epsilon))



        self.tabnet_layers = layers



    def call(self, inputs, training=None):

        x = self.tabnet_layers[0](inputs, training=training)



        for layer_idx in range(1, self.num_layers):

            x = self.tabnet_layers[layer_idx](x, training=training)



        return x



    @property

    def tabnets(self):

        return self.tabnet_layers



    @property

    def feature_selection_masks(self):

        return [tabnet.feature_selection_masks

                for tabnet in self.tabnet_layers]



    @property

    def aggregate_feature_selection_mask(self):

        return [tabnet.aggregate_feature_selection_mask

                for tabnet in self.tabnet_layers]



    

class StackedTabNetClassifier(tf.keras.Model):



    def __init__(self, feature_columns,

                 num_classes,

                 num_layers=1,

                 feature_dim=64,

                 output_dim=64,

                 num_features=None,

                 num_decision_steps=5,

                 relaxation_factor=1.5,

                 sparsity_coefficient=1e-5,

                 norm_type='group',

                 batch_momentum=0.98,

                 virtual_batch_size=None,

                 num_groups=2,

                 epsilon=1e-5,

                 **kwargs):

        """

        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)

        Stacked variant of the TabNet model, which stacks multiple TabNets into a singular model.

        # Hyper Parameter Tuning (Excerpt from the paper)

        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting

        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter

        selection:

            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and

            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from

            overfitting and yield poor generalization.

            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off

            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A

            very high value of Nd and Na may suffer from overfitting and yield poor generalization.

            - An optimal choice of γ can have a major role on the overall performance. Typically a larger

            Nsteps value favors for a larger γ.

            - A large batch size is beneficial for performance - if the memory constraints permit, as large

            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically

            much smaller than the batch size.

            - Initially large learning rate is important, which should be gradually decayed until convergence.

        Args:

            feature_columns: The Tensorflow feature columns for the dataset.

            num_classes: Number of classes.

            num_layers: Number of TabNets to stack together.

            feature_dim (N_a): Dimensionality of the hidden representation in feature

                transformation block. Each layer first maps the representation to a

                2*feature_dim-dimensional output and half of it is used to determine the

                nonlinearity of the GLU activation where the other half is used as an

                input to GLU, and eventually feature_dim-dimensional output is

                transferred to the next layer. Can be either a single int, or a list of

                integers. If a list, must be of same length as the number of layers.

            output_dim (N_d): Dimensionality of the outputs of each decision step, which is

                later mapped to the final classification or regression output.

                Can be either a single int, or a list of

                integers. If a list, must be of same length as the number of layers.

            num_features: The number of input features (i.e the number of columns for

                tabular data assuming each feature is represented with 1 dimension).

            num_decision_steps(N_steps): Number of sequential decision steps.

            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each

                feature at different decision steps. When it is 1, a feature is enforced

                to be used only at one decision step and as it increases, more

                flexibility is provided to use a feature at multiple decision steps.

            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.

                Sparsity may provide a favorable inductive bias for convergence to

                higher accuracy for some datasets where most of the input features are redundant.

            norm_type: Type of normalization to perform for the model. Can be either

                'batch' or 'group'. 'group' is the default.

            batch_momentum: Momentum in ghost batch normalization.

            virtual_batch_size: Virtual batch size in ghost batch normalization. The

                overall batch size should be an integer multiple of virtual_batch_size.

            num_groups: Number of groups used for group normalization.

            epsilon: A small number for numerical stability of the entropy calculations.

        """

        super(StackedTabNetClassifier, self).__init__(**kwargs)



        self.num_classes = num_classes



        self.stacked_tabnet = StackedTabNet(feature_columns=feature_columns,

                                            num_layers=num_layers,

                                            feature_dim=feature_dim,

                                            output_dim=output_dim,

                                            num_features=num_features,

                                            num_decision_steps=num_decision_steps,

                                            relaxation_factor=relaxation_factor,

                                            sparsity_coefficient=sparsity_coefficient,

                                            norm_type=norm_type,

                                            batch_momentum=batch_momentum,

                                            virtual_batch_size=virtual_batch_size,

                                            num_groups=num_groups,

                                            epsilon=epsilon)



        self.clf = tf.keras.layers.Dense(num_classes, activation='sigmoid', use_bias=False)



    def call(self, inputs, training=None):

        self.activations = self.stacked_tabnet(inputs, training=training)

        out = self.clf(self.activations)



        return out

def stabnet(model_config):

    feature_columns = []

    for col_name in model_config['feature_columns']:

        feature_columns.append(tf.feature_column.numeric_column(col_name))

    model = StackedTabNetClassifier(num_classes=model_config['n_labels'],  feature_columns=feature_columns, num_layers=model_config['n_layers'], num_features=model_config['n_features'], feature_dim=model_config['feature_dim'], output_dim=model_config['output_dim'])

    

    optimizer = tfa.optimizers.RectifiedAdam()

    if model_config['use_loookahead']:

        optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=10)

    

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    #model.build((None, model_config['n_features']))

    

    return model



def densenn(model_config):

    model = tf.keras.Sequential()

    

    layers = model_config['n_layers']

    use_weight_normalization =  model_config['use_weight_normalization']

    use_loookahead = model_config['use_loookahead']

    units = model_config['units']

    use_batch_norm = model_config['use_batch_norm']

    use_mish = model_config['use_mish']

    drop_rate= model_config['drop_rate']

    n_features = model_config['n_features']

    n_labels = model_config['n_labels']

    

    for l in range(layers):

        

        if use_batch_norm:

             model.add(tf.keras.layers.BatchNormalization())

                

        dense = tf.keras.layers.Dense(units=units)

        

        if use_weight_normalization:

             dense = tfa.layers.WeightNormalization(dense)

        model.add(dense)

        

        if use_mish:

            model.add(tf.keras.layers.Activation(tfa.activations.mish))

        else:

            model.add(tf.keras.layers.Activation('relu'))

            

        model.add(tf.keras.layers.Dropout(drop_rate))

    

    dense = tf.keras.layers.Dense(n_labels, activation='sigmoid')

    if use_weight_normalization:

         dense = tfa.layers.WeightNormalization(dense)

            

    model.add(dense)

    #learning_rate = hp.Float('learning_rate', min_value=3e-4, max_value=3e-3)

    

    optimizer = tfa.optimizers.RectifiedAdam()

    if use_loookahead:

        optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=10)

    

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.build((None, n_features))

    

    return model
def mixednn(model_config):

    input_cell = tf.keras.layers.Input((10,10,1))

    input_gen = tf.keras.layers.Input((193,4,1))

    input_cats = tf.keras.layers.Input((4,))

    

    """Parameters"""

    

    layers_cell = model_config['layers_cell']

    layers_gen = model_config['layers_gen']

    layers_final = model_config['layers_final']

    use_mish = model_config['use_mish']

    

    use_weight_normalization =  model_config['use_weight_normalization']

    use_loookahead = model_config['use_loookahead']

    num_units = model_config['units']

    use_batch_norm = model_config['use_batch_norm']

    drop_rate= model_config['drop_rate']

    n_features = model_config['n_features']

    n_labels = model_config['n_labels']

    mask_size = model_config['mask_size']

    

    """Cell layer"""

    for lc in range(layers_cell):

        

        conv_layer = tf.keras.layers.Conv2D(num_units, (mask_size, mask_size), activation='linear', padding='same')

        if use_weight_normalization:

            output_cell = tfa.layers.WeightNormalization(conv_layer)(input_cell)

        else:

            output_cell = conv_layer(input_cell)

        

        if use_batch_norm:

            output_cell = tf.keras.layers.BatchNormalization()(output_cell)

            

        if use_mish:

            output_cell = tf.keras.layers.Activation(tfa.activations.mish)(output_cell)

        else:

            output_cell = tf.keras.layers.Activation('relu')(output_cell)

            

        output_cell = tf.keras.layers.Dropout(drop_rate)(output_cell)

    

    """Gen layer"""

    for gc in range(layers_gen):



        conv_layer = tf.keras.layers.Conv2D(num_units, (mask_size, mask_size), activation='linear', padding='same')

        

        if use_weight_normalization:

             output_gen = tfa.layers.WeightNormalization(conv_layer)(input_gen)

        else:

            output_gen = conv_layer(input_gen)

        

        if use_batch_norm:

            output_gen = tf.keras.layers.BatchNormalization()(output_gen)

        

        if use_mish:

            output_gen = tf.keras.layers.Activation(tfa.activations.mish)(output_gen)

        else:

            output_gen = tf.keras.layers.Activation('relu')(output_gen)

            

        output_gen = tf.keras.layers.Dropout(drop_rate)(output_gen)



    output_cell = tf.keras.layers.Flatten()(output_cell)

    output_gen = tf.keras.layers.Flatten()(output_gen)



    output = tf.keras.layers.Concatenate()([output_cell, output_gen, input_cats])

    

    """Final layer"""

    for lf in range(layers_final):

        

        dense = tf.keras.layers.Dense(num_units, activation='linear')(output)

        if use_weight_normalization:

             output = tfa.layers.WeightNormalization(dense)(output)

        else:

            output = dense(output)

            

        if use_batch_norm:

            output = tf.keras.layers.BatchNormalization()(output)

            

        if use_mish:

            output = tf.keras.layers.Activation(tfa.activations.mish)(output)

        else:

            output = tf.keras.layers.Activation('relu')(output)

            

        output = tf.keras.layers.Dropout(drop_rate)(output)

    

    output = tf.keras.layers.Dense(n_labels, activation='sigmoid')(output)



    model = tf.keras.Model(inputs=[input_cell, input_gen, input_cats], outputs=[output])



    optimizer = tfa.optimizers.RectifiedAdam()

    if use_loookahead:

        optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=10)

    bce = tf.keras.losses.BinaryCrossentropy() 



    model.compile(optimizer=optimizer,

              loss=bce,

              metrics=['accuracy'])

    

    return model
model_configs = [

        {'type': 'stabnet', 'n_layers': 1, 'feature_dim': 512, 'output_dim': 256,'use_loookahead': True, 'feature_columns': list(train_short.columns),  'n_features': train_short.shape[1],'n_labels': labels_short.shape[1]},

    {'type': 'nn-mixed', 'layers_cell': 1, 'layers_gen': 1, 'layers_final': 0, 'units': 32, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': False, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'mask_size':3, 'use_mish':False},

      {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': False, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':False},

    {'type': 'rf', 'max_depth': 20},

    {'type': 'lr', 'max_iter': 100, 'solver': 'lbfgs', 'C': 1},

    {'type': 'xgb', 'max_depth': 20, 'learning_rate': 0.1, 'n_estimators':  50, 'booster':'gbtree'},

    {'type': 'catboost', 'iterations': 10},

    {'type': 'knn', 'k': 3}

]

models = [get_model(config) for config in model_configs]

models
def preprocess(data, experiment_config, transformations=[], prefit=False, is_testing=False):

    

    pipeline = get_pipeline(experiment_config)        

    

    train_data = data[:2]

    if not is_testing:

        valid_data = data[2:]

    

    fitted_transformations = []

    for idx, (name, transformation) in enumerate(pipeline):

        

        if len(transformations)>0:

            transformer = transformations[idx]

        else:

            transformer = None

            

        train_data, transformer, experiment_config = transformation(train_data, experiment_config, prefit, transformer=transformer)

        if not is_testing and name != 'do_mlsmote':

            valid_data, _, _ = transformation(valid_data, experiment_config, prefit=True, transformer=transformer)

        

        fitted_transformations.append(transformer)

    

    data_processed = []

    data_processed.extend(train_data)

    if not is_testing:

        data_processed.extend(valid_data)

        

    return data_processed, fitted_transformations, experiment_config
def get_pipeline(experiment_config):

    

    pipeline = []

    

    if experiment_config.get('do_mlsmote', False):

        pass

    if experiment_config.get('do_power_transform', False):

        pipeline.append(('do_power_transform',power_transform))

    if experiment_config.get('do_pca', False):

        pipeline.append(('do_pca',pca_transform))

    if experiment_config.get('do_scale', False):

        pipeline.append(('do_scale', scaler_transform)) 

    if experiment_config.get('do_mixed', False):

        pipeline.append(('do_mixed',mixed_representation))

    if experiment_config.get('do_tf_dataset', False):

        pipeline.append(('do_tf_dataset',make_tf_dataset))

        

    return pipeline



def scaler_transform(data, experiment_config, prefit=False, transformer=None):

    features, labels = data

    if prefit:

        new_features = transformer.transform(features)

    else:

        transformer = MinMaxScaler()

        new_features = transformer.fit_transform(features)

    

    return [new_features, labels], transformer, experiment_config



def power_transform(data, experiment_config, prefit=False, transformer=None):

    features, labels = data

    if prefit:

        new_features = transformer.transform(features)

    else:

        transformer = PowerTransformer()

        new_features = transformer.fit_transform(features)

    

    return [new_features, labels], transformer, experiment_config



def pca_transform(data, experiment_config, prefit=False, transformer=None):

    features, labels = data

    if prefit:

        new_features = transformer.transform(features)

    else:

        transformer = PCA(n_components=experiment_config['min_variance'], random_state=123)

        new_features = transformer.fit_transform(features)

        experiment_config['n_features'] = new_features.shape[1]

    

    return [new_features, labels], transformer, experiment_config



def transform_map(x, y, experiment_config):

    features = tf.unstack(x)

    labels = y

    

    x = dict(zip(experiment_config['columns'], features))

    y = labels

    return x, y



def make_tf_dataset(data, experiment_config, prefit=False, transformer=None):

    features, labels = data

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    

    if not prefit: # Not shuffle the validation data

        dataset = dataset.shuffle(labels.shape[0], seed=123)

    

    if experiment_config.get('ds_map', False):

        dataset = dataset.map(lambda x, y: transform_map(x, y, experiment_config))

        

    dataset = dataset.batch(experiment_config['batch_size'])#.prefetch(experiment_config['batch_size'])

    

    return [dataset], transformer, experiment_config



def mixed_representation(data, experiment_config, prefit=False, transformer=None):

    features, labels = data

    

    cell_data_train = features[:, experiment_config['cell_cols_mask']].reshape(-1, 10,10)

    gen_data_train = features[:, experiment_config['gen_cols_mask']].reshape(-1, 193,4)

    cat_data_train = features[:, experiment_config['cat_cols_mask']]

    new_features = {'input1':cell_data_train, 'input2': gen_data_train, 'input3':cat_data_train}

    

    return [new_features, labels], transformer, experiment_config
data = [train_short.values, labels_short.values]

experiment_config = {'do_power_transform': False, 'do_pca': False, 'do_scale': True, 'do_tf_dataset': False, 'min_variance': 0.9}

data_np = iterative_train_test_split(train.values, labels.values, .2)

data, pipeline, experiment_config = preprocess(data_np, experiment_config)

print(data_np[0])

print(data[0].min())

print(data[0].max())
data = [train_short.values, labels_short.values]



cell_cols_mask = ['c-' in col for col in train_short.columns]

gen_cols_mask = ['g-' in col for col in train_short.columns]

cat_cols_mask = [('g-' not in col and 'c-' not in col) for col in train_short.columns]



experiment_config = {'do_power_transform': False, 'do_pca': False, 'do_scale': True, 'do_tf_dataset': True, 'min_variance': 0.9, 'batch_size': 128, 'do_mixed': True,

                    'cell_cols_mask': cell_cols_mask, 'gen_cols_mask': gen_cols_mask, 'cat_cols_mask': cat_cols_mask}

data, pipeline,experiment_config = preprocess([X_train, Y_train, X_valid, Y_valid], experiment_config)

print(data)

x, y = data

for x, y in data[0]:

    print(x['input1'].shape)

    print(y.shape)

    break
def train_model(model, data, experiment_config, has_valid=True):

    

    metrics = pd.DataFrame(columns = ['log_loss', 'hamming_loss', 'accuracy'])

    

    best_model = None



    if experiment_config.get('do_tf_dataset', False) or experiment_config.get('is_tf', False):

        if has_valid:

            train_dataset, valid_dataset  = data

            model.fit(train_dataset, validation_data=valid_dataset, callbacks=experiment_config['callbacks'], epochs=experiment_config['epochs'], verbose=0)



            Y_valid = np.concatenate(np.array(list(valid_dataset))[:,1], axis=0)

            predictions = model.predict(valid_dataset)

        else:

            train_dataset = data[0]

            model.fit(train_dataset, callbacks=experiment_config['callbacks'], epochs=experiment_config['epochs'], verbose=0) 



    else:

        if has_valid:

            X_train, Y_train, X_valid, Y_valid = data

        else:

            X_train, Y_train = data

            

        model.fit(X_train, Y_train)

        

        if has_valid:

            predictions = model.predict_proba(X_valid)



            if type(predictions) == csr_matrix or type(predictions) == lil_matrix:

                predictions = predictions.toarray()

            elif len(np.array(predictions).shape)>2:

                predictions = np.array(predictions)[:,:,1].T 

                

    if has_valid:

        metrics = calculate_metrics(Y_valid, np.array(predictions))



    return metrics, model
def log_loss_score(actual, predicted,  eps=1e-15):

        

    p1 = actual * np.log(predicted+eps)

    p0 = (1-actual) * np.log(1-predicted+eps)

    loss = p0 + p1



    return -loss.mean()



def log_loss_multi(y_true, y_pred):

    n_labels = y_true.shape[1]

    results = np.zeros(n_labels)

    for i in range(n_labels):

        true = np.argwhere(y_true[:,i] > 0.01)

        print(true)

        results[i] = log_loss(y_true[:,i], y_pred[:,i])

    return results.mean()

        



def calculate_metrics(real, predictions):

    index = 0

    metrics = {'log_loss': 0, 'hamming_loss': 0, 'accuracy': 0}

    n_labels = real.shape[1]

    

    predictions = np.clip(np.array(predictions), 1e-15, 1-1e-15)

    predictions_minus_1 = np.clip(np.array(1-predictions), 1e-15, 1-1e-15)

    metrics['log_loss'] = -(real*np.log(predictions)+ (1-real)*np.log(predictions_minus_1)).mean() #log_loss_multi(real, predictions)

    metrics['hamming_loss'] = hamming_loss(real, predictions.astype(int))

    metrics['accuracy'] = accuracy_score(real, predictions.astype(int))

    

    metrics = pd.DataFrame(metrics, index=[index])

    index += 1

    return metrics



def update_metrics(name, current_metrics, metrics=None):

    if metrics is None:

        metrics = pd.DataFrame(columns = ['id', 'log_loss', 'hamming_loss', 'accuracy'])

        

    current_metrics['id'] = name

    metrics = pd.concat([metrics, current_metrics])

    

    return metrics
experiment_config = {'name':'Dense-NN','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': None, 'epochs': 30,  'do_pca': False,  'min_variance': 0.8}





data_np = iterative_train_test_split(train_short.values, labels_short.values, .2)

data, pipeline, experiment_config = preprocess(data_np, experiment_config)



model_config =  {'type': 'nn-dense', 'n_layers': 5, 'units': 512, 'drop_rate': 0.2, 

     'use_weight_normalization': True, 'use_loookahead': True, 'use_batch_norm': True, 'n_features': train.shape[1], #experiment_config['n_features'],

     'n_labels': labels_short.shape[1], 'use_mish':True}



model = get_model(model_config)

train_dataset, valid_dataset  = data



metrics, model = train_model(model, data, experiment_config)

metrics
cell_cols_mask = ['c-' in col for col in train.columns]

gen_cols_mask = ['g-' in col for col in train.columns]

cat_cols_mask = [('g-' not in col and 'c-' not in col) for col in train.columns]



experiment_configs = [

                    {'name':'Tabnet','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': None, 'epochs': 30, 'ds_map': True,'columns': list(train_short.columns)},

                    {'name':'NN-Mixed', 'is_tf': True, 'do_scale': True, 'do_tf_dataset': True, 'batch_size': 128, 'callbacks': None, 'epochs': 30, 'do_mixed': True,

                     'cell_cols_mask': cell_cols_mask, 'gen_cols_mask': gen_cols_mask, 'cat_cols_mask': cat_cols_mask}, 

    

                     {'name':'Dense-NN','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': None, 'epochs': 30},

                     

                    {'name':'KNN','do_scale': True},

                     {'name':'RandomForest'},

                     {'name':'LogisticRegression','do_scale': True,}, {'name':'XGB'}, {'name':'Catboost'}

                     

                    ]



cell_cols_mask = ['c-' in col for col in train_short.columns]

gen_cols_mask = ['g-' in col for col in train_short.columns]

cat_cols_mask = [('g-' not in col and 'c-' not in col) for col in train_short.columns]



model_configs = [

    {'type': 'stabnet', 'n_layers': 1, 'feature_dim': 512, 'output_dim': 256,'use_loookahead': True, 'feature_columns': list(train_short.columns),  'n_features': train_short.shape[1],'n_labels': labels_short.shape[1]},

    {'type': 'nn-mixed', 'layers_cell': 1, 'layers_gen': 1, 'layers_final': 0, 'units': 32, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': True, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'mask_size':3, 'use_mish':True},

    {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': True, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':True},

    

    {'type': 'knn', 'k': 3},

    { 'type': 'rf', 'max_depth': 20, 'n_estimators':  100},

    {'type': 'lr', 'max_iter': 200, 'solver': 'lbfgs', 'C': 1},

    {'type': 'xgb', 'max_depth': 20, 'learning_rate': 0.1, 'n_estimators':  100, 'booster':'gbtree'},

    {'type': 'catboost', 'iterations': 100}

]



configurations = zip(model_configs, experiment_config)
%%time



metrics = None



configurations = zip(model_configs, experiment_configs)

for model_config, experiment_config in configurations:     

    config = {'model_config': model_config, 'experiment_config': experiment_config} 

    data = iterative_train_test_split(train_short.values, labels_short.values, .2)

    model = get_model(model_config)

    experiment_metrics, _ = run_experiment(data, config)

    metrics = update_metrics(experiment_config['name'], experiment_metrics, metrics)

metrics
experiment_configs = [

                     {'name':'LogisticRegression-Base','do_scale': True},

                     {'name':'LogisticRegression-Power','do_scale': True, 'do_power_transform': True},

                     {'name':'LogisticRegression-PCA','do_scale': True, 'do_pca': True,  'min_variance': 0.8},

                    {'name':'LogisticRegression-Power+PCA','do_scale': True, 'do_pca': True, 'do_power_transform': True, 'min_variance': 0.8}

                    ]



model_configs = [

    {'type': 'lr', 'max_iter': 500, 'solver': 'lbfgs', 'C': 1},

    {'type': 'lr', 'max_iter': 500, 'solver': 'lbfgs', 'C': 1},

    {'type': 'lr', 'max_iter': 500, 'solver': 'lbfgs', 'C': 1},

    {'type': 'lr', 'max_iter': 500, 'solver': 'lbfgs', 'C': 1}

]



configurations = zip(model_configs, experiment_config)
%%time



metrics_lr = None



configurations = zip(model_configs, experiment_configs)

for model_config, experiment_config in configurations:

    config = {'model_config': model_config, 'experiment_config': experiment_config} 

    data = iterative_train_test_split(train_short.values, labels_short.values, .2)

    model = get_model(model_config)

    print(model)

    experiment_metrics, _ = run_experiment(data, config)

    metrics_lr = update_metrics(experiment_config['name'], experiment_metrics, metrics_lr)

metrics_lr




early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', min_lr = 1e-7,  factor=0.1, patience=3, verbose=1)

callbacks = [early_stop, reduce_lr]



experiment_configs = [

                     {'name':'LogisticRegression-Base','do_scale': True},

                        

                     {'name':'Dense-NN-original','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': callbacks, 'epochs': 30},

                     {'name':'Dense-NN-lookahead','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': callbacks, 'epochs': 30},

                     {'name':'Dense-NN-batchnorm','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': callbacks, 'epochs': 30},

                     {'name':'Dense-NN-weightnorm','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': callbacks, 'epochs': 30},

                     {'name':'Dense-NN-mish','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': callbacks, 'epochs': 30},

                     {'name':'Dense-NN-all','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': callbacks, 'epochs': 30},

                     {'name':'Dense-NN-PCA','is_tf': True, 'do_scale': True, 'do_tf_dataset': True, 'do_pca': True, 'batch_size': 128, 'callbacks': callbacks, 'epochs': 30, 'min_variance': 0.8},

                     {'name':'Dense-NN-Power','is_tf': True, 'do_scale': True, 'do_tf_dataset': True, 'do_power_transform': True, 'batch_size': 128, 'callbacks': callbacks, 'epochs': 30},

                     {'name':'Dense-NN-PCA+lookahead+mish','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': callbacks, 

                          'epochs': 30, 'do_pca': True, 'do_power_transform': False, 'min_variance': 0.8 }, 

                     {'name':'Dense-NN-Power+lookahead+mish','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': callbacks, 

                          'epochs': 30, 'do_pca': False, 'do_power_transform': True, 'min_variance': 0.8 }, 

                     {'name':'Dense-NN-improved','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': callbacks, 

                          'epochs': 30, 'do_pca': True, 'do_power_transform': True, 'min_variance': 0.8 },

                    

                    {'name':'NN-Mixed-original', 'is_tf': True, 'do_scale': True, 'do_tf_dataset': True, 'batch_size': 128, 'callbacks': None, 'epochs': 30, 'do_mixed': True,

                                     'cell_cols_mask': cell_cols_mask, 'gen_cols_mask': gen_cols_mask, 'cat_cols_mask': cat_cols_mask},

                    {'name':'NN-Mixed-lookahead', 'is_tf': True, 'do_scale': True, 'do_tf_dataset': True, 'batch_size': 128, 'callbacks': None, 'epochs': 30, 'do_mixed': True,

                                     'cell_cols_mask': cell_cols_mask, 'gen_cols_mask': gen_cols_mask, 'cat_cols_mask': cat_cols_mask},

                    {'name':'NN-Mixed-batchnorm', 'is_tf': True, 'do_scale': True, 'do_tf_dataset': True, 'batch_size': 128, 'callbacks': None, 'epochs': 30, 'do_mixed': True,

                                     'cell_cols_mask': cell_cols_mask, 'gen_cols_mask': gen_cols_mask, 'cat_cols_mask': cat_cols_mask},

                    {'name':'NN-Mixed-weightnorm', 'is_tf': True, 'do_scale': True, 'do_tf_dataset': True, 'batch_size': 128, 'callbacks': None, 'epochs': 30, 'do_mixed': True,

                                     'cell_cols_mask': cell_cols_mask, 'gen_cols_mask': gen_cols_mask, 'cat_cols_mask': cat_cols_mask},

                    {'name':'NN-Mixed-mish', 'is_tf': True, 'do_scale': True, 'do_tf_dataset': True, 'batch_size': 128, 'callbacks': None, 'epochs': 30, 'do_mixed': True,

                                     'cell_cols_mask': cell_cols_mask, 'gen_cols_mask': gen_cols_mask, 'cat_cols_mask': cat_cols_mask},

                    {'name':'NN-Mixed-all', 'is_tf': True, 'do_scale': True, 'do_tf_dataset': True, 'batch_size': 128, 'callbacks': None, 'epochs': 30, 'do_mixed': True,

                                     'cell_cols_mask': cell_cols_mask, 'gen_cols_mask': gen_cols_mask, 'cat_cols_mask': cat_cols_mask},

                    {'name':'NN-Mixed-Power', 'is_tf': True, 'do_scale': True, 'do_tf_dataset': True, 'batch_size': 128, 'callbacks': None, 'epochs': 30, 'do_mixed': True,

                                     'cell_cols_mask': cell_cols_mask, 'gen_cols_mask': gen_cols_mask, 'cat_cols_mask': cat_cols_mask, 'do_power_transform': True,}

     

                    ]





model_configs = [

        {'type': 'lr', 'max_iter': 500, 'solver': 'lbfgs', 'C': 1},

        

    {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': True, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':False},

    {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': True, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':False},

    {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': False, 'use_batch_norm': True, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':False},

    {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': True, 'use_loookahead': False, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':False},

    {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': False, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':True},

    {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': True, 'use_loookahead': True, 'use_batch_norm': True, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':True},

    {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': False, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':False},

    {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': False, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':False},

    {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': True, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':True},

    {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': True, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':True},

    {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': True, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'use_mish':True},

    

    {'type': 'nn-mixed', 'layers_cell': 1, 'layers_gen': 1, 'layers_final': 0, 'units': 32, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': False, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'mask_size':3, 'use_mish':False},

    {'type': 'nn-mixed', 'layers_cell': 1, 'layers_gen': 1, 'layers_final': 0, 'units': 32, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': True, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'mask_size':3, 'use_mish':False},

    {'type': 'nn-mixed', 'layers_cell': 1, 'layers_gen': 1, 'layers_final': 0, 'units': 32, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': False, 'use_batch_norm': True, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'mask_size':3, 'use_mish':False},

    {'type': 'nn-mixed', 'layers_cell': 1, 'layers_gen': 1, 'layers_final': 0, 'units': 32, 'drop_rate': 0.3, 

     'use_weight_normalization': True, 'use_loookahead': False, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'mask_size':3, 'use_mish':False},

    {'type': 'nn-mixed', 'layers_cell': 1, 'layers_gen': 1, 'layers_final': 0, 'units': 32, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': False, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'mask_size':3, 'use_mish':True},

    {'type': 'nn-mixed', 'layers_cell': 1, 'layers_gen': 1, 'layers_final': 0, 'units': 32, 'drop_rate': 0.3, 

     'use_weight_normalization': True, 'use_loookahead': True, 'use_batch_norm': True, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'mask_size':3, 'use_mish':True},

    {'type': 'nn-mixed', 'layers_cell': 1, 'layers_gen': 1, 'layers_final': 0, 'units': 32, 'drop_rate': 0.3, 

     'use_weight_normalization': False, 'use_loookahead': False, 'use_batch_norm': False, 'n_features': train_short.shape[1],

     'n_labels': labels_short.shape[1], 'mask_size':3, 'use_mish':False}

]

   



configurations = zip(model_configs, experiment_config)
%%time



metrics_nn = None



configurations = zip(model_configs, experiment_configs)

for model_config, experiment_config in configurations:

    config = {'model_config': model_config, 'experiment_config': experiment_config} 

    data = iterative_train_test_split(train.values, labels.values, .2)

    model = get_model(model_config)

    experiment_metrics, _ = run_experiment(data, config)

    metrics_nn = update_metrics(experiment_config['name'], experiment_metrics, metrics_nn)

metrics_nn
def dense_nn_kt(hp, n_features, n_labels):

    model = tf.keras.Sequential()

    layers = hp.Int('layers',min_value=1,max_value=10,step=1)

    use_weight_normalization = hp.Boolean('use_weight_norm', default=False)

    use_loookahead = hp.Boolean('use_loookahead', default=False)

    use_mish = hp.Boolean('use_mish', default=False)

    

    for l in range(layers):

        use_batch_norm = True #hp.Boolean(f'use_batch_norm{l}', default=False)

        dense = tf.keras.layers.Dense(units=hp.Int(f'units_{l}',min_value=32,max_value=1024,step=32))

        if use_weight_normalization:

             dense = tfa.layers.WeightNormalization(dense)

        model.add(dense)

        if use_batch_norm:

             model.add(tf.keras.layers.BatchNormalization())

                

        if use_mish:

            model.add(tf.keras.layers.Activation(tfa.activations.mish))

        else:

            model.add(tf.keras.layers.Activation('relu'))

        

        drop_rate= hp.Float(f'dropout_{l}',min_value=0.0,max_value=0.5,default=0.25,step=0.05 )

        model.add(tf.keras.layers.Dropout(drop_rate))

        

    model.add(tf.keras.layers.Dense(n_labels, activation='sigmoid'))

    

    learning_rate = hp.Float('learning_rate', min_value=3e-4, max_value=3e-2)

    sync_period = hp.Int('sync_period',min_value=5,max_value=30,step=1)

    

    optimizer = optimizer = tfa.optimizers.RectifiedAdam(learning_rate)

    if use_loookahead:

        optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=sync_period)

    

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', min_lr = 1e-7,  factor=0.1, patience=3, verbose=1)

callbacks = [early_stop, reduce_lr]



experiment_config =  {'name':'Dense-NN-batchnorm','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': callbacks, 'epochs': 30}







tuner = kt.Hyperband(partial(dense_nn_kt,n_features=train.shape[1], n_labels=labels.shape[1]), objective = 'val_loss', max_epochs = 100, factor = 10, seed=123,

                            directory=f'./nn_tuner', project_name='nn_tuner')



data_np = iterative_train_test_split(train.values, labels.values, .2)

data, pipeline, experiment_config = preprocess(data_np, experiment_config)



train_dataset, valid_dataset = data



tuner.search(train_dataset, validation_data=valid_dataset, callbacks=callbacks, verbose=0)

model = tuner.get_best_models()[0]



Y_valid = np.concatenate(np.array(list(valid_dataset))[:,1], axis=0)

predictions = model.predict(valid_dataset)

calculate_metrics(Y_valid, predictions)
tries = 5

early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', min_lr = 1e-7,  factor=0.1, patience=3, verbose=0)

callbacks = [early_stop, reduce_lr]



experiment_config_nn = {'name':'Dense-NN-batchnorm','is_tf': True, 'do_scale': True, 'do_tf_dataset': True,  'batch_size': 128, 'callbacks': callbacks, 'epochs': 30}

experiment_config_lr =  {'name':'LogisticRegression-Base','do_scale': True}



model_config_lr={'type': 'lr', 'max_iter': 500, 'solver': 'lbfgs', 'C': 1}

model_config_nn = {'type': 'nn-dense', 'n_layers': 2, 'units': 256, 'drop_rate': 0.3, 'use_weight_normalization': False, 

                    'use_loookahead': False, 'use_batch_norm': True, 'n_features': train_short.shape[1],'n_labels': labels_short.shape[1], 'use_mish':False}

                                        

_, transformations, _ = preprocess([train.values, labels.values], experiment_config_lr, is_testing=True)

data_test, _, _ = preprocess([test.values, None], experiment_config_lr, transformations=transformations, prefit=True, is_testing=True)



predictions = np.zeros((test.shape[0], labels.shape[1]))

for i in range(tries):

    

    data_np = iterative_train_test_split(train.values, labels.values, .2)

    data_lr, _, experiment_config_lr = preprocess(data_np, experiment_config_lr)

    data_nn, _, experiment_config_nn = preprocess(data_np, experiment_config_nn)

    model_nn = dense_nn_kt(tuner.get_best_hyperparameters()[0], n_features=train.shape[1], n_labels=labels.shape[1])

    model_lr = get_model(model_config_lr)

    #model_nn = get_model(model_config_nn)



    _, model_lr = train_model(model_lr, data_lr, experiment_config_lr)

    _, model_nn = train_model(model_nn, data_nn, experiment_config_nn)



    test_dataset = data_test[0]

    predictions += (model_lr.predict(test_dataset) + model_nn.predict(test_dataset))/2

    

predictions = predictions/tries
submission =  pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

submission = submission.set_index('sig_id')

submission.loc[:,:] = np.zeros(submission.shape)

submission.loc[test.index,:] = predictions



submission.to_csv('submission.csv')

submission