import warnings

warnings.filterwarnings("ignore")



import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



import os

import gc

import datetime

import numpy as np

import pandas as pd

import tensorflow as tf

tf.random.set_seed(42)

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import tensorflow_addons as tfa

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import log_loss

from scipy.optimize import minimize

from tqdm.notebook import tqdm

from time import time



print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
MIXED_PRECISION = False

XLA_ACCELERATE = True



if MIXED_PRECISION:

    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    if tpu: policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')

    else: policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')

    mixed_precision.set_policy(policy)

    print('Mixed precision enabled')



if XLA_ACCELERATE:

    tf.config.optimizer.set_jit(True)

    print('Accelerated Linear Algebra enabled')
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')



ss = pd.read_csv('../input/lish-moa/sample_submission.csv')



cols = [c for c in ss.columns.values if c != 'sig_id']
def preprocess(df):

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    del df['sig_id']

    return df



# [Fast Numpy Log Loss] https://www.kaggle.com/gogo827jz/optimise-blending-weights-4-5x-faster-log-loss

def log_loss_metric(y_true, y_pred):

    loss = 0

    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)

    for i in range(y_pred.shape[1]):

        loss += - np.mean(y_true[:, i] * np.log(y_pred_clip[:, i]) + (1 - y_true[:, i]) * np.log(1 - y_pred_clip[:, i]))

    return loss / y_pred.shape[1]



train = preprocess(train_features)

test = preprocess(test_features)



del train_targets['sig_id']
top_feats = [  0,   1,   2,   3,   5,   6,   8,   9,  10,  11,  12,  14,  15,

        16,  18,  19,  20,  21,  23,  24,  25,  27,  28,  29,  30,  31,

        32,  33,  34,  35,  36,  37,  39,  40,  41,  42,  44,  45,  46,

        48,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,

        63,  64,  65,  66,  68,  69,  70,  71,  72,  73,  74,  75,  76,

        78,  79,  80,  81,  82,  83,  84,  86,  87,  88,  89,  90,  92,

        93,  94,  95,  96,  97,  99, 100, 101, 103, 104, 105, 106, 107,

       108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,

       121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134,

       135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,

       149, 150, 151, 152, 153, 154, 155, 157, 159, 160, 161, 163, 164,

       165, 166, 167, 168, 169, 170, 172, 173, 175, 176, 177, 178, 180,

       181, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 195,

       197, 198, 199, 202, 203, 205, 206, 208, 209, 210, 211, 212, 213,

       214, 215, 218, 219, 220, 221, 222, 224, 225, 227, 228, 229, 230,

       231, 232, 233, 234, 236, 238, 239, 240, 241, 242, 243, 244, 245,

       246, 248, 249, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260,

       261, 263, 265, 266, 268, 270, 271, 272, 273, 275, 276, 277, 279,

       282, 283, 286, 287, 288, 289, 290, 294, 295, 296, 297, 299, 300,

       301, 302, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313, 315,

       316, 317, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331,

       332, 333, 334, 335, 338, 339, 340, 341, 343, 344, 345, 346, 347,

       349, 350, 351, 352, 353, 355, 356, 357, 358, 359, 360, 361, 362,

       363, 364, 365, 366, 368, 369, 370, 371, 372, 374, 375, 376, 377,

       378, 379, 380, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,

       392, 393, 394, 395, 397, 398, 399, 400, 401, 403, 405, 406, 407,

       408, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422,

       423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435,

       436, 437, 438, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450,

       452, 453, 454, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465,

       466, 468, 469, 471, 472, 473, 474, 475, 476, 477, 478, 479, 482,

       483, 485, 486, 487, 488, 489, 491, 492, 494, 495, 496, 500, 501,

       502, 503, 505, 506, 507, 509, 510, 511, 512, 513, 514, 516, 517,

       518, 519, 521, 523, 525, 526, 527, 528, 529, 530, 531, 532, 533,

       534, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547,

       549, 550, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563,

       564, 565, 566, 567, 569, 570, 571, 572, 573, 574, 575, 577, 580,

       581, 582, 583, 586, 587, 590, 591, 592, 593, 595, 596, 597, 598,

       599, 600, 601, 602, 603, 605, 607, 608, 609, 611, 612, 613, 614,

       615, 616, 617, 619, 622, 623, 625, 627, 630, 631, 632, 633, 634,

       635, 637, 638, 639, 642, 643, 644, 645, 646, 647, 649, 650, 651,

       652, 654, 655, 658, 659, 660, 661, 662, 663, 664, 666, 667, 668,

       669, 670, 672, 674, 675, 676, 677, 678, 680, 681, 682, 684, 685,

       686, 687, 688, 689, 691, 692, 694, 695, 696, 697, 699, 700, 701,

       702, 703, 704, 705, 707, 708, 709, 711, 712, 713, 714, 715, 716,

       717, 723, 725, 727, 728, 729, 730, 731, 732, 734, 736, 737, 738,

       739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751,

       752, 753, 754, 755, 756, 758, 759, 760, 761, 762, 763, 764, 765,

       766, 767, 769, 770, 771, 772, 774, 775, 780, 781, 782, 783, 784,

       785, 787, 788, 790, 793, 795, 797, 799, 800, 801, 805, 808, 809,

       811, 812, 813, 816, 819, 820, 821, 822, 823, 825, 826, 827, 829,

       831, 832, 833, 834, 835, 837, 838, 839, 840, 841, 842, 844, 845,

       846, 847, 848, 850, 851, 852, 854, 855, 856, 858, 860, 861, 862,

       864, 867, 868, 870, 871, 873, 874]

print(len(top_feats))
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



        if feature_dim < output_dim:

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



        # if num_decision_steps > 1:

            # features_for_coeff = feature_dim - output_dim

            # print(f"[TabNet]: {features_for_coeff} features will be used for decision steps.")



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





class TabNetClassifier(tf.keras.Model):



    def __init__(self, feature_columns,

                 num_classes,

                 num_features=None,

                 feature_dim=64,

                 output_dim=64,

                 num_decision_steps=5,

                 relaxation_factor=1.5,

                 sparsity_coefficient=1e-5,

                 norm_type='group',

                 batch_momentum=0.98,

                 virtual_batch_size=None,

                 num_groups=1,

                 epsilon=1e-5,

                 multi_label=False, 

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

            num_classes: Number of classes.

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

                'group' or 'group'. 'group' is the default.

            batch_momentum: Momentum in ghost batch normalization.

            virtual_batch_size: Virtual batch size in ghost batch normalization. The

                overall batch size should be an integer multiple of virtual_batch_size.

            num_groups: Number of groups used for group normalization.

            epsilon: A small number for numerical stability of the entropy calculations.

        """

        super(TabNetClassifier, self).__init__(**kwargs)



        self.num_classes = num_classes



        self.tabnet = TabNet(feature_columns=feature_columns,

                    num_features=num_features,

                    feature_dim=feature_dim,

                    output_dim=output_dim,

                    num_decision_steps=num_decision_steps,

                    relaxation_factor=relaxation_factor,

                    sparsity_coefficient=sparsity_coefficient,

                    norm_type=norm_type,

                    batch_momentum=batch_momentum,

                    virtual_batch_size=virtual_batch_size,

                    num_groups=num_groups,

                    epsilon=epsilon,

                    **kwargs)

        

        if multi_label:

            

            self.clf = tf.keras.layers.Dense(num_classes, activation='sigmoid', use_bias=False, name='classifier')

            

        else:

            

            self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False, name='classifier')



    def call(self, inputs, training=None):

        self.activations = self.tabnet(inputs, training=training)

        out = self.clf(self.activations)



        return out



    def summary(self, *super_args, **super_kwargs):

        super().summary(*super_args, **super_kwargs)

        self.tabnet.summary(*super_args, **super_kwargs)





class TabNetRegressor(tf.keras.Model):



    def __init__(self, feature_columns,

                 num_regressors,

                 num_features=None,

                 feature_dim=64,

                 output_dim=64,

                 num_decision_steps=5,

                 relaxation_factor=1.5,

                 sparsity_coefficient=1e-5,

                 norm_type='group',

                 batch_momentum=0.98,

                 virtual_batch_size=None,

                 num_groups=1,

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

            num_regressors: Number of regression variables.

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

                'group' or 'group'. 'group' is the default.

            batch_momentum: Momentum in ghost batch normalization.

            virtual_batch_size: Virtual batch size in ghost batch normalization. The

                overall batch size should be an integer multiple of virtual_batch_size.

            num_groups: Number of groups used for group normalization.

            epsilon: A small number for numerical stability of the entropy calculations.

        """

        super(TabNetRegressor, self).__init__(**kwargs)



        self.num_regressors = num_regressors



        self.tabnet = TabNet(feature_columns=feature_columns,

                             num_features=num_features,

                             feature_dim=feature_dim,

                             output_dim=output_dim,

                             num_decision_steps=num_decision_steps,

                             relaxation_factor=relaxation_factor,

                             sparsity_coefficient=sparsity_coefficient,

                             norm_type=norm_type,

                             batch_momentum=batch_momentum,

                             virtual_batch_size=virtual_batch_size,

                             num_groups=num_groups,

                             epsilon=epsilon,

                             **kwargs)



        self.regressor = tf.keras.layers.Dense(num_regressors, use_bias=False, name='regressor')



    def call(self, inputs, training=None):

        self.activations = self.tabnet(inputs, training=training)

        out = self.regressor(self.activations)

        return out



    def summary(self, *super_args, **super_kwargs):

        super().summary(*super_args, **super_kwargs)

        self.tabnet.summary(*super_args, **super_kwargs)





# Aliases

TabNetClassification = TabNetClassifier

TabNetRegression = TabNetRegressor



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

                 multi_label = False, 

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

        if multi_label:

            

            self.clf = tf.keras.layers.Dense(num_classes, activation='sigmoid', use_bias=False)

        

        else:

            

            self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False)



    def call(self, inputs, training=None):

        self.activations = self.stacked_tabnet(inputs, training=training)

        out = self.clf(self.activations)



        return out





class StackedTabNetRegressor(tf.keras.Model):



    def __init__(self, feature_columns,

                 num_regressors,

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

            num_regressors: Number of regressors.

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

        super(StackedTabNetRegressor, self).__init__(**kwargs)



        self.num_regressors = num_regressors



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



        self.regressor = tf.keras.layers.Dense(num_regressors, use_bias=False)



    def call(self, inputs, training=None):

        self.activations = self.tabnet(inputs, training=training)

        out = self.regressor(self.activations)

        return outl
N_STARTS = 5

N_SPILTS = 10



res = train_targets.copy()

ss.loc[:, train_targets.columns] = 0

res.loc[:, train_targets.columns] = 0



for seed in range(N_STARTS):

    

    for n, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits = N_SPILTS, random_state = seed, shuffle = True).split(train_targets, train_targets)):

        

        start_time = time()

        x_tr, x_val = train.values[tr][:, top_feats], train.values[te][:, top_feats]

        y_tr, y_val = train_targets.astype(float).values[tr], train_targets.astype(float).values[te]

        x_tt = test_features.values[:, top_feats]

        

        model = StackedTabNetClassifier(feature_columns = None, num_classes = 206, num_layers = 2, 

                                        feature_dim = 128, output_dim = 64, num_features = len(top_feats),

                                        num_decision_steps = 1, relaxation_factor = 1.5,

                                        sparsity_coefficient = 1e-5, batch_momentum = 0.98,

                                        virtual_batch_size = None, norm_type = 'group',

                                        num_groups = -1, multi_label = True)



        model.compile(optimizer = tfa.optimizers.Lookahead(tf.optimizers.Adam(1e-3), sync_period = 10), 

                      loss = 'binary_crossentropy')

        

        rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 0, 

                                min_delta = 1e-4, mode = 'min')

        ckp = ModelCheckpoint(f'TabNet_{seed}_{n}.hdf5', monitor = 'val_loss', verbose = 0, 

                              save_best_only = True, save_weights_only = True, mode = 'min')

        es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 10, mode = 'min', 

                           baseline = None, restore_best_weights = True, verbose = 0)

        

        model.fit(x_tr, y_tr, validation_data = (x_val, y_val), epochs = 100, batch_size = 128,

                  callbacks = [rlr, ckp, es], verbose = 0)

        

        model.load_weights(f'TabNet_{seed}_{n}.hdf5')

        ss.loc[:, train_targets.columns] += model.predict(x_tt, batch_size = x_tt.shape[0]) / (N_SPILTS * N_STARTS)

        fold_pred = model.predict(x_val, batch_size = x_val.shape[0])

        res.loc[te, train_targets.columns] += fold_pred / N_STARTS

        fold_score = log_loss_metric(train_targets.loc[te].values, fold_pred)

        print(f'[{str(datetime.timedelta(seconds = time() - start_time))[2:7]}] TabNet: Seed {seed}, Fold {n}:', fold_score)

        

        K.clear_session()

        del model

        x = gc.collect()
print(f'TabNet OOF Metric: {log_loss_metric(train_targets.values, res.values)}')

res.loc[train['cp_type'] == 1, train_targets.columns] = 0

ss.loc[test['cp_type'] == 1, train_targets.columns] = 0

print(f'TabNet OOF Metric with postprocessing: {log_loss_metric(train_targets.values, res.values)}')
ss.to_csv('submission.csv', index = False)