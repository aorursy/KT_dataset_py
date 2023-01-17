import math

import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer

from tensorflow.keras.initializers import Constant

from tensorflow.python.keras.utils import tf_utils





def _resolve_training(layer, training):

    if training is None:

        training = K.learning_phase()

    if isinstance(training, int):

        training = bool(training)

    if not layer.trainable:

        # When the layer is not trainable, override the value

        training = False

    return tf_utils.constant_value(training)
class ArcFace(Layer):

    """

    Implementation of ArcFace layer. Reference: https://arxiv.org/abs/1801.07698

    

    Arguments:

      num_classes: number of classes to classify

      s: scale factor

      m: margin

      regularizer: weights regularizer

    """

    def __init__(self,

                 num_classes,

                 s=30.0,

                 m=0.5,

                 regularizer=None,

                 name='arcface',

                 **kwargs):

        

        super().__init__(name=name, **kwargs)

        self._n_classes = num_classes

        self._s = float(s)

        self._m = float(m)

        self._regularizer = regularizer



    def build(self, input_shape):

        embedding_shape, label_shape = input_shape

        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),

                                  initializer='glorot_uniform',

                                  trainable=True,

                                  regularizer=self._regularizer,

                                  name='cosine_weights')



    def call(self, inputs, training=None):

        """

        During training, requires 2 inputs: embedding (after backbone+pool+dense),

        and ground truth labels. The labels should be sparse (and use

        sparse_categorical_crossentropy as loss).

        """

        embedding, label = inputs



        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)

        label = tf.reshape(label, [-1], name='label_shape_correction')



        # Normalize features and weights and compute dot product

        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')

        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')

        cosine_sim = tf.matmul(x, w, name='cosine_similarity')



        training = resolve_training_flag(self, training)

        if not training:

            # We don't have labels if we're not in training mode

            return self._s * cosine_sim

        else:

            one_hot_labels = tf.one_hot(label,

                                        depth=self._n_classes,

                                        name='one_hot_labels')

            theta = tf.math.acos(K.clip(

                    cosine_sim, -1.0 + K.epsilon(), 1.0 - K.epsilon()))

            selected_labels = tf.where(tf.greater(theta, math.pi - self._m),

                                       tf.zeros_like(one_hot_labels),

                                       one_hot_labels,

                                       name='selected_labels')

            final_theta = tf.where(tf.cast(selected_labels, dtype=tf.bool),

                                   theta + self._m,

                                   theta,

                                   name='final_theta')

            output = tf.math.cos(final_theta, name='cosine_sim_with_margin')

            return self._s * output
class CosFace(Layer):

    """

    Implementation of CosFace layer. Reference: https://arxiv.org/abs/1801.09414

    

    Arguments:

      num_classes: number of classes to classify

      s: scale factor

      m: margin

      regularizer: weights regularizer

    """

    def __init__(self,

                 num_classes,

                 s=30.0,

                 m=0.35,

                 regularizer=None,

                 name='cosface',

                 **kwargs):



        super().__init__(name=name, **kwargs)

        self._n_classes = num_classes

        self._s = float(s)

        self._m = float(m)

        self._regularizer = regularizer



    def build(self, input_shape):

        embedding_shape, label_shape = input_shape

        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),

                                  initializer='glorot_uniform',

                                  trainable=True,

                                  regularizer=self._regularizer)



    def call(self, inputs, training=None):

        """

        During training, requires 2 inputs: embedding (after backbone+pool+dense),

        and ground truth labels. The labels should be sparse (and use

        sparse_categorical_crossentropy as loss).

        """

        embedding, label = inputs



        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)

        label = tf.reshape(label, [-1], name='label_shape_correction')



        # Normalize features and weights and compute dot product

        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')

        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')

        cosine_sim = tf.matmul(x, w, name='cosine_similarity')



        training = _resolve_training(self, training)

        if not training:

            # We don't have labels if we're not in training mode

            return self._s * cosine_sim

        else:

            one_hot_labels = tf.one_hot(label,

                                        depth=self._n_classes,

                                        name='one_hot_labels')

            final_theta = tf.where(tf.cast(one_hot_labels, dtype=tf.bool),

                                   cosine_sim - self._m,

                                   cosine_sim,

                                   name='cosine_sim_with_margin')

            return self._s * output
class AdaCos(Layer):

    """

    Implementation of AdaCos layer. Reference: https://arxiv.org/abs/1905.00292

    

    Arguments:

      num_classes: number of classes to classify

      is_dynamic: if False, use Fixed AdaCos. Else, use Dynamic Adacos.

      regularizer: weights regularizer

    """

    def __init__(self,

                 num_classes,

                 is_dynamic=True,

                 regularizer=None,

                 **kwargs):



        super().__init__(**kwargs)

        self._n_classes = num_classes

        self._init_s = math.sqrt(2) * math.log(num_classes - 1)

        self._is_dynamic = is_dynamic

        self._regularizer = regularizer



    def build(self, input_shape):

        embedding_shape, label_shape = input_shape

        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),

                                  initializer='glorot_uniform',

                                  trainable=True,

                                  regularizer=self._regularizer)

        if self._is_dynamic:

            self._s = self.add_weight(shape=(),

                                      initializer=Constant(self._init_s),

                                      trainable=False,

                                      aggregation=tf.VariableAggregation.MEAN)



    def call(self, inputs, training=None):

        embedding, label = inputs



        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)

        label = tf.reshape(label, [-1])



        # Normalize features and weights and compute dot product

        x = tf.nn.l2_normalize(embedding, axis=1)

        w = tf.nn.l2_normalize(self._w, axis=0)

        logits = tf.matmul(x, w)



        # Fixed AdaCos

        is_dynamic = tf_utils.constant_value(self._is_dynamic)

        if not is_dynamic:

            # _s is not created since we are not in dynamic mode

            output = tf.multiply(self._init_s, logits)

            return output



        training = _resolve_training(self, training)

        if not training:

            # We don't have labels to update _s if we're not in training mode

            return self._s * logits

        else:

            theta = tf.math.acos(

                    K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))

            one_hot = tf.one_hot(label, depth=self._n_classes)

            b_avg = tf.where(one_hot < 1.0,

                             tf.exp(self._s * logits),

                             tf.zeros_like(logits))

            b_avg = tf.reduce_mean(tf.reduce_sum(b_avg, axis=1))

            theta_class = tf.gather_nd(

                    theta,

                    tf.stack([

                        tf.range(tf.shape(label)[0]),

                        tf.cast(label, tf.int32)

                    ], axis=1))

            mid_index = tf.shape(theta_class)[0] // 2 + 1

            theta_med = tf.nn.top_k(theta_class, mid_index).values[-1]



            # Since _s is not trainable, this assignment is safe. Also,

            # tf.function ensures that this will run in the right order.

            self._s.assign(

                    tf.math.log(b_avg) /

                    tf.math.cos(tf.minimum(math.pi/4, theta_med)))



            # Return scaled logits

            return self._s * logits