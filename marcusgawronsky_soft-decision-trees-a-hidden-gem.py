from typing import List, Tuple

from operator import add

from itertools import chain



import tensorflow as tf

import pandas as pd

import holoviews as hv

from sklearn.datasets import load_digits, load_boston

from sklearn.metrics import confusion_matrix

from toolz.curried import *





hv.extension('bokeh')



digits = load_digits()



digits_X = tf.convert_to_tensor(digits.data.astype('float32'))

digits_y = tf.convert_to_tensor(digits.target.astype('float32'))

digits_y_sparse = tf.one_hot(tf.dtypes.cast(digits_y, 'int32'), 9)
class Node(tf.keras.layers.Layer):



    def __init__(self, units:int=1, alpha:float = 0.1, l1:float = 0.1):

        super(Node, self).__init__()

        self.units = units

        self.alpha = alpha

        self.l1 = l1



    def build(self, input_shape):

        self.w = self.add_weight(shape=(input_shape[-1], self.units),

                                 initializer='random_normal',

                                 trainable=True)

        

        self.b = self.add_weight(shape=(1,self.units),

                                 initializer='random_normal',

                                 trainable=True)



    def call(self, inputs, probability):

        

        pr = tf.nn.sigmoid(tf.matmul(inputs, self.w) + self.b)

        pr_one = tf.reshape(pr, (-1,))



        

        l1 = tf.reduce_sum(tf.abs(self.w))

        self.add_loss(self.l1 * l1)

        

        p = tf.reduce_mean(pr)

        self.add_loss(self.alpha * tf.keras.losses.binary_crossentropy([0.5], p))



        

        return (pr * inputs, pr_one * probability), ( (1 - pr) * inputs, (1-pr_one) * probability)



    def get_config(self):

        return {'units': self.units}



class Leaf(tf.keras.layers.Layer):



    def __init__(self, units:int =32, l1:float = 0.1):

        super(Leaf, self).__init__()

        self.units = units

        self.l1 = l1



    def build(self, input_shape):

        self.w = self.add_weight(shape=(input_shape[-1], self.units),

                                 initializer='random_normal',

                                 trainable=True)

        

        self.b = self.add_weight(shape=(1,self.units),

                                 initializer='random_normal',

                                 trainable=True)



    def call(self, inputs: tf.Tensor, probability):

        

        pr = tf.matmul(inputs, self.w) + self.b

        

        l1 = tf.reduce_sum(0. * tf.abs(self.w))

        

        self.add_loss(l1)

        

        return pr, probability



    def get_config(self):

        return {'units': self.units}



    

class RegressionHead(tf.keras.layers.Layer):



    def __init__(self, units:int = 8):

        super(RegressionHead, self).__init__()



    def build(self, input_shape):

        pass



    def call(self, inputs: tf.Tensor):

        

        weighted_inputs = [x[0] * tf.reshape(x[1], (-1,1)) for x in inputs]

        

        output = reduce(tf.add, weighted_inputs)

        

        return (output)



    def get_config(self):

        return {'units': self.units}



class ClassificationHead(tf.keras.layers.Layer):



    def __init__(self, units:int = 8):

        super(ClassificationHead, self).__init__()



    def build(self, input_shape):

        pass



    def call(self, inputs: tf.Tensor):

        

        weighted_inputs = [tf.nn.softmax(x[0]) * tf.reshape(x[1], (-1,1)) for x in inputs]

        

        output = reduce(tf.add, weighted_inputs)

        

        return (output)



    def get_config(self):

        return {'units': self.units}



class SoftTree(tf.keras.Model):



    def __init__(self, max_depth:int = 3, classes: int = 1, alpha:float = 0.01, l1:float=0., head = ClassificationHead(), **kwargs):

        super(SoftTree, self).__init__(**kwargs)

        

        self.max_depth = max_depth

            

        self.nodes = [[Node(1, alpha=alpha, l1=l1) for _ in range(2**layer)] for layer in range(self.max_depth)]

        self.leaves = [[Leaf(classes, l1=l1) for _ in range(2**(self.max_depth))]]

        

        self.tree = self.nodes + self.leaves

        

        self.head = ClassificationHead() if head is None else head

        

    def prototype(self, inputs: tf.Tensor):

        input_to_layers = [ [[( inputs, tf.ones((tf.shape(inputs)[0],)) )]] ] + self.tree

        proto_output = reduce(lambda x, f: self.forward(x,f), input_to_layers[:-1])

        

        return [x[0] for x in list(chain(*proto_output))]

    

    def leaf_probabilty(self, inputs: tf.Tensor):

        input_to_layers = [ [[( inputs, tf.ones((tf.shape(inputs)[0],)) )]] ] + self.tree

        proto_output = reduce(lambda x, f: self.forward(x,f), input_to_layers)

        

        return [x[1] for x in proto_output]

    

    def leaf(self, inputs: tf.Tensor):

        

        input_to_layers = [ [[( inputs, tf.ones((tf.shape(inputs)[0],)) )]] ] + self.tree

        proto_output = reduce(lambda x, f: self.forward(x,f), input_to_layers)

        

        leaf_preductions = [x[0] for x in proto_output]

    

        return list(map(tf.nn.softmax, leaf_preductions))

        

    def forward(self, inputs: List[Tuple[tf.Tensor]], layer: List[tf.keras.Model]):

        inputs = list(chain(*inputs))

        joined = zip(inputs, layer)

        return [f(x[0], x[1]) for x, f in joined]

    

    def call(self, inputs: tf.Tensor):

        input_to_layers = [ [[( inputs, tf.ones((tf.shape(inputs)[0],)) )]] ] + self.tree

        

        leaf_output = reduce(lambda x, f: self.forward(x,f), input_to_layers)

        

        return self.head(leaf_output)

    

class SoftRegressionTree(SoftTree):

    def __init__(self,  max_depth:int = 3, classes: int = 1, alpha:float = 0.01, l1:float=0.,**kwargs):

        super(SoftRegressionTree, self).__init__(max_depth, classes, alpha, l1, head=RegressionHead(), **kwargs)

        

class SoftClassificationTree(SoftTree):

    def __init__(self,  max_depth:int = 3, classes: int = 1, alpha:float = 0.01, l1:float=0.,**kwargs):

        super(SoftClassificationTree, self).__init__(max_depth, classes, alpha, l1, head=ClassificationHead(), **kwargs)
digits_tree = SoftClassificationTree(max_depth=4, classes=9, alpha=0.025)



digits_tree.compile(loss='categorical_crossentropy', optimizer='adam')



digits_tree.fit(digits_X, digits_y_sparse, epochs=250, validation_split=0.1, batch_size=200)



digits_y_hat = digits_tree.predict(digits_X)
# Leaf probabilities for first datapoint

lead_prob = digits_tree.leaf_probabilty(digits_X)



P = tf.concat([tf.reshape(x, (-1,1)) for x in lead_prob], 1)



pd.np.random.choice(range(1000), 5)



hv.Bars(tf.random.shuffle(P)[1,:].numpy())
pd.DataFrame(confusion_matrix(digits_y_sparse.numpy().argmax(axis=1), 

                              digits_y_hat.argmax(axis=1)),

             index = range(1,10),

             columns = range(1,10))
prototype_labels = []

for s, p in zip(digits_tree.leaf(digits_X), digits_tree.leaf_probabilty(digits_X)):

    prototype_labels.append(s * tf.reshape(p, (-1,1)))



prototype_labels = list(map(lambda x: (tf.reduce_mean(x, 0)

                                 .numpy()

                                 .argmax(0)), 

                            prototype_labels))

prototypes = list(map(lambda x: tf.reduce_mean(x, 0), digits_tree.prototype(digits_X)))
images = []

for i, (proto, label) in enumerate(zip(prototypes, prototype_labels)):

    grid = (proto

            .numpy()

            .reshape((8,8)))

    standard_grid = (grid - grid.min())/grid.max()

    

    images.append(hv.Image(standard_grid).opts(title=f'Digit: {str(label)} ------------------------------ Leaf : {i+1}',

                                               xlabel='', ylabel='', 

                                               xaxis=None, yaxis=None))



reduce(add, images)
boston = load_boston()



boston_X = tf.convert_to_tensor(boston.data.astype('float32'))

boston_X_z_score = tf.math.divide_no_nan(boston_X - tf.reduce_mean(boston_X, 0), tf.math.reduce_std(boston_X, 0))

boston_y = tf.convert_to_tensor(boston.target.astype('float32'))

boston_tree = SoftRegressionTree(max_depth=1, classes=1, alpha=0.1, l1=1.)



boston_tree.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.025))

boston_tree.fit(boston_X_z_score, boston_y, epochs=50, validation_split=0.1, batch_size=200)



y_hat = boston_tree.predict(boston_X_z_score)
def weight_plots(index: int, leaf: tf.keras.layers.Layer) -> hv.plotting.Plot:

    return (pipe(zip(boston.feature_names, 

                    leaf.w

                        .numpy()

                        .reshape((-1,))

                        .tolist()),

                hv.Bars)

            .opts(xrotation=90, 

                  xlabel='Features', 

                  ylabel='Weights', 

                  title=f'Feature Importances at Leaf {index}'))
pipe(enumerate(boston_tree.tree[-1]), 

     map(lambda leaf: weight_plots(*leaf)),

     reduce(add))
pipe(boston_X_z_score, # the data

     boston_tree.leaf_probabilty, # compose leaf probabilities

     map(lambda x: tf.reshape(x, (-1,1))), # reshape probabilities

     list,

     partial(tf.concat, axis=1), # contatenate vector

     partial(tf.reduce_mean, axis=0), # get mean accross index

     lambda x: x.numpy(), # convert to numpy

     hv.Bars # gen polts

    ).opts(title='% sample per leaf', 

           xlabel='Leaves', 

           ylabel='%')