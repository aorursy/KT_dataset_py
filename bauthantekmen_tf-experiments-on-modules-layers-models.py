# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from datetime import datetime



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# All of the tensorflow high level functions(layers, models) 

# are defined on (inherits) the same baseline class: tf.Module



#example

class SimpleModule(tf.Module):

    def __init__(self, name_=None):

        super().__init__(name=name_)

        self.a_variable = tf.Variable(5.0, name="train_meh")

        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="not_train_meh")

    def __call__(self,x):

        return self.a_variable*x + self.non_trainable_variable

    

simple_module = SimpleModule(name_ = "simple")



print(simple_module(tf.constant(5.0)))

print()

print("trainables = ", simple_module.trainable_variables)

print("variables = ", simple_module.variables)

# lets define simple model with dense layers



class Dense(tf.Module):

    def __init__(self, in_features, out_features, name="dense"):

        super().__init__(name=name)

        self.w = tf.Variable(tf.random.normal([in_features, out_features], name="w"))

        self.b = tf.Variable(tf.zeros([out_features], name="b"))

    def __call__(self, x):

        y = tf.matmul(x,self.w) + self.b

        return tf.nn.relu(y)

    

class SeqModule(tf.Module):

    def __init__(self, name="seqmodule"):

        super().__init__(name=name)

        

        self.dense1 = Dense(3,3, name="d1")

        self.dense2 = Dense(3,2, name="d2")

        

    def __call__(self, x):

        x = self.dense1(x)

        x = self.dense2(x)

        return x



my_model = SeqModule("mymodule")

print("Results = ", my_model(tf.constant([[2.,2.,2.]])))

        

# All tf.Variable objects and tf.Module objects can be found

print("\n","#"*100)

print("used submodules =", my_model.submodules)

print("\n\n used vars =", my_model.variables)



#also note that by entering is.built=False we can build while calling, and

# this gives flexibility to our models such as not specifying input size
chkpoint_path = "my_checkpointlol"

checkpoint = tf.train.Checkpoint(model = my_model) # get the checkpoint of variables and submodules

# save weights (variables etc.)

checkpoint.write(chkpoint_path)

# save index which is metadata about checkpoint

checkpoint.write(chkpoint_path)



# see files in directory

! ls . && echo "\n"



# check checkpoint to be sure variables are saveed correctly

tf.train.list_variables("my_checkpointlol")
# "When you load models back in, you overwrite the values in your Python object." (tf official page)



# Restore weights from my_checkpointlol

new_model = SeqModule()

new_checkpoint = tf.train.Checkpoint(model=new_model)

new_checkpoint.restore("my_checkpointlol")



# Note that results are the same when weights are saved with my_model with same input

new_model(tf.constant([[2.,2.,2.]]))



# Also there are more deep-into guide on checkpoint on the tf guides page
# Firstly define model whose graph will be builded:



class GraphSeqModule(tf.Module):

    def __init__(self, name="seqmodule"):

        super().__init__(name=name)

        

        self.dense1 = Dense(3,3, name="d1")

        self.dense2 = Dense(3,2, name="d2")

    

    @tf.function

    def __call__(self, x):

        x = self.dense1(x)

        x = self.dense2(x)

        return x

        

model_on_graph = GraphSeqModule(name="graph")



# Bam! Run the same model on the graphs!

print(model_on_graph([[2.0,2.0,2.0]]))

print(model_on_graph([[2.0,2.0,2.0], [1.,2.,5.]]))  #with batch = 2
# LEARN TENSORBOARD LATER

# Visualize graph with TensorBoard summary



# It is great idea to set loggings or generate unique

# file/object names by stamp



### Bu kısmı bir ara araştıralım

# Set up logging.

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

logdir = "logs/func/%s" % stamp

writer = tf.summary.create_file_writer(logdir)



# Create a new model to get a fresh trace

# Otherwise the summary will not see the graph.

new_model = GraphSeqModule()



# Bracket the function call with

# tf.summary.trace_on() and tf.summary.trace_export().

tf.summary.trace_on(graph=True, profiler=True)

# Call only one tf.function when tracing.

z = print(new_model(tf.constant([[2.0, 2.0, 2.0]])))

with writer.as_default():

  tf.summary.trace_export(

      name="my_func_trace",

      step=0,

      profiler_outdir=logdir)
# This is the recommended way to save & share completely trained model



# create a saved model

tf.saved_model.save(model_on_graph, "saved_model")



# check

!ls -l saved_model

!echo "\n"

!ls -l saved_model/variables



#The saved_model.pb file is a protocol buffer describing the functional tf.Graph

new_model = tf.saved_model.load("saved_model")



#model is reloaded without any class knowledge

print(isinstance(new_model, GraphSeqModule)) #false



print("hello : ", model_on_graph(tf.constant([[2.,2.,2.]])))

print("hello : ", new_model(tf.constant([[2.,2.,2.]])))



#Thus, using tf.saved_model, we are able to save weights graphs using tf.Module and reload them
# we can transform our layer to keras layer just by swapping

# superclass with tf.keras.layers.Layer, and __call__ with call



class MyDense(tf.keras.layers.Layer):

    def __init__(self, in_dim, out_dim, **kwargs):

        super().__init__(**kwargs)

        

        self.w = tf.Variable( tf.random.normal([in_dim, out_dim]), name='w')

        self.b = tf.Variable(tf.zeros([out_dim]), name='b')

        

    def call(self, x):

        y = tf.matmul(x,self.w) + self.b

        return tf.nn.relu(y)





simple_layer = MyDense(name="simple", in_dim=5, out_dim=4)



# no change in the functionality

print(simple_layer([[2.0, 2.0, 2.0, 2.0, 2.]]))



# build is called exactly once, and it is called with the shape of the input. 

# It's usually used to create variables (weights)



class FlexibleDense(tf.keras.layers.Layer):

  # Note the added `**kwargs`, as Keras supports many arguments

  def __init__(self, out_dim, **kwargs):

    super().__init__(**kwargs)

    self.out_dim = out_dim



  # Magic happpens here!

  def build(self, input_shape):  # Create the state of the layer (weights)

    self.w = tf.Variable(

      tf.random.normal([input_shape[-1], self.out_dim]), name='w')

    self.b = tf.Variable(tf.zeros([self.out_dim]), name='b')



  def call(self, inputs):  # Defines the computation from inputs to outputs

    return tf.matmul(inputs, self.w) + self.b



# Create the instance of the layer

flexible_dense = FlexibleDense(out_dim=4)



#there is no vairables, model hasnt been built

print(flexible_dense.variables)



# call it

print(flexible_dense(tf.constant([[2.,2.,2.,2.,2.,2.,2.],[3.,3.,3.,3.,3.,3.,3.]])))

print()

print(flexible_dense.variables)
class KerasModel(tf.keras.Model):

    def __init__(self, **kwargs):

        super().__init__(kwargs)

        

        self.dense1 = FlexibleDense(5)

        self.dense2 = FlexibleDense(2)

        

    def call(self, x):

        x = self.dense1(x)

        x = self.dense2(x)

        return tf.nn.softmax(x)

    

model = KerasModel()

print(model(tf.constant([[2.,2.,2.]])))

print()

print(model.variables)

print()

print(model.submodules)
# Functional model with keras



inputs = tf.keras.Input(shape=[3,])



x = FlexibleDense(5)(inputs)

out = FlexibleDense(2)(x)



functional_model = tf.keras.Model(inputs=inputs, outputs=out)

functional_model.summary()



functional_model(tf.constant([[2.,2.,2.]]))
# Save keras functional model (applicaple to all keras models)

functional_model.save("functional_model")



loaded = tf.keras.models.load_model("functional_model")



print(loaded(tf.constant([[2.,2.,2.]])))



# Keras saved models also save metric, loss and optimizer states ??? so what are them ???