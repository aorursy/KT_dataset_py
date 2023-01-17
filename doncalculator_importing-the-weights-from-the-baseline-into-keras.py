import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.saved_model import tag_constants
tf.disable_eager_execution()
import numpy as np
import pandas as pd

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=config) as sess:
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
            '../baseline/submission/baseline_landmark_retrieval_model/',
        )
        ops = graph.get_operations()
        gvs = tf.global_variables()
        node_vars = [n.name for n in graph.as_graph_def().node]
        var_names = [gv.name for gv in tf.global_variables()]
        
        vars = sess.run(tf.global_variables())

#exporting the names and shapes of the variables
baseline_map = []
for i in range(523):
    baseline_map.append([gvs[i].name, gvs[i].shape])
baseline_map = pd.DataFrame(baseline_map, columns=["Weight_name", "Weight_shape"])

#the baseline didn't use biases in the conv2d layers, so I'm adding a placeholder for it. This is needed because the tf.keras.applications.Resnet101 uses bias
for i in range(baseline_map.shape[0]):
    baseline_map_d.append(["",baseline_map.Weight_name.loc[i], baseline_map.Weight_shape.loc[i]])
    if baseline_map.Weight_name.loc[i].split("/")[-1] == "weights:0":
        baseline_map_d.append(["Dummy","Bias", "?"])
        
baseline_map_d = pd.DataFrame(baseline_map_d, columns=["is_dummy", "Name", "Shape"])
baseline_map_d.to_csv("baseline_map_d.csv", index=False)

#saving the weights
vars = np.array(vars)
np.save("baseline_w.npy", vars)

#saving the names of the weights
var_names = np.array(var_names)
np.save("baseline_w_names.npy", var_names)
import tensorflow as tf
import pandas as pd

model = tf.keras.applications.ResNet101V2(include_top=False, weights="imagenet")

layer_map = []
layer_len = len(model.layers)

for i in range(layer_len):
    layer = model.layers[i]
    if (isinstance(layer, tf.python.keras.layers.convolutional.Conv2D) | 
        isinstance(layer, tf.python.keras.layers.normalization_v2.BatchNormalization)):
        w = layer.weights
        for j in range(len(w)):
            layer_map.append([i, layer.name, layer.weights[j].name, layer.weights[j].shape])
            

keras_map = pd.DataFrame(layer_map, columns=["Layer_id", "Layer_name", "Weight_name", "Weights_shape"])
keras_map.to_csv("keras_mapv2.csv", index=False)
import tensorflow as tf
import pandas as pd
import numpy as np


b_weights = np.load("../input/landmark2020-weight-export-intermediate/baseline_w.npy", allow_pickle=True)
b_weight_names = np.load("../input/landmark2020-weight-export-intermediate/baseline_w_names.npy", allow_pickle=True)

#this is the lookup table I created
df = pd.read_csv("../input/landmark2020-weight-export-intermediate/dict_converter_v2.csv", sep=";")
df.head()
class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.p = 3.0
        self.eps = 1e-6

    def call(self, inputs):
        inputs = tf.clip_by_value(inputs, clip_value_min=1e-6, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1./self.p)
        return inputs
    
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = tf.keras.applications.ResNet101(include_top=False, weights=None)
        #to make sure we won't forget to turn the biases off, let's do it here :)
        layer_len = len(self.backbone.layers)

        for i in range(layer_len):
            layer = self.backbone.layers[i]
            if isinstance(layer, tf.python.keras.layers.convolutional.Conv2D):
                self.backbone.layers[i].use_bias = False
        
        self.pooling = GeMPoolingLayer()
        self.dense = tf.keras.layers.Dense(2048, name='features')
        
    def call(self, x, training=False):
        x = self.backbone(x, training)
        x = self.pooling(x)
        return self.dense(x)
    
model = Model()
model.build([None,None,None,3])
###importing the resnet weights
layer_names = df.Layer_name.unique()

for l in layer_names:
    temp = []
    w_count = len(model.backbone.get_layer(l).weights)
    for i in range(w_count):
        w_name = model.backbone.get_layer(l).weights[i].name
        if "bias" in w_name:
            temp.append(model.backbone.get_layer(l).weights[i].numpy())
        else:
            bw_name = df[df.Keras_name == w_name].Baseline_Name.values[0]
            w_id = np.where(b_weight_names==bw_name)[0][0]
            temp.append(b_weights[w_id])
    model.backbone.get_layer(l).set_weights(temp)
    
####setting dense weights
model.dense.set_weights([b_weights[-2], b_weights[-1]])
#saving the weights
model.save_weights("landmark2020_baseline.h5")
x = tf.random.normal((1,224,224,3), dtype=tf.float32)
baseline = tf.saved_model.load('../input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model/')
baseline = baseline.prune(
    feeds=["ResizeBilinear:0"],
    fetches=["l2_normalization:0"],
)
baseline_out = baseline(x)
keras_out = tf.math.l2_normalize(model(x, training=False))
keras_out
baseline_out
