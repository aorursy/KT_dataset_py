import tensorflow as tf

import numpy as np
model = tf.saved_model.load('../input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model')
len(model.variables)
model.graph.get_collection('variables')[:10]
var_names = [var.name for var in model.graph.get_collection('variables')]
test_in = tf.constant(np.uint8(np.random.randn(300, 300, 3)))
var_names_to_fetch = [

    var_name[:-2] + '/read' + var_name[-2:] for var_name in var_names]



weight_fetcher = model.prune(

    feeds=["input_image:0"],

    fetches=var_names_to_fetch

)



weights = weight_fetcher(test_in)
len(weights) == len(model.graph.get_collection('variables'))
weights[0].numpy().shape, weights[0].numpy()