import tensorflow as tf
baseline_0271 = tf.saved_model.load('../input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model')
baseline_0277 = tf.saved_model.load('../input/google-2020-baseline/∩╗┐basline')
baseline_0271_embedding_model = baseline_0271.prune(

    feeds=["ResizeBilinear:0"],

    fetches=["l2_normalization:0"],

)
baseline_0277_embedding_model = baseline_0277.model
test_input = tf.random.normal((1, 300, 300, 3))
baseline_0271_embedding_model(test_input)
baseline_0277_embedding_model(test_input)