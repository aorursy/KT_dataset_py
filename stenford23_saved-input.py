import tensorflow as tf
score_model = tf.saved_model.load('/kaggle/input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model')

served_function = score_model.signatures['serving_default']
tf.saved_model.save(

      score_model, export_dir="", signatures={'serving_default': served_function})