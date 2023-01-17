import os
from types import SimpleNamespace
import tensorflow as tf
import numpy as np

CONFIG = {
  'model_dir': '../input/boat-types-retraining/boats_on_inception_v3/',
  'num_top_predictions': 5
}
FLAGS = SimpleNamespace(**CONFIG)
def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def create_graph():
  # Creates graph from saved graph_def.pb.
  filename = None
  filename = os.path.join(FLAGS.model_dir, 'retrained_graph.pb')
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    labels = load_labels(os.path.join(FLAGS.model_dir, 'retrained_labels.txt'))

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
run_inference_on_image('../input/boat-types-examples/jack-ward-527052-unsplash.jpg')
run_inference_on_image('../input/boat-types-examples/mike-arney-437116-unsplash.jpg')
run_inference_on_image('../input/boat-types-examples/kalen-emsley-94118-unsplash.jpg')