import tensorflow as tf
INPUT_DIR =  '../input/transfer-learning/'
SAVED_MODEL = f'{INPUT_DIR}exp_saved_model'

# convert model to tf lite format + Post Training Quantization
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # Create input data generator 
# # Ops that do not have quantized implementations will automatically be left in floating point
# def representative_data_gen():
#     for input_value, _ in test_batches.take(100):
#         yield [input_value]
        
# converter.representative_dataset = representative_data_gen
# To require the converter to only output integer operations
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]


tflite_model = converter.convert()

# interprete
tflite_model_file = 'converted_model.tflite'

with open(tflite_model_file, "wb") as f:
    f.write(tflite_model)
# interprete
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
import tensorflow_datasets as tfds
# Use tfds to load data
DATASET_NAME = 'cats_vs_dogs'
ds_builder = tfds.builder(DATASET_NAME)
ds_builder.download_and_prepare()
ds_info = ds_builder.info
test_examples = ds_builder.as_dataset(split=[ "train[90%:100%]"], as_supervised=True)[0]
def format_image(image, label):
    image = tf.image.resize(image, (224, 224)) / 255
    return image, label
test_batches = test_examples.map(format_image).batch(1)
predictions = []

test_label, test_imgs = [], []
for img, label in test_batches.take(10):
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_index))
    test_label.append(label.numpy()[0])
    test_imgs.append(img)
# Utilities functions for plotting

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    img = np.squeeze(img)
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
        
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)
import matplotlib.pyplot as plt
import numpy as np

class_names = ['cat', 'dog']

index = 8
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(index, predictions, test_label, test_imgs)
plt.show()
# output the model
with open('labels.txt', 'w') as f:
    f.write('\n'.join(class_names))
    
