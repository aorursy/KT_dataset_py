import tensorflow as tf
tf.__version__
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
# configuration parameters 
TEST_DATA_DIR = './val'
MODEL_PATH = "../input/saved-model1/best2_model (1).h5"
TFLITE_MODEL_DIR = "./models/tflite/"
TEST_SAMPLES = 1912
image_size = 256
model = load_model('../input/saved-model1/best2_model (1).h5')
# create a directory to save the tflite models if it does not exists
if not os.path.exists(TFLITE_MODEL_DIR):
    os.makedirs(TFLITE_MODEL_DIR)
# convert a keras model to tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# write the model to a tflite file as binary file
tflite_no_quant_file = TFLITE_MODEL_DIR + "Resnet50_no_quant.tflite"
with open(tflite_no_quant_file, "wb") as f:
    f.write(tflite_model)
# convert a tf.Keras model to tflite model with INT8 quantization 
# Note: INT8 quantization is by default! 
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# write the model to a tflite file as binary file
tflite_no_quant_file = TFLITE_MODEL_DIR + "Resnet50_weights_int8_quant.tflite"
with open(tflite_no_quant_file, "wb") as f:
    f.write(tflite_model)
    
# Note: you should see roughly 4x times reduction in the model size
# convert a tf.Keras model to tflite model with INT8 quantization 
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# set optimization to DEFAULT and set float16 as the supported type on 
# the target platform
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# write the model to a tflite file as binary file
tflite_no_quant_file = TFLITE_MODEL_DIR + "Resnet50_weights_float16_quant.tflite"
with open(tflite_no_quant_file, "wb") as f:
    f.write(tflite_model)
    
# Note: you should see roughly 2x times reduction in the model size
# create an image generator with a batch size of 1 
test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = test_data_generator.flow_from_directory(
    "../input/mushroom-validation-set/val",
    target_size=(image_size, image_size),
    batch_size=1,
    class_mode='categorical',# classes to predict
    shuffle =False
    ) 

def represent_data_gen():
    """ it yields an image one by one """
    for ind in range(len(validation_generator.filenames)):
        img_with_label = validation_generator.next() # it returns (image and label) tuple
        yield [np.array(img_with_label[0], dtype=np.float32, ndmin=2)] # return only image
# convert a tf.Keras model to tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# assign the custom image generator fn to representative_dataset
converter.representative_dataset = represent_data_gen 

# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# write the model to a tflite file as binary file
tflite_both_quant_file = TFLITE_MODEL_DIR + "Resnet50_both_int8_quant.tflite"
with open(tflite_both_quant_file, "wb") as f:
    f.write(tflite_model)
# convert a tf.Keras model to tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16] # save them in float16
converter.representative_dataset = represent_data_gen
tflite_model = converter.convert()

# write the model to a tflite file as binary file
tflite_both_quant_file = TFLITE_MODEL_DIR + "Resnet50_both_fp16_quant.tflite"
with open(tflite_both_quant_file, "wb") as f:
    f.write(tflite_model)
!ls "../input/tfmodel/models/tflite" -lh
!ls "../input/inputoutput8bit/" -lh
!zip -r ./models.zip ./models/
!cp -a ../input/tfmodel/models ./
# choose the model type here
QUANT_TYPE = "both_int8" # no_quant, w_int8, w_fp16, both_int8, both_fp16
LABELS = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Exidia',
                   'Hygrocybe', 'Inocybe','Lactarius', 'Pluteus','Russula', 'Suillus']
          
QUANT_NAME_MAP = {"no_quant": "no quantization", "w_int8": "weights 8-bit INT quantized", 
                  "w_fp16": "weights 16-bit FP quantized", "both_int8": "both weights and activations INT8 quantized", 
                 "both_fp16": "both weights and activations FP-16 quantized"}
if QUANT_TYPE == "no_quant":
    # model without any quantization
    print(f"interpreter for {QUANT_NAME_MAP[QUANT_TYPE]} loading ...")
    interpret = tf.lite.Interpreter(model_path = TFLITE_MODEL_DIR + "Resnet50_no_quant.tflite")
    # Learn about its input and output details
    input_details = interpret.get_input_details()
    output_details = interpret.get_output_details()

    interpret.resize_tensor_input(input_details[0]['index'], (1, 256, 256, 3))
    interpret.resize_tensor_input(output_details[0]['index'], (1, 12))
    interpret.allocate_tensors() # allocate memory to the model
    
elif QUANT_TYPE == "w_int8":
    # model with weights INT8 quantization
    print(f"interpreter for {QUANT_NAME_MAP[QUANT_TYPE]} loading ...")
    interpret = tf.lite.Interpreter(model_path = TFLITE_MODEL_DIR + "Resnet50_weights_int8_quant.tflite")
    # Learn about its input and output details
    input_details = interpret.get_input_details()
    output_details = interpret.get_output_details()

    interpret.resize_tensor_input(input_details[0]['index'], (1, 256, 256, 3))
    interpret.resize_tensor_input(output_details[0]['index'], (1, 12))
    interpret.allocate_tensors() # allocate memory to the model
    

    
elif QUANT_TYPE == "w_fp16":
    # model with weights FP16 quantization 
    print(f"interpreter for {QUANT_NAME_MAP[QUANT_TYPE]} loading ...")
    interpret = tf.lite.Interpreter(model_path = TFLITE_MODEL_DIR + "Resnet50_weights_float16_quant.tflite")
    # Learn about its input and output details
    input_details = interpret.get_input_details()
    output_details = interpret.get_output_details()

    interpret.resize_tensor_input(input_details[0]['index'], (1, 256, 256, 3))
    interpret.resize_tensor_input(output_details[0]['index'], (1, 12))
    interpret.allocate_tensors() # allocate memory to the model

    
elif QUANT_TYPE == "both_int8":
    # model with both weights and activations INT8 quantization 
    print(f"interpreter for {QUANT_NAME_MAP[QUANT_TYPE]} loading ...")
#     interpret = tf.lite.Interpreter(model_path = TFLITE_MODEL_DIR + "Resnet50_both_int8_quant.tflite")
    interpret = tf.lite.Interpreter(model_path = "../input/inputoutput8bit/Resnet50_both_int8_quant.tflite")
    # Learn about its input and output details
    input_details = interpret.get_input_details()
    output_details = interpret.get_output_details()

    
    interpret.resize_tensor_input(input_details[0]['index'], (1, 256, 256, 3))
    interpret.resize_tensor_input(output_details[0]['index'], (1, 12))
    interpret.allocate_tensors() # allocate memory to the model
    

    input_type = interpret.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpret.get_output_details()[0]['dtype']
    print('output: ', output_type)
    
elif QUANT_TYPE == "both_fp16":
    # model with both weights and activations INT8 quantization 
    print(f"interpreter for {QUANT_NAME_MAP[QUANT_TYPE]} loading ...")
    interpret = tf.lite.Interpreter(model_path = TFLITE_MODEL_DIR + "Resnet50_both_fp16_quant.tflite")
    # Learn about its input and output details
    input_details = interpret.get_input_details()
    output_details = interpret.get_output_details()

    interpret.resize_tensor_input(input_details[0]['index'], (1, 256, 256, 3))
    interpret.resize_tensor_input(output_details[0]['index'], (1, 12))
    interpret.allocate_tensors() # allocate memory to the model



else:
    print(f"Wrong quantization type has been chosen for {QUANT_NAME_MAP[QUANT_TYPE]}")
    sys.exit(0)
# get indices of input and output tensors for each model 
input_ind = interpret.get_input_details()[0]["index"]
out_ind   = interpret.get_output_details()[0]["index"]
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

image_size = 256
test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = test_data_generator.flow_from_directory(
    '../input/mushroom-validation-set/val',
    target_size=(image_size, image_size),
    batch_size=1,
    class_mode='categorical',# classes to predict
    shuffle =False
    ) 
%%time
# save the predicted class label (highest probability one) in a list
import numpy as np
results = []
accuracy_count = 0
TEST_SAMPLES =1912
for i in range(TEST_SAMPLES): 
    
    print(f"computing results for {i}th image ...")
    import time
    start = time.time()


    # generate a batch of images 
    test_image = validation_generator.next()
    
    # set the input image to the input index 
    interpret.set_tensor(input_ind, test_image[0].astype(np.uint8))
    
    # run the inference 
    interpret.invoke() 
    
    # read the predictions from the output tensor
    predictions = interpret.tensor(out_ind) # or, get_tensor(out_ind)
    
    # get the highest predicted class
    pred_class = np.argmax(predictions()[0])
    end = time.time()

    dt = end - start
    print("time to make prediction",dt)
    #print("predicted class: ", pred_class, " and actual class: ", test_generator.classes[i])
    
    results.append(pred_class)
    
    if pred_class == validation_generator.classes[i]:
        accuracy_count += 1 
# compute the accuracy percentage
print(f"accuracy percentage: {round((accuracy_count / TEST_SAMPLES) * 100, 3)}% \n")
# Plot confusion matrix, classification report
from sklearn.metrics import classification_report, confusion_matrix
print("-"*50)
print(f"Confusion matrix for {QUANT_NAME_MAP[QUANT_TYPE]}: \n")
print(confusion_matrix(y_true=validation_generator.classes, y_pred=results))
print("-"*50)
print(f"Classification report for {QUANT_NAME_MAP[QUANT_TYPE]} : \n")
target_names =['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Exidia',
                   'Hygrocybe', 'Inocybe','Lactarius', 'Pluteus','Russula', 'Suillus']
print(classification_report(y_true=validation_generator.classes, y_pred=results, target_names=target_names))
print("-"*50)















# # A generator that provides a representative dataset
# def representative_data_gen():
#   dataset_list = tf.data.Dataset.list_files('../input/mushroom-validation-set/val'+ '/*/*')
#   for i in range(1912):
#     image = next(iter(dataset_list))
#     image = tf.io.read_file(image)
#     image = tf.io.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [image_size, image_size])
#     image = tf.cast(image / 255., tf.float32)
#     image = tf.expand_dims(image, 0)
#     yield [image]


# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # This ensures that if any ops can't be quantized, the converter throws an error
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# # These set the input and output tensors to uint8
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# # And this sets the representative dataset so we can quantize the activations
# converter.representative_dataset = representative_data_gen
# tflite_model = converter.convert()

# with open('quant.tflite', 'wb') as f:
#   f.write(tflite_model)
# batch_images, batch_labels = next(validation_generator)

# logits = model(batch_images)
# prediction = np.argmax(logits, axis=1)
# truth = np.argmax(batch_labels, axis=1)

# keras_accuracy = tf.keras.metrics.Accuracy()
# keras_accuracy(prediction, truth)

# print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))
# def set_input_tensor(interpreter, input):
#   input_details = interpreter.get_input_details()[0]
#   tensor_index = input_details['index']
#   input_tensor = interpreter.tensor(tensor_index)()[0]
#   # Inputs for the TFLite model must be uint8, so we quantize our input data.
#   # NOTE: This step is necessary only because we're receiving input data from
#   # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
#   # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
#   #   input_tensor[:, :] = input
#   scale, zero_point = input_details['quantization']
#   input_tensor[:, :] = np.uint8(input / scale + zero_point)

# def classify_image(interpreter, input):
#   set_input_tensor(interpreter, input)
#   interpreter.invoke()
#   output_details = interpreter.get_output_details()[0]
#   output = interpreter.get_tensor(output_details['index'])
#   # Outputs from the TFLite model are uint8, so we dequantize the results:
#   scale, zero_point = output_details['quantization']
#   output = scale * (output - zero_point)
#   top_1 = np.argmax(output)
#   return top_1

# interpreter = tf.lite.Interpreter('quant.tflite')
# # input_details = interpreter.get_input_details()
# # output_details = interpreter.get_output_details()

# interpreter.resize_tensor_input(input_details[0]['index'], (1, 256, 256, 3))
# interpreter.resize_tensor_input(output_details[0]['index'], (1, 12))
# interpreter.allocate_tensors()

# # Collect all inference predictions in a list
# batch_prediction = []
# batch_truth = np.argmax(batch_labels, axis=1)

# for i in range(len(batch_images)):
#   import time
#   start = time.time()
#   prediction = classify_image(interpreter, batch_images[i])
#   batch_prediction.append(prediction)
#   end = time.time()
#   dt = end - start
#   print("time to make prediction",dt)


