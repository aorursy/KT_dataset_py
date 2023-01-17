import numpy as np
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

%matplotlib inline
MAX_WIDTH = 512
content_img_url = 'http://www.bhmpics.com/walls/radha_with_krishna_glass_painting-normal.jpg'
content_img_path = tf.keras.utils.get_file(fname='./content.jpg', origin=content_img_url)
content_img = Image.open(content_img_path)
style_img_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Lubang_Jeriji_Sal%C3%A9h_cave_painting_of_Bull.jpg/440px-Lubang_Jeriji_Sal%C3%A9h_cave_painting_of_Bull.jpg'
style_img_path = tf.keras.utils.get_file(fname='./style.jpg', origin=style_img_url)
style_img = Image.open(style_img_path)
def get_content_style_img_arrays(content_img, style_img, WIDTH, HEIGHT):
    content_img = content_img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    style_img = style_img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    content_img = content_img.convert('RGB')
    style_img = style_img.convert('RGB')
    content_img_array = np.asarray(content_img, dtype=np.float64)[:, :, 0:3]
    style_img_array = np.asarray(style_img, dtype=np.float64)[:, :, 0:3]
    return content_img_array, style_img_array
content_width, content_height = content_img.size
if content_width > MAX_WIDTH:
    WIDTH = MAX_WIDTH
    HEIGHT = int(content_height * MAX_WIDTH / content_width)
else:
    WIDTH = content_width
    HEIGHT = content_height
content_img_array, style_img_array = get_content_style_img_arrays(content_img, style_img, WIDTH, HEIGHT)
# Preview Content Image
plt.imshow(content_img_array / 255)
# Preview Style Image
plt.imshow(style_img_array / 255)
vgg_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg_model.trainable = False
vgg_model.summary()
CONTENT_LAYERS = ['block2_conv1', 'block2_conv2']
STYLE_LAYERS = ['block2_conv1', 'block2_conv2', 'block3_conv1', 'block4_conv1', 'block5_conv1']
def get_layer_outputs(layer_names, input_data):
    intermediate_layer_model = tf.keras.Model(inputs=[vgg_model.input], 
          outputs=[vgg_model.get_layer(layer_name).output for layer_name in layer_names])
    return intermediate_layer_model(input_data)
def gram_matrix(input_tensor):
    assert tf.rank(input_tensor) == 3
    channels = tf.shape(input_tensor)[2]
    assert channels >= 2
    height = tf.shape(input_tensor)[0]
    width = tf.shape(input_tensor)[1]
    input_tensor_flattened = tf.reshape(input_tensor, [height * width, channels])
    return tf.matmul(tf.transpose(input_tensor_flattened), input_tensor_flattened)
content_img_targets = {}
content_img_preprocessed = tf.keras.applications.vgg19.preprocess_input(content_img_array.reshape((1, HEIGHT, WIDTH, 3)))
for i, output in enumerate(get_layer_outputs(CONTENT_LAYERS, content_img_preprocessed)):
    content_img_targets[CONTENT_LAYERS[i]] = output[0]

style_img_targets = {}
style_img_preprocessed = tf.keras.applications.vgg19.preprocess_input(style_img_array.reshape((1, HEIGHT, WIDTH, 3)))
for i, output in enumerate(get_layer_outputs(STYLE_LAYERS, style_img_preprocessed)):
    style_img_targets[STYLE_LAYERS[i]] = gram_matrix(output[0])
def compute_content_cost(generated_img, content_cost_weight):
    content_cost = tf.Variable(0.0)
    generated_img_preprocessed = tf.keras.applications.vgg19.preprocess_input(tf.reshape(generated_img, [1, HEIGHT, WIDTH, 3]))
    for i, output in enumerate(get_layer_outputs(CONTENT_LAYERS, generated_img_preprocessed)):
        content_cost = tf.math.add(content_cost, tf.math.reduce_mean(tf.math.squared_difference(content_img_targets[CONTENT_LAYERS[i]], 
                            output)))
    return tf.math.multiply(content_cost, tf.constant(content_cost_weight / len(CONTENT_LAYERS)))
def compute_style_cost(generated_img, style_cost_weight):
    style_cost = tf.Variable(0.0)
    generated_img_preprocessed = tf.keras.applications.vgg19.preprocess_input(tf.reshape(generated_img, [1, HEIGHT, WIDTH, 3]))
    for i, output in enumerate(get_layer_outputs(STYLE_LAYERS, generated_img_preprocessed)):
        style_cost = tf.math.add(style_cost, tf.math.reduce_mean(tf.math.squared_difference(style_img_targets[STYLE_LAYERS[i]], 
                            gram_matrix(output[0]))))
    return tf.math.multiply(style_cost, tf.constant(style_cost_weight / len(STYLE_LAYERS)))
def compute_total_cost(generated_img, content_cost_weight=0.6, style_cost_weight=0.4):
    return tf.math.add(compute_content_cost(generated_img, content_cost_weight), compute_style_cost(generated_img, style_cost_weight))
adam_optimizer = tf.optimizers.Adam(learning_rate=4)
def train_step(step, generated_img):
  with tf.GradientTape() as tape:
    loss = compute_total_cost(generated_img, 5e4, 2e-8)
    if (step + 1) % 10 == 0 :
        print('Loss at step {} = {}'.format(step + 1, loss.numpy()))
  grad = tape.gradient(loss, generated_img)
  adam_optimizer.apply_gradients([(grad, generated_img)])
  generated_img.assign(generated_img)
generated_img = tf.Variable(np.random.random(content_img_array.shape), trainable=True)
for i in range(100):
    train_step(i, generated_img)
plt.imshow(generated_img.numpy() / 255)
generated_img_array = generated_img.numpy()
generated_img_array = np.maximum(generated_img_array, 0)
generated_img_array = np.minimum(generated_img_array, 255)
generated_img_array = np.uint8(generated_img_array)
generated_img_PIL = Image.fromarray(generated_img_array)
generated_img_PIL.save('./generated/styled.jpg')
!mkdir ./generated
