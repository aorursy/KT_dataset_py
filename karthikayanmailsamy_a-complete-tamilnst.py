%matplotlib inline

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

lr = 23
size = 360
iterations = 250
style_wt = 0.0008
content_wt = 0.8

content_image_path = "../input/tamil-nst/TamilContentImages/C_image7.jpg"
style_image_path = "../input/tamil-nst/TamilStyleImages/S_image1.jpg"

style_layer_wts = [0.8,0.9,1.0,0.9,0.8]
model = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=(size, size, 3))
model.trainable = False
model.summary()
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(size, size))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return np.expand_dims(img, axis = 0)
def deprocess(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

def display_image(image):
    if len(image.shape) == 4:
        image = image[0,:,:,:]

    img = deprocess(image)
    
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()
display_image(preprocess_image(style_image_path))
display_image(preprocess_image(content_image_path))
content_layer = 'block3_conv4'

content_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=model.get_layer(content_layer).output
)
style_layers = [
    'block1_conv1', 'block2_conv2',
    'block3_conv3', 'block4_conv4',
    'block5_conv2'
    ]

style_models = [
    tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer).output)
    for layer in style_layers
]
def content_cost(content_img, generated_img):
    C = content_model(content_img)
    G = content_model(generated_img)
    cost = tf.reduce_mean(tf.square(C - G))/(4*G.shape[0]*G.shape[1]*3)
    return cost
def gram_matrix(M):
    num_channels = tf.shape(M)[-1]
    M = tf.reshape(M, shape=(-1, num_channels))
    n = tf.shape(M)[0]
    G = tf.matmul(tf.transpose(M), M)
    return G 
def style_cost(style_img, generated_img):
    total_cost = 0
    
    for i, style_model in enumerate(style_models):
        S = style_model(style_img)
        G = style_model(generated_img)
        GS = gram_matrix(S)
        GG = gram_matrix(G)
        current_cost = style_layer_wts[i] * tf.reduce_mean(tf.square(GS - GG))/(2*GS.shape[0]*GS.shape[0]*3)**2
        total_cost += current_cost
    return total_cost
content_image_preprocessed = preprocess_image(content_image_path)
style_image_preprocessed = preprocess_image(style_image_path)
generated_image = tf.Variable(content_image_preprocessed, dtype=tf.float32)

generated_images = []
costs = []

min_cost=1*10**12
optimizer = tf.optimizers.Adam(learning_rate=lr)

for i in range(iterations):
    
    with tf.GradientTape() as tape:
        J_content = content_cost(content_img=content_image_preprocessed, generated_img=generated_image)
        J_style = style_cost(style_img=style_image_preprocessed, generated_img=generated_image)
        J_total = content_wt * J_content + style_wt * J_style
    
    gradients = tape.gradient(J_total, generated_image)
    optimizer.apply_gradients([(gradients, generated_image)])
    
    costs.append(J_total.numpy())
    
    if i % 10 == 0:
        if(J_total<min_cost):
            generated_images.append(generated_image.numpy())
            min_cost=J_total
        print("Iteration:{}/{}, Total Cost:{}, Style Cost: {}, Content Cost: {}".format(i+1, iterations, J_total, J_style, J_content))
plt.plot(range(iterations), costs)
plt.xlabel("Iterations")
plt.ylabel("Total Cost")
plt.show()
image = Image.fromarray(deprocess(generated_images[-1][0]))
plt.figure(figsize=(24,8))
dict_title={1:"Content_image",2:"Generated_image",3:"Style_image"}
images={1:tf.keras.preprocessing.image.load_img(content_image_path),2:image,3:tf.keras.preprocessing.image.load_img(style_image_path)}
for i in range(1,4):
    plt.subplot(2,4,i)
    plt.imshow(images[i])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.title(dict_title[i])
plt.savefig('out.png')
plt.show