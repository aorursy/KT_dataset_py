import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import scipy.io
!wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O vgg19.mat
def load_vgg_model(path):
    """
    Returns a model for the purpose of 'painting' the picture.
    Takes only the convolution layer weights and wrap using the TensorFlow
    Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
    the paper indicates that using AveragePooling yields better results.
    The last few fully connected layers are not used.
    Here is the detailed configuration of the VGG model:
        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu    
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
    """
    
    vgg = scipy.io.loadmat(path)

    vgg_layers = vgg['layers']
    
    def _weights(layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        """
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

        return W, b

    def _relu(conv2d_layer):
        """
        Return the RELU function wrapped over a TensorFlow layer. Expects a
        Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        """
        Return the Conv2D layer using the weights, biases from the VGG
        model at 'layer'.
        """
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG
        model at 'layer'.
        """
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph
class CONFIG:
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    COLOR_CHANNELS = 3
!wget http://www.francetourism.com/themes/site16/img/6.jpg -O con.jpg
!wget https://duranvirginia.files.wordpress.com/2012/11/vdblog_the_great_wave_off_kanagawa.jpg -O sty1.jpg
!wget https://duranvirginia.files.wordpress.com/2012/11/vdblog_the_scream.jpg -O sty2.jpg
def content_cost(c,g):
    _,nh,nw,nc=g.get_shape().as_list()
    ccost=tf.reduce_sum(tf.square(c-g))
    ccost=ccost * (1/2)    #(1/(4*nh*nw*nc))
    return ccost
def gram_matrix(x):
    return tf.matmul(x,tf.transpose(x))
def layer_style_cost(s,g):
    _,nh,nw,nc=g.get_shape().as_list()
    S=tf.reshape(s,[nh*nw,nc])
    G=tf.reshape(g,[nh*nw,nc])
    S=gram_matrix(tf.transpose(S))
    G=gram_matrix(tf.transpose(G))
    scost=tf.reduce_sum(tf.square(S-G))
    scost=scost * (1/((2*nh*nw*nc)**2))
    return scost
style_layers=[
    ('conv1_1',0.2),
    ('conv2_1',0.2),
    ('conv3_1',0.2),
    ('conv4_1',0.2),
    ('conv5_1',0.2)
]
def style_cost(model,style_layers):
    jstyle=0.
    for layer,coeff in style_layers:
        out=model[layer]
        s=sess.run(out)
        g=out
        layerstylecost=layer_style_cost(s,g)*coeff
        jstyle=jstyle+layerstylecost
    return jstyle
def total_cost(j_content,j_style,alpha,beta):
    return (j_content*alpha) + (j_style*beta)
mean_vgg=np.array([103.939,116.779,123.68]).reshape(1,1,1,3)
mean_vgg
# BGR
content=image.load_img('./con.jpg',target_size=(224,224))
con_arr=np.asarray(content)
con_arr=np.expand_dims(con_arr,axis=0)
con_arr=con_arr[:,:,:,::-1]
con_arr=con_arr-mean_vgg
print(con_arr.shape)
plt.imshow(content)
style=image.load_img('./sty1.jpg',target_size=(224,224))
sty_arr=np.asarray(style)
sty_arr=np.expand_dims(sty_arr,axis=0)
sty_arr=sty_arr[:,:,:,::-1]
sty_arr=sty_arr-mean_vgg
print(sty_arr.shape)
plt.imshow(style)
init_gen_arr=np.random.uniform(1,255,(1,224,224,3))
plt.imshow(init_gen_arr[0,:,:,:])
init_gen_arr=init_gen_arr-128
print(init_gen_arr.shape)
tf.reset_default_graph()
sess=tf.InteractiveSession()
model=load_vgg_model('./vgg19.mat')
model
sess.run(model['input'].assign(con_arr))
out=model['conv4_2']
c=sess.run(out)

g=out

j_content=content_cost(c,g)
sess.run(model['input'].assign(sty_arr))
j_style=style_cost(model,style_layers)
j_total=total_cost(j_content,j_style,1,100)
optimizer=tf.train.AdamOptimizer(3.)
train_step=optimizer.minimize(j_total)
init=tf.global_variables_initializer()
def print_pic(gen_arr):
    gen_arr=gen_arr+mean_vgg
    gen_arr=gen_arr[:,:,:,::-1]
    gen_arr=np.clip(gen_arr,0,255)
    plt.imshow(gen_arr[0,:,:,:])
def train(sess,model,iterations,input_image):
    sess.run(model['input'].assign(input_image))
    sess.run(init)
    pics={}
    j=1
    for i in range(1,iterations+1):
        sess.run(train_step)
        if i%100==0:
            jc=sess.run(j_content)
            js=sess.run(j_style)
            jt=sess.run(j_total)
            print(i,j,jc,js,jt)
            gen_arr=sess.run(model['input'])
            pics[str(j)]=gen_arr
            j=j+1
    print_pic(gen_arr)
    return pics
pics=train(sess,model,2000,init_gen_arr)
