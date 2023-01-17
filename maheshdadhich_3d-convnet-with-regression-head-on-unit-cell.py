import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
# For reading .xyz extention files 

def get_xyz_data(filename):

    """source - https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates

    Thanks Tony Y. for this function"""

    pos_data = []

    lat_data = []

    with open(filename) as f:

        for line in f.readlines():

            x = line.split()

            if x[0] == 'atom':

                pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])

            elif x[0] == 'lattice_vector':

                lat_data.append(np.array(x[1:4], dtype=np.float))

    return pos_data, np.array(lat_data)
def image_dimension(train_xyz, print_ = 0):

    """function to return the values min-max for each dimension"""

    x_cordinates = list()

    y_cordinates = []

    z_cordinates = []

    for i in range(train_xyz.__len__()):

        x_cordinates.append(train_xyz[i][0][0]) 

        y_cordinates.append(train_xyz[i][0][1])

        z_cordinates.append(train_xyz[i][0][2])

    min_x = min(x_cordinates) 

    max_x = max(x_cordinates)

    min_y = min(y_cordinates)

    max_y = max(y_cordinates)

    min_z = min(z_cordinates)

    max_z = max(z_cordinates)

    if print_ == 1:

        print("min_max of X is {} - {}".format(min_x, max_x))

        print("min_max of Y is {} - {}".format(min_y, max_y))

        print("min_max of Z is {} - {}".format(min_z, max_z))

    return(min_x, max_x, min_y, max_y, min_z, max_z)



def new_cordinates(point , train_xyz, pixel =20):

    """function to give nw cordinates for unit cell 3d image"""

    min_x, max_x, min_y, max_y, min_z, max_z = image_dimension(train_xyz)

    old_x = point[0][0]

    old_y = point[0][1]

    old_z = point[0][2]

    new_x = 0 + pixel*abs((old_x-min_x)/(max_x - min_x))

    new_y = 0 + pixel*abs((old_y-min_y)/(max_y - min_y))

    new_z = 0 + pixel*abs((old_z-min_z)/(max_z - min_z))

    return(int(new_x), int(new_y), int(new_z))



def image_generator(train_xyz, pixel = 20):

    """function to create a 3d image for convnet regression"""

    rgb = np.zeros((pixel+1, pixel+1, pixel+1, 3), dtype=np.uint8)

    rgb[..., 0] = 0.0

    rgb[..., 1] = 0.0

    rgb[..., 2] = 0.0

    # Ininitally we have initiated a unit cell with all zeros = > No atoms 

    # Now, we will gradually fill atoms, and color code them for 3DConvNet

    for j in range(train_xyz.__len__()):

        #print(j)

        point = train_xyz[j]

        #print(point)

        new_x, new_y, new_z = new_cordinates(point, train_xyz, pixel)

        #if j ==45:

        #print(new_x, new_y, new_z)

        #print(train_xyz[j][1])

        if train_xyz[j][1]=='Al':

            #print(j)

            # Al is red colored

            rgb[new_x,new_y,new_z,0] = 255.0

            #print(rgb[new_x][new_y][new_z])

            rgb[new_x][new_y][new_z][1] = 0.0

            rgb[new_x][new_y][new_z][2] = 0.0

        elif train_xyz[j][1] == 'Ga':

            #print(j)

            #Ga is green

            rgb[new_x][new_y][new_z][0] = 0.0

            rgb[new_x][new_y][new_z][1] = 255.0

            rgb[new_x][new_y][new_z][2] = 0.0

        elif train_xyz[j][1] == 'In':

            #print(j)

            #Ga is green

            rgb[new_x][new_y][new_z][0] = 0.0

            rgb[new_x][new_y][new_z][1] = 0.0

            rgb[new_x][new_y][new_z][2] = 255.0

        else:

            #print(j)

            #print(new_x, new_y, new_z)

            rgb[new_x][new_y][new_z][0] = 255.0

            rgb[new_x][new_y][new_z][1] = 255.0

            rgb[new_x][new_y][new_z][2] = 255.0

    #print(rgb)

    return(rgb)



# function check

#img = image_generator(train_xyz, pixel = 20)          

ids = list(range(1, 2401))

img_array = []

for id_ in ids:

    fn = "../input/train/{}/geometry.xyz".format(id_)

    train_xyz, train_lat = get_xyz_data(fn)

    #print(id_)

    img_temp = image_generator(train_xyz, pixel = 20)

    #print(img_temp.shape)

    img_array.append(img_temp)

img_ar = np.array(img_array)

img_ar.shape
np.save('position_unitcell_21.npy', img_ar)
# Visualizing unit cell Layers 

fig = plt.figure(1,figsize=(15,15))

for i in range(21):

    ax = fig.add_subplot(7,3,i+1)

    arr = img_ar[0][:][i]

    ax.imshow(arr,cmap='inferno')

    

#plt.show()

#plt.imshow(img[:][0])

plt.show()
train_df = pd.read_csv("../input/train.csv")

# bandgap numpy array for 3D convnet

bandgap = np.array(train_df.bandgap_energy_ev.values)

bandgap.shape

bandgap[0]

np.save('bandgap.npy', bandgap)





# bandgap = np.array(train_df.bandgap_energy_ev.values)

ef = np.array(train_df.formation_energy_ev_natom.values)

ef.shape

ef[0]

np.save('ef.npy', ef)
# Start TensorFlow InteractiveSession

import tensorflow as tf

import numpy as np

IMG_SIZE_PX = 21

SLICE_COUNT = 21

n_input = 1

n_classes = 1

batch_size = 30

X = tf.placeholder(tf.float32, [None, 21, 21, 21, 3], name='images')

Y = tf.placeholder(tf.float32, [None, 1], name='labels')

keep_rate = 0.6
def conv3d(x, W):

    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='VALID')



def maxpool3d(x):

    #size of window movement of window as you slide about

    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')

def convolutional_neural_network(x):

    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.

    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,3,32])),

               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.

               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),

               #                                  64 features

               'W_fc':tf.Variable(tf.random_normal([3*3*3*64,1024])),

               'out':tf.Variable(tf.random_normal([1024, n_classes]))}



    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),

               'b_conv2':tf.Variable(tf.random_normal([64])),

               'b_fc':tf.Variable(tf.random_normal([1024])),

               'out':tf.Variable(tf.random_normal([n_classes]))}



    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])

    conv1 = maxpool3d(conv1)





    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])

    conv2 = maxpool3d(conv2)



    fc = tf.reshape(conv2,[-1, 3*3*3*64]) #3*3*3*64

    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

    fc = tf.nn.dropout(fc, keep_rate)



    output = tf.matmul(fc, weights['out'])+biases['out']



    return output
import numpy as np

from IPython.display import clear_output, Image, display, HTML



def strip_consts(graph_def, max_const_size=32):

    """Strip large constant values from graph_def."""

    strip_def = tf.GraphDef()

    for n0 in graph_def.node:

        n = strip_def.node.add() 

        n.MergeFrom(n0)

        if n.op == 'Const':

            tensor = n.attr['value'].tensor

            size = len(tensor.tensor_content)

            if size > max_const_size:

                tensor.tensor_content = "<stripped %d bytes>"%size

    return strip_def



def show_graph(graph_def, max_const_size=32):

    """Visualize TensorFlow graph."""

    if hasattr(graph_def, 'as_graph_def'):

        graph_def = graph_def.as_graph_def()

    strip_def = strip_consts(graph_def, max_const_size=max_const_size)

    code = """

        <script src="//cdnjs.cloudflare.com/ajax/libs/polymer/0.3.3/platform.js"></script>

        <script>

          function load() {{

            document.getElementById("{id}").pbtxt = {data};

          }}

        </script>

        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>

        <div style="height:600px">

          <tf-graph-basic id="{id}"></tf-graph-basic>

        </div>

    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))



    iframe = """

        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>

    """.format(code.replace('"', '&quot;'))

    display(HTML(iframe))
g = tf.Graph()



with g.as_default():

    x = tf.placeholder(tf.float32, name="X")

    n_classes =1

    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,3,32])),

               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.

               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),

               #                                  64 features

               'W_fc':tf.Variable(tf.random_normal([3*3*3*64,1024])),

               'out':tf.Variable(tf.random_normal([1024, n_classes]))}



    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),

               'b_conv2':tf.Variable(tf.random_normal([64])),

               'b_fc':tf.Variable(tf.random_normal([1024])),

               'out':tf.Variable(tf.random_normal([n_classes]))}



    #                            image X      image Y        image Z

    #x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 3])

    with tf.name_scope("Layer1"):

        conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])

        conv1 = maxpool3d(conv1)



    with tf.name_scope("Layer2"):

        conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])

        conv2 = maxpool3d(conv2)

    with tf.name_scope("Layer3"):

        fc = tf.reshape(conv2,[-1, 3*3*3*64]) #3*3*3*64

        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

        fc = tf.nn.dropout(fc, keep_rate)

    with tf.name_scope("Layer4"):

        output = tf.matmul(fc, weights['out'])+biases['out']

    

    

    

tf.summary.FileWriter("logs", g).close()



show_graph(g)
full_data = np.load('./position_unitcell_21.npy')

full_y = np.load('./bandgap.npy')

print(full_data.shape, full_y.shape)

full_y = full_y*10000 # normalizing y for easy convergence 

full_y[1:5]
train_data = full_data[:-240]

train_y = full_y[:-240].reshape(2160,1)

print(train_data.shape, train_y.shape)

validation_data = full_data[-240:]

validation_y = full_y[-240:].reshape(240,1)

print(validation_data.shape, validation_y.shape)
def test_neural_network(x):

    prediction = convolutional_neural_network(X)

    cost = tf.reduce_mean(tf.square(Y-prediction))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=3)

        ckpt = tf.train.latest_checkpoint('.')

        if ckpt:

            saver.restore(sess, ckpt)

            print('Model restored')

        X_test = test_data

        Y_test = test_y

        preds, loss = sess.run([prediction,cost], feed_dict={X: X_test, Y: Y_test})

        print(preds, loss)

        np.save("Predictions.npy", preds)

    



def train_neural_network(x):

    prediction = convolutional_neural_network(x)

    cost = tf.reduce_mean(tf.square(Y-prediction))

    global_step = tf.Variable(0, trainable=False)

    starter_learning_rate = 1e-4

    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,

                                           20*72, 0.10, staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)

    train_op = optimizer.minimize(cost, global_step=global_step)

    hm_epochs = 3

    with tf.Session() as sess:

        saver = tf.train.Saver(max_to_keep=3)

        sess.run(tf.global_variables_initializer())

        

        successful_runs = 0

        total_runs = 0

    

        for epoch in range(hm_epochs):

            epoch_loss = 0.0

            num_batches = len(train_data)//batch_size

            print("Number of batches {}".format(num_batches))

            print("Learning_rate {}".format(learning_rate.eval()))

            indices = np.arange(len(train_data))

            np.random.permutation(indices)

            for batch in range(num_batches):

                batch_indices = indices[batch*batch_size:(batch+1)*batch_size]

                X_batch = train_data[batch_indices]

                Y_batch = train_y[batch_indices]

                _, p, c = sess.run([train_op, prediction, cost], feed_dict={X: X_batch, Y: Y_batch})

                epoch_loss += c

            epoch_loss /= num_batches

            print("Epoch: %d \t Loss: %f" % (epoch+1, epoch_loss))

            print("-"*52)

            

            if (epoch+1)%10 == 0:

                X_val = validation_data

                Y_val = validation_y

                val_pred, val_loss = sess.run([prediction,cost], feed_dict={X: X_val, Y: Y_val})

                print('Validation loss : %f' % val_loss)

            

            saver.save(sess, './model.ckpt', global_step=epoch)

            
train_neural_network(X)