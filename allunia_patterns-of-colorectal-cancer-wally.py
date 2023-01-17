import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="darkgrid")



from os import listdir

print(listdir("../input"))



import tensorflow as tf



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
data = pd.read_csv('../input/colorectal-histology-mnist/hmnist_64_64_L.csv')

data.head()
print("The data consists of {} samples".format(data.shape[0]))
class_names = {1: "Tumor", 2: "Stroma", 3: "Complex", 4: "Lympho",

               5: "Debris", 6: "Mucosa", 7: "Adipose", 8: "Empty"}

class_numbers = {"Tumor": 1, "Stroma": 2, "Complex": 3, "Lympho": 4,

               "Debris": 5, "Mucosa": 6, "Adipose": 7, "Empty": 8}

class_colors = {1: "Red", 2: "Orange", 3: "Gold", 4: "Limegreen",

                5: "Mediumseagreen", 6: "Darkturquoise", 7: "Steelblue", 8: "Purple"}



label_percentage = data.label.value_counts() / data.shape[0]

class_index = [class_names[idx] for idx in label_percentage.index.values]



plt.figure(figsize=(20,5))

sns.barplot(x=class_index, y=label_percentage.values, palette="Set3");

plt.ylabel("% in data");

plt.xlabel("Target cancer class");

plt.title("How is cancer distributed in this data?");
fig, ax = plt.subplots(2,4, figsize=(25,10))

for n in range(2):

    for m in range(4):

        class_idx = n*4+(m+1)

        sns.distplot(data[data.label == class_idx].drop("label", axis=1).values.flatten(),

                     ax=ax[n,m],

                     color=class_colors[class_idx])

        ax[n,m].set_title(class_names[class_idx])

        ax[n,m].set_xlabel("Intensity")

        ax[n,m].set_ylabel("Density")
from sklearn.model_selection import train_test_split



temp_data, val = train_test_split(data, test_size=0.2, stratify=data.label.values, random_state=2019)

train, test = train_test_split(temp_data, test_size=0.2, stratify=temp_data.label.values, random_state=2020)





label_counts = pd.DataFrame(index=np.arange(1,9), columns=["train", "test", "val"])

label_counts["train"] = train.label.value_counts().sort_index()

label_counts["test"] = test.label.value_counts().sort_index()

label_counts["val"] = val.label.value_counts().sort_index()
plt.figure(figsize=(10,5))

sns.heatmap(label_counts.transpose(), cmap="YlGnBu", annot=True, cbar=False, fmt="g");

plt.xlabel("Cancer label number")
class Wally:

    

    def __init__(self, num_classes,

                 n_features,

                 image_width,

                 image_height,

                 image_color_channels,

                 learning_rate):

        # The number of pixels per image

        self.n_features = image_width * image_height * image_color_channels

        # Image dimensions

        self.image_width = image_width

        self.image_height = image_height

        self.color_channels = image_color_channels

        # The number of unique cancer classes 

        self.num_classes = num_classes

        # The learning rate of our loss optimizer

        self.learning_rate = learning_rate

    

    def create_placeholder(self):

        with tf.name_scope("placeholder"):

            self.inputs = tf.placeholder(tf.float32, shape=[None, self.n_features], name="inputs")

            self.targets = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="targets")

    

    # Let's define a method to feed Wally with data and fill in placeholders:

    def feed(self, x, t):

        food = {self.inputs: x, self.targets: t}

        return food

    

    def build_wally(self):

        self.graph = tf.Graph()

        with self.graph.as_default():

            self.create_placeholder()

            self.body()

            self.heart_beat()

            self.wonder()

            self.blood()

    

    def blood(self):

        self.init_op = tf.initialize_all_variables()

    

    def body(self):

        with tf.name_scope("body"):

            

            image = tf.reshape(self.inputs, shape=[-1,

                                                   self.image_height,

                                                   self.image_width,

                                                   self.color_channels]

                              )

        

            self.conv1, self.conv1_weights, self.conv1_bias = self.get_convolutional_block(

                image, self.color_channels, 5, "convolution1"

            )

            self.active_conv1 = tf.nn.relu(self.conv1)

        

            flatten1 = tf.reshape(self.active_conv1, [-1, self.image_height * self.image_width * 5])

            fc1 = self.get_dense_block(

                flatten1, "fullyconnected1", self.image_height * self.image_width * 5, 20

            )

        

            self.logits = self.get_output_block(fc1, "output", 20)

            self.predictions = tf.nn.softmax(self.logits)

    

    def heart_beat(self):

        with tf.name_scope("heartbeat"):

            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(

                logits=self.logits, labels=self.targets))

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            self.train_op = optimizer.minimize(self.loss_op)

    

    def wonder(self):

        with tf.name_scope("wonder"):

            correct_pred = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.targets, 1))

            self.evaluation_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    

    def sleep(self):

        self.sess.close()

    

    def learn(self, x_train, t_train, x_test, t_test, max_steps):

        self.train_losses = []

        self.test_losses = []

        self.train_scores = []

        self.test_scores = []

        

        x_train = x_train/255 - 0.5

        x_test = x_test/255 - 0.5

        

        self.sess = tf.Session(graph=self.graph)

        self.sess.run(self.init_op)

        

        for step in range(max_steps):

            _, train_loss, train_score = self.sess.run(

                [self.train_op, self.loss_op, self.evaluation_op],

                feed_dict=self.feed(x=x_train, t=t_train)

            )

            

            if step % 5 == 0:

                test_loss, test_score = self.sess.run(

                    [self.loss_op, self.evaluation_op],

                    feed_dict=self.feed(x=x_test, t=t_test)

                )

                train_loss = np.round(train_loss, 4)

                self.train_losses.append(train_loss)

                test_loss = np.round(test_loss, 4)

                self.test_losses.append(test_loss)

                train_score = np.round(train_score, 2)

                self.train_scores.append(train_score)

                test_score = np.round(test_score, 2)

                self.test_scores.append(test_score)

                

                print("Learning step {}".format(step))

                print("Train loss: {}, and train score: {}.".format(train_loss, train_score))

                print("Test loss: {}, and test score: {}.".format(test_loss, test_score))

        

        self.first_kernel, self.first_hidden_neurons, self.first_neurons = self.sess.run(

            [self.conv1_weights, self.conv1, self.active_conv1],

            feed_dict=self.feed(x=x_train, t=t_train)

        )

         

        

    def get_convolutional_block(self, images, in_channel, out_channel, blockname):

        with tf.variable_scope(blockname):

            weights = tf.Variable(

                tf.truncated_normal(shape=[3,3,in_channel,out_channel], mean=0, stddev=0.01, seed=0),

                name="weights")

            bias = tf.Variable(

                tf.truncated_normal(shape=[out_channel], mean=0, stddev=0.01, seed=0),

                name="bias")

            conv_neurons = tf.nn.conv2d(images, weights,

                                        strides=[1,1,1,1],

                                        padding="SAME",

                                        data_format='NHWC',

                                        name="conv_neurons")

            hidden_neurons = tf.nn.bias_add(conv_neurons, bias, name="hidden_neurons")

        return hidden_neurons, weights, bias

    

    def get_dense_block(self, flatten, blockname, n_inputs, n_outputs):

        with tf.variable_scope(blockname):

            weights = tf.Variable(

                tf.truncated_normal(shape=[n_inputs, n_outputs], mean=0, stddev=0.01, seed=1),

                name="weights")

            bias = tf.Variable(

                tf.truncated_normal(shape=[n_outputs], mean=0, stddev=0.01, seed=1),

                name="bias")

            fc_neurons = tf.add(tf.matmul(flatten, weights), bias)

        return fc_neurons

    

    def get_output_block(self, flatten, blockname, n_inputs):

        with tf.variable_scope(blockname):

            weights = tf.Variable(

                tf.truncated_normal(shape=[n_inputs, self.num_classes], mean=0, stddev=0.01, seed=2),

                name="weights")

            bias = tf.Variable(

                tf.truncated_normal(shape=[self.num_classes], mean=0, stddev=0.01, seed=2),

                name="bias")

            hidden_output = tf.add(tf.matmul(flatten, weights), bias, name="logits")

        return hidden_output

    

    def tell_fortune(self, x):

        x = x/255 - 0.5

        predictions = self.sess.run(self.predictions, feed_dict={self.inputs: x})

        return predictions

    
from sklearn.preprocessing import OneHotEncoder



t_train = train.label.values

t_test = test.label.values

t_val = val.label.values



x_train = train.drop("label", axis=1).values

x_test = test.drop("label", axis=1).values

x_val = val.drop("label", axis=1).values



encoder = OneHotEncoder(sparse=False)

t_train = encoder.fit_transform(t_train.reshape(-1,1))

t_test = encoder.transform(t_test.reshape(-1,1))

t_val = encoder.transform(t_val.reshape(-1,1))
your_cancer = "Tumor"

seed = 0
def norm_image(image):

    return (image - np.mean(image)) / np.std(image)



def min_max_scaling(image, new_min=-0.5, new_max=0.5):

    return new_min + (image - np.min(image)) * (new_max - new_min) /(np.max(image) - np.min(image))
f = min_max_scaling
image_ids = data[data.label == class_numbers[your_cancer]].index.values

selected_ids = np.random.RandomState(seed).choice(image_ids, 4)

sns.set()



fig, ax = plt.subplots(4,4, figsize=(20,22))

for n in range(4):

    image = data.loc[selected_ids[n]].drop("label").values

    original = image.reshape((64,64))

    ax[0,n].imshow(original, cmap="gray")

    ax[0,n].set_title("Original image of class \n {}, Id:{}".format(your_cancer, selected_ids[n]))

    

    sns.distplot(image, ax=ax[1,n], color="midnightblue")

    ax[1,n].axvline(np.mean(image), c="r")

    ax[1,n].set_title("Original intensity distribution")

    

    normed_image = f(image)

    

    ax[2,n].imshow(np.reshape(normed_image, (64,64)), cmap="gray", vmin=-2, vmax=5)

    sns.distplot(normed_image, ax=ax[3,n], color="cornflowerblue")

    #ax[3,n].set_xlim([-2,5])

    ax[3,n].axvline(np.mean(normed_image), c="r")

    ax[3,n].set_title("Per-Image normalized \n intensity distribution")

    
num_classes = len(train.label.unique())

n_features = x_train.shape[1]

image_height = 64

image_width = 64

image_color_channels = 1

eta = 0.01

max_steps = 150
robot = Wally(num_classes = num_classes,

              n_features = n_features,

              image_width = image_width,

              image_height = image_height,

              image_color_channels = image_color_channels,

              learning_rate = eta)
robot.build_wally()

robot.learn(x_train, t_train, x_test, t_test, max_steps)
sns.set()

plt.figure(figsize=(20,5))

plt.plot(np.arange(0,max_steps,5), robot.train_losses, '+--', label="Train loss")

plt.plot(np.arange(0,max_steps,5), robot.test_losses, '+--', label="Test loss")

plt.xlabel("Learning steps")

plt.ylabel("Loss")

plt.legend();
example_kernel = np.squeeze(robot.first_kernel)



fig, ax = plt.subplots(1,5,figsize=(20,5))

for n in range(5):

    ax[n].imshow(example_kernel[:,:,n], cmap="coolwarm")

    ax[n].set_title("Weight kernel {}".format(n+1))
p_val = robot.tell_fortune(x_val)
robot.sleep()
class BigEyeWally(Wally):

    

    def __init__(self, num_classes,

                 n_features,

                 image_width,

                 image_height,

                 image_color_channels,

                 learning_rate):

        super().__init__(num_classes,

                 n_features,

                 image_width,

                 image_height,

                 image_color_channels,

                 learning_rate)

        

    def body(self):

        with tf.name_scope("body"):

            

            image = tf.reshape(self.inputs, shape=[-1,

                                                   self.image_height,

                                                   self.image_width,

                                                   self.color_channels]

                              )

        

            self.conv1, self.conv1_weights, self.conv1_bias = self.get_convolutional_block(

                image, self.color_channels, 10, "convolution1"

            )

            self.active_conv1 = tf.nn.relu(self.conv1)

            

            self.conv2, self.conv2_weights, self.conv2_bias = self.get_convolutional_block(

                self.active_conv1, 10, 5, "convolution2"

            )

            

            self.active_conv2 = tf.nn.relu(self.conv2)

            

            flatten1 = tf.reshape(self.active_conv2, [-1, self.image_height * self.image_width * 5])

            fc1 = self.get_dense_block(

                flatten1, "fullyconnected1", self.image_height * self.image_width * 5, 20

            )

        

            self.logits = self.get_output_block(fc1, "output", 20)

            self.predictions = tf.nn.softmax(self.logits)
robot = BigEyeWally(num_classes = num_classes,

              n_features = n_features,

              image_width = image_width,

              image_height = image_height,

              image_color_channels = image_color_channels,

              learning_rate = eta)
robot.build_wally()

robot.learn(x_train, t_train, x_test, t_test, max_steps)
robot.sleep()
sns.set()

plt.figure(figsize=(20,5))

plt.plot(np.arange(0,max_steps,5), robot.train_losses, '+--', label="Train loss")

plt.plot(np.arange(0,max_steps,5), robot.test_losses, '+--', label="Test loss")

plt.xlabel("Learning steps")

plt.ylabel("Loss")

plt.legend();
sample_maps = robot.first_hidden_neurons



fig, ax = plt.subplots(10,5,figsize=(20,50))

for m in range(10):

    for n in range(5):

        ax[m, n].imshow(sample_maps[m,:,:,n], cmap="coolwarm")

    