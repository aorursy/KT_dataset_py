import numpy as np

import tensorflow as tf



from glob import glob

from tqdm import tqdm

from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
model = tf.keras.models.load_model("../input/tripletmodel/triplet_inception_resnet_v1_0.h5")  # load model

input_shape = model.layers[0].input_shape[0][1:]  # get input(image) shape
def load_image(path):  # function to load images

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, (input_shape[0], input_shape[1]), method="nearest")



    return tf.cast(image, tf.float32)/255.
def triplet_loss(output):  # function for triplet loss

    anchor, positive, negative = tf.unstack(tf.reshape(output, (-1, 3, 512)), num=3, axis=1)



    positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)

    negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)



    loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), 0.2)

    loss = tf.reduce_mean(tf.maximum(loss_1, 0.0), 0)



    return loss
ac1 = load_image("../input/pins-face-recognition/105_classes_pins_dataset/pins_alycia dabnem carey/alycia dabnem carey0_0.jpg")  # load face of alycia

ac2 = load_image("../input/pins-face-recognition/105_classes_pins_dataset/pins_alycia dabnem carey/alycia dabnem carey100_3.jpg")  # load face of alycia

ad3 = load_image("../input/pins-face-recognition/105_classes_pins_dataset/pins_Alexandra Daddario/Alexandra Daddario0_214.jpg")  # load face of alexandra
fig=plt.figure(figsize=(64,64))  # display faces



fig.add_subplot(1, 3, 1)

plt.imshow(ac1)

fig.add_subplot(1, 3, 2)

plt.imshow(ac2)

fig.add_subplot(1, 3, 3)

plt.imshow(ad3)
outputs = model(tf.concat([tf.expand_dims(ac1, axis=0), tf.expand_dims(ac2, axis=0), tf.expand_dims(ad3, axis=0)], axis=0), training=False)  # get embeddings

outputs = tf.nn.l2_normalize(outputs, 1, 1e-10)  # apply l2 normaization(optional)
loss = triplet_loss(outputs)  # get the loss value

print(f"Loss --> {loss}")  # PERFECT SCORE!
print(tf.norm(outputs[0] - outputs[1]))  # Distance between image 1 and image 2 (alycia-alycia)

print(tf.norm(outputs[0] - outputs[2]))  # Distance between image 1 and image 3 (alycia-Alexandra)

print(tf.norm(outputs[1] - outputs[2]))  # Distance between image 1 and image 2 (alycia-Alexandra)



# As you can see from the output, embeddings are closer when they compared between same person.
# Let's make a space!



y_map = {}  # ["label name": label_id]

i = 0  # id iterator



x_data, y_data = [], []



for path in tqdm(glob("../input/pins-face-recognition/105_classes_pins_dataset/*/*.*")):

    label_str = path.split("/")[-2]

    

    if not label_str in y_map.keys():  # Add to map if it is not already in it

        y_map[label_str] = i

        i += 1

        

    label_int = y_map[label_str]

    image = load_image(path)

    

    # get every path, load the image from path

    # extract label from path, turn label into integer

    

    x_data.append(image)

    y_data.append(label_int)

def chunks(lst, n):  # from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks

    """Yield successive n-sized chunks from lst."""

    for i in range(0, len(lst), n):

        yield lst[i:i + n]

        

embed_map = {}

for key in y_map:

    embed_map[y_map[key]] = []

    

outputs = []

for images in tqdm(chunks(x_data, 32)):

    for output in model(tf.convert_to_tensor(images), training=False):

        outputs.append(output)

        

outputs = tf.convert_to_tensor(outputs)  # Get embeddings from model by iterating data with batch size 32
pca = PCA(n_components=2)  # compress 512-D data to 2-D, we need to do that if we want to display data.

pC = pca.fit_transform(outputs)
fig,ax=plt.subplots(figsize=(10,10))

fig.patch.set_facecolor('white')

for l in np.unique(y_data)[:10]:

    ix=np.where(y_data==l)

    ax.scatter(pC[:,0][ix],pC[:,1][ix])



plt.xlabel("First Principal Component",fontsize=14)

plt.ylabel("Second Principal Component",fontsize=14)

plt.legend()

plt.show()



# As you can see, the results are not good. Because we are compressing 512-D data to 2-D so there happens a lot of data loss. 

# Space splitted to 2 as you can see, i think this a seperation of man and woman. Let's check :)
for key in list(y_map.keys())[:10]:  # YES! There are 5 man and 5 woman. And with our logic, if we get first 5 outputs, there should be 1 man and 4 woman. Let's check that too.

    print(key)
fig,ax=plt.subplots(figsize=(10,10))

fig.patch.set_facecolor('white')

for l in np.unique(y_data)[:5]:

    ix=np.where(y_data==l)

    ax.scatter(pC[:,0][ix],pC[:,1][ix])



plt.xlabel("First Principal Component",fontsize=14)

plt.ylabel("Second Principal Component",fontsize=14)

plt.legend()

plt.show()  # YES! Seems like those woman are not interested with Chris Pratt at all :)