#The below code is included to make the plots expandable.
%matplotlib notebook

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random
# import module we'll need to import our custom module
from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/pcaploty/pca_plotter.py", dst = "../working/pca_plotter.py")

# import all our functions
from pca_plotter import PCAPlotter
#The image data is taken , normalized and splitted into train and validation sets of 128*128 resolution images.
train_datagen=ImageDataGenerator(rescale=1/255.,validation_split=0.1)
train_set=train_datagen.flow_from_directory("../input/age-prediction/20-50/20-50/train",target_size=(128,128),shuffle=True,batch_size=2048,class_mode='categorical',subset='training')
validation_set=train_datagen.flow_from_directory("../input/age-prediction/20-50/20-50/train",target_size=(128,128),batch_size=4000,class_mode='categorical',subset='validation')
def plot_triplets(examples,example_labels):
    plt.figure(figsize=(10, 3))
    for i in range(3):
        plt.subplot(1, 4, 1 + i)
        plt.imshow(np.reshape(examples[i], (128,128,3)))
        plt.xticks([])
        plt.yticks([])
        plt.title(example_labels[i])
    plt.show()
#The train_set shape is(batchs,2(i.e [image data,labels]),2048,128*128 for image data (or) 1 for labels)
plot_triplets([train_set[0][0][1], train_set[1][0][0], train_set[2][0][4]],[20+np.argmax(train_set[0][1][1]),20+np.argmax(train_set[1][1][0]),20+np.argmax(train_set[2][1][4])])
# b_c = total_number_of_batches-1 i.e 30103//2048
def create_batch(batch_size=256,b_c=14):
    #arrays for triplet images is created
    x_anchors = np.zeros((batch_size*b_c, 128,128,3))
    x_positives = np.zeros((batch_size*b_c, 128,128,3))
    x_negatives = np.zeros((batch_size*b_c, 128,128,3))
    #arrays for respective labels.
    y_ind_pos = []
    y_ind_neg = []
    
    #the j for loop goes over batches
    for j in range(1,b_c+1):
        #the i for loop goes over the batch images
        for i in range(0, batch_size):
            # We need to find an anchor, a positive example and a negative example
            random_index = random.randint(0, 2048 - 1) #a random index is generated.
            x_anchor = train_set[j-1][0][random_index]
            y = np.argmax(train_set[j-1][1][random_index],axis=-1) #np.argmax is used as the labels are one-hot encodings i.e 2 = [0 0 1 0 0] for 5 category set.
            
            indices_for_pos = np.squeeze(np.where(np.argmax(train_set[j-1][1],axis=-1) == y)) #Only the index of images that contains the same label as anchor is chosen.
            indices_for_neg = np.squeeze(np.where(np.argmax(train_set[j-1][1],axis=-1) != y)) #The index of images that contains the label other than that of anchor is chosen.
            
            x_positive = train_set[j-1][0][indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
            x_negative_ind =random.randint(0, len(indices_for_neg) - 1)
            x_negative = train_set[j-1][0][indices_for_neg[x_negative_ind]]

            x_anchors[i*j] = x_anchor
            x_positives[i*j] = x_positive
            x_negatives[i*j] = x_negative
            #20 is added just for our reference as we are dealing with people between 20 and 50 ages.
            y_ind_pos.append(20+y)
            y_ind_neg.append(20+np.argmax(train_set[j-1][1][x_negative_ind],axis=-1))
        
    return [x_anchors, x_positives, x_negatives],[y_ind_pos[0],y_ind_pos[0],y_ind_neg[0]]
examples = create_batch(1,1)
#example[0] contains the image data and example[1] contains the labels for the respective images.
print(examples[0],examples[1])
plot_triplets(examples[0],examples[1])
#A tensorflow model that returns an embeddiing of size 64 is created.
emb_size = 64
embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(128,128,3)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(emb_size, activation='sigmoid')
])

embedding_model.summary()
example = np.expand_dims(train_set[0][0][1], axis=0)
example_emb = embedding_model.predict(example)
#Prints the embedding of an image from train_set
print(example_emb)
#input layers for respective elements in the triplet.
input_anchor = tf.keras.layers.Input(shape=(128,128,3))
input_positive = tf.keras.layers.Input(shape=(128,128,3))
input_negative = tf.keras.layers.Input(shape=(128,128,3))
#An embedding layer is created as above.
embedding_anchor = embedding_model(input_anchor)
embedding_positive = embedding_model(input_positive)
embedding_negative = embedding_model(input_negative)
#Concatenating the embeddings obtained from above.
output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)
net.summary()
alpha = 0.2 #This value is not mandatory to change.

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    #After calculating the mean squared error , loss is calculated.
    return tf.maximum(positive_dist - negative_dist + alpha, 0.) 
def data_generator(batch_size=256):
    while True:
        x,_= create_batch(batch_size)
        y = np.zeros((batch_size, 3*emb_size)) #As the model does not need the labels for training although the framework does need it to be mentioned, sparse array is sent.
        yield x, y
#It will take some time for training as triplets are formed only before training for each epoch.
batch_size = 4
epochs = 16

net.compile(loss=triplet_loss, optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.002))

_ = net.fit(
    data_generator(batch_size),
    steps_per_epoch=1,
    epochs=epochs, verbose=1,
    callbacks=[
        #PCAPlot is used to plot the embeddings and loss for each epoch of both train_set and validation_set using pca.
        PCAPlotter(
            plt, embedding_model,
            train_set[0][0], np.argmax(train_set[0][1],axis=-1)),
        PCAPlotter(
            plt, embedding_model,
            validation_set[0][0], np.argmax(validation_set[0][1],axis=-1)
        )]
)
#I have included two weights after training it along this notbook and used the best one below.
net.load_weights("../input/agebasedfacecategorizer/saved_final_weights.h5")
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
pred=[]
plt.figure(figsize=(8,4))
for j in range(14):
    p=embedding_model.predict(train_set[j][0])
    pca_out=PCA(n_components=2).fit_transform(p)
    pred.append(pca_out)

pred=np.concatenate(np.array(pred),axis=0)
pca_df= pd.DataFrame(pred,columns=['pca1','pca2'])
pca_cat = pd.concat([pca_df,pd.DataFrame({"Labels":np.argmax(train_set[j][1],axis=-1)})],axis=1)
cmap=sns.cubehelix_palette(n_colors=31,dark=.4, light=.8,as_cmap=True)
ax=sns.scatterplot(x='pca1',y='pca2',hue="Labels",data= pca_cat,palette="Set2",size="Labels",sizes=(20, 200))
plt.show()