from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D

from keras.layers import Activation, Dropout, Flatten, Dense, Input, BatchNormalization

from keras.regularizers import l2

from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.optimizers import RMSprop, Adam

from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import OneHotEncoder

from keras.losses import BinaryCrossentropy

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from PIL import Image

import seaborn as sns
#Load Data

images_dir = '../input/germantrees/data'

data_csv_name = 'classes.csv'

df=pd.read_csv(images_dir + '/' + data_csv_name)

df.head()
sns.countplot(x='class', data=df)

df['class'].value_counts()
df = df[(df['class'] == 'oak') | (df['class'] == 'maple')]


#Show some images to get a better impressive of the Data set

def plot_pics(pics,title=None,rows=5,cols=7, color=True, image_size=(384,512)):

    fig, axis = plt.subplots(figsize=(3*cols,4*rows),nrows=rows ,ncols=cols)

    for img, ax in zip(pics.sample(rows*cols),axis.flatten()):

        if color:

            im = Image.open(img)

        else :

            im = Image.open(img).convert('LA')

        im=im.resize(image_size, Image.ANTIALIAS)

        ax.imshow(im, cmap='gray')

        ax.set_xticklabels([])

        ax.set_yticklabels([])

    if title!=None: fig.suptitle(title, fontsize=20)

    plt.subplots_adjust(wspace=-0.1,hspace=0)





maple_pics = images_dir+'/'+df[df['class']=='maple']['image_name']

plot_pics(maple_pics,'Maple Leaves')

oak_pics = images_dir+'/'+df[df['class']=='oak']['image_name']

plot_pics(oak_pics,'Oak Leaves')
#Transform some oak and maple images and get rif off the color.

plot_pics(maple_pics,'Maple Leaves',rows=2,cols=7,color=False)

plot_pics(oak_pics,'Oak Leaves',rows=2,cols=7, color=False)
#Split the data into a train set, val set and a test set

seed=532413

train_df, val_df = train_test_split(df, test_size=0.3, random_state=seed)

val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=seed)
# We set up the generators which are responsible to feed our models with the data.

def create_generators(dataframe, directory_path, preprocessing_function=None, augmentation=False, color_mode="rgb",

                      just_generator=False):

    x_col, y_col = dataframe.columns

    # If augmentation==True, then use augmentation techniques which do not alter the shape of the leafes.

    if augmentation:

        generator = ImageDataGenerator(preprocessing_function=preprocessing_function,

                                                  rescale=1./255.,

                                                  rotation_range=90,

                                                  vertical_flip=True,

                                                  horizontal_flip=True,

                                                  zoom_range=0.2,

                                                  width_shift_range=0.2,

                                                  height_shift_range=0.2)

    else:

        generator = ImageDataGenerator(preprocessing_function=preprocessing_function, 

                                       rescale=1. / 255.)

     

    

    return generator.flow_from_dataframe(dataframe, directory_path, x_col=x_col, y_col=y_col,

                                         target_size=image_size, batch_size=batch_size, color_mode=color_mode)
def show_augmentation_effects(image):

    #Create a generator with the same augmentations as in the function create_generator

    generator = ImageDataGenerator(preprocessing_function=None,

                                                  rescale=1./255.,

                                                  rotation_range=90,

                                                  vertical_flip=True,

                                                  horizontal_flip=True,

                                                  zoom_range=0.2,

                                                  width_shift_range=0.2,

                                                  height_shift_range=0.2)

    

    #Present the effects of image augmentation

    im = load_img(images_dir + '/' + image)

    image_size=(128, 128)

    im=im.resize(image_size, Image.ANTIALIAS)

    x = img_to_array(im)

    original_shape = x.shape

    x=x.reshape((1,)+x.shape)



    #create an iterator

    flow=iter(generator.flow(x, batch_size=1))

    rows, cols = (1,6)

    fig, axes = plt.subplots(figsize=(3*cols,4*rows),nrows=rows,ncols=cols)

    fig.suptitle(image, fontsize=20)

    axes[0].imshow(im)

    axes[0].set_xticklabels([])

    axes[0].set_yticklabels([])

    for ax in axes.flatten()[1:]:

        z=next(flow)

        z=z.reshape(original_shape)

        ax.imshow(z)

        ax.set_xticklabels([])

        ax.set_yticklabels([])



show_augmentation_effects('maple137.jpg')

show_augmentation_effects('maple1000.jpg')

show_augmentation_effects('oak138.jpg')

show_augmentation_effects('oak1010.jpg')
class classic_model(Sequential):

    def __init__(self, input_shape, parameters, l2_reg_factor=0, dropout=0, batch_normalization=False, num_classes=2):

        super().__init__()

        #save the constuctor parameters

        self.parameters = parameters

        self.l2_reg_factor = l2_reg_factor

        self.dropout = dropout

        self.batch_normalization = batch_normalization

        

        #extract the information

        filters, kernels, strides, paddings, pools, dense = parameters.values()



        #build the model

        self.add(Input(shape=(input_shape[0],input_shape[1],input_shape[2],),name='input'))

        #Build convolution layers

        for i in range(len(filters)):

            self.add(Conv2D(filters[i], (kernels[i],kernels[i]), (strides[i],strides[i]),

                            activation='relu',padding=paddings[i],name='conv_'+str(i+1)))

            self.add(MaxPooling2D((pools[i],pools[i]), name='pool_'+str(i+1)))

            self.add(SpatialDropout2D(dropout))

            if batch_normalization:

                self.add(BatchNormalization(momentum=0.9))

                

        self.add(Flatten(name='flatten'))

        

        #Build dense layers

        for i in range(len(dense)):

            self.add(Dense(dense[i], activation='relu', kernel_regularizer=l2(l2_reg_factor), name='dense_'+str(i+1)))

            self.add(Dropout(dropout, name='dropout_'+str(i+1)))

            if batch_normalization:

                self.add(BatchNormalization(momentum=0.9))



        #Output Layer

        self.add(Dense(num_classes, activation='softmax', name='output'))

    

    def get_signature(self):

        #The signature consists of the filter sizes and the sizes of the fully connected layers

        filters, kernels, strides, paddings, pools, dense = self.parameters.values()

        sig = 'Conv'

        for i in range(len(filters)): sig = sig + ':' + str(filters[i])

        sig = sig + ';Full'

        for i in range(len(dense)): sig = sig + ':' + str(dense[i])

        if self.batch_normalization: sig =  sig + '+BN'

        #Convert the numbers dropout in L2 to strings, but we need to replace '.' with ',',

        #to avoid conflicts.

        if self.dropout != 0: 

            sig = sig + '+DP:' + str(self.dropout).replace('.',',')

        if self.l2_reg_factor !=0: 

            sig = sig + '+L2:' + str(self.l2_reg_factor).replace('.',',')

        return sig

    

#Presentation:

par={'filters':[32,32,64],

                'kernels': [3]*3,

                'strides': [1]*3,

                'paddings': ['same']*3,

                'pools': [2]*3,

                'dense': [64]}

        

model = classic_model(input_shape=(128,128,3),parameters=par,l2_reg_factor=0.1, dropout=0.1)

print('Model Signature:' + model.get_signature())

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.summary()
def perform_experiment_one(parameters, image_size, optimizer=RMSprop(), augmentation=False, batch_normalization=False, 

                           dropout=0, l2=0, color_mode="rgb", name=None):

    

    if color_mode == 'rgb':

        input_shape=(image_size[0],image_size[1],3)

    

    if color_mode == 'grayscale':

        input_shape=(image_size[0],image_size[1],1)

    

    # Generator

    gen_train = create_generators(train_df, images_dir, augmentation=augmentation, color_mode=color_mode)

    gen_val = create_generators(val_df, images_dir, color_mode=color_mode)



    #build the model

    model = classic_model(input_shape, parameters, dropout=dropout, batch_normalization=batch_normalization, l2_reg_factor=l2)



    # compile shallow model

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    

    #create signature

    if name == None: name=model.get_signature()

    name = name + '|' + color_mode + str(image_size)

    if augmentation: name = name + '+Aug'

    print(name)

    

    # fit the model

    hist = model.fit(gen_train, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=gen_val,verbose = 0)

    return (model,hist,name)





def plot_history(hist, name='Model', n=8):

    fig, ax = plt.subplots(figsize=(10,16), nrows=2)

    fig.suptitle(name, fontsize=20)

    #create a plot of the history of the error:

    ax[0].plot(hist.history['val_loss'], label='val', zorder=20)

    ax[0].plot(hist.history['loss'], label='train', zorder=30)

    ax[0].legend(loc='upper right')

    ax[0].set(ylabel='loss',

           ylim=[0, 2],

           xlabel='epoch')

    ax[0].yaxis.label.set_size(20)

    ax[0].xaxis.label.set_size(20)

    ax[0].legend(loc='upper right',prop={'size': 15})



    #create a plot of the history of accuracy:

    ax[1].plot(hist.history['val_accuracy'], label='val', zorder=20)

    ax[1].plot(hist.history['accuracy'], label='train', zorder=30)

    h = hist.history['val_accuracy']

    avg_h = [(sum(h[i:(i + n)])) / n for i in range(len(h) - n)]

    ax[1].plot(np.arange(n, len(h)), avg_h, color='red', label='trend_val', zorder=40)

    ax[1].set(ylabel='accuracy',

           ylim=[0.6, 1.05],

           xlabel='epoch')

    ax[1].yaxis.label.set_size(20)

    ax[1].xaxis.label.set_size(20)

    ax[1].legend(loc='upper left',prop={'size': 15})

    plt.axhline(0.8, color='steelblue', zorder=10, alpha=0.5)

    plt.axhline(0.9, color='crimson', zorder=10, alpha=0.5)

    plt.savefig(name)

    plt.show()





def save_results(model, hist, df_hist, name=None):

    if name == None: name=model.get_signature()

    plot_history(hist,name)

    df_hist[name + '-Train'] = hist.history['accuracy']

    df_hist[name + '-Val'] = hist.history['val_accuracy']

    model.save_weights(name +'.h5')

    



def test_model(model):

    gen_test = create_generators(test_df, images_dir, color_mode=color_mode)

    test_res = model.evaluate(gen_test, verbose = 0)

    print('Test-Accuracy: ' +str(test_res[1]))
#Our model

final_par= {'filters':[32,32,64,64,64],

                    'kernels': [3]*5,

                    'strides': [1]*5,

                    'paddings': ['same']*5,

                    'pools': [2]*5,

                    'dense': [120,20,20]}

color_mode = 'rgb'

image_size = (128,128)

model = classic_model(input_shape=(128,128,3),parameters=final_par,l2_reg_factor=0.1, dropout=0.1, batch_normalization=True)

print('Model Signature:' + model.get_signature())

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.summary()


#Training variables

batch_size = 64

epochs = 400

steps_per_epoch = int(train_df.shape[0] / batch_size)



df_hist = pd.DataFrame()

model, hist, name = perform_experiment_one(final_par, image_size, optimizer=Adam(),augmentation = True, 

                                           batch_normalization=True, dropout=0.1, l2=0.1, color_mode=color_mode)

save_results(model, hist, df_hist, name = name)

test_model(model)