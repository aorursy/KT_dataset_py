%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import pickle
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing import image
from sklearn.utils import shuffle
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from skimage import img_as_ubyte
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_test.head()
h, b = np.histogram(df_train['label'], bins=10, range=(0,10))
sigma = h.std()/np.sqrt(10.0)

plt.figure(figsize=(9,6))
plt.bar(b[:-1], h, width=0.5, fc='r', alpha=0.5)
plt.hlines(h.mean(), -1, 10, colors='b', linestyle='dashed', label='Average')
plt.hlines(h.mean(), -1, 10, colors='b', linewidth=sigma, alpha=0.15)
plt.xlim([-0.7, 9.7])
plt.ylim([0,5500])
plt.xlabel('Digit', fontsize=15)
plt.ylabel('Occurance', fontsize=15)
plt.xticks([0,1,2,3,4,5,6,7,8,9], fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=13)
plt.show()
def plot_digit(x, y):
    '''
    Thus function plots the image of x and, optionally, prints its label if provided.
    Input:
        x: Grey scale image of the digit, i.e. we are assuming there is only one channel. 
           The dimension could be (28, 28) or (28, 28, 1). 
        y: A list of category probabilities. The label is the category that has the 
           maximum probability.
    '''
    plt.imshow(x.reshape(28,28), cmap = matplotlib.cm.binary)
    plt.axis('off')
    print('label:', np.argmax(y))    
    plt.show()
    
    
# -------------------------------------------------------------------------------
def train_val_split(df, size, height, width, channels, seed=0):
    '''
    This function splits the training data into training and validation sets.
    In production mode, set seed=None. Otherwise, the code will generate the same random split.
    '''
    
    norm = 255.0
    # Fetch column names for digit pixels and digit labels
    pixels_col_name = df.columns[1:]
    labels_col_name = df.columns[0]
    
    # Split the dataframe into training and validation sets using a list of shuffled Booleans.
    arr = np.array( [0]*(len(df) - size) + [1]*size )
    arr = (shuffle(arr, random_state=seed)).astype(bool)
    
    train_images = (df[pixels_col_name][arr].values)/norm
    train_images = train_images.reshape(size, height, width, channels)
    train_labels = to_categorical( df[labels_col_name][arr].values )
    
    val_images = (df[pixels_col_name][~arr].values)/norm
    val_images = val_images.reshape(len(df)-size, height, width, channels)
    val_labels = to_categorical( df[labels_col_name][~arr].values )
    
    print('   training tensor shape:', train_images.shape)
    print(' validation tensor shape:', val_images.shape)
    
    return train_images, train_labels, val_images, val_labels



# -------------------------------------------------------------------------------
def test_set(df, height, width, channels):
    '''
    The function preprocesses the test set for the CNN
    '''
    norm = 255.0    
    test_images = (df.values)/norm
    test_images = test_images.reshape(len(df), height, width, channels)
    print('       test tensor shape:', test_images.shape)
    
    return test_images


# -------------------------------------------------------------------------------
def plot_history(history, ymin=0.9, ymax=1.0):
    '''
    Simple function that plots metrics in the Keras history class.
    '''
    acc = history.history['acc']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc)+1)
    
    plt.figure(figsize=(8,10))
    plt.subplots_adjust(hspace=0.2)
    
    plt.subplot(211)
    plt.plot(epochs, acc,     'bo-', markersize=3, label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', markersize=3, label='Validation acc')
    plt.ylim(ymin, ymax)
    plt.xlim(0, len(acc)+1)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend()
    
    plt.subplot(212)
    plt.plot(epochs, loss,     'bo-', markersize=3, label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', markersize=3, label='Validation loss')
    plt.xlim(0, len(acc)+1)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    
    plt.show()
    
    
    
# -------------------------------------------------------------------------------
def plot_predictions(flag, x, y, y_pred, nrows, ncols, verbose=False):
    '''
    For inspection, this function visualizes correctly or wrongly predicted labels depending on the flag.
    Inputs --
         flag: 0 -> show wrong predictions
               1 -> show correction predictions
            x: target digit images of shape (batch, height, width, channel)
            y: target digit labels, assumes the probability form.
       y_pred: label predictions, assumes the probability form.
        nrows: number of rows in the final plot
        ncols: number of columns in the final plot
      verbose: optional.
    '''
    assert flag <= 1, 'The first argument has to be 0 or 1'
    height = 28
    width = 28
    header = 10
    
    pred = np.argmax(y_pred, axis=1)
    ans = np.argmax(y, axis=1)
    fltr = ( np.array(pred) == np.array(ans) )
    if verbose:
        print(' number of correct predictions:', len(x[fltr]))
        print('   number of wrong predictions:', len(x[~fltr]))
    if flag == 0:
        digits = x[~fltr]
        labels_pred = np.array(pred)[~fltr]
        labels_ans = np.array(ans)[~fltr]
    elif flag == 1:
        digits = x[fltr]
        labels_pred = np.array(pred)[fltr]
        labels_ans = np.array(ans)[fltr]
        
    # Reshape the images for plotting    
    digits = digits.reshape(len(digits), 28, 28)
          
    # Prepare the figure  
    d = np.zeros(((height+header)*nrows, width*ncols))
    fig, ax = plt.subplots(figsize=(14,14), sharex=True, sharey=True)
    ax.axis('off')

    for i in range(nrows):
        for j in range(ncols):
            idx= i*ncols + j            
            d[i*(height+header)+header:(i+1)*(height+header), j*width:(j+1)*width] = digits[idx]
            idx = j*ncols + i
            if flag == 0:
                ax.text(i*width+9,j*(height+header)+9,labels_pred[idx], 
                        ha="center", va="center", color="b", fontsize=18)
                ax.text(i*width+19,j*(height+header)+9,labels_ans[idx], 
                        ha="center", va="center", color="r", fontsize=18)        
            elif flag == 1:
                ax.text(i*width+14,j*(height+header)+9,labels_pred[idx], 
                        ha="center", va="center", color="b", fontsize=18)                
    ax.imshow(d, cmap=matplotlib.cm.binary)
    if flag == 0:
        plt.title('blue: predictions       red: ground truth ', fontsize=20)
        plt.suptitle('Samples of wrong predictions', y=0.94, fontsize=24)
    elif flag == 1:
        plt.suptitle('Samples of correct predictions', y=0.91, fontsize=24)
    plt.show()    

    
    
# -------------------------------------------------------------------------------
def get_wrong_predictions(x, y, y_pred):
    '''
    This functions returns digits that are wrongly predicted.
    Corresponding prediction and ground truth are also returned.
    '''
    pred = np.argmax(y_pred, axis=1)
    ans = np.argmax(y, axis=1)
    fltr = ( np.array(pred) == np.array(ans) )
    print(' number of correct predictions:', len(x[fltr]))
    print('   number of wrong predictions:', len(x[~fltr]))

    digits = x[~fltr]
    labels_pred = np.array(pred)[~fltr]
    labels_ans = np.array(ans)[~fltr]
    
    return digits.reshape(len(digits), 28,28), labels_ans, labels_pred



# -------------------------------------------------------------------------------
def image_enhancement(image):
    '''
    This function demos several image enhancement methods provided by the skimage library.
    '''
    selem = disk(3.5)
    cmap = matplotlib.cm.binary
    p2, p98 = np.percentile(image, (2, 98))
    percent = 0.20

    image_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    image_adapteq = exposure.equalize_adapthist(img_as_ubyte(image), clip_limit=0.02)
    image_eq = exposure.equalize_hist(image)
    image_al = rank.autolevel(image, selem=selem)
    image_alp = rank.autolevel_percentile(image, selem=selem, p0=percent, p1=(1.0-percent))
    image_ec = rank.enhance_contrast(image, disk(2))
    image_ecp = rank.enhance_contrast_percentile(image, disk(2), p0=percent, p1=(1.0-percent))
    image_otsu = rank.otsu(image, disk(1))

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8,8),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    title_list = ['original',  'contrast streching',
                  'auto_level','auto-level {0:3.1f}%'.format(percent*100),
                  'histogram equalizer','adaptive histogram equalizer',
                  'enhance contrast', 'enhance contrast {0:3.1f}%'.format(percent*100),
                  'Otsu threshold']
    image_list = [image, image_rescale,              
                  image_al, image_alp,
                  image_eq, image_adapteq,
                  image_ec, image_ecp,
                  image_otsu]

    for i in range(0, len(image_list)):
        ax[i].imshow(image_list[i], cmap=cmap)
        ax[i].set_title(title_list[i])
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()    
train_size = 30000
height = 28
width = 28
channels = 1

x_train, y_train, x_val, y_val = train_val_split(df_train, train_size, height, width, channels, seed=0)
test_images = test_set(df_test, height, width, channels)
idx = 3722
print('maximum element of the input image array:', np.amax(x_train[idx]))
print('minimum element of the input image array:', np.amin(x_train[idx]))
plot_digit(x_train[idx], y=y_train[idx])
learning_rate = 5e-4

model = models.Sequential()
model.add(layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(28,28,1)))
model.add(layers.MaxPool2D(2))
model.add(layers.Conv2D(64, 3, activation='relu', padding='same'))
model.add(layers.MaxPool2D(2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer = optimizers.RMSprop(lr=learning_rate), 
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.save_weights('mnist_CNN_initial_weights.h5')
history = model.fit(x_train, 
                    y_train,
                    batch_size = 64,
                    epochs = 40,
                    validation_data = (x_val, y_val))

model.save('mnist_CNN_model.h5')

with open('mnist_CNN_history.pickle', 'wb') as history_file:
    pickle.dump(history, history_file)
h = pickle.load( open('mnist_CNN_history.pickle', 'rb'))
plot_history(h, ymin=0.9)
m = models.load_model('mnist_CNN_model.h5')
m.evaluate(x_val, y_val, verbose=1)
y_pred = m.predict(x_val, verbose=1)
wrong, ans, pred = get_wrong_predictions(x_val, y_val, y_pred)
plot_predictions(1, x_val, y_val, y_pred, 10, 10)
plot_predictions(0, x_val, y_val, y_pred, 10, 10)
id = 72
print('\n   prediction:', pred[id])
print(' ground truth:', ans[id], '\n')
image_enhancement(wrong[id])
id = 91
digit = x_val[id]
plt.imshow(digit.reshape(28,28), cmap=matplotlib.cm.binary)
plt.axis('off')
plt.show()
print('shape:', digit.shape)
digit = digit.reshape((1, ) + digit.shape)
digit.shape
img_generator = image.ImageDataGenerator(
                    rotation_range     = 40,
                    width_shift_range  = 0.1,
                    height_shift_range = 0.1,
                    shear_range        = 10,
                    zoom_range         = 0.2,
                    horizontal_flip    = False,
                    vertical_flip      = False )
i = 0
image_list = []
for x in img_generator.flow(digit, batch_size=1):
    image_list.append(x.reshape(28,28))
    i += 1
    if i%9 ==0:
        break
        
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8,8),
                             sharex=True, sharey=True)
ax = axes.ravel()                      
for i in range(0, len(image_list)):
    ax[i].imshow(image_list[i], cmap=plt.cm.binary)
plt.tight_layout()
plt.show()
def PercentileAutoLevel(image):
    '''
    This function demos several image enhancement methods provided by the skimage library.
    '''
    selem = disk(3.5)
    percent = 0.20

    output = []
    count = 0
    for x in image:
        x = rank.autolevel_percentile(x.reshape(28,28), selem=selem, p0=percent, p1=(1.0-percent))/255.0
        count += 1
        output.append(x)
        if count%1000 == 0:
            print('batch {} processed.'.format(count))
    return np.array(output).reshape(image.shape)
x_train_autoleveled = PercentileAutoLevel(x_train)
img_generator.fit(x_train_autoleveled)
train_generator = img_generator.flow(x_train_autoleveled, y_train, batch_size=64)
model.load_weights('mnist_CNN_initial_weights.h5')
steps = len(x_train_autoleveled) // 64
print('  Training steps per epoch:', steps)

history2 = model.fit_generator( train_generator,
                                steps_per_epoch = steps,
                                epochs = 40,
                                validation_data = (x_val, y_val) )

model.save('mnist_CNN_model_2.h5')

with open('mnist_CNN_history_2.pickle', 'wb') as history_file:
    pickle.dump(history2, history_file)
h2 = pickle.load( open('mnist_CNN_history_2.pickle', 'rb'))
plot_history(h2, ymin=0.5)
from keras import backend as K
# Credit: F. Chollet
def layer_filter(model, layer_name, filter_index, size=28):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])    
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random( (1, size, size, 1))*20 + 128.    
    
    step = 1.0
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.1
    img += 0.5
    img = np.clip(img, 0, 1)
    
    return img

def ConvnetFilters(model, layer_name, filter_size, margin, nrows, ncols, figsize=(20,20)):
    results = np.zeros((nrows*filter_size+(nrows-1)*margin, ncols*filter_size+(ncols-1)*margin, 1))
    for i in range(nrows):
        for j in range(ncols):
            filter_img = layer_filter(model, layer_name, i+(j*nrows), size=filter_size)        
            h_start = i*filter_size + i*margin
            h_end = h_start + filter_size
            v_start = j*filter_size + j*margin
            v_end = v_start + filter_size
            results[h_start:h_end, v_start:v_end,:] = filter_img

    plt.figure(figsize=figsize)
    h, w = results.shape[0], results.shape[1]
    plt.title('Filter patterns for layer: {}'.format(layer_name), fontsize=24)
    plt.imshow(results.reshape(h,w))
    plt.show()
m3 = models.load_model('mnist_CNN_model.h5')
m3.summary()
ConvnetFilters(m3, 'conv2d_1', filter_size=28, margin=1, nrows=4, ncols=8, figsize=(16,16))
ConvnetFilters(m3, 'conv2d_2', filter_size=28, margin=1, nrows=8, ncols=8, figsize=(16,16))
learning_rate=5e-4
m5 = models.Sequential()
m5.add(layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(28,28,1)))
m5.add(layers.Conv2D(32, 3, activation='relu', padding='same'))
m5.add(layers.BatchNormalization())
m5.add(layers.MaxPool2D(2))
m5.add(layers.Conv2D(64, 3, activation='relu', padding='same'))
m5.add(layers.Conv2D(64, 3, activation='relu', padding='same'))
m5.add(layers.BatchNormalization())
m5.add(layers.MaxPool2D(2))
m5.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
m5.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
m5.add(layers.BatchNormalization())
m5.add(layers.Flatten())
m5.add(layers.Dropout(0.5))
m5.add(layers.Dense(128, activation='relu'))
m5.add(layers.Dense(32, activation='relu'))
m5.add(layers.BatchNormalization())
m5.add(layers.Dense(10, activation='softmax'))

m5.compile(optimizer = optimizers.RMSprop(lr=learning_rate), 
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
m5.summary()
h5 = m5.fit_generator( train_generator,
                  steps_per_epoch = steps,
                  epochs = 40,
                  validation_data = (x_val, y_val),
                  verbose = 1 )

m5.save('mnist_CNN_model_3.h5')

with open('mnist_CNN_history_3.pickle', 'wb') as history_file:
    pickle.dump(h5, history_file)
h5 = pickle.load( open('mnist_CNN_history_3.pickle', 'rb'))
plot_history(h5, ymin=0.85)
ConvnetFilters(m5, 'conv2d_3', filter_size=28, margin=1, nrows=4, ncols=8, figsize=(16,16))
ConvnetFilters(m5, 'conv2d_6', filter_size=28, margin=1, nrows=8, ncols=8, figsize=(16,16))
ConvnetFilters(m5, 'conv2d_8', filter_size=28, margin=1, nrows=16, ncols=8, figsize=(32,32))
