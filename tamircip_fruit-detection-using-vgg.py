#%% Libraries import
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img #preprocessing for image
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16 #for transfer learning
import matplotlib.pyplot as plt
from glob import glob
train_path = "../input/fruits/fruits-360/Training"
test_path = "/kaggle/input/fruits/fruits-360/Test"

img = load_img('../input/fruits/fruits-360/Training/Avocado/0_100.jpg')
plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)
print(x.shape)


numberOfClass = len(glob('../input/fruits/fruits-360/Training'+'/*'))
vgg = VGG16()

print(vgg.summary())
print(type(vgg))

vgg_layer_list = vgg.layers
model = Sequential() 
for i in range(len(vgg_layer_list)-1): 
    model.add(vgg_layer_list[i]) 
print(model.summary())
# Transfer Learning
for layers in model.layers:
    layers.trainable = False
    
model.add(Dense(numberOfClass, activation="softmax")) 

print(model.summary())
model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])


train_data = ImageDataGenerator().flow_from_directory(train_path,target_size = (224,224)) 
test_data = ImageDataGenerator().flow_from_directory(test_path,target_size = (224,224)) 

batch_size = 32

hist = model.fit_generator(train_data,
                           steps_per_epoch=1600//batch_size,
                           epochs= 25,
                           validation_data=test_data,
                           validation_steps= 800//batch_size,)
model.save_weights("./weights.h5")
plt.title('Loss Scores')
print(hist.history.keys())
plt.plot(hist.history["loss"],label = "training loss")
plt.plot(hist.history["val_loss"],label = "validation loss")
plt.legend()
plt.show()
plt.figure()
plt.title('Accuracy Scores')
plt.plot(hist.history["accuracy"],label = "training acc")
plt.plot(hist.history["val_accuracy"],label = "validation acc")
plt.legend()
plt.show()

model = Sequential() 
for i in range(len(vgg_layer_list)-1): 
    model.add(vgg_layer_list[i]) 

for layers in model.layers:
    layers.trainable = False
    
model.add(Dense(numberOfClass, activation="softmax")) 

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])


train_data = ImageDataGenerator(
                            featurewise_center=False, #set input mean to 0
                           samplewise_center=False,  #set each sample mean to 0
                           featurewise_std_normalization=False, #divide input datas to std
                           samplewise_std_normalization=False,  #divide each datas to own std
                           zca_whitening=False,  #dimension reduction
                           rotation_range=0.5,    #rotate 5 degree
                           zoom_range=0.5,        #zoom in-out 5%
                           width_shift_range=0.5, #shift 5%
                           height_shift_range=0.5,
                           horizontal_flip=False,  #randomly flip images
                           vertical_flip=False
                   )
train_data = train_data.flow_from_directory(train_path,target_size = (224,224))
test_data = ImageDataGenerator(
                            featurewise_center=False, #set input mean to 0
                           samplewise_center=False,  #set each sample mean to 0
                           featurewise_std_normalization=False, #divide input datas to std
                           samplewise_std_normalization=False,  #divide each datas to own std
                           zca_whitening=False,  #dimension reduction
                           rotation_range=0.5,    #rotate 5 degree
                           zoom_range=0.5,        #zoom in-out 5%
                           width_shift_range=0.5, #shift 5%
                           height_shift_range=0.5,
                           horizontal_flip=False,  #randomly flip images
                           vertical_flip=False
)

test_data = test_data.flow_from_directory(test_path,target_size = (224,224)) 

batch_size = 32

hist = model.fit_generator(train_data,
                           steps_per_epoch=1600//batch_size,
                           epochs= 25,
                           validation_data=test_data,
                           validation_steps= 800//batch_size,)