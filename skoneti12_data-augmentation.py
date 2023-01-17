#import the important liibraries
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array
#create an object 
datagen=ImageDataGenerator(width_shift_range=0.2,vertical_flip=True,height_shift_range=0.2,shear_range=0.4,rotation_range=40)
#load the image 
from keras.preprocessing.image import load_img
dimage=load_img(r'../input/cat-image/download.jpg')
#convert the image into array
x=img_to_array(dimage)
x.shape
#the flow function takes a 4d numpy array so we reshape it using reshape function
x=x.reshape((1,)+x.shape)

#now start the process
i=0
for batch in datagen.flow(x,batch_size=1,save_prefix='cat',save_to_dir='./',save_format='jpeg'):
  i=i+1
  if i>20:
    break

