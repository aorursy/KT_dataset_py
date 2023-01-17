base_dir = '/menwomen-classification'

train_dir = os.path.join(base_dir, 'traindata/traindata')
test_dir = os.path.join(base_dir, 'testdata/testdata')


train_men_dir = os.path.join(train_dir, 'men')
train_women_dir = os.path.join(train_dir, 'women')


test_men_dir = os.path.join(test_dir, 'men')
test_women_dir = os.path.join(test_dir, 'women')
print('Training (men) :', len(os.listdir(train_men_dir ) ))
print('Training (women) :', len(os.listdir(train_women_dir ) ))

print('Testing (men) :', len(os.listdir(test_men_dir ) ))
print('Testing (women) :', len(os.listdir(test_women_dir ) ))
%matplotlib inline

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#img_path=os.path.join(train_men_dir, '00000493.jpg')
img_path=os.path.join(train_women_dir, '00000023.jpg')
img = mpimg.imread(img_path)
plt.imshow(img)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    target_size=(150, 150))     

test_generator =  test_datagen.flow_from_directory(test_dir,
                                                         batch_size=32,
                                                         class_mode  = 'binary',
                                                         target_size=(150, 150))
   


import tensorflow as tf

model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')        
])
model.summary()
model.compile(optimizer='adam',
              loss='BinaryCrossentropy',
              metrics = ['accuracy'])
history = model.fit(train_generator,
                              validation_data=test_generator,
                              steps_per_epoch=100,
                              epochs=10,
                              validation_steps=50,
                              verbose=2)
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from keras.preprocessing import image

uploaded=files.upload()

for fn in uploaded.keys():
  path='/content/' + fn       #Save the image to content folder
  img=image.load_img(path, target_size=(150, 150))    #load the image
  
  x=image.img_to_array(img)    
  x=np.expand_dims(x, axis=0)   
  images = np.vstack([x])
  
  classes = model.predict(images, batch_size=32)  #predict the label for the image
  plt.imshow(img)
  print(classes[0])     #Print the label, remember it will be either one or zero
  
  if classes[0]>0:
    print(fn + " is a men")     #print human readable label
    
  else:
    print(fn + " is a women") 