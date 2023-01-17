import zipfile
import os
import pandas as pd

zip_ref = zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r')
zip_ref.extractall()
zip_ref.close()

labels = []
img_path = []


for img in os.listdir('train/'):
    img_path.append(os.path.join('train/',img))
    
    if img.startswith("cat"):
        labels.append("cat")
    
    elif img.startswith("dog"):
        labels.append("dog")
df = pd.DataFrame({
    'image' : img_path,
    'class' : labels
})
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop,Adam
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
train_generator = train_datagen.flow_from_dataframe(dataframe = df,
                                                   x_col = 'image',
                                                   y_col = 'class',
                                                   batch_size = 20,
                                                   class_mode = 'binary',
                                                   target_size=(150, 150),
                                                   shuffle = True)
asd = pd.DataFrame( {   
    'image' : img_path,
    'label' : train_generator.classes})
asd.head()
model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')  
    
])
model.summary()
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy'])
model.fit_generator(train_generator,
                   steps_per_epoch=100,
                   epochs=15,
                   verbose=2)
zip_ref = zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r')
zip_ref.extractall()
zip_ref.close()

img_pa = []

for img in os.listdir('test1/'):
    img_pa.append(os.path.join('test1/', img))
    
df_test = pd.DataFrame({'image_path':img_pa})
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_data_generator =  test_datagen.flow_from_dataframe(dataframe=df_test,
                                                       x_col='image_path',
                                                       y_col=None,
                                                       batch_size=20,
                                                       target_size=(150,150),
                                                       class_mode=None)
pred = model.predict(test_data_generator,verbose=1)
prediction = 1*(pred >0.5)
prediction
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
fig = plt.figure(figsize = (20,20))
fig.suptitle("predction", fontsize = 16)


for i, img in enumerate(df_test.image_path[:10]):
    plt.subplot(5, 5, i + 1)
    img = mpimg.imread(img)
    plt.imshow(img)
    if prediction[i] == 1:
        a = "dog"
    else:
        a = "cat"
    plt.title(a)
    plt.xticks([])
    plt.yticks([])
    