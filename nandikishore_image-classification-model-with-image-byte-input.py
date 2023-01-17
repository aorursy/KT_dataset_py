!ls '/kaggle/input/apparel-images-dataset'
import os
print(len(os.listdir('/kaggle/input/apparel-images-dataset')))

data_dir = '/kaggle/input/apparel-images-dataset'
black_shirt_dir = os.path.join(data_dir,'black_shirt')
imglist  = os.listdir(black_shirt_dir)

import matplotlib.pyplot as plt
import matplotlib.image  as img

j = 0
for i in imglist[0:16]:
   j = j + 1 
   sp = plt.subplot(4,4,j)    
   image = img.imread(os.path.join(black_shirt_dir,i))
   plt.imshow(image)
    
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.show()
import tensorflow as tf
import numpy      as np

image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1.00/255.0,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,vertical_flip=True,fill_mode='nearest',validation_split=0.3)
tf.__version__
image_gen  = image_datagen.flow_from_directory(data_dir,target_size=(200,200),class_mode='categorical',color_mode='rgb',batch_size = 32,shuffle=True)
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),input_shape=(200,200,3),activation='relu'),tf.keras.layers.MaxPooling2D(2,2),tf.keras.layers.Conv2D(32,(3,3),activation='relu'),tf.keras.layers.MaxPooling2D(2,2),tf.keras.layers.Conv2D(64,(3,3),activation='relu'),tf.keras.layers.MaxPooling2D(2,2),tf.keras.layers.Flatten(),tf.keras.layers.Dense(512,activation='relu'),tf.keras.layers.Dropout(0.2),tf.keras.layers.Dense(24,activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),loss=tf.keras.losses.categorical_crossentropy,metrics=['acc'])
model.summary()
class accuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('acc')> 0.95):
            print("Max accuracy reached stopping Training process")
            self.model.stop_training = True
accu = accuracy()

model.fit(image_gen,shuffle=True,verbose=1,callbacks=[accu],steps_per_epoch=20,epochs=1000,validation_steps=10)
model.save('Multiclass')
!saved_model_cli show --dir /kaggle/working/Multiclass --all
model.inputs

model.outputs
image = img.imread('/kaggle/input/blue-shirt/8783505_fpx.jpeg')
plt.imshow(image)
plt.show()
classes = os.listdir('/kaggle/input/apparel-images-dataset')
classes.sort()
classes
img     = tf.keras.preprocessing.image.load_img('/kaggle/input/blue-shirt/8783505_fpx.jpeg',target_size=(200,200))
img_pix = tf.keras.preprocessing.image.img_to_array(img)
img_pix = img_pix/255
img_pix = np.expand_dims(img_pix,axis=0)
img_pix = np.vstack([img_pix])

op      = model.predict(img_pix)

op
op_class = model.predict_classes(img_pix)
op_class
print([cls for i,cls in enumerate(classes) if i == op_class[0]])
class ConvertModel(tf.keras.Model):
    def __init__(self,model):
        super().__init__(self)
        self.model = model
        
    @tf.function(input_signature=[tf.TensorSpec([None],dtype=tf.string)])    
    def img_serve(self,images):
        def input_to_actual_shape(img):
            img = tf.io.decode_jpeg(img,channels=3)
            img = tf.image.convert_image_dtype(img,tf.float32)
            img = tf.image.resize_with_pad(img,200,200)
            return img
        
        img = tf.map_fn(input_to_actual_shape,images,dtype=tf.float32)
        op  = self.model(img)
        return {"OUTPUT":op}
serving_model = ConvertModel(model)
tf.saved_model.save(serving_model,'/kaggle/working/CustomMulticlass',signatures={'serving_default':serving_model.img_serve})
!saved_model_cli show --dir /kaggle/working/CustomMulticlass/ --all
reload_model = tf.saved_model.load('/kaggle/working/CustomMulticlass')
infer = reload_model.signatures["serving_default"]
import io
from PIL import Image
img = Image.open('/kaggle/input/blue-shirt/8783505_fpx.jpeg',mode='r')
img = img.resize((200,200))
img_byte = io.BytesIO()
img.save(img_byte,format='JPEG')
img_byte = img_byte.getvalue()
img_byte
op = infer(tf.constant([img_byte]))
op
op_list = np.array(op['OUTPUT']).tolist()[0]
Class_index = op_list.index(max(op_list))
print(Class_index)
classes[Class_index]