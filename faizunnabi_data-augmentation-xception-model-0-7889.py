import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.applications.xception import preprocess_input
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.callbacks import ReduceLROnPlateau
import datetime as dt

%matplotlib inline
whale_labels= pd.read_csv('../input/train.csv')
whale_labels.head()
top_populars = whale_labels.groupby('Id').count().sort_values(by='Image',ascending=False).reset_index().head(10)
top_populars
sns.set_style('whitegrid')
plt.figure(figsize=(16,9))
sns.barplot(x='Id',y='Image',data=top_populars)
top_populars.describe()
top_categories = list(top_populars['Id'].values)
df = whale_labels[whale_labels["Id"].isin(top_categories)]
cn = df.groupby('Id').count().reset_index()
cn
df.head()
width = 150
height = 150
channels = 3
batch_size = 16

base_model = Xception(include_top=False,input_shape=(width,height,3))
base_model.summary()
datagene = ImageDataGenerator(rotation_range=20,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              zoom_range=0.2,
                              rescale=1./255,
                              horizontal_flip=True)

lbe = LabelEncoder()
yl = df['Id'].values

yl = lbe.fit_transform(yl)

onhe = OneHotEncoder()
yl = onhe.fit_transform(yl.reshape(-1,1))
img_array = np.zeros(shape=(1028,width,width,3))
label_array = yl.toarray()
i=0
for index,row in df.iterrows():
    img = cv2.imread('../input/train/'+row["Image"],0)
    img = cv2.resize(img,(150,150))
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = img_to_array(img)
    img_array[i] = img
train_f = datagene.flow(img_array,label_array,batch_size=batch_size)
def extract_features(sample_count, datagen):
    start = dt.datetime.now()
    features = np.zeros(shape=(sample_count, 5, 5, 2048))
    labels = np.zeros(shape=(sample_count,10))
    generator = datagen
    i = 0
    for inputs_batch,labels_batch in generator:
        stop = dt.datetime.now()
        time = (stop - start).seconds
        print('\r',
        'Extracting features from batch', str(i+1), '/', len(datagen),
        '-- run time:', time,'seconds',
        end='')
        features_batch = base_model.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    print("\n")
    return features,labels
train_features, train_labels = extract_features(1028, train_f)
flat_dim = 5 * 5 * 2048

train_features = np.reshape(train_features, (1028, flat_dim))
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

callbacks = [reduce_learning_rate]
model = Sequential()

model.add(Dense(512, activation='relu', input_dim=flat_dim))
model.add(Dropout(0.5))
model.add(Dense(units=10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(train_features,
                    train_labels,
                    epochs=30,
                    batch_size=batch_size,
                    shuffle=True,
                    callbacks=callbacks)
acc = history.history['acc']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
f,(ax1,ax2)=plt.subplots(1,2,figsize=(16,6))
ax1.set_title('Training accuracy')
ax1.plot(epochs, acc, 'red', label='Training acc')
ax2.set_title('Training loss')
ax2.plot(epochs, loss, 'blue', label='Training loss')
plt.show()
