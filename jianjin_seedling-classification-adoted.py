import glob
from PIL import Image,ImageOps
import numpy as np
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
!dir
print("--------------------------------------------------------------")
!dir 'D:\MyPython\SeedlingClassification\'
fns = glob.glob(r'train/*/*.*')
fns[0:5]
labels = []
fitted_imgs=[]
for fn in fns:
    if fn[-3:]!='png':
        continue
    labels.append(fn.split('\\')[-2])
    fitted_imgs.append(
        ImageOps.fit(Image.open(fn),(48,48),Image.ANTIALIAS).convert('RGB'))
imgs_array = np.array([np.array(fitted_img) 
                       for fitted_img in fitted_imgs])/255
lb = LabelBinarizer().fit(labels)
lb_array = lb.transform(labels)
trainX, validX, trainY, validY = train_test_split(imgs_array,lb_array
                                                 ,test_size = 0.05
                                                  ,random_state=42)
from keras.layers import Dropout, Input, Dense,Activation,GlobalMaxPooling2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
IM_input = Input((48, 48, 3))
IM = Conv2D(16, (3, 3))(IM_input)
IM = BatchNormalization(axis = 3)(IM)
IM = Activation('relu')(IM)
IM = Conv2D(16, (3, 3))(IM)
IM = BatchNormalization(axis = 3)(IM)
IM = Activation('relu')(IM)
IM = MaxPooling2D((2, 2), strides=(2, 2))(IM)
IM = Conv2D(32, (3, 3))(IM)
IM = BatchNormalization(axis = 3)(IM)
IM = Activation('relu')(IM)
IM = Conv2D(32, (3, 3))(IM)
IM = BatchNormalization(axis = 3)(IM)
IM = Activation('relu')(IM)
IM = GlobalMaxPooling2D()(IM)

IM = Dense(64, activation='relu')(IM)
IM = Dropout(0.5)(IM)
IM = Dense(32, activation='relu')(IM)
IM = Dropout(0.5)(IM)
IM = Dense(12, activation='softmax')(IM)
model = Model(inputs=IM_input, outputs=IM)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4), metrics=['acc'])
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint
batch_size = 64
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
earlystop = EarlyStopping(patience=10)
modelsave = ModelCheckpoint(
    filepath='model.h5', save_best_only=True, verbose=1)
model.fit(
    trainX, trainY, batch_size=batch_size,
    epochs=200, 
    validation_data=(validX, validY),
    callbacks=[annealer, earlystop, modelsave]
)
z = glob.glob('test//*.png')

test_imgs = []
names = []
for fn in z:
    if fn[-3:] != 'png':
        continue
    names.append(fn.split('\\')[-1])
    new_img = Image.open(fn)
    test_img = ImageOps.fit(new_img, (48, 48), Image.ANTIALIAS).convert('RGB')
    test_imgs.append(test_img)
model = load_model('model.h5')
timgs = np.array([np.array(im) for im in test_imgs])
testX = timgs.reshape(timgs.shape[0], 48, 48, 3) / 255
yhat = model.predict(testX)
test_y = lb.inverse_transform(yhat)
import pandas as pd
df = pd.DataFrame(data={'file': names, 'species': test_y})
df_sort = df.sort_values(by=['file'])
df_sort.to_csv('results.csv', index=False)
fns = glob.glob(r'train/*/*.*')
fns[0:5]
for fn in fns:
    print(fn.split('\\')[-2])
labels = []
labels = [fn.split('\\')[-2] for fn in fns]
list(labels)
lb = LabelBinarizer()
lb.fit_transform(labels)