# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
tf.test.gpu_device_name()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
import cv2
import os
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3
model = InceptionV3()
model.summary()
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import decode_predictions
N = 224
chair = []

for filepath in tqdm(sorted(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/chair/'))):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/chair/{0}'.format(filepath),1)
    src = cv2.resize(src, (N, N))
    chair.append(src)

chair = np.array(chair)

bed = []

for filepath in tqdm(sorted(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/bed/'))):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/table/{0}'.format(filepath),1)
    src = cv2.resize(src, (N, N))
    bed.append(src)

bed = np.array(bed)

table = []

for filepath in tqdm(sorted(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/table/'))):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/table/{0}'.format(filepath),1)
    src = cv2.resize(src, (N, N))
    table.append(src)

table = np.array(table)

sofa = []

for filepath in tqdm(sorted(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/sofa/'))):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/sofa/{0}'.format(filepath),1)
    src = cv2.resize(src, (N, N))
    sofa.append(src)

sofa = np.array(sofa)

swivle_chair = []

for filepath in tqdm(sorted(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/swivelchair/'))):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/swivelchair/{0}'.format(filepath),1)
    src = cv2.resize(src, (N, N))
    swivle_chair.append(src)

swivle_chair = np.array(swivle_chair)


print(chair.shape, table.shape, bed.shape, swivle_chair.shape, sofa.shape)
image = chair[0]

plt.imshow(image)

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

image = preprocess_input(image)

model = InceptionV3()

yhat = model.predict(image)

label = decode_predictions(yhat)

label = label[0][0]
print('%s (%.5f%%)' % (label[1], label[2]*100))
print(yhat.shape, '\n', yhat)
def label_image(images):

    images = preprocess_input(images)

    model = InceptionV3()

    yhat = model.predict(images, verbose=1)
    
    return yhat
    
yhat_chair = label_image(chair)
yhat_bed = label_image(bed)
yhat_sofa = label_image(sofa)
yhat_swivel_chair = label_image(swivle_chair)
yhat_table = label_image(table)
x_inception = np.array([i for i in yhat_chair] + [i for i in yhat_swivel_chair] + [i for i in yhat_table] + [i for i in yhat_bed] + [i for i in yhat_sofa])
y_inception = np.array([1 for _ in range(yhat_chair.shape[0])] + [3 for _ in range(yhat_swivel_chair.shape[0])] + [4 for _ in range(yhat_table.shape[0])] + [0 for _ in range(yhat_bed.shape[0])] + [2 for _ in range(yhat_sofa.shape[0])])

np.random.seed(0)
indexes = np.arange(x_inception.shape[0])
np.random.shuffle(indexes)

x_inception = x_inception[indexes]
y_inception = y_inception[indexes]

print(x_inception.shape)
print(y_inception.shape)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0, n_estimators=500) #for reproducibility
clf.fit(x_inception[:int(0.85*x_inception.shape[0])], y_inception[:int(0.85*x_inception.shape[0])])
y_rf = clf.predict(x_inception)
print('Validation',clf.score(x_inception[int(0.85*x_inception.shape[0]):], y_inception[int(0.85*x_inception.shape[0]):]))
print('Training',clf.score(x_inception[:int(0.85*x_inception.shape[0])], y_inception[:int(0.85*x_inception.shape[0])]))
groups = {0:[], 1:[], 2:[], 3:[], 4:[]}
for i,j in zip(y_inception, y_rf):
    groups[i].append(j)
    
p_error = {0:[], 1:[], 2:[], 3:[], 4:[]}
for key, item in groups.items():
    for i in range(5):
        p_error[key].append(item.count(i))
    total = sum(p_error[key])
    p_error[key] = [round(i/total, 5) for i in p_error[key]]

_ = [print(key,':',item) for key, item in p_error.items()]
import networkx as nx 

def error_graph(p_error):
    
    G = nx.DiGraph() 
    for i in range(5):
        G.add_node(i)
    
    for key, item in p_error.items():
        for i,x in enumerate(item):
            x = round(x, 3)
            if i!=key and x>0.01:
                G.add_edge(key, i, weight=x)
    
    plt.figure(figsize =(4, 4)) 
    nx.draw_networkx(G, with_label = True, node_color ='green')
    
error_graph(p_error)
x = np.array([i for i in chair[:int(0.9*chair.shape[0])]] + [i for i in table[:int(0.9*table.shape[0])]] + [i for i in bed[:int(0.9*bed.shape[0])]] + [i for i in swivle_chair[:int(0.9*swivle_chair.shape[0])]] + [i for i in sofa[:int(0.9*sofa.shape[0])]] + [i for i in chair[int(0.9*chair.shape[0]):]] + [i for i in table[int(0.9*table.shape[0]):]] + [i for i in bed[int(0.9*bed.shape[0]):]] + [i for i in swivle_chair[int(0.9*swivle_chair.shape[0]):]] + [i for i in sofa[int(0.9*sofa.shape[0]):]])
y_sparse = np.array([1 for _ in range(int(0.9*chair.shape[0]))] + [4 for _ in range(int(0.9*table.shape[0]))] + [0 for _ in range(int(0.9*bed.shape[0]))] + [3 for _ in range(int(0.9*swivle_chair.shape[0]))] + [2 for _ in range(int(0.9*sofa.shape[0]))] + [1 for _ in range(chair.shape[0]-int(0.9*chair.shape[0]))] + [4 for _ in range(table.shape[0]-int(0.9*table.shape[0]))] + [0 for _ in range(bed.shape[0]-int(0.9*bed.shape[0]))] + [3 for _ in range(swivle_chair.shape[0] - int(0.9*swivle_chair.shape[0]))] + [2 for _ in range(sofa.shape[0] - int(0.9*sofa.shape[0]))])
y_encoded = []

for i in y_sparse:
    a = [0 for _ in range(5)]
    a[i] = 1
    y_encoded.append(a)
y_encoded = np.array(y_encoded)

print(x.shape, y_encoded.shape)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, Concatenate, Dropout
import tensorflow as tf
model = InceptionV3(include_top=False, input_shape=(N, N, 3))
model.trainable = True

gap1 = GlobalAveragePooling2D()(model.layers[-1].output)
flat1 = Flatten()(gap1)
bn = BatchNormalization()(flat1)
class1 = Dense(1024, activation='relu')(bn)
class2 = Dense(256, activation='relu')(class1)
class3 = Dense(32, activation='relu')(class2)
output = Dense(5, activation='softmax')(class3)

model = Model(inputs=model.inputs, outputs=output)

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.75, patience=4, verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=5*1e-5,
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc', min_delta=0, patience=15, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)
model.fit(x[:int(0.9*x.shape[0])], y_encoded[:int(0.9*x.shape[0])], epochs=25, shuffle=True, callbacks=[reduce_lr, early_stop], validation_data=(x[x.shape[0]-int(0.9*x.shape[0]):], y_encoded[x.shape[0]-int(0.9*x.shape[0]):]))
test = []
files = sorted(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/test/'))
for filepath in tqdm(files):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/test/{0}'.format(filepath),1)
    src = cv2.resize(src, (N, N))
    test.append(src)

test = np.array(test)
y_pred = model.predict(test, verbose=1)
y_pred = [np.argmax(i) for i in y_pred]
submission = pd.DataFrame({'image': files, 'target':y_pred})
submission.to_csv('submission_basic.csv', index=False)
model1 = InceptionV3(include_top=False, input_shape=(N, N, 3))
model1.trainable = True

gap1 = GlobalAveragePooling2D()(model1.layers[-1].output)
flat1 = Flatten()(gap1)
flat1 = BatchNormalization()(flat1)

out1 = Dense(1024, activation='relu')(flat1)

input2 = tf.keras.layers.Input([N, N, 3])
x = tf.keras.applications.inception_v3.preprocess_input(input2)
model2 = tf.keras.applications.InceptionV3()
out2 = model2(x)
out2 = BatchNormalization()(out2)

mergedOut = Concatenate()([out1,out2])
class1 = Dense(1228, activation='relu')(mergedOut)
class1 = Dense(1024, activation='relu')(class1)
class1 = Dense(516, activation='relu')(class1)
class2 = Dense(256, activation='relu')(class1)
class3 = Dense(32, activation='relu')(class2)
output = Dense(5, activation='softmax')(class3)


model = Model(inputs=[model1.inputs, input2], outputs=[output])
model.summary()
x = np.array([i for i in chair[:int(0.9*chair.shape[0])]] + [i for i in table[:int(0.9*table.shape[0])]] + [i for i in bed[:int(0.9*bed.shape[0])]] + [i for i in swivle_chair[:int(0.9*swivle_chair.shape[0])]] + [i for i in sofa[:int(0.9*sofa.shape[0])]] + [i for i in chair[int(0.9*chair.shape[0]):]] + [i for i in table[int(0.9*table.shape[0]):]] + [i for i in bed[int(0.9*bed.shape[0]):]] + [i for i in swivle_chair[int(0.9*swivle_chair.shape[0]):]] + [i for i in sofa[int(0.9*sofa.shape[0]):]])
y_sparse = np.array([1 for _ in range(int(0.9*chair.shape[0]))] + [4 for _ in range(int(0.9*table.shape[0]))] + [0 for _ in range(int(0.9*bed.shape[0]))] + [3 for _ in range(int(0.9*swivle_chair.shape[0]))] + [2 for _ in range(int(0.9*sofa.shape[0]))] + [1 for _ in range(chair.shape[0]-int(0.9*chair.shape[0]))] + [4 for _ in range(table.shape[0]-int(0.9*table.shape[0]))] + [0 for _ in range(bed.shape[0]-int(0.9*bed.shape[0]))] + [3 for _ in range(swivle_chair.shape[0] - int(0.9*swivle_chair.shape[0]))] + [2 for _ in range(sofa.shape[0] - int(0.9*sofa.shape[0]))])
y_encoded = []

for i in y_sparse:
    a = [0 for _ in range(5)]
    a[i] = 1
    y_encoded.append(a)
y_encoded = np.array(y_encoded)
print(x.shape, y_encoded.shape)
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0025), loss='categorical_crossentropy', metrics=['acc'])
model.fit([x, x], y_encoded, epochs=5, validation_split=0.1, shuffle=True)
test = []
files = sorted(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/test/'))
for filepath in tqdm(files):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/test/{0}'.format(filepath),1)
    src = cv2.resize(src, (N, N))
    test.append(src)

test = np.array(test)
y_pred = model.predict([test, test], verbose=1)
y_peed = [np.argmax(i) for i in y_pred]
submission = pd.DataFrame({'image': files, 'target':y_peed})
submission.to_csv('submission_skipped.csv', index=False)
for index in range(10):
    
    np.random.shuffle(chair)
    np.random.shuffle(table)
    np.random.shuffle(bed)
    np.random.shuffle(sofa)
    np.random.shuffle(swivle_chair)

    x = np.array([i for i in chair[:int(0.9*chair.shape[0])]] + [i for i in table[:int(0.9*table.shape[0])]] + [i for i in bed[:int(0.9*bed.shape[0])]] + [i for i in swivle_chair[:int(0.9*swivle_chair.shape[0])]] + [i for i in sofa[:int(0.9*sofa.shape[0])]] + [i for i in chair[int(0.9*chair.shape[0]):]] + [i for i in table[int(0.9*table.shape[0]):]] + [i for i in bed[int(0.9*bed.shape[0]):]] + [i for i in swivle_chair[int(0.9*swivle_chair.shape[0]):]] + [i for i in sofa[int(0.9*sofa.shape[0]):]])
    y_sparse = np.array([1 for _ in range(int(0.9*chair.shape[0]))] + [4 for _ in range(int(0.9*table.shape[0]))] + [0 for _ in range(int(0.9*bed.shape[0]))] + [3 for _ in range(int(0.9*swivle_chair.shape[0]))] + [2 for _ in range(int(0.9*sofa.shape[0]))] + [1 for _ in range(chair.shape[0]-int(0.9*chair.shape[0]))] + [4 for _ in range(table.shape[0]-int(0.9*table.shape[0]))] + [0 for _ in range(bed.shape[0]-int(0.9*bed.shape[0]))] + [3 for _ in range(swivle_chair.shape[0] - int(0.9*swivle_chair.shape[0]))] + [2 for _ in range(sofa.shape[0] - int(0.9*sofa.shape[0]))])
    y_encoded = []

    for i in y_sparse:
        a = [0 for _ in range(5)]
        a[i] = 1
        y_encoded.append(a)
    y_encoded = np.array(y_encoded)
    
    model = InceptionV3(include_top=False, input_shape=(N, N, 3))
    model.trainable = True

    gap1 = GlobalAveragePooling2D()(model.layers[-1].output)
    flat1 = Flatten()(gap1)
    bn = BatchNormalization()(flat1)
    class1 = Dense(1024, activation='relu')(bn)
    class2 = Dense(256, activation='relu')(class1)
    class3 = Dense(32, activation='relu')(class2)
    output = Dense(5, activation='softmax')(class3)

    model = Model(inputs=model.inputs, outputs=output)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
    model.fit(x[:int(0.9*x.shape[0])], y_encoded[:int(0.9*x.shape[0])], epochs=20, shuffle=True, callbacks=[reduce_lr, early_stop], validation_data=(x[x.shape[0]-int(0.9*x.shape[0]):], y_encoded[x.shape[0]-int(0.9*x.shape[0]):]))
    
    y_pred = model.predict(test, verbose=1)
    y_pred = [np.argmax(i) for i in y_pred]
    submission = pd.DataFrame({'image': files, 'target':y_pred})
    submission.to_csv('submission_basic_'+str(index)+'.csv', index=False)
    
    del model    
