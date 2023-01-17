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
url = '../input/thai-mnist-classification/'
os.listdir(url)
import pandas as pd
import numpy as np
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras import Model, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
# from skimage.io import imread, imsave
# from skimage.filters import threshold_otsu
# from skimage.color import rgb2gray
# from skimage.util import invert, img_as_ubyte, img_as_bool
# from skimage.morphology import skeletonize, binary_closing
# def get_binary(img):    
#     thresh = threshold_otsu(img)
#     binary = img > thresh
#     return binary
 
# def to_skeleton(fn):    
#     im = img_as_bool(get_binary(invert(imread(fn))))
#     out = binary_closing(skeletonize(im))
#     imsave(fn, img_as_ubyte(out))
 
# to_skeleton('file.png')
np.random.seed(1)
df = pd.read_csv('/kaggle/input/thai-mnist-classification/mnist.train.map.csv')
df['path'] = '/kaggle/input/thai-mnist-classification/train/' + df['id']
df['category'] = df['category'].astype('str')
df['rand'] = np.random.rand(len(df))
train_df = df[df['rand'] <= 0.8]
val_df = df[df['rand'] > 0.8]

len(train_df), len(val_df)
base = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

for l in base.layers:
    l.trainable = False
    
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dense(10, activation='softmax')(x)
model = Model(base.input, x)
model.summary()
datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
train_gen = datagen.flow_from_dataframe(train_df, x_col='path', y_col='category', batch_size=64)
val_gen = datagen.flow_from_dataframe(val_df, x_col='path', y_col='category', shuffle=False, batch_size=64)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
        validation_data=val_gen,
        steps_per_epoch=train_gen.n // train_gen.batch_size,
        epochs=30,
        verbose=True)
def plotAccHistory(history):
    history_df = pd.DataFrame(history.history)
#     history_df = history_df.iloc[::5]
    sns.pointplot(data=history_df, y='accuracy', x=history_df.index, color='red', label='train')
    sns.pointplot(data=history_df, y='val_accuracy', x=history_df.index, color='blue', label='val')
    plt.legend()
    plt.show()
plotAccHistory(history)
rule_df = pd.read_csv('/kaggle/input/thai-mnist-classification/train.rules.csv')
rule_df.head()
def extractFeature(df, dataset='train'):
    f1_df = df[['feature1']].dropna()
    f2_df = df[['feature2']].dropna()
    f3_df = df[['feature3']].dropna()

    f1_df['path'] = '/kaggle/input/thai-mnist-classification/' + dataset + '/' + f1_df['feature1']
    f2_df['path'] = '/kaggle/input/thai-mnist-classification/' + dataset + '/' + f2_df['feature2']
    f3_df['path'] = '/kaggle/input/thai-mnist-classification/' + dataset + '/' + f3_df['feature3']

    # mock for datagen
    f1_df['mock_label'] = (np.round(np.random.rand(len(f1_df)) * 10) % 10).astype('int').astype('str')
    f2_df['mock_label'] = (np.round(np.random.rand(len(f2_df)) * 10) % 10).astype('int').astype('str')
    f3_df['mock_label'] = (np.round(np.random.rand(len(f3_df)) * 10) % 10).astype('int').astype('str')

    return f1_df, f2_df, f3_d
rule_f1, rule_f2, rule_f3 = extractFeature(rule_df, 'train')
rule_f1_gen = datagen.flow_from_dataframe(rule_f1, x_col='path', y_col='mock_label', batch_size=256, shuffle=False)
rule_f2_gen = datagen.flow_from_dataframe(rule_f2, x_col='path', y_col='mock_label', batch_size=256, shuffle=False)
rule_f3_gen = datagen.flow_from_dataframe(rule_f3, x_col='path', y_col='mock_label', batch_size=256, shuffle=False)
rule_pred_f1 = model.predict_generator(rule_f1_gen, verbose=True)
rule_pred_f2 = model.predict_generator(rule_f2_gen, verbose=True)
rule_pred_f3 = model.predict_generator(rule_f3_gen, verbose=True)
f1 = rule_f1.join(df[['id', 'category']].set_index('id'), on='feature1')['category'].rename('f1').astype('int')
f2 = rule_f2.join(df[['id', 'category']].set_index('id'), on='feature2')['category'].rename('f2').astype('int')
f3 = rule_f3.join(df[['id', 'category']].set_index('id'), on='feature3')['category'].rename('f3').astype('int')
def createOnehot(_y, cat=100):
    y = np.zeros((_y.shape[0], cat))
    for i in range(len(_y)):
        label = _y[i]
        y[i, label] = 1
    return y
# rule_f1['f1'] = np.argmax(rule_pred_f1, axis=1)
# rule_f2['f2'] = np.argmax(rule_pred_f2, axis=1)
# rule_f3['f3'] = np.argmax(rule_pred_f3, axis=1)


# rule_feature_df = pd.concat([ rule_df, rule_f1['f1'], rule_f2['f2'], rule_f3['f3'] ], axis=1)
rule_feature_df = pd.concat([ rule_df, f1, f2, f3 ], axis=1)
rule_feature_df = rule_feature_df[['id', 'f1', 'f2', 'f3', 'predict']]
rule_feature_df = rule_feature_df.fillna(-1)

rule_feature_df['rand'] = np.random.rand(len(rule_feature_df))
train_rule_df = rule_feature_df[rule_feature_df['rand'] <= 0.8]
val_rule_df = rule_feature_df[rule_feature_df['rand'] > 0.8]

X_train_rule = train_rule_df[['f1', 'f2', 'f3']].values
y_train_rule = train_rule_df['predict'].values
# y_train_rule = createOnehot(y_train_rule)

X_val_rule = val_rule_df[['f1', 'f2', 'f3']].values
y_val_rule = val_rule_df['predict'].values
# y_val_rule = createOnehot(y_val_rule)
np.random.seed(2)
x_in = Input(3)
x = Dense(64, activation='sigmoid')(x_in)
# x = Dense(8, activation='sigmoid')(x)
x = Dense(1, activation='relu')(x)
rule_model = Model(x_in, x)
rule_model.summary()
rule_model.compile(loss='MSE', optimizer=optimizers.SGD(learning_rate=0.005)) 
history = rule_model.fit(X_train_rule, y_train_rule, validation_data=(X_val_rule, y_val_rule), epochs=500, verbose=True)
y_val_pred = np.round(rule_model.predict(X_val_rule))
y_val_pred = y_val_pred.reshape(y_val_pred.shape[0])
# y_val_rule
np.mean(y_val_pred == y_val_rule)
def plotLossHistory(history):
    history_df = pd.DataFrame(history.history)
    history_df = history_df.iloc[::10]
    sns.pointplot(data=history_df, y='loss', x=history_df.index, color='red', label='train')
    sns.pointplot(data=history_df, y='val_loss', x=history_df.index, color='blue', label='val')
    plt.legend()
    plt.show()
test_rule_df = pd.read_csv('/kaggle/input/thai-mnist-classification/test.rules.csv')
f1_df = test_rule_df[['feature1']].dropna()
f2_df = test_rule_df[['feature2']].dropna()
f3_df = test_rule_df[['feature3']].dropna()

f1_df['path'] = '/kaggle/input/thai-mnist-classification/test/' + f1_df['feature1']
f2_df['path'] = '/kaggle/input/thai-mnist-classification/test/' + f2_df['feature2']
f3_df['path'] = '/kaggle/input/thai-mnist-classification/test/' + f3_df['feature3']

f1_df['mock_label'] = (np.round(np.random.rand(len(f1_df)) * 10) % 10).astype('int').astype('str')
f2_df['mock_label'] = (np.round(np.random.rand(len(f2_df)) * 10) % 10).astype('int').astype('str')
f3_df['mock_label'] = (np.round(np.random.rand(len(f3_df)) * 10) % 10).astype('int').astype('str')
test_f1_gen = datagen.flow_from_dataframe(f1_df, x_col='path', y_col='mock_label', batch_size=256, shuffle=False)
test_f2_gen = datagen.flow_from_dataframe(f2_df, x_col='path', y_col='mock_label', batch_size=256, shuffle=False)
test_f3_gen = datagen.flow_from_dataframe(f3_df, x_col='path', y_col='mock_label', batch_size=256, shuffle=False)
pred_f1 = model.predict_generator(test_f1_gen, verbose=True)
pred_f2 = model.predict_generator(test_f2_gen, verbose=True)
pred_f3 = model.predict_generator(test_f3_gen, verbose=True)
f1_df['f1'] = np.argmax(pred_f1, axis=1)
f2_df['f2'] = np.argmax(pred_f2, axis=1)
f3_df['f3'] = np.argmax(pred_f3, axis=1)


test_rule_feature_df = pd.concat([ test_rule_df, f1_df['f1'], f2_df['f2'], f3_df['f3'] ], axis=1)
test_rule_feature_df = test_rule_feature_df[['id', 'f1', 'f2', 'f3', 'predict']]
test_rule_feature_df = test_rule_feature_df.fillna(-1)
X_test = test_rule_feature_df[['f1', 'f2', 'f3']].values
X_test
y_pred = rule_model.predict(X_test)
y_pred = np.round(y_pred).reshape(y_pred.shape[0])
y_pred
test_rule_feature_df['predict'] = y_pred.astype('int')
test_rule_feature_df
submit_df = test_rule_feature_df[['id', 'predict']]
submit_df
submit_df.to_csv('/kaggle/working/submit.csv', index=False)
import os
from IPython.display import FileLink

os.chdir(r'/kaggle/working')
FileLink(r'submit.csv')