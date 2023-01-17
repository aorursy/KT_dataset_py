import time
start = time.time()
import numpy as np 
import pandas as pd 
import os
from keras.layers import Dense, Flatten, Dropout, Lambda, Input, Concatenate, concatenate
from keras.models import Model
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import regularizers
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d train
!unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip -d test
print(os.listdir('../'))
filenames = os.listdir("../working/train/train")

labels = []
for file in filenames:
    category = file.split('.')[0]
    if category == 'cat':
        labels.append('cat')
    else:
        labels.append('dog')
df = pd.DataFrame({
    'filename': filenames,
    'label': labels
})
df.head(10)
def get_class_counts(df):
    grp = df.groupby(['label']).nunique()
    return {key: grp[key] for key in list(grp.keys())}

def get_class_proportions(df):
    class_counts = get_class_counts(df)
    return {val[0]: round(val[1]/df.shape[0],4) for val in class_counts.items()}
print("Dataset class counts", get_class_counts(df))
print("Dataset class proportions", get_class_proportions(df))
train_df, validation_df = train_test_split(df, 
                                           test_size=0.1, 
                                           stratify=df['label'],
                                           random_state = 42)
print("Train data class proportions : ", get_class_proportions(train_df))
print("Validation data class proportions : ", get_class_proportions(validation_df))
train_df.head(10)
train_df = train_df.reset_index(drop=True)

validation_df = validation_df.reset_index(drop=True)
train_df.head(10)
batch_size = 64
train_num = len(train_df)
validation_num = len(validation_df)

print("The number of training set is {}".format(train_num))
print("The number of validation set is {}".format(validation_num))
def two_image_generator(generator, df, directory, batch_size,
                        x_col = 'filename', y_col = None, model = None, shuffle = False,
                        img_size1 = (224, 224), img_size2 = (299,299)):
    gen1 = generator.flow_from_dataframe(
        df,
        directory,
        x_col = x_col,
        y_col = y_col,
        target_size = img_size1,
        class_mode = model,
        batch_size = batch_size,
        shuffle = shuffle,
        seed = 1)
    gen2 = generator.flow_from_dataframe(
        df,
        directory,
        x_col = x_col,
        y_col = y_col,
        target_size = img_size2,
        class_mode = model,
        batch_size = batch_size,
        shuffle = shuffle,
        seed = 1)
    
    while True:
        X1i = gen1.next()
        X2i = gen2.next()
        
        if y_col:
            yield [X1i[0], X2i[0]], X1i[1]  #X1i[1] is the label
        else:
            yield [X1i, X2i]
        
ex_df = pd.DataFrame()
ex_df['filename'] = filenames[:5]
ex_df['label'] = labels[:5]
ex_df.head()

train_aug_datagen = ImageDataGenerator(
    rotation_range = 20,
    shear_range = 0.1,
    zoom_range = 0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True
)
e1 = two_image_generator(train_aug_datagen, ex_df, '../working/train/train/',
                                      batch_size = 2, y_col = 'label',
                                      model = 'binary', shuffle = True)

fig = plt.figure(figsize = (10,10))
batches = 0
rows = 5
cols = 5
i = 0
j = 0
indices_a = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
indices_b = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]
for [x_batch, x_batch2], y_batch in e1:
    for image in x_batch:
        fig.add_subplot(rows, cols, indices_a[i])
        i += 1
        plt.imshow(image.astype('uint8'))
        
    for image in x_batch2:
        fig.add_subplot(rows, cols, indices_b[j])
        j += 1
        plt.imshow(image.astype('uint8'))
    
    batches += 1
    if batches >= 6:
        break
plt.show()


train_aug_datagen = ImageDataGenerator(
    rotation_range = 20,
    shear_range = 0.1,
    zoom_range = 0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True
)
train_generator = two_image_generator(train_aug_datagen, train_df, '../working/train/train/',
                                      batch_size = batch_size, y_col = 'label',
                                      model = 'binary', shuffle = True)
validation_datagen = ImageDataGenerator()

validation_generator = two_image_generator(validation_datagen, validation_df,
                                           '../working/train/train/', batch_size = batch_size,
                                           y_col = 'label',model = 'binary', shuffle = True)
def create_base_model(MODEL, img_size, lambda_fun = None):
    inp = Input(shape = (img_size[0], img_size[1], 3))
    x = inp
    if lambda_fun:
        x = Lambda(lambda_fun)(x)
    
    base_model = MODEL(input_tensor = x, weights = 'imagenet', include_top = False, pooling = 'avg')
        
    model = Model(inp, base_model.output)
    return model
#define vgg + resnet50 + densenet
model1 = create_base_model(nasnet.NASNetLarge, (224, 224), nasnet.preprocess_input)
model2 = create_base_model(inception_resnet_v2.InceptionResNetV2, (299, 299), inception_resnet_v2.preprocess_input)
model3 = create_base_model(xception.Xception, (299, 299), xception.preprocess_input)
model1.trainable = False
model2.trainable = False
model3.trainable = False

inpA = Input(shape = (224, 224, 3))
inpB = Input(shape = (299, 299, 3))
out1 = model1(inpA)
out2 = model2(inpA)
out3 = model3(inpB)

x = Concatenate()([out1, out2, out3])                
x = Dropout(0.6)(x)
x = Dense(1, activation='sigmoid')(x)
multiple_pretained_model = Model([inpA, inpB], x)

multiple_pretained_model.compile(loss = 'binary_crossentropy',
                          optimizer = 'rmsprop',
                          metrics = ['accuracy'])

multiple_pretained_model.summary()
checkpointer = ModelCheckpoint(filepath='dogcat.weights.best.hdf5', verbose=1, 
                               save_best_only=True, save_weights_only=True)
%%time
multiple_pretained_model.fit_generator(
    train_generator,
    epochs = 3,
    steps_per_epoch = train_num // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_num // batch_size,
    verbose = 1,
    callbacks = [checkpointer]
)
multiple_pretained_model.load_weights('dogcat.weights.best.hdf5')
test_filenames = os.listdir("../working/test/test")
test_df = pd.DataFrame({
    'filename': test_filenames
})

num_test = len(test_df)

test_datagen = ImageDataGenerator()

test_generator = two_image_generator(test_datagen, test_df, '../working/test/test/', batch_size = batch_size)
prediction = multiple_pretained_model.predict_generator(test_generator, 
                                         steps=np.ceil(num_test/batch_size))
prediction = prediction.clip(min = 0.005, max = 0.995)
submission_df = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')

for i, fname in enumerate(test_filenames):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    submission_df.at[index-1, 'label'] = prediction[i]
submission_df.to_csv('Cats&DogsSubmission.csv', index=False)
submission_df.head()
## print run time
end = time.time()
print(round((end-start),2), "seconds")