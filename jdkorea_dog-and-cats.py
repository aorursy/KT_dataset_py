import os, shutil



original_dataset_dir = '../input/train/train'

base_dir = '../cats_and_dogs_small'



os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')

os.mkdir(validation_dir)

test_dir = os.path.join(base_dir,'test')

os.mkdir(test_dir)
train_cats_dir = os.path.join(train_dir, 'cats')

os.mkdir(train_cats_dir)



train_dogs_dir = os.path.join(train_dir, 'dogs')

os.mkdir(train_dogs_dir)



validation_cats_dir = os.path.join(validation_dir, 'cats')

os.mkdir(validation_cats_dir)



validation_dogs_dir = os.path.join(validation_dir, 'dogs')

os.mkdir(validation_dogs_dir)



test_cats_dir = os.path.join(test_dir, 'cats')

os.mkdir(test_cats_dir)



test_dogs_dir = os.path.join(test_dir, 'dogs')

os.mkdir(test_dogs_dir)



fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_cats_dir, fname)

    shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_cats_dir, fname)

    shutil.copyfile(src, dst)

    

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_cats_dir, fname)

    shutil.copyfile(src, dst)    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_dogs_dir, fname)

    shutil.copyfile(src, dst)

    

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_dogs_dir, fname)

    shutil.copyfile(src, dst)

    

fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_dogs_dir, fname)

    shutil.copyfile(src, dst)
print('훈련용 고양이 이미지 전체 개수:', len(os.listdir(train_cats_dir)))

print('훈련용 강아지 이미지 전체 개수:', len(os.listdir(train_dogs_dir)))

print('검증용 고양이 이미지 전체 개수:', len(os.listdir(validation_cats_dir)))

print('검증용 강아지 이미지 전체 개수:', len(os.listdir(validation_dogs_dir)))

print('테스트용 고양이 이미지 전체 개수:', len(os.listdir(test_cats_dir)))

print('테스트용 강아지 이미지 전체 개수:', len(os.listdir(test_dogs_dir)))

from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu',

                       input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
from keras import optimizers

model.compile(loss='binary_crossentropy',

             optimizer=optimizers.RMSprop(lr=1e-4),

             metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

                    train_dir,

                    target_size=(150,150),

                    batch_size=20,

                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(

                    validation_dir,

                    target_size=(150,150),

                    batch_size=20,

                    class_mode='binary')
for data_batch, labels_batch in train_generator:

    print('배치 데이터 크기:', data_batch.shape)

    print('배치 레이블 크기:', labels_batch.shape)

    break
history = model.fit_generator(

train_generator,

steps_per_epoch=100,

epochs = 30,

validation_data = validation_generator,

validation_steps=50)
model.save('cats_and_dogs_small_1.h5')
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and Validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validaiton loss')

plt.title('Training and Validation loss')

plt.legend()



plt.show()
datagen = ImageDataGenerator(

rotation_range=20,

width_shift_range=0.1,

height_shift_range=0.1,

shear_range=0.1,

zoom_range=0.1,

horizontal_flip=True,

fill_mode='nearest')
# 랜덤으로 증식된 훈련 이미지 그리기

from keras.preprocessing import image



fnames = sorted([os.path.join(train_cats_dir, fname) for 

                fname in os.listdir(train_cats_dir)])

img_path = fnames[3]

img = image.load_img(img_path, target_size=(150, 150))



x = image.img_to_array(img) # (150,150,3) 크기의 넘파이 배열로 변환함

x = x.reshape((1,) + x.shape) # (1, 150, 150, 3) 크기로 변환



i = 0

for batch in datagen.flow(x, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 ==0:

        break

plt.show()
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu',

                       input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

             optimizer=optimizers.RMSprop(lr=1e-4),

             metrics=['acc'])

train_datagen = ImageDataGenerator(

rescale=1./255,

rotation_range=40,

width_shift_range=0.2,

height_shift_range=0.2,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True,)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

                    train_dir,

                    target_size=(150,150),

                    batch_size=32,

                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(

                    validation_dir,

                    target_size=(150,150),

                    batch_size=32,

                    class_mode='binary')



history = model.fit_generator(

train_generator,

steps_per_epoch=100,

epochs = 100,

validation_data = validation_generator,

validation_steps=50)

model.save('cats_and_dogs_small_2.h5')
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and Validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validaiton loss')

plt.title('Training and Validation loss')

plt.legend()



plt.show()