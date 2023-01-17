import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import os

import shutil



import re



from collections import Counter



from IPython.display import display, Image



from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Conv2D, Dense, Flatten



IMAGE_SIZE = 32

TRAIN_PERCENT = 0.8

BATCH_SIZE = 20

EPOCHS = 5
!ls '../input/classification-of-handwritten-letters'
path1 = '../input/classification-of-handwritten-letters/letters.csv'

path2 = '../input/classification-of-handwritten-letters/letters2.csv'

path3 = '../input/classification-of-handwritten-letters/letters3.csv'

df1 = pd.read_csv(path1)

df2 = pd.read_csv(path2)

df3 = pd.read_csv(path3)

df = pd.concat([df1, df2, df3])

df.head()
df['background'].unique()
columns = ['letter', 'label']

letters_df = df[columns]

letters_df = letters_df.drop_duplicates()

letters_df = letters_df.set_index('label')



letters_dict = letters_df.to_dict()

letters_dict = letters_dict['letter']

print(letters_dict)
n_classes = len(letters_dict)

n_classes
bg_imgs = df.drop_duplicates(subset='background')[['background', 'file']].set_index('background').to_dict()['file']

bg_imgs
base_path = '../input/classification-of-handwritten-letters/letters'



for bg in bg_imgs:

    path1 = base_path + '/' + bg_imgs[bg]

    path2 = base_path + '2/' + bg_imgs[bg]

    path3 = base_path + '3/' + bg_imgs[bg]

    if os.path.isfile(path1):

        display(Image(path1))

    elif os.path.isfile(path2):

        display(Image(path2))

    elif os.path.isfile(path3):

        display(Image(path3))
def show_img(ax, path, title):

    img = load_img(path, color_mode='grayscale')

    array = img_to_array(img)

    array = array.reshape(IMAGE_SIZE, IMAGE_SIZE)

    array = array / 255



    ax.imshow(array)

    ax.set_title(title)
fig, axs = plt.subplots(1, 4, figsize=(10, 40))

axs = axs.ravel()



for bg, ax in zip(bg_imgs, axs):

    path1 = base_path + '/' + bg_imgs[bg]

    path2 = base_path + '2/' + bg_imgs[bg]

    path3 = base_path + '3/' + bg_imgs[bg]

    title = 'Background ' + str(bg)

    if os.path.isfile(path1):

        show_img(ax, path1, title)

    elif os.path.isfile(path2):

        show_img(ax, path2, title)

    elif os.path.isfile(path3):

        show_img(ax, path3, title)



plt.tight_layout()

plt.show()
path1 = '../input/classification-of-handwritten-letters/letters'

path2 = '../input/classification-of-handwritten-letters/letters2'

path3 = '../input/classification-of-handwritten-letters/letters3'



my_path = '../manel'

try:

    os.mkdir(my_path)

except:

    pass



for bg in bg_imgs:

    bg_path = my_path + '/' + str(bg)

    for end in ['', '/train', '/test']:

        new_path = bg_path + end

        try:

            os.mkdir(new_path)

        except:

            pass
my_regex = r'[_](\d+)[.]'

df['number'] = df['file'].str.extract(my_regex)

df['number'] = pd.to_numeric(df['number'])

df.tail()
all_columns = ['background', 'letter', 'file']

group_columns = ['background', 'letter']

group_df = df[all_columns].groupby(group_columns).count()

group_df = group_df.reset_index()

group_df = group_df.drop(columns='letter')

group_df = group_df.set_index('background')

group_df = group_df.drop_duplicates()

group_df
group_df['end'] = group_df['file'].cumsum()

group_df['beginning'] = group_df['end'].shift(1) + 1

group_df.loc[0, 'beginning'] = 1

group_df['beginning'] = group_df['beginning'].astype('int64')

group_df
group_df['train'] = group_df['file'] * TRAIN_PERCENT

group_df['train'] = group_df['train'].astype('int64')

group_df['cut'] = group_df['beginning'] + group_df['train']

group_df
group_df = group_df[['cut']]

TRAIN_TEST_CUT = group_df.to_dict()['cut']

print(TRAIN_TEST_CUT)
def copy_images(path1, path2):

    

    # Gets at least one (+) digit (\d) at start of string (^)

    get_label = r'^\d+'

    # Gets at least one digit between underscore (_) and dot (\.)

    get_number = r'_(\d+)\.'

    

    # Loop over files in path1 to copy them to path2

    for dirname, _, filenames in os.walk(path1):

        for filename in filenames:

            

            # For each file, this is it complete path of origin

            origin = dirname + '/' + filename

            

            # Gets label of letter (01 is a, 02 is b, etc)

            label = re.search(get_label, filename).group()

            # Gets number of sample

            number = int(re.search(get_number, filename).group(1))

            

            # Consults dataframe to get type of background of image

            mask = df['file'] == filename

            bg = df.loc[mask, 'background'].item()

            

            # Depending on number of sample and background, decides train or test

            if number < TRAIN_TEST_CUT[bg]:

                train_or_test = 'train'

            else:

                train_or_test = 'test'

            

            

            # This is the path of destination

            destination = path2 + '/' + str(bg) + '/' + train_or_test + '/' + label

            

            # Try to make directory if it doesn't exist

            try:

                os.mkdir(destination)

            except:

                pass

            

            # Copy file from origin to destination

            shutil.copy(origin, destination)
copy_images(path1, my_path)

copy_images(path2, my_path)

copy_images(path3, my_path)
TRAIN_PATH = '../manel/2/train'

TEST_PATH = '../manel/2/test'
imgdatagen = ImageDataGenerator(rescale=1/255)

train_gen = imgdatagen.flow_from_directory(TRAIN_PATH,

                                           target_size=(IMAGE_SIZE, IMAGE_SIZE),

                                           batch_size=BATCH_SIZE,

                                           class_mode='categorical',

                                           color_mode='grayscale')
test_gen = imgdatagen.flow_from_directory(TEST_PATH,

                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),

                                          batch_size=BATCH_SIZE,

                                          class_mode='categorical',

                                          color_mode='grayscale')
print(train_gen.class_indices)
class_ind = train_gen.class_indices

letters_ind = {}



for key in class_ind:

    letters_ind[class_ind[key]] = letters_dict[int(key)]



print(letters_ind)
TRAIN_STEPS = train_gen.samples // BATCH_SIZE

TEST_STEPS = test_gen.samples // BATCH_SIZE
for x, y in train_gen:

    plt.imshow(x[0].reshape(IMAGE_SIZE, IMAGE_SIZE))

    plt.show()

    break
for x, y in test_gen:

    plt.imshow(x[0].reshape(IMAGE_SIZE, IMAGE_SIZE))

    plt.show()

    break
model = Sequential()

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_gen, epochs=EPOCHS, steps_per_epoch=TRAIN_STEPS, validation_data=test_gen, validation_steps=TEST_STEPS)
def decode_preds(preds, top):

    output = []

    for pred in preds:

        pred_dict = {}

        for key in letters_ind:

            pred_dict[letters_ind[key]] = '{:.2f} %'.format(pred[key] * 100)

        pred_dict = Counter(pred_dict)

        output.append(pred_dict.most_common(top))

    return output
def make_preds(arrays, letters, top):

    

    preds = model.predict(arrays)

    dec_preds = decode_preds(preds, top)

    fig, axs = plt.subplots(2, 4, figsize=(15, 60))

    axs = axs.ravel()

    for array, letter, pred, ax in zip(arrays, letters, dec_preds, axs):

        array = array.reshape(IMAGE_SIZE, IMAGE_SIZE)

        array = array / 255

        ax.set_title(letter + ', ' + str(pred))

        ax.imshow(array)

    

    plt.tight_layout()

    plt.show()
path = '../input/classification-of-handwritten-letters/letters2'

get_label = r'^\d+'

arrays = []

letters = []

i = 4



for dirname, _, filenames in os.walk(path):

    for filename in filenames:

        if i > 0:

            img = load_img(dirname + '/' + filename, color_mode='grayscale')

            array = img_to_array(img)

            arrays.append(array)

            label = re.search(get_label, filename).group()

            label = int(label)

            letters.append(letters_dict[label])

            i = i - 1

        else:

            arrays = np.array(arrays)

            break



make_preds(arrays, letters, 2)