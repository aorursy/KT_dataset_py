import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import seaborn as sns

from tqdm import tqdm

from numpy import random



import matplotlib as mpl

import matplotlib.pyplot as plt



from skimage.transform import rotate



from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler

from skimage.restoration.inpaint import inpaint_biharmonic



import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import BatchNormalization

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers.core import Dense, Flatten, Dropout, Lambda



from keras.utils.np_utils import to_categorical
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

sample = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
target = train.label.values

# Drop the label feature

train = train.drop("label",axis=1)
scaler = MinMaxScaler()

train = scaler.fit_transform(train)

test = scaler.transform(test)
del scaler
n = np.random.randint(0, 42000, 10)



digits = train[n]

labels = target[n]

print(labels)

# code for rendering

f, ax = plt.subplots(

    2, 5, figsize=(12,5),

    gridspec_kw={'wspace':0.03, 'hspace':0.01}, 

    squeeze=True

)



index = 0

for r in range(2):

    for c in range(5):

        ax[r,c].axis("off")

        image = digits[index].reshape(28, 28)

        ax[r,c].imshow(image, cmap='gray')

        ax[r,c].set_title('No. {0}'.format(labels[index]))

        index+=1

        

plt.show()

plt.close()
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)

rnd_clf.fit(train, target)
sns.set(rc={'figure.figsize': (15, 10)})



def plot_digit(data):

    image = data.reshape(28, 28)

    plt.imshow(image, cmap = mpl.cm.hot,

               interpolation="nearest")

    plt.axis("off")

    

plot_digit(rnd_clf.feature_importances_)



cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])

cbar.ax.set_yticklabels(['Not important', 'Very important'])

plt.show()
del rnd_clf
k_index =[39010, 18519, 37412, 26715, 1257, 25694, 17832, 33398, 24817, 1548, 326, 27648, 4989, 34042, 27653,

         35860, 36694, 23727, 1048, 38721, 985, 22155, 35557]



digits = train[k_index]

labels = target[k_index]

print(labels)

# code for rendering

f, ax = plt.subplots(

    3, 7, figsize=(12,5),

    gridspec_kw={'wspace':0.03, 'hspace':0.01}, 

    squeeze=True

)



index = 0

for r in range(3):

    for c in range(7):

        ax[r,c].axis("off")

        image = digits[index].reshape(28, 28)

        ax[r,c].imshow(image, cmap='gray')

        ax[r,c].set_title('No. {0}'.format(labels[index]))

        index+=1

        

plt.show()

plt.close()
sns.set(rc={'figure.figsize': (7, 5)})

sns.set_style({'axes.grid' : False}) 



eight_test_images_restoration = train[34042].reshape(28, 28)

plt.imshow(eight_test_images_restoration, cmap='gray')
mask = np.zeros((28, 28))

mask[5:12, 18:24] = 1.0



tset_image_with_defect = eight_test_images_restoration.copy()

tset_image_with_defect[np.where(mask)] = 0



image_inpainted = inpaint_biharmonic(tset_image_with_defect, mask) #multichannel=True)
plt.imshow(image_inpainted.reshape(28, 28), cmap='gray')
nine_test_images_restoration = train[24817].reshape(28, 28)

plt.imshow(nine_test_images_restoration, cmap='gray')
mask = np.zeros((28, 28))

mask[7:11, 17:19] = 1.0



test_images_restoration_defect = nine_test_images_restoration.copy()

test_images_restoration_defect[np.where(mask)] = 0



image_inpainted = inpaint_biharmonic(test_images_restoration_defect, mask)
plt.imshow(image_inpainted.reshape(28, 28), cmap='gray')
index_target_8 = np.array(np.where(target == 8)[0])

index_target_9 = np.array(np.where(target == 9)[0])



index_98 = np.random.randint(0, 4063, 2000) 



index_target_8 = index_target_8[index_98]

index_target_9 = index_target_9[index_98]
for index_8 in tqdm(index_target_8):

    eight_images_restoration = train[index_8]

    eight_images_restoration = eight_images_restoration.reshape(28, 28)

    

    mask = np.zeros((28, 28))

    mask[5:12, 18:24] = 1

    

    eight_image_with_defect = eight_images_restoration.copy()

    eight_image_with_defect[np.where(mask)] = 0

    

    image_inpainted = inpaint_biharmonic(eight_image_with_defect, mask)

    

    train[index_8] = image_inpainted.reshape(784)
for index_9 in tqdm(index_target_9):

    nine_images_restoration = train[index_9]

    nine_images_restoration = nine_images_restoration.reshape(28, 28)

    

    mask = np.zeros((28, 28))

    mask[7:11, 17:19] = 1.0

    

    nine_image_with_defect = nine_images_restoration.copy()

    nine_image_with_defect[np.where(mask)] = 0

    

    image_inpainted = inpaint_biharmonic(nine_image_with_defect, mask)

    

    train[index_9] = image_inpainted.reshape(784)
unknown_images = train[30352].reshape(28, 28)

plt.imshow(unknown_images, cmap='gray')
train = np.delete(train, 30352, axis = 0)

target = np.delete(target, 30352, axis = 0)
print(len(train))

print(len(target))
# causes zooming in the x direction,

def matrix_хах(a):                   

    matrix_хах = np.array([[a, 0], 

                           [0, 1]])

    return matrix_хах



# in this example we will transform both coordinates

def matrix_хах_ydy(a, d):            

    matrix_хах_ydy = np.array([[a, 0],  

                               [0, d]])

    return matrix_хах_ydy



# point mapping relative to both x y axes

def revers_on_xy():

    revers_on_xy = np.array([[-1, 0],  

                             [0, -1]])

    return revers_on_xy



# the shift is proportional to x

def shift_on_x(b):

    shift_on_x = np.array([[1, b],  

                           [0, 1]])

    return shift_on_x

'''

# the shift is proportional to y

def shift_on_y(c):

    shift_on_y = np.array([[1, 0],  

                           [c, 1]])

    return shift_on_y

'''

# Turns 90 and 180



def Turn_on90():

    turn_on90 = np.array([[0, 1],  

                          [-1, 0]])

    return turn_on90



def Turn_on180():

    turn_on180 = np.array([[-1, 0],  

                           [0, -1]])

    return turn_on180
n = np.random.randint(0, 41999, 7000)



digits = train[n] 

labels = target[n]
labels
random_transformation_matrix = [

                                matrix_хах(np.random.randint(2, 5)), 

    

                                matrix_хах_ydy(2, 2), 

                                matrix_хах_ydy(np.random.randint(0, 3), 

                                               np.random.randint(0, 3)), 

    

                                revers_on_xy(), 

                    

                                shift_on_x(np.random.randint(1, 4)), 

                                #shift_on_y(np.random.randint(1, 4)), 

    

                                Turn_on90(),

                                Turn_on180()

                               ]



Frame_new_alis = []



for ind_new_alis, mas_digits in tqdm(enumerate(digits)):

    h = random_transformation_matrix[np.random.randint(0, 7)]

    alis = []

    for d in range(0, len(list(mas_digits)), 28):

        alis.append(mas_digits[d:d+28]) 



    new_alis = np.zeros((200, 200)) 

    

    for i, t in enumerate(alis):

        for ind_t in range(len(t)):

            new_koar = np.array([i, ind_t]).dot(h)

            new_alis[abs(new_koar[0])][abs(new_koar[1])] = t[ind_t]

            

    Frame_new_alis.append(new_alis)
f, ax = plt.subplots(

    2, 5, figsize=(12,5),

    gridspec_kw={'wspace':0.03, 'hspace':0.01}, 

    squeeze=True

)



index = 0

for r in range(2):

    for c in range(5):

        ax[r,c].axis("off")

        image = Frame_new_alis[index].reshape(200, 200)

        ax[r,c].imshow(image, cmap='gray')

        index+=1

        

plt.show()

plt.close()
mas_DF_newalias_dict = {}



for di in tqdm(range(len(Frame_new_alis))):

    mas_DF_newalias = []

    for st_di in Frame_new_alis[di]: 

        for st_di_i in st_di:

            mas_DF_newalias.append(st_di_i)

    mas_DF_newalias_dict['{}'.format(di)] = mas_DF_newalias

    

    

df2 = pd.DataFrame(data=mas_DF_newalias_dict) 

    

del mas_DF_newalias_dict   
df2 = df2.loc[:, (df2 != 0).any(axis=0)]
pd.to_numeric(df2.columns)
target_X_reduced_new = labels[pd.to_numeric(df2.columns)]

print(len(target_X_reduced_new))
target_X_reduced_new
print(df2.shape)

print(len(df2))

df2.head() 
df2 = df2.T
dict_abbreviated = {}

for index_df2 in tqdm(range(df2.shape[0])): 

    abbreviated_mas = []

    for nd_df2 in df2.iloc[index_df2].values.reshape(200, 200):

        for truncated_array in nd_df2[:100]:

            abbreviated_mas.append(truncated_array)

    dict_abbreviated[index_df2] = abbreviated_mas
del df2

df3 = pd.DataFrame(data=dict_abbreviated).T

df3 = df3.iloc[:,0:10000]

del dict_abbreviated
n_df3 = np.random.randint(0, df3.shape[0], 6)

digits_df3 = df3.values[n_df3]

labels_df3 = target_X_reduced_new[n_df3]



# code for rendering

f, ax = plt.subplots(

    2, 3, figsize=(12,5),

    gridspec_kw={'wspace':0.03, 'hspace':0.01}, 

    squeeze=True

)



index = 0

for r in range(2):

    for c in range(3):

        ax[r,c].axis("off")

        image = digits_df3[index].reshape(100, 100)

        ax[r,c].imshow(image, cmap='gray')

        ax[r,c].set_title('No. {0}'.format(labels_df3[index]))

        index+=1

        

plt.show()

plt.close()
n_rotate = np.random.randint(0, 41999, 30000)

digits_rotate = train[n_rotate] 

labels_rotate = target[n_rotate]
rotatet_x = []

for x in digits_rotate:

    angle = np.random.rand() * 30 - 15

    rot_img = rotate(x.reshape(28, 28), angle)

    rotatet_x.append(rot_img.flatten())

    

rotatet_x = pd.DataFrame(data=rotatet_x)
print(rotatet_x.shape)

rotatet_x.head()
n_rotatet_x = np.random.randint(0, rotatet_x.shape[0], 6)

digits_rotatet_x = rotatet_x.values[n_rotatet_x]

labels_rotatet_x = labels_rotate[n_rotatet_x]



# код для отрисовки

f, ax = plt.subplots(

    2, 3, figsize=(12,5),

    gridspec_kw={'wspace':0.03, 'hspace':0.01}, 

    squeeze=True

)



index = 0

for r in range(2):

    for c in range(3):

        ax[r,c].axis("off")

        image = digits_rotatet_x[index].reshape(28, 28)

        ax[r,c].imshow(image, cmap='gray')

        ax[r,c].set_title('No. {0}'.format(labels_rotatet_x[index]))

        index+=1

        

plt.show()

plt.close()
df3_value = df3.values

del df3
pca = PCA(n_components=784)

pca.fit(df3_value)

X_reduced_new = pca.transform(df3_value)
len(X_reduced_new[0])
print(len(target_X_reduced_new ))

target_X_reduced_new # target variables of converted images
print(len(target))

target
c = np.concatenate((train, X_reduced_new, rotatet_x.values), axis=0) 

target_c = np.concatenate((target, target_X_reduced_new, labels_rotate), axis=0) 
print(len(c))

print(len(target_c))
def ret(a):

    return  a
c = c.reshape([-1,28,28, 1])

target_c = to_categorical(target_c, num_classes= 10)
X_train, y_train, X_test, y_test = train_test_split(c, target_c, test_size=0.4, random_state=42)
model= Sequential()



model.add(Lambda(ret, input_shape = (28,28, 1)))



model.add(Conv2D(64, (3,3), padding= 'same', activation = 'relu'))

model.add(BatchNormalization())



model.add(Conv2D(64, (3,3), padding= 'same', activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

model.add(BatchNormalization())



model.add(Conv2D(64, (3,3), padding= 'same', activation = 'relu'))

model.add(BatchNormalization())



model.add(Conv2D(32, (3,3), padding= 'same', activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

model.add(BatchNormalization())

model.add(Flatten())





model.add(Dense(400, activation = 'relu'))

model.add(BatchNormalization())

#model.add(Dropout(0.4))

model.add(Dense(300, activation = 'relu'))

model.add(BatchNormalization())

#model.add(Dropout(0.4))



model.add(Dense(100, activation = 'softmax'))

model.add(BatchNormalization())

#model.add(Dropout(0.4))



model.add(Dense(10, activation = 'softmax'))



model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = tf.keras.optimizers.SGD())

model_fit = model.fit(X_train, X_test, validation_data=(y_train, y_test), epochs=15) 
sns.set(rc={'figure.figsize': (15, 10)})

plt.plot(model_fit.history['accuracy'], label='train')

plt.plot(model_fit.history['val_accuracy'], label='test')

plt.legend()

plt.show()
test = test.reshape([-1,28, 28, 1])
predictions = model.predict(test)
len(predictions)
predictions = np.argmax(predictions, axis = 1)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("predictions.csv", index=False, header=True) 
submissions.shape  
submissions.head() 