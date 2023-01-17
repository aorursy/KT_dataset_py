import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from glob import glob

from PIL import Image



from warnings import filterwarnings



np.random.seed(101)

filterwarnings('ignore')

sns.set_style('darkgrid')
base_loc = '../input'

image_paths = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(base_loc, '*', '*.jpg'))}
df_skin = pd.read_csv(os.path.join(base_loc, 'HAM10000_metadata.csv'))

df_skin.head()
df_skin['image_path'] = df_skin['image_id'].map(image_paths.get)
lesion_types = {

    'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}



df_skin['type'] = df_skin['dx'].map(lesion_types.get)



# Converting the type to the categorical values

df_skin['type_id'] = pd.Categorical(df_skin['type']).codes



df_skin.head()
sns.heatmap(df_skin.isna(), cbar = False, cmap= 'plasma')

print(df_skin.isna().sum())
print('The % of missing values is only {:.2f} so we will consider filling them with mean valueof the entire column'.format(df_skin['age'].isna().sum()/len(df_skin)*100))

df_skin['age'] = df_skin['age'].fillna(df_skin['age'].mean())

df_skin.head()
df_skin.info()
plt.figure(figsize = (20,14))



plt.subplot(2,2,1)

fig = sns.countplot(y = df_skin['type'], order = df_skin['type'].value_counts().index, palette= 'viridis')

plt.xticks(fig.get_xticks())

plt.title('Most Frequent Type of Cells')



plt.subplot(2,2,2)

fig = sns.countplot(x = df_skin['dx_type'], order = df_skin['dx_type'].value_counts().index, palette= 'autumn')

plt.xticks(fig.get_xticks())

plt.title('The Technical Validation')



plt.subplot(2,2,3)

fig = sns.countplot(x = df_skin['localization'], order = df_skin['localization'].value_counts().index, palette= 'inferno')

plt.xticks(fig.get_xticks(),rotation = 90)

plt.title('Most Frequent Localizations')



plt.subplot(2,2,4)

fig = sns.countplot(x = df_skin['sex'], order = df_skin['sex'].value_counts().index, palette= 'summer')

plt.xticks(fig.get_xticks(),rotation = 90)

plt.title('Sex')



# plt.tight_layout()

plt.show()
plt.figure(figsize=(18, 12))

plt.suptitle('Distribution of Age')



plt.subplot(2,2,1)

sns.distplot(df_skin['age'], color= 'green')

plt.title('Overall Distribution')

plt.xticks(list(range(0,100,10)))



plt.subplot(2,2,2)

sns.kdeplot(df_skin['age'], shade = True, color = 'green')

plt.title('Overall Distribution')

plt.xticks(list(range(0,100,10)))



plt.subplot(2,2,2)

sns.kdeplot(df_skin['age'], shade = True, color = 'green')

plt.xticks(list(range(0,100,10)))



plt.subplot(2,2,3)

sns.kdeplot(df_skin[df_skin['sex'] == 'male']['age'],label = 'Male', shade = True, color = 'blue')

plt.xticks(list(range(0,100,10)))

plt.title('Distribution Among Males')



plt.subplot(2,2,4)

sns.kdeplot(df_skin[df_skin['sex'] == 'female']['age'],label = 'Female', shade = True, color = 'red')

plt.title('Distribution Among Females')

plt.xticks(list(range(0,100,10)))





plt.show()
plt.figure(figsize=(9, 7))

sns.scatterplot(df_skin['age'], df_skin['type_id'])

plt.title('Types vs Age')

plt.show()
Image.open(df_skin['image_path'][1])
Image.open(df_skin['image_path'][1]).resize((100,75))
print(np.asarray(Image.open(df_skin['image_path'][0]).resize((100,75)))[:4])
df_skin['image'] = df_skin['image_path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
df_skin.head()
plt.figure(figsize= (20,10))

for i,img in enumerate(np.random.random_integers(0, 10000, 25)):

    plt.subplot(5,5,i+1)

    plt.imshow(df_skin['image'][img])

    plt.title(df_skin['type'][img])

    plt.xticks([])

    plt.yticks([])

    plt.grid()

plt.tight_layout()

plt.show()
df_skin.head()
features = df_skin.drop(columns=['type_id'],axis=1)

target = df_skin['type_id']
from sklearn.model_selection import train_test_split

X_train_, X_test_, y_train_, y_test_ = train_test_split(features, target, test_size = 0.20, random_state = 101)

print('The length of training Set is {}\nThe length of the test set is {}\nThe ratio is {}'.format(len(X_train), len(X_test), '80/20'))
X_train = np.asarray(X_train_['image'].tolist())

X_test = np.asarray(X_test_['image'].tolist())



X_train = (X_train - X_train.mean())/X_train.std()

X_test = (X_test - X_test.mean())/X_test.std()
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train_, num_classes= 7)

y_test = to_categorical(y_test_, num_classes= 7)
y_test[:10]
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding



from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau
model = Sequential()

model.add(Conv2D(32, kernel_size= (3,3), activation= 'relu', padding= 'Same', input_shape = (75, 100, 3)))

model.add(Conv2D(32, kernel_size= (3,3), activation= 'relu', padding= 'Same'))

model.add(MaxPool2D(pool_size= (2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size= (3,3), activation= 'relu', padding= 'Same'))

model.add(Conv2D(64, kernel_size= (3,3), activation= 'relu', padding= 'Same'))

model.add(MaxPool2D(pool_size= (2, 2)))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(128, activation= 'relu'))

model.add(Dropout(0.5))



model.add(Dense(7, activation= 'softmax'))

model.summary()

model.compile(Adam(), loss = 'categorical_crossentropy', metrics = ['mae', 'acc'])
annealer = ReduceLROnPlateau(monitor= 'val_acc')
mod = model.fit(x = X_train, y = y_train, epochs= 50, callbacks= [annealer])
import pickle

pickle.dump(model,open('model.pkl','wb'))
loss, mae, acc = model.evaluate(X_test, y_test)

print("The accuracy of the model is {:.3f}\nThe Loss in the model is {:.3f}".format(acc,loss))
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)

plt.plot(mod.history['acc'], color = 'green')

plt.title('The Training Accuracy')

plt.subplot(1,2,2)

plt.plot(mod.history['loss'], color = 'red')

plt.title('The Training Loss')

plt.show()