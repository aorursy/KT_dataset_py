# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2

import matplotlib.pyplot as plt
os.listdir('/kaggle/input/animals10/animals/raw-img')

foldernames = os.listdir('/kaggle/input/animals10/animals/raw-img')

categories = []

files = []

i = 0

for folder in foldernames:

    filenames = os.listdir("../input/animals10/animals/raw-img/" + folder);

    for file in filenames:

        files.append("../input/animals10/animals/raw-img/" + folder + "/" + file)

        categories.append(i)

    i = i + 1

        

        

df = pd.DataFrame({

    'filename': files,

    'category': categories

    

})
train_df = pd.DataFrame(columns=['filename', 'category'])

for i in range(10):

    train_df = train_df.append(df[df.category == i].iloc[:400,:])



del df

train_df.head()

train_df = train_df.reset_index(drop=True)
len(train_df)
y = train_df['category']

train_df['category'].value_counts()
from sklearn.utils import class_weight, shuffle



x = train_df['filename']

y = train_df['category']



x, y = shuffle(x, y, random_state=8)
from tqdm import tqdm_notebook as tqdm





def centering_image(img):

    size = [256,256]

    

    img_size = img.shape[:2]

    

    # centering

    row = (size[1] - img_size[0]) // 2

    col = (size[0] - img_size[1]) // 2

    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)

    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img



    return resized



images = []

with tqdm(total=len(train_df)) as pbar:

    for i, file_path in enumerate(train_df.filename.values):

        #read image

        img = cv2.imread(file_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



        #resize

        if(img.shape[0] > img.shape[1]):

            tile_size = (int(img.shape[1]*256/img.shape[0]),256)

        else:

            tile_size = (256, int(img.shape[0]*256/img.shape[1]))



        #centering

        img = centering_image(cv2.resize(img, dsize=tile_size))



        #out put 224*224px 

        img = img[16:240, 16:240]

        images.append(img)

        pbar.update(1)



images = np.array(images)
rows,cols = 2,5

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20,20))

for i in range(10):

    path = train_df[train_df.category == i].values[2]

#     image = cv2.imread(path[0])

    axes[i//cols, i%cols].set_title(path[0].split('/')[-2] + str(path[1]))

    axes[i//cols, i%cols].imshow(images[train_df[train_df.filename == path[0]].index[0]])

    

'''

0 -> gatto -> cat

1 -> ragno -> spider

2 -> cane -> dog

3 -> galina -> chicken

4 -> pecora -> sheep

5 -> elepahnte -> hathi

6 -> scioattola -> squirel

7 -> farfalia -> butterfly

8 -> mucca -> cow

9 -> cavalo -> horse



'''
data_num = len(y)

random_index = np.random.permutation(data_num)



x_shuffle = []

y_shuffle = []

for i in range(data_num):

    x_shuffle.append(images[random_index[i]])

    y_shuffle.append(y[random_index[i]])

    

x = np.array(x_shuffle) 

y = np.array(y_shuffle)
val_split_num = int(round(0.2*len(y)))

x_train = x[val_split_num:]

y_train = y[val_split_num:]

x_test = x[:val_split_num]

y_test = y[:val_split_num]



print('x_train', x_train.shape)

print('y_train', y_train.shape)

print('x_test', x_test.shape)

print('y_test', y_test.shape)
from keras.utils import to_categorical

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255
from keras.models import Sequential, Model, load_model

from keras import applications

from keras import optimizers

from keras.layers import Dropout, Flatten, Dense

from keras.utils import to_categorical





img_rows, img_cols, img_channel = 224, 224, 3



base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
add_model = Sequential()

add_model.add(Flatten(input_shape=base_model.output_shape[1:]))

add_model.add(Dense(256, activation='relu'))

add_model.add(Dense(10, activation='softmax'))



model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])



# model.summary()
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint



batch_size = 32

epochs = 50



train_datagen = ImageDataGenerator(

        rotation_range=30, 

        width_shift_range=0.1,

        height_shift_range=0.1, 

        horizontal_flip=True)

train_datagen.fit(x_train)





history = model.fit_generator(

    train_datagen.flow(x_train, y_train, batch_size=batch_size),

    steps_per_epoch=x_train.shape[0] // batch_size,

    epochs=epochs,

    validation_data=(x_test, y_test),

    callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)]

)
from urllib.request import urlopen

from PIL import Image

animals = ['cat', 'ragno', 'dog', 'chicken', 'sheep', 'hathi', 'squirell', 'cow'

            'horse']

'''

0 -> gatto -> cat

1 -> ragno -> spider

2 -> cane -> dog

3 -> galina -> chicken

4 -> pecora -> sheep

5 -> elepahnte -> hathi

6 -> scioattola -> squirel

7 -> farfalia -> butterfly

8 -> mucca -> cow

9 -> cavalo -> horse



'''

cat = "https://jngnposwzs-flywheel.netdna-ssl.com/wp-content/uploads/2019/05/Transparent-OrangeWhiteCat-764x1024.png"

dog = "data:image/jpeg;base64,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"

spider = "https://media.australianmuseum.net.au/media/dd/images/Wolf_spider_-__Allocosa_obscuroides.width-800.b29a27b.jpg"

cow = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMTEhUTExMVFhUWFxcWGRgYFRcaFhcXFxcXGBUVFxoYHyggGB4nGxgYITEhJSkrLi4uFyIzODMtNygtLisBCgoKDg0OGhAQGi0lHyUrLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIALwBDAMBIgACEQEDEQH/xAAcAAABBQEBAQAAAAAAAAAAAAADAAECBAUGBwj/xABBEAABAwIEAwUGAwcCBQUAAAABAAIRAyEEEjFBBVFhBiJxgZETMqGx0fAHFMEjQlJiguHxM7IVJENjklNyosLS/8QAGQEAAwEBAQAAAAAAAAAAAAAAAAECAwQF/8QAJhEAAgICAgICAQUBAAAAAAAAAAECEQMhEjEiQQRRYRMUIzJxQv/aAAwDAQACEQMRAD8A7FSDUwKddxzE2hO1RapBABKZUwgp0gDBycVQggKQaigDCqFMVAq4YpBiVAWBUUmvVcBOAigss5wmDkIKQRQWEzJZkOUpRQWEzJSoEJ8pQAi5OCmLSmypiJ502dRhLKjQCL0+dLKmyI0GyXtEs6iWJsqNBskaig6ryT5E2VKgBvcSptcUsqeEUMmCnUE8ooLHJUZKdNCAM4MThicNUg1AEcikGKYaphqAB5FIMUg1PCAGDVMMTSU2coAlkKkGqGcpZktgEAUgELMnDk6YBISAUJSlABUkJKUUKw0p8yBKSKCw+ZRzISSKFYXMlmQ0k6CyeZLMoQnDUUFslmTZk2QpZUaC2PmTZk0JQgLHzJSlCUICxpSlKEoQGxSlKZJAWAaByRWtChCUFZlhYSQKr8olzgANSYAHiSuIPaericS/DYepSYwugVHe9Fg4svDjMxbzUyko7Y1Fs72U11knjlFo9kJdUZbvNDSYGsiyjhuNsLczsrerjAG0zYLFfJj9GrwM2E0rAqdrsG0w7EUvJ2bp+7I+K1himOAc13dcAQbEQdFos8X2Q8UizISLgsji3EqWHaHVqzKbTpmJlxGoAiSfBYbe3OBJj8yPEsqgf7Vopxfsjgzs5CQXIDttgQYOJb4htQj1DVbpdtcC6zcVS/qdl/3wq5IKZ0ySxXdqMI0S7E0AOtan9VSxHb7h7Qf+ZY6Nmhzp6DKIlFoVM6hOCub4F2roYx/s8O51R8F2UgMMCJtUIJ12larauIL8gwlb/wB5dRDPXOT8FLyRXsrhL6L8hOIVqlw63ePe3i48JMShmgLzI6SLrN/IgvZSxSYEjqmjqoYgZbw4jpB1WFxjtfgsMctXEAOtLGgueOha2SPNOOeD0mJ4pL0dBCFicQKbczugAGridGgbkrjan4l8OMRUqdR7J9vGR8kLgXbRuNxjaLaOVgDn03OPflou7KLCRIiTab3ITnlSjaCOO3R19LiRIk0y0SRd142MAIvDeL06xe1hOZkZmkQRNweRB5glBqYKWtDZaJJIEGc2vPmuI7cV63DqlDEUdJc186PBg5HcpAN9iFy488nKpG08UUtHpkp1wlH8VcC6kXxWD/8A0vZ98+DgckeLggcP/FjCPdlqU61L+Yhr2jxynN8F1ckYcWehFwUSQoYeq17WvY4Oa4BzXAyCCJBBRYVCIEJoRE2VOxUDhLKiZU8I5BQHKllRoTQjkFFKE8JpSlSMwO3+ILMBWgA5gGHoHEAnxj4rlOwHDxNXEuAlgyNto4jvkToQIv1K6X8RWzgam8OpkDn3xaN7SfJcV+H3FnnEvwsgDENztO+drSYG12zfoubPZviLvFscym82vPvP90WOp3te/TmuT43iX1nNlxcBuDIB3gQI9N1vdtOB1qDiajSWOOYuF2i8QSW2PisTgnC69chtCi55LsswQ0Tu43DbX1WMEkrNJP0af4edl24zFhlUn2bQaj41cARDSdpJE9F6x2jfh6ThmLKbGANvDWNFgByFoVjsJ2aGBoHOWms+HVHNmLaMbN4Em+5M+HkX4r8SNbFkHRjQBI0Jg/qE35OhLRkdv6rquMLszHMyMNLK7M0UiJbfYkySOvmud/LmdkjWIty9UvzK1SpEsg9pGqEXJ3vJ1UQExDKTUgEVgiEAe6fg/wBjvy9MYuqP21ZndB/6dJ0GI/idAJ5QBzn0stAC5D8MuLuxHD6T3ucXtzU3Odq4tPvdbELpsVUgADXTpfmueUtuzVKydSsG3nVVHVJJmyfGs0byXmv4jduWU2VcHRk1SMj3jRgPvN55otbSeainJ0VpKwvb7t8MP+xwxa6ofefq1l3COrrLxbHPz1DUdPfcXOjWXGXRPiYlWiZvqhRzXTGKj0ZNtkOI0GNINOoHtcJGuZpFiHgixWr2e4pVwzmV6ZGdk2OjmmQ9ruhEiRcLPFMAiBufkmt7t8x0vYXnzsSmlqhH0dwPGNr0adZnu1GhwnablvkV5z+NnFc1Snh2/wDTbmebxmfo2NLNvP8AN4z0n4Q4s1MI6i4jNQNhzY8lwPUB0j0XlH4g8R9tjq7h7rXll/8At9w/EFZRjUi29AOBYBgdTdi89KhUu2plcWuE82yQOsbLS4H2aOLxLmUGkUS8nMZ7lPNIBJ1OW3Ur1/szhw3BYalUa0ltCmHSJE5BIWrRyMEMa0Dk0AD4LoWLy5WYvJqguAwzaNNlJghjGhrR0AgSj5lVdiOSH7ZxWxnRezJSq7Kh3KcV+aVhQeUi5CNYKvXrTomFFyUpWc2qQpfmCimIJCeEIvPL4p2u5qbGUe0dMHDVQ6IyHUTHWF59+DXAX1MRTxQb3KLagLiO7nc0tDQd3Q4m23iF6PxVhdRqgCZY63Oxsrf4e8JOE4fQoOEPDc7x/PUJe4eUgeS58ro2x9G7UaHNggGdtVncR4hRw1M1Kjgxgubbk8huStJ1Fo0G8+a47th2bdiwAHZe9mzROXXa0+u65n3s2M7hf4lMxVY0WsNJkOhziAXGQGiNtz5BcZ2+4E9tV1VlN5Y7vF+oa4kzMaDTVZ2L4DVHEHYbCONStTZ7Q6N/aMGYtG27BJOrrmy7bs23G4/DcQp1XNDm/sG0wPdqU7uBdMQXC5k6naFpVO0T+DxmuyCVAU5W1xrhhpVXsJa5zHFpLTLcw94A2mDI8lmAQVuQANFLIrUSZNgEFwsXbIAGQiMCE0o9IoA3Oz/H6+FqNfTee5mhpJy94d63p6L3bs5xc4nCU6z3NDnNl8bG/PRfOgcrI4jUDDTD3Bh1aCcp5SN1nOPIqLo+iuJ8apjDVarHtd7Om90gg/6YJ26hfNdYFxLiZJkk8yTJPqtCljXta4NcQHNLSATcHUKkw3ITjGhN2V6dnQlVEDzRK7e9I5JPEt8fv6KxCH6n4wqrnXnfVWcw+fyEK1wrhZxEsptJqhzI/hc17msLTs0tJDp3BdyCV0B6h+BzSfzFWDlyU2TsXS4kDwAB/qC8y7e4F1LH4phGtV7hbUVO+IH9S9n4HxLC8NonBB5qVKRms4CJe4NJIHKMoAvYdCuY7W1MJj8VQdTJbiGua73TDmMe0lrjod4PUqIvyKa0dzRoZWNbN2tA9AAp5eqmGlIsPJddnPQM+KaeqJkKcjonYgQcead3ipZeicMHJFgDzdVLzU46JoHJKx0QLUsgTnoEydiJB3gmz+CrseCAQQQbgjQ+CkpGHcMwLYBm0TYzaF0wXJ0q7W1KYJu57QBvr8uq36NWMwnlHmuXP2bYug1apsq9XEtECboWIqS4a+PVc5+Tqvq2qODR3jGhGupXM2bpHQcL4DQpVquKYyKtaM7pNwBEAH3RaTGp12Wf264ycBgq1ehTaKhIiGiM9Rwb7R8awL31IAW7hXj2bYOlr2k7rkPxKxFb8jWbSbmc/LTgCTlqPDHRyMO121VJ7RLR4mzGAsDnPzVHOMty6Cxzl03JJNgNj0Q69DldV+N8MqYSsaNUQ9rWE7jvNBMHeCSPJQw+OLeq6l+DIJU0hBxGgCsiqx15gqNTKfBMCrSI0Kk5vJO6mNkN7kAGc7ZBqVUiEXDcPqVn5KTHVHxOVovAIH6hADYfERrorTHDX4qk5kaouGdFigAuKrEaH4KocUYhRfPWFCL2ugAtN+/UH4FanZniZo4yjUBjvhrtYLSYIt5ecKlxXAPoFjaghzmZy3dslzQD1tKHwnhz8RWZSZ71RwaDym7neAAJ8ktNB0egdp+E18dVdiMHQOV7zmc5zWnO2JIzG7evOQj9iexWJp1fbV8tPYNz5n6zMN7u1r+S9GweGZSY2mwQ1jQ0DoOfVT9q3mI8R97H0VRhS2Q5sstqwmNboqQxtPOGZxmNwAeSsK6RNscvKZpKSSoROU0qMIONxbKVN1SocrGAucTsB8z03SAPKRKpnF1HYeniKNCo9tQAlhhlZsm3cdZ3Oxk2gGVzvC+3tCpXfRqt9iWe0BdmLmk05LhGVrxZp1bNohSporizrZTqNN4cARoQCDzB0KkqJPKcLSqUhlZVcBe5kASNp0norg4jiKfu1iQOcRG1juj+yd16Xup16eYzEWA1OwA+P6leV+7Zv4fZb4LxSpWxdEOg5STIbBADTOmv913Jqkzl13+q47szSP5gbBoJPK8AD5n+ldw9gayRAm5hVzclbLil6MnHYhx950Bt4bv0uuU4j2rqspn2QAcSTmNwAPdA0na6nx7El9QtBMD3r+gWDiu9mYdojrafqpT3Q+SuifAOOYirUzVsVUADgTLW5QXE5QRECYMeC9JwuPa8DM5rp/eFvUbryGrgzBiQDBI2JEwSN4k+pW7wd7mMaZMgeSJyS2Dko9mD274kzFVTRrU20MRSqPaKpn2ZoycgfAJ0vIB0tqVw7qVzBkTY3vyN7r2PH8OoYwsfU7tamRlJAIcBcNdzHxuqXsadN4LsPQzD/ttg7yLLaOeKWiVHl0zy3EYZ9MgPEEgGJuJ0DgNDF45EIbXkL1LH4HCYh+epQh5uS17mZidyAbqrj+zuFc0NFL2YF8zT3/Mmc3nKtZ4g4M84NQrp+x3ZSpi3B74p0AZc9094DVtMau5ToPGyuMwGEoulrC8jeoc3h3bN+C1sNx86SfCbAJTza8UEYr2aFX8O8K/L7N1X3iXHM24JNgC20eq6/sj2VwmFLfZCXOIDnOMuN9JOg6CAuUpdpwLTAV7AcdpvcHCplcCIBOp28VzSyTfZslFGf+L3ZOlQqMxVMQKri1zLZQ8CQQNpA06SvP6PDM72sYC6o8hoa3Ul1gAvdMdgGcUNAOeB+XxDK1Smf3mAOGUjcH01ldEeC4anVdiG0KYrO1eG97SDHK3JdanqzCtnhuP/AA3rNoteajRWc9jfZgDKxrjBLnzqJmwixElEHYhtDiIs84em0VRn3dfLSDhrBgz06yu37e4wtDQN3DeJsSq/B2fmDmrO0i07BuUfCfCOqhzkOtmPV7C/8Q9pUHcc0ZWGZBIvlPMddpWNwHheI4fiM9WiGkMcxrS4Ey6JcMpI2jzK9Pp8SZhsIzIO9UzFvqe8elrLjcXNR2dxJJJN+Z3P3ssv1+GgnxvY9ftNXdLQBBESBp9NVkVcXWdEAkQGg+G3UXKvNw4m8eV7+CLUpSbF06To0DoApfyrJ8DPp+0BDm2cJMzcEzeYt08Uajj69K4nNe88xfX70R24ex7zvQ+WqZ2EBi7j4pL5TQ3wIs41iotUMxF9gSk7jOKMftiIbFo9Ta/inbQbP71vC6k+n3XDNExGgi99E/3X5F/GAqcbxOWDXdpB7o6CBzuNbfFVqvbJ+Fa6ZrPLAKYcf2bCHg53DU6WAI3uFdpYDPlYHAucQ0WEXtB6yQuN43w91XFPw9J7XmnDSRZvds/LMyA4xOpLtFtiyc+2N8f+T1LhnbQYzBZq9MMdUzjLJLarGZQ9zdwMzi2DexidvOuG8Kea+as3MBSe3M4hxeHhzW3BlxFM5b36xlXbUsLTOGgAgUAxlJx95mZrmOeIs6ReDIkzsCsylw8sBc10k5RJOgBBdr0SWaK6E3ZE8UxTMgD82Roa20GBFu7c6Ac1pYbthVY0NqMJeNTlg9JFot0VNmHMznMjQ6eWvOPRN+Wcbmpc+aiPy67G1B+yTCZvPja/ojPb4xHNCB3j0hDfSzG5PyXBZgdJwrDezLRuTJO8+6AD0zRtcHmug4hiMlIk6wdSsfhNNopteQemskxfrqSqXaXFyfZg3MSBF+fyXYmuJ1KkjB9tOZ5drLj+t55LncNiicRJ/e2na/1WxicC4tfkJzEWl1p5R1iFR7K8HfWqtdExBO0QbA8p08jyTwNNSZjHbN2rggMOajjrZoE6zEnpKakRAEiwH9tFq9qaIDqTNI7wEWIAjn11jZY7m8z5QsMr9Ble6CNqiLO9D9VJ9xe/mgjaRf4pVHPGkTyIWSdEJtFXE0Kn/SewE/xMB9CPoqtThVZ5l9YHyt6SLK97St09PUckYZjsD9/WVf6kkPnIy29nhvVPk0D5kqxS4LTGuY+Lv/zCute6Ls8e9HzUms0ufn96pPJN+xcmNSwzW+4GDrv6qbm5rGD5J/XVN7I/5U2Kzo+xtQDFUzDWnK5sie+0icpGnvNaQekbr0HF3afAryjhTor0i0Qc7IM/zDZeuFsgrt+NJyi0axlZ5Z+J5LH0oEkmPRrlg8LqVAc7oAgiDy+a7jt/Qn2b+R15WIPwXHgO3NuUQoz5HF8UKU2mWMTjc/ecR/CBaABoANh0Qg220Jw075fS6URu3bbbpzXK3eyLGdTG4FuiWSdvQfqhNc3cg+EqXd5umOcD78UhEzQGsD4kqRIG3wshwNb+spZR19UgJlmiiL2/UbeASkDWbeaRbvb/AMfuUwD4Or7N7XalpzDTUXbPnCw+AcDFA1HPPtXvgzAAEXGpMmb+Q5LUNMi5jxIAJ5aKbQNiJ9VcckoqkOyYAgkgyIy3EDnMaqNWqIAHnJjWdPgnnmQm9o2dRPgpU6FyB5jaACOpHwTPB/hHok6s0ch1v5KTHgiZn0+qVhYAsm8E+BP6J3Ni3y+V1B9E7AD0S9k6NYPQR5W6WQB2PDb4elv7xPhPzgArkcTVz1XvM3MAWNgvQ6uDGHwZNiWs16wb+YXnLaRvzj+2y6cvjFI2m9JEg7oZ5/pot3sn3WVsoE5i71aCZPr6rDDDry+HLVanB6xY4g/vAGN40I+Xos8TqRGN0wfF6zn1ZI91pj1B8tVmVhUOkep89F0PaGj3Kb/5nMO0cv8Ab8Vz5N7g366c5+Ceb+wsnYmtdzN+h35RqnFA7k77eep0RGxc315wLJBw2tr0v4rEkiwOkifX5KbmHS41587qJn+/0J0UTyk+vrvcoYEX0js6B1H+FN7AR753Gv0UKlEEQZP3/hRFOBAbPptHPxQIj+WJj9pPKNeWsojMJ/MfCU7KRG4B6DQff3unFMmO8frvHTdFgbHZeiDiqOY2zT5tBLfiB6r1KpWDWyTAC8q7PU/+ZowbZ2nxgz+i7/FVA9xoHV7HOH9JaD/uC7PjOos2x9HK9r+MMc4U2kOLnCY2sT9+K59zv1v4KlxThdenjh7UQ0BzgZtaWjz73wVsvB9dr+Kx+Q/Jf4Tk/sOW89Uz6Ddx0k/fkgvqSbNnnY7i33yQX1n6AHQfQ+CxILDGtPXadrdU7GN1geuqhSYLd0AifAWRWstoPKPNICPsdLGD580gwb/fgnNQ9fvw5fqgvG8kXnSRG8oEEeGxYhvWEIOaCM1QeimQ7WDryupNZl0yjbQSmMi4gj3vhbUKLMLqd+kBFLmg6jlJjl96qWe0A9dUrERFIjV30PxTS3Vzx0GydoEa/wCFAUGiA3xPl5JjCNaCYtfpb4JPw4/hB9f0CgzDQPeJPjZTNPnf1/RAUis8EjWPrJ56KxwvD5qtNneOZ7R6nX0+SrhoscwsOc8/vzWjwusGVG1TcN73kYFh0Dk0lexrs7ntriAMO9o/lAjUyYXnDKhB38Vucd4t7d1JrZ/1Gz4D/IXNCu42AHPW4tYfP0C3zyUmqLm1ZbqgnQi88tBzReGPirTJ5hpk/wAVv1lU21XETGztL+gTtc7UNvYySBHUTp98lgtMhM7bjGFP5Utf7wuDJ2Nr+FlxjacHU+s+Nyu0ZxBuLoEHuug90kZgdjY35rgsfW9lUNMg5gQLczpP3t4rpzR5JNGmRe0XIuAXONuduiWTYX8pWazF1J92NB63ifCUatWqaBh6G8WtBPhC5uLMi2en+PTVSIHPlOnwVQU3kXgGNI0tYXKBUwtT96oBJ25XmPghJfYGiWxv+uyHk6z57fJV6IA/fm3P5j0KLngHUkn+w+SQBvZm1z5Df/JUmmO6TJjeNT180H8zLT3STeY+Ik9Aq/tCXRlJ38QJAA+OvJFNgdb2Co58Tmm9NjnHlLhlHnDvgtp+KjidBpOrag9Q79QFidgcc2n7Vo/1HlrhOrmiRbmATp1CBXxx/wCJU3EyGuaCSIiSZj1C7MfjBf6bJVE2/wARqOUU6nUi3ULiPzw0vPKLrsPxKquNOmR7rHSfCLR5wPNcM0RuLbjXclRnS5WTkTTJO4sIsDE/L/KpV+MO2gC4i8o5gdRNhHL+5U3jQtaL7nYC8LNcV6Myg3ir2677ESLn4beqMOJv1I+PKZ081KnQeZz5QOYk6Tp6oNem0GASW3AETcTM+uivxfoRL/iVSTc8vC5sEMYqpM+0P3qVCrhnEmGcybkX2j73VingxGYm19j5JvglYUQ/P1YjMLffJFbjXzcA+Sl+VZItJ3kHxj0RvZEu5Tc2iRMLNuHpBRTxFcE3ZcdXAa7qJ4g5oADQL2v8VpeytEQJ3dJnwOqFUwOgzdJ2G+/r5oUoewKgx51LW7HUkorOKu/hbb1R3YNseM/d/BDbwpgGrp5fqE/42Am8UdJzWEbHdTHFhvKI3h7Du7SQY6pmcOYBck+tuilqAUydPBgES6edugv0iCreEaO8BuWtE7TcW/p+CxqlWq4uDDGbKS4yALTvfSLbTdXW1hSaYLiXEEE6lwOUQLDV0ocZeyqLgqNY4jTKH35y0gehj0QDXzGBcnlfeRfwn0VClVJa5x94kiL30dPWwddEYKhkaEw2dBcwQDvyP01XD7Eab6wBEX2F9/pEeM9VXrVRldmA63nSJ+NvP0zxmIgOgC8m8fXbx80WqwZZmTLSZH0P8Rny6J8EOi77Vg089tYsOt9U7CC4ktgyDePGellWrYkQNzN7DcknX7une+1+pMWMRMfGJ6pcWAZ8WsD3pB201Hx9EOpjGwRYRJPXSw9RfqFn1MTnkXBkX5AbW8rW1vpIqkuktANr7aBwk21Mq1i+xUadHHNJINi0Eug+7Bm/M3+EbWtONNwMHoPrdY1aifZhoHv2LuRJEA+Ji5WhSwvswMzhOYibzNha9vDoiUI9oVFpmHbYC+8nXnfp9EJtEyIyk+gAE7C26A0CQWmACc1ydTYddhCelWLSJMmb6TO/6HxUOL9AX20I3J+qE+s1gMyLHQbHU/FCOMdJbIsBeREkAhp9QZ6JnPIaZtckiLwJkdYPrCng/YE8PxVodmY6CP3rcz9VYxBzmXHM6Tob/uqnQZTnOYueRM2u375FXKeJEzaLOm03AvPh+ibtdByZru46X0W0nMBgNaSd8oGvLRYbqTZJ90EyBykxtJ+7KXtxzOu3OJHluoNrEkjLpFzyOnx+SHKT7Byb7FUDReJ89SLkxCPSqAnSw6bTG2n9kJ9U+kDS339FL8wYNoy38RmAJ9TopoQZzQf3RF+fLTpqVCqA27W6XgD0k7+XVDo4k3I0aYA5QYcrHtIEwADtN+TvK49UUx9leq14sPemNLCSfhyhEyuymwmwIjS4vHr6KDqw3MgkA9Tt63T1cX3g1upbmsLW58h9eqOxUNTLpbmZ1toNJnysjClzHXWTbSPvkguxJyh8k7EfE/ohsqwIPI2jw73yRQy49m0xr7u5Hjt9FH2R1kGw2B6E/P1CB+ZvE8o6nT0+7pCuN/DnpvHigLCHB7+niPnopGlsJBMX+/u6hTxANpNvmNh5/d0R9rzO2xAdYR1iNtUbGM1pJy3sDIjQDX4ITMOTq0D+r+yLTLjJkWjSDvz5GyNTqNv0MXn70QCMbBOL9G5XQfeJO9iREtEz6HSCEOoM0CQbmCTBvYFxm3hzHRWyYZlFgGg23/1CZ+91mYqrlyWBzCTM8zpBH2At+3oqyy2tTYBSBzGSS4id7STMQCRz8VGkQ9wucsAkkRYWmNpOUb6qnh4L2CAA4vcQJuW5o3W0aIblaNPe6mRnjwn4IkqY6vZSqvaC9pMkS4taOWxItt49E+CLS8TOrbZogWsTF9XHaYCoU6p9rH8VR02EmZBE67fFaLcO1pe0WuzSAfd6eJCckkgoY1gIIAMi28iAbReDG2pKjXrEjILzd2XQC3M8tB0jwpVAMhsIYyAIkGYuZ1MW8lcB7wGxAGg3AM/FLomx2NaG90mZEZZLgNt9ry6Od+TOzFpLGNHe1OYm0m2pjQ9Y9W4e6bneLbWaD8/kgYiqWDM2xsfMnKT6E+qa7Gy/RrezEOu4hrSJ70jQRGvunoq9S/vTpMHmCT5E3+7JsHTEZjc5ZvzIBPzVzGCGsO5qZdBZsZoHmlLTEyvSb3REkuMRoSXQ06cg43tHLkQBrRmI92CbmxgWAJPzUa9Qga6Aj7jxPgm4dSEOH80anQgSLo7TYBCyXaEE7k3I3sBbRSFVtibe9JIvl0J111t0TU3RPUj4GyGKQLL7t/8AtFv/ABCTE2QOKdlzBtiZE6g6iOmqkcUbzEQSNIIsBI8TH3CNiKAblAJu8t12D3ARy0CyqRsX6k5PIEusOQECydIRZ/NEX5bnWxEW63gdVOliHDMddwI1J0g77p6Yl7m/wwAbT7pMnrPyU6lPMSCdATtJ8THQeiKRSjYOi/M2SfEDwIIB5W+KuUKLtOYg+AIBmDbn5KOHbd7RYAN06uaDr0R6lgWjQmTzJEaqJBRFpi7Y97uzpB1J5mRKHi6wPswTbL5GTEnlPdT03d1pt7xEbQc5TR3gf5m/EhJaJHdBcQQIkAncZbGdtTte8JGu0WbIbe8TLhBBMzaR4eO4Krs5vvndYnUGx62MXUMEA7NmAJZ7p5Hu38blPiBdrQ1ga2BE2gk9JAuPBU62MDiYAgd3MSB4yeg226qGOF2XIHdsCYuL2329FWdhmikHic2UumdCXsaY5b9bq4RXbGlYWhjGZXEGR5yTI0v6c1bJDmgDew6kybeJj0WdQ9e+BfvEgO3zTPNbPA7wTe5PnYfKyJxSE0PQwoDoBvEjoeRG1p84UxSdD2tP7s+B+YM/eik0w9oFrn4kC/oD5KRMknd4g/8AwEjrBWQFOh7zMp1LZdty+cfFExEE3qNbymZP81uZlHp0xccjl8jJPxAVrD4Rr5JtDi23IWCdItQs/9k="

horse = "https://images2.minutemediacdn.com/image/upload/c_crop,h_1194,w_2121,x_0,y_34/f_auto,q_auto,w_1100/v1553786510/shape/mentalfloss/539787-istock-879570436.jpg"



test_images = []

for animal in [cat, dog, spider, cow, horse]:

    img = Image.open(urlopen(animal)).convert('RGB') 

    img = np.array(img) 

    # Convert RGB to BGR 

    img = img[:, :, ::-1].copy() 



    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    #resize

    if(img.shape[0] > img.shape[1]):

        tile_size = (int(img.shape[1]*256/img.shape[0]),256)

    else:

        tile_size = (256, int(img.shape[0]*256/img.shape[1]))



    #centering

    img = centering_image(cv2.resize(img, dsize=tile_size))



    #out put 224*224px 

    img = img[16:240, 16:240]

    test_images.append(img)

        

plt.imshow(img)
a = model.predict(img.reshape(-1, 224,224,3))[0]

a.max(), np.where(a.max() == a)[0][0], animals[np.where(a.max() == a)[0][0]]
test_images = np.array(test_images).reshape(-1,224,224,3)

something = model.predict(test_images)
for pred in something:

    print(

        pred.max(), np.where(pred.max() == a)[0][0],

        animals[np.where(pred.max() == pred)[0][0]]

    )

    print()
a = something