import numpy as np

import os

import glob

import re

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from random import shuffle

from skimage.util.shape import view_as_blocks

from skimage import io, transform

from sklearn.model_selection import train_test_split

# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential   

from keras.layers import Dense

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD

import warnings

warnings.filterwarnings('ignore')







train_size = 600

test_size = 30



train = glob.glob("../input/dataset/train/*.jpeg")

test = glob.glob("../input/dataset/test/*.jpeg")



shuffle(train)

shuffle(test)



train = train[:train_size]

test = test[:test_size]

















 









            

























###################################################################################









###################################################################################









################################################################################################################################################





    



    
###########Functions###########%%%%%%

def fen_from_filename(filename):

  base = os.path.basename(filename)

  return os.path.splitext(base)[0]





def img_cut64(img):

    temp = []

    for x in range(0,8):

        temp2 = img[:,(x*25):((x+1)*25)]

        for y in range(0,8):

            temp.append(temp2[(y*25):((y+1)*25),:].flatten())

    return temp



def rgb_to_grey(img):

    temp = img.mean(axis = -1)

    np.where(temp>128,255,0)

    return temp



def take_img(rgb):

    img_read = rgb_to_grey(io.imread(rgb)[::2, 1::2])

    return img_cut64(img_read)



def boardmatrix_to_fen(boardmatrix):

    string = ''

    for line in boardmatrix:

        count = 0

        for j in line:

            if(j == 0):

                count = count+ 1

            else:

                if(count != 0):

                    string = string + str(count)

                    count=0

                string = string + piece_decoder[j]

        if(count != 0):

            string = string + str(count)

            count=0

        string = string + '-'

    return string[:-1]   



def decode_fen(str):

    temp = np.zeros(64)

    strlist = str.split("-")

    for i in range(0,8):

        board_line = fen_to_board(strlist[i])

        for j in range(0,8):

            temp[((j*8)+i)] = board_line[j]

    return temp.tolist()



def group_to_color(y):

    y_c = np.zeros(len(y))

    for i in range(0, len(y)):

        if(y[i]>6):

            y_c[i] = 1

    return y_c



def clear_extra(X,y,gap):

    count = 0

    indexes = []

    for i in range(0,len(X)):

        if(y[i]==0):

            if(count==gap):

                count=0

            else:

                indexes.append(i)

                count = count + 1

    X = np.delete(X, np.array(indexes), axis=0)

    y = np.delete(y, np.array(indexes), axis=0)

    return X,y



def fen_to_board(str):

    temp = np.zeros(8)

    index=0

    for i in str:

        if(i.isnumeric()):

            index =index + int(i,10)

        else:

            temp[index] = set_chess_piece_index(i)

            index = index+1

        

    return temp



def set_chess_piece_index(c):

    val = 0;

    if(c.islower()):

        val = val + 6

    c = c.upper()

    if(c=='K'):

        val = 1 + val

    elif(c=='Q'):

        val = 2 + val

    elif(c=='B'):

        val = 3 + val

    elif(c=='N'):

        val = 4 + val

    elif(c=='R'):

        val = 5 + val

    elif(c=='P'):

        val = 6 + val

    return val



piece_decoder = {

           1:'P',

           2:'K',

           3:'Q',

           4:'B',

           5:'N',

           6:'R',

           7:'p',

           8:'k',

           9:'q',

           10:'b',

           11:'n',

           12:'r'

        }
X= []

y= []

for a in train:

    y.extend(decode_fen(fen_from_filename(a)))

    X.extend(take_img(a))



X = np.array(X)

y = np.array(y)







X,y = clear_extra(X,y,2)

y_p = np.zeros(len(y))

for i in range(0,len(y)):

    if(y[i]!=0):

        y_p[i] = 1

        
#LOGISTIC REGRESSION TO SEPRATE COLOUR BOX CONTAINING PIECE FROM EMPTY

from sklearn.linear_model import LogisticRegression

piece_detector = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X , y_p)
X,y = clear_extra(X,y,100000000000)

y_c = group_to_color(y)



#LOGISTIC REGRESSION TO SEPRATE COLOUR BOX CONTAINING PIECE FROM EMPTY

color_detector = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X , y_c)
y_t = y%6







#Initializing Neural Network

type_detector = Sequential()





# Adding the input layer and the first hidden layer

type_detector.add(BatchNormalization())

type_detector.add(Dense(output_dim = len(X), init = 'uniform', activation = 'relu', input_dim = 625))





# Adding the second hidden layer

type_detector.add(BatchNormalization())

type_detector.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))





# Adding the output layer

type_detector.add(BatchNormalization())

type_detector.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu'))





# Adding the output layer

type_detector.add(BatchNormalization())

type_detector.add(Dense(output_dim = 6, init = 'uniform', activation = 'softmax'))





sgd = SGD(lr=0.5, momentum=0.03, decay=0.2, nesterov=False)



# Compiling Neural Network

type_detector.compile(optimizer = sgd, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])



# Fitting our model 

type_detector.fit(X, y_t, batch_size = 100000, nb_epoch = 30)
for a in test:

    actual_fen = fen_from_filename(a)

    list_square = np.array(take_img(a))

    

    y_p = piece_detector.predict(list_square)

    list_piece,temp = clear_extra(list_square,y_p,100000000000)

    

    y_c = color_detector.predict(list_piece)

    

    y_t = type_detector.predict(list_piece)

    y_t = np.argmax(y_t,axis = 1) + 1

    

    boardmatrix = np.zeros((8,8))

    index = 0

    k = -1

    for i in range(0,8):

        for j in range(0,8):

            k = k + 1

            value = 0

            if(y_p[k]==1):                

                if(y_c[index] == 1):

                    value = 6

                value = y_t[index] + value

                index = index + 1

            boardmatrix[j,i] = value

     

    print("Predicted Fen:", boardmatrix_to_fen(boardmatrix),'\nActual Fen:',actual_fen,'\n\n')   