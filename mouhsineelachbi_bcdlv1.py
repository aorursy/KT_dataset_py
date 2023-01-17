!curl http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz -o all-mias.tar.gz
!ls

!tar xvzf all-mias.tar.gz

!mkdir all-mias

!mv *.pgm all-mias
no_angles = 360

url ='/kaggle/input/mias-mammography/all-mias/'

def save_dictionary(path,data):

    print('saving catalog...')

    #open('u.item', encoding="utf-8")

    import json

    with open(path,'w') as outfile:

      json.dump(str(data), fp=outfile)

      # save to file:

    print(' catalog saved')



def read_image():

    import cv2

    info = {}

    for i in range(322):

        if i<9:

            image_name='mdb00'+str(i+1)

        elif i<99:

            image_name='mdb0'+str(i+1)

        else:

            image_name = 'mdb' + str(i+1)

        # print(image_name)

        image_address= url+image_name+'.pgm'

        #print(image_address)

        #print(image_address)

        img = cv2.imread(image_address, 0)

        # print(i)

        img = cv2.resize(img, (64,64))   #resize image

        rows, cols = img.shape

        info[image_name]={}

        for angle in range(no_angles):

            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)    #Rotate 0 degree

            img_rotated = cv2.warpAffine(img, M, (cols, rows))

            info[image_name][angle]=img_rotated

    return (info)



def read_lable():

    filename = url+'Info.txt'

    text_all = open(filename).read()

    #print(text_all)

    lines=text_all.split('\n')

    info={}

    for line in lines:

      words=line.split(' ')

      if len(words)>3:

          if (words[3] == 'B'):

              info[words[0]] = {}

              for angle in range(no_angles):

                info[words[0]][angle] = 0

          if (words[3] == 'M'):

            info[words[0]] = {}

            for  angle in range(no_angles):

              info[words[0]][angle] = 1

    return (info)





def training():

    from sklearn.model_selection import train_test_split

    import numpy as np

    lable_info=read_lable()

    image_info=read_image()

    #print(image_info[1][0])

    ids=lable_info.keys()   #ids = acceptable labeled ids

    #print(type(ids))

    del lable_info['Truth-Data:']

    #print(lable_info)

    #print(ids)

    X=[]

    Y=[]

    for id in ids:

        for angle in range(no_angles):

            X.append(image_info[id][angle])

            Y.append(lable_info[id][angle])

    X=np.array(X)

    Y=np.array(Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

    (a,b,c)=x_train.shape  # (60000, 28, 28)

    x_train = np.reshape(x_train, (a, b, c, 1))  #1 for gray scale

    (a, b, c)=x_test.shape

    x_test = np.reshape(x_test, (a, b, c, 1))   #1 for gray scale

    cancer_prediction_cnn(x_train, y_train, x_test, y_test)

    

def cancer_prediction_cnn(x_train,y_train,x_test,y_test):

    import numpy as np

    from keras.models import Sequential

    from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten

    from keras import optimizers

    from keras import losses

    rows, cols,color = x_train[0].shape

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(rows, cols, 1)))

    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))

    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=40, batch_size=128)

    loss_value , metrics = model.evaluate(x_test, y_test)

    print('Test_loss_value = ' +str(loss_value))

    print('test_accuracy = ' + str(metrics))

    #print(model.predict(x_test))

    #model.save('breast_cance_model.h5')

    save_dictionary('history1.dat', history.history)

    

training()
!ls