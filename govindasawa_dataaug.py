# All the Required Import statements

import gc

from tensorflow.keras.layers import Dense, Flatten, Conv2D,add, MaxPool2D,AveragePooling2D, Dropout, BatchNormalization,concatenate

from tensorflow.keras import Input

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image

import pandas

import numpy





# Collect Unrefereced Memory and Free them using gc.collect()

gc.collect()



def convert_data(data,name = 'train'):

    data.index = data['id_code']

    data = data.drop(['id_code'],axis=1)

    if name == 'test':

        return data

    data = data.astype("category")

    data = pandas.get_dummies(data)

    return data



# Hyperparameters

url = '/kaggle/input'



shape = (224,224)



train_data = convert_data(pandas.read_csv(url + '/train.csv'))

test_data = convert_data(pandas.read_csv(url + '/test.csv'), 'test')





# This Method Splits the data depending on which set you want to choose from (i.e, train or test)

# and giving a start and end point will give you the data required only upto that amount

def split_data(name, start=0, end=-1):



    if name == 'train':

        data = train_data



    elif name == 'test':

        data = test_data



    else:

        print('wrong split name')

        return None



    if end > data.shape[0] or end == -1:

        end = data.shape[0]

    

    x = [(Image.open(url+f'/{name}_images/{img}.png')).resize(shape) for img in data.index[start:end]]

    x = numpy.array([numpy.array(img_obj) for img_obj in x])

    

    if name == 'train':

        return x, train_data.iloc[start:end,:]



    elif name == 'test':

        return x



def renet_block(x,filter_size=3,filters=32,stride=1):

    conv1 = Conv2D(filters,kernel_size=filter_size,strides=stride,padding='same',activation='relu')(x)

    conv1 = Conv2D(filters,kernel_size=filter_size,strides=stride,padding='same',activation='relu')(conv1)

    res = add([ conv1, x])



    return res

    

def incep_resnetv2(x,filter_1x1,filter_1x1_3,filter_3x3,filter_1x1_5,filter_5x5):

    conv_1x1 = Conv2D(filters = filter_1x1,

                      kernel_size = (1,1),

                      strides = (1,1),

                      padding = 'same',

                      activation = 'relu'

                     )(x)

    

    conv_1x1_3 = Conv2D(filters = filter_1x1_3,

                      kernel_size = (1,1),

                      strides = (1,1),

                      padding = 'same',

                      activation = 'relu'

                       )(x)

    

    conv_1x1_3 = Conv2D(filters = filter_3x3,

                      kernel_size = (3,3),

                      strides = (1,1),

                      padding = 'same',

                      activation = 'relu'

                       )(conv_1x1_3)

    # we apply the fact that 5x5 conv is 2.78 times more expensive than 3x3 conv

    conv_1x1_5 = Conv2D(filters = filter_1x1_5,

                      kernel_size = (1,1),

                      strides = (1,1),

                      padding = 'same',

                      activation = 'relu'

                       )(x)

    conv_1x1_5 = Conv2D(filters = filter_5x5,

                      kernel_size = (3,3),

                      strides = (1,1),

                      padding = 'same',

                      activation = 'relu'

                       )(conv_1x1_5)

    

    conv_1x1_5 = Conv2D(filters = filter_5x5,

                      kernel_size = (3,3),

                      strides = (1,1),

                      padding = 'same',

                      activation = 'relu'

                       )(conv_1x1_5)

        

    concat = concatenate([conv_1x1,conv_1x1_3,conv_1x1_5])

    return add([concat,x])



def create_model():

    input_layer = Input(shape=(224,224,3))

    

    x = Conv2D(64,(5,5),(2,2),activation='relu')(input_layer)

    x = Conv2D(64,(3,3),(1,1),padding='same',activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

#     x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    

    res1 = renet_block(x,filters = 64,filter_size=(3,3))

    x = Conv2D(72,(3,3),(1,1),padding='same',activation='relu')(res1)

    x = Dropout(0.2)(x)

    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    

    res2 = renet_block(x,filters =72,filter_size=(3,3))

    x = Conv2D(96,(1,1),(1,1),padding='same',activation='relu')(res2)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    

    res3 = renet_block(x,filters=96,filter_size=(3,3))

    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(res3)

    

    x = Conv2D(128,(1,1),(1,1),padding='same',activation='relu')(x)

    incep_res1 = incep_resnetv2(x,64,64,32,64,32)

    

    x = Conv2D(192,(1,1),(1,1),padding='same',activation='relu')(incep_res1)

    incep_res2 = incep_resnetv2(x,64,64,64,64,64)

    x = BatchNormalization()(incep_res2)

    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    x = Dropout(0.2)(x)

    

    x = Conv2D(256,(3,3),(1,1),padding='same',activation='relu')(x)

    incep_res3 = incep_resnetv2(x,128,32,64,32,64)

    

    x = Conv2D(300,(1,1),(1,1),padding='same',activation='relu')(incep_res3)

    x = AveragePooling2D(pool_size = (2,2),strides=(2,2))(x)

    

    x = Flatten()(x)

    x = Dense(1000,activation='relu')(x)

    x = Dense(500,activation='relu')(x)

    x = Dropout(0.2)(x)

    output = Dense(5,activation='softmax')(x)



    model = Model(input_layer,output)

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    

    return model

    

def model_fitting(model, batch_size=250, epochs=5, validation=False):



        end = int(train_data.shape[0] * 0.9) if validation else train_data.shape[0]

        data_generator1 = ImageDataGenerator(rotation_range = 90)

                                            # shear_range = 0.2,

                                            # zoom_range = 0.2)

                                            # vertical_flip = True,

                                            # width_shift_range = 0.25,

                                            # height_shift_range = 2

        data_generator2 = ImageDataGenerator(horizontal_flip = True,

                                            shear_range = 0.2)

        

        for j in range(0, end, batch_size):

            last = end if (j + batch_size) > end else (j + batch_size)

            

            x,y = split_data('train', j, last)

            

            print("without DataAug")

            model.fit(x, y, batch_size = 50, epochs = epochs, shuffle = True)

            

            imgs_augment = 30

            print("applying DataAug")

            print(" "*2+"DataGen1")

            train_gen = data_generator1.flow(x,y,

                                            batch_size = imgs_augment)

            

            model.fit_generator(train_gen,

                                steps_per_epoch=x.shape[0]/imgs_augment,

                                epochs=epochs)

            

            print(" "*2+"DataGen2")

            train_gen = data_generator2.flow(x,y,

                                            batch_size = imgs_augment)

            

            model.fit_generator(train_gen,

                                steps_per_epoch=x.shape[0]/imgs_augment,

                                epochs=epochs)



            del x, y,train_gen

            gc.collect()



            print("(" + str(last) + "/" + str(end) + ") images have been Scanned!!")



        if validation:

            validate_x, validate_y = split_data('train', end)

            model.evaluate(validate_x,validate_y)

            

            del  validate_x, validate_y

            

        gc.collect()



        return model



# Submitting the predicted values can be done using this method

# Since the test_y values are not given we have to submit the model to check for the accuracy achieved

def submit(model,batch_size = 300):

    

    end = test_data.shape[0]

    final_data = pandas.DataFrame(columns=['id_code','diagnosis'])

    

    for i in range(0,end,batch_size):

        last = end if (i+batch_size)>end else (i+batch_size)



        test_x = split_data('test',i,last)

        test_y = model.predict(test_x)



        test_y = numpy.array([numpy.argmax(i) for i in test_y])



        predicted_data = pandas.DataFrame({ 'id_code' : test_data.index[i:last].values, 'diagnosis' : test_y})

        final_data = final_data.append(predicted_data,ignore_index=True)



        del test_x, test_y

        gc.collect()

    

    final_data.to_csv('submission.csv',index=False)



if __name__ == '__main__':



    print('Images are resized to', shape, '\n\n')

    print(train_data.shape)

    model = create_model()

    print(model.summary())

    model = model_fitting(model,batch_size = 300, epochs = 40, validation = False)

    submit(model)