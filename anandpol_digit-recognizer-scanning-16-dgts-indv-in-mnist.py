import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
tf.__version__
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../input/mnistasjpg/trainingSet/trainingSet',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('../input/mnistasjpg/testSet',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'input')
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, epochs = 10)
import numpy as np
from PIL import Image
from keras.preprocessing import image
test_image = image.load_img('../input/mnistasjpg/testSet/testSet/img_10.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
#test_image
#test_image.show()
training_set.class_indices
if result[0][0] == 1:
    prediction = 0
elif result[0][1] == 1:
    prediction = 1
elif result[0][2] == 1:
    prediction = 2
elif result[0][3] == 1:
    prediction = 3
elif result[0][4] == 1:
    prediction = 4
elif result[0][5] == 1:
    prediction = 5
elif result[0][6] == 1:
    prediction = 6
elif result[0][7] == 1:
    prediction = 7
elif result[0][8] == 1:
    prediction = 8
elif result[0][9] == 1:
    prediction = 9

print(prediction)
from PIL import Image
test_image = Image.open('../input/mnist-digits/mnist_digits/samples_0000.png').crop((0, 0, 95, 95))
coordinate = x, y = 0, 0
print(test_image.getpixel(coordinate))
test_image = test_image.convert('RGB')
test_image.save('/kaggle/working/img_new1.png')
test_image = image.load_img('/kaggle/working/img_new1.png', target_size = (64, 64))
test_image.save('/kaggle/working/img_new1.png')
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
if result[0][0] == 1:
    prediction = 0
elif result[0][1] == 1:
    prediction = 1
elif result[0][2] == 1:
    prediction = 2
elif result[0][3] == 1:
    prediction = 3
elif result[0][4] == 1:
    prediction = 4
elif result[0][5] == 1:
    prediction = 5
elif result[0][6] == 1:
    prediction = 6
elif result[0][7] == 1:
    prediction = 7
elif result[0][8] == 1:
    prediction = 8
elif result[0][9] == 1:
    prediction = 9

print(prediction)

lst = []
rs = []
x_cursor = 73
y_cursor = 69
for i in range(0,4):
    for j in range(0, 4):
        test_image = Image.open('../input/gan-generated/samples/samples_0000.png').crop((x_cursor, y_cursor, x_cursor + 95, y_cursor + 95))
        test_image = test_image.convert('RGB')
        #test_image.show()
        test_image.save('/kaggle/working/img_new1.jpg')
        test_image = image.load_img('/kaggle/working/img_new1.jpg', target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        rs = cnn.predict(test_image/255)
        if result[0][0] == 1:
            prediction = 0
        elif result[0][1] == 1:
            prediction = 1
        elif result[0][2] == 1:
            prediction = 2
        elif result[0][3] == 1:
            prediction = 3
        elif result[0][4] == 1:
            prediction = 4
        elif result[0][5] == 1:
            prediction = 5
        elif result[0][6] == 1:
            prediction = 6
        elif result[0][7] == 1:
            prediction = 7
        elif result[0][8] == 1:
            prediction = 8
        elif result[0][9] == 1:
            prediction = 9    
       # print(rs)    
        if rs.all() < 0.7:
            lst.append(10)
        else:    
            lst.append(prediction)
        x_cursor += 115

    y_cursor += 115
    x_cursor = 72
        


np.reshape(lst, (-1, 4))
lst1 = []
x_cursor = 37
y_cursor = 35
for i in range(0,4):
    for j in range(0, 4):
        test_image = Image.open('../input/kaggle-mnist-digits/mnist_digits_kaggle/img_0.png').crop((x_cursor, y_cursor, x_cursor + 48, y_cursor + 48))
        test_image = test_image.convert('RGB')
        test_image.show()
        test_image.save('/kaggle/working/img_new1.jpg')
        test_image = image.load_img('/kaggle/working/img_new1.jpg', target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        if result[0][0] == 1:
            prediction = 0
        elif result[0][1] == 1:
            prediction = 1
        elif result[0][2] == 1:
            prediction = 2
        elif result[0][3] == 1:
            prediction = 3
        elif result[0][4] == 1:
            prediction = 4
        elif result[0][5] == 1:
            prediction = 5
        elif result[0][6] == 1:
            prediction = 6
        elif result[0][7] == 1:
            prediction = 7
        elif result[0][8] == 1:
            prediction = 8
        elif result[0][9] == 1:
            prediction = 9
        
        lst1.append(prediction)
        
        x_cursor += 58
    y_cursor += 56
    x_cursor = 37
        


np.reshape(lst1, (-1, 4))
'''lst_test = []
import os
for filename in os.listdir('../input/mnistasjpg/testSet/testSet'):
    img_path = os.path.join('../input/mnistasjpg/testSet/testSet', filename)
    
    
    test_image = image.load_img(img_path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    if result[0][0] == 1:
        prediction = 0
    elif result[0][1] == 1:
        prediction = 1
    elif result[0][2] == 1:
        prediction = 2
    elif result[0][3] == 1:
        prediction = 3
    elif result[0][4] == 1:
        prediction = 4
    elif result[0][5] == 1:
        prediction = 5
    elif result[0][6] == 1:
        prediction = 6
    elif result[0][7] == 1:
        prediction = 7
    elif result[0][8] == 1:
        prediction = 8
    elif result[0][9] == 1:
        prediction = 9
        
    lst_test.append(prediction)'''
        
    
        


    
    
#lst_test
test_list = []
import os
for filename in os.listdir('../input/kaggle-mnist-digits/mnist_digits_kaggle'):
    img_path = os.path.join('../input/kaggle-mnist-digits/mnist_digits_kaggle', filename)
    
    lst1 = []
    x_cursor = 37
    y_cursor = 35
    for i in range(0,4):
        for j in range(0, 4):
            test_image = Image.open(img_path).crop((x_cursor, y_cursor, x_cursor + 48, y_cursor + 48))
            test_image = test_image.convert('RGB')
            test_image.show()
            test_image.save('/kaggle/working/img_new1.jpg')
            test_image = image.load_img('/kaggle/working/img_new1.jpg', target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = cnn.predict(test_image)
            if result[0][0] == 1:
                prediction = 0
            elif result[0][1] == 1:
                prediction = 1
            elif result[0][2] == 1:
                prediction = 2
            elif result[0][3] == 1:
                prediction = 3
            elif result[0][4] == 1:
                prediction = 4
            elif result[0][5] == 1:
                prediction = 5
            elif result[0][6] == 1:
                prediction = 6
            elif result[0][7] == 1:
                prediction = 7
            elif result[0][8] == 1:
                prediction = 8
            elif result[0][9] == 1:
                prediction = 9
        
            lst1.append(prediction)
        
            x_cursor += 58
    
        y_cursor += 56
        x_cursor = 37
    lst1 = np.reshape(lst1, (-1, 4))
    test_list.append(lst1)
    lst1 = []
    
        


    
    
#print(test_list)
np.reshape(test_list, (-1, 4))
os.listdir('../input/gan-generated/samples')
#os.path.join('../input/gan-generated/samples', 'samples_0057.png')
test_list_gan = []
import os
for filename in os.listdir('../input/gan-generated/samples'):
    img_path = os.path.join('../input/gan-generated/samples', filename)
    
    
    lst_gan = []
    x_cursor = 72
    y_cursor = 70
    for i in range(0,4):
        for j in range(0, 4):
            test_image = Image.open(img_path).crop((x_cursor, y_cursor, x_cursor + 95, y_cursor + 95))
            test_image = test_image.convert('RGB')
            #test_image.show()
            test_image.save('/kaggle/working/img_new1.jpg')
            test_image = image.load_img('/kaggle/working/img_new1.jpg', target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = cnn.predict(test_image)
            if result[0][0] == 1:
                prediction = 0
            elif result[0][1] == 1:
                prediction = 1
            elif result[0][2] == 1:
                prediction = 2
            elif result[0][3] == 1:
                prediction = 3
            elif result[0][4] == 1:
                prediction = 4
            elif result[0][5] == 1:
                prediction = 5
            elif result[0][6] == 1:
                prediction = 6
            elif result[0][7] == 1:
                prediction = 7
            elif result[0][8] == 1:
                prediction = 8
            elif result[0][9] == 1:
                prediction = 9
        
            lst_gan.append(prediction)
        
            x_cursor += 115

        y_cursor += 115
        x_cursor = 72
    lst_gan = np.reshape(lst_gan, (-1, 4))
    test_list_gan.append(lst_gan)
        
    
print(test_list_gan)
from PIL import Image
test_image = Image.open('../input/read-plate/plateread.png')
x_len, y_len = test_image.size
test_image.getpixel((0, 0))!=(0, 0, 0)
x_before = []
x_after = [0, ]
k = 0
i = 0 
i_plus = 0
while i < x_len:
    #print(i)
    j=0
    flag = False
    i_plus = i + 1
    while flag==False and j < y_len:
        if test_image.getpixel((i, j))!=(0, 0, 0):
            x_before.append(i)
            flag = True
            
            flag1 = False
            k = i
            while k < x_len:
                #print(k)
                j = 0
                flag1 = True
                while flag1 == True and j < y_len:
                    #
                    #print(j)
                    if test_image.getpixel((k, j)) != (0, 0, 0):
                        flag1 = False
                    
                    j += 1
                
                if flag1 == True:
                    x_after.append(k)
                    i_plus = k
                    #print(i_plus)
                    k = x_len
                    
                k += 1
            
            
        j += 1
        #print(i, j)
    
    i = i_plus
    

    
x_before.append(x_len)    
x_before
x_after
x_cor_values = []
for i in range(len(x_before)):
    x_cor_values.append(int((x_before[i] + x_after[i])/2))
x_cor_values
y_before = []
y_after = []
k = 0
i = 0 
i_plus = 0

for i in range(len(x_before)-1):
    flag = True
    l=0
    while flag and l < y_len:
        j = x_cor_values[i]
        flag = True
        while flag and j < x_cor_values[i+1]:
            if test_image.getpixel((j, l)) != (0, 0, 0):
                flag=False
                
                y_before.append(l/2)
                
                k = l
                while k < y_len:
                    flag1 = True
                    j = x_cor_values[i]
                    while flag1 and j < x_cor_values[i+1]:
                        if test_image.getpixel((j, k)) != (0, 0, 0):
                            flag1 = False
                            
                        j += 1    
                    
                    if flag1 == True:
                        y_after.append((k + y_len)/2)
                        k = y_len
                
                    k += 1
                    
                
            j += 1
            
        l += 1
            
                
y_before
y_after
lst = []
import numpy as np
from keras.preprocessing import image
for i in range(len(x_before)-1):
    test_image = Image.open('../input/read-plate/plateread.png').crop((x_cor_values[i], y_before[i],x_cor_values[i+1], y_after[i]))
    test_image = test_image.convert('RGB')
    test_image.save('/kaggle/working/img_new1.png')
    test_image = image.load_img('/kaggle/working/img_new1.png', target_size = (64, 64))
    test_image.save('/kaggle/working/img_new1.png')
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    
    if result[0][0] == 1:
        prediction = 0
    elif result[0][1] == 1:
        prediction = 1
    elif result[0][2] == 1:
        prediction = 2
    elif result[0][3] == 1:
        prediction = 3
    elif result[0][4] == 1:
        prediction = 4
    elif result[0][5] == 1:
        prediction = 5
    elif result[0][6] == 1:
        prediction = 6
    elif result[0][7] == 1:
        prediction = 7
    elif result[0][8] == 1:
        prediction = 8
    elif result[0][9] == 1:
        prediction = 9

    print(prediction)
    lst.append(prediction)
    
lst