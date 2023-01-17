import os

import cv2 

import tensorflow as tf

import numpy as np
def  get_images(img_dir):

    good_extensions=['jpg', 'jpeg', 'gif', 'tiff', 'bmp', 'png']

    img_list=[] # empty list that will hold the input images

    file_list=[] # empty list to store names of image files

    image_list=os.listdir(img_dir) # list of image files

    for f in image_list: # iterate through the images

        ext=f[f.rfind('.') +1:] # get the files extension

        if ext in good_extensions: 

            img_path=os.path.join(img_dir, f)  # create path to the image file

            try: # try to input the image with cv2 generate exception if it can not be read

                img=cv2.imread(img_path)

                size=img.shape

            except:

                print(f'file {img_path} is not a valid image file and will not be processed')

            else:

                img=cv2.resize(img, (128,128)) # resize image to be the same size as was used to train the model

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # cv2 images are BGR convert to rgb

                img=tf.keras.applications.mobilenet.preprocess_input(img) # scale  the pixels between 1 and -1 

                img_list.append(img)

                file_list.append(f) # save the names of the image files       

    img_array=np.array(img_list)   # convert the list to an np array

    return img_array, file_list

    
def print_results(predictions,file_list):

    # print header for table of results

    msg='{0}{1:^15s}{0:2s}{2:^28s}{0:2s}{3:^12s}'

    msg=msg.format( ' ', 'File Name', 'Predicted Class', 'Probability')

    print (msg)

    for i, p in enumerate (predictions):

        file_name=file_list[i]

        class_index=p.argmax()  # get the class index with the highest probability

        klass='beautiful' if class_index==1 else 'average' 

        probability= 100*p[class_index]

        msg='{0}{1:^15s}{0:2s}{2:^28s}{0:2s}{3:^12.2f}'

        print (msg.format(' ',file_name, klass, probability ))

    
def classifier (img_dir, model_location):

    img_array, file_list =get_images(img_dir)

    length=img_array.shape[0]    

    # determine batch size and stepsper epoch based on the number of images to classify

    batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=100],reverse=True)[0]  

    steps=int(length/batch_size)

    print ( f'{length} valid images were found -- loading model with batch_size= {batch_size}, and {steps} steps ')

    model = tf.keras.models.load_model( model_location) # load the trained model

    print ('model is loaded-- making predictions \n')

    predictions=model.predict(img_array,batch_size=batch_size, steps=steps, verbose=0 ) # make predictions on the images

    print_results(predictions,file_list)
img_dir='../input/beauty-detection-data-set/exampledir' # directory with images first 5 files are in class average, next 5 are in class beautiful

model_loc='../input/beauty-detection-data-set/exampledir' # directory where trained model is stored has folders assests and variables and file saved_model.pb

results=classifier(img_dir, model_loc)