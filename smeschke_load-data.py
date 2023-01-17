import cv2, numpy as np, os



#create lists to save the labels (the name of the shape)

labels, images, shape_dir = [],[], '../input/shapes/'

shapes = ['square', 'circle', 'star', 'triangle']



#iterate through each shape

for shape in shapes:

    print('Getting data for: ', shape)

    #iterate through each file in the folder

    for path in os.listdir(shape_dir+shape):

        #add the image to the list of images

        images.append(cv2.imread(shape_dir+shape+'/'+path, 0))

        #add an integer to the labels list 

        labels.append(shapes.index(shape))



#break data into training and test sets

train_test_ratio, to_train = 5,0

train_images, test_images, train_labels, test_labels = [],[],[],[]

for image, label in zip(images, labels):

    if to_train<train_test_ratio: 

        train_images.append(image)

        train_labels.append(label)

        to_train+=1

    else:

        test_images.append(image)

        test_labels.append(label)

        to_train = 0



print('Number of training images: ', len(train_images))

print('Number of test images: ', len(test_images))