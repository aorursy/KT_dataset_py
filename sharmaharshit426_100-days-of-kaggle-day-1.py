# Importing the bread and butter for every Deep-Learning problem

import numpy as np

import pandas as pd



# Importing OS for accessing data

import os



# For plotting stuff

import matplotlib.pyplot as plt



# For reading images

import cv2



%matplotlib inline



# Setting the random seed so that y'all don't get different images when you run the notebook yourself

np.random.seed(0)
# Setting the root directory, so that we don't have to type the same path everytime

root_dir = '/kaggle/input/global-wheat-detection/'



#Let's see what files do we have here

for file in os.listdir(root_dir):

    print(file)
train = pd.read_csv(root_dir+'train.csv')

train.head()
nncols = 4

train_path = os.path.join(root_dir, 'train') 

image_files_list = os.listdir(train_path)

main_title = "Just some pictures of Wheat"



# Here's a generic function to plot multiple images at once

def plot_random_images(image_files_list, image_path, nrows=3, ncols=4, main_title=""):

    """

    image_files_list : List containing all of the image files, stacked on axis 0

    nrows            : Number of rows of pictures

    ncols            : Number of columns of pictures

    main_title       : Main title for sub-plots

    """

    

    # Selecting a random number of images from the given list

    random_img_list = np.random.choice(image_files_list, nrows * ncols)

    

    # Reading images and appending them to the image_matrix_list

    image_matrix_list = []

    for file in random_img_list:

        img = cv2.imread(os.path.join(image_path, file))

        image_matrix_list.append(img)

        

    # Setting the subplots as per inputs provided

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize = (20, 15), squeeze=False)

    fig.suptitle('Wheat', fontsize=30)

    num=0

    for i in range(nrows):

        for j in range(ncols):

            axes[i][j].imshow(image_matrix_list[num])

            axes[i][j].set_title(random_img_list[num], fontsize=14)

            num += 1

    plt.show()

    

# Finally plotting the images

plot_random_images(image_files_list, train_path, 3, 4, main_title)
train.describe()
"""

For starters, we will restructure the way bounding boxes are stored, so that it's easier for us to plot them on the images

Let's get these bb's



Here, we apply a lambda function to the bbox column, to extract the 4 numbers from the string. Since the first and the last characters

are '[' and ']' respectively, we started the slicing from index 1, and left out the last index, and using "," as seperator



"""

bboxes = np.stack(train['bbox'].apply(lambda x : np.fromstring(x[1:-1], sep = ",")))



# Now we add new columns, namely x_min and y_min to the csv file

for i, column in enumerate(['x_min', 'y_min', 'width', 'height']):

    train[column] = bboxes[:, i]

    

# These lines will add the columns x_max and y_max to the csv file, and fill them with appropriate data

train["x_max"] = train.apply(lambda col: col.x_min+col.width, axis=1)

train["y_max"] = train.apply(lambda col: col.y_min + col.height, axis=1)



# Let's see if how our operation looks like for now

train.head()
train.drop(columns=['bbox', 'width', 'height'], inplace=True)

train.head()
x_max = np.array(train["x_max"].values.tolist())

y_max = np.array(train["y_max"].values.tolist())



"""

np.where works like a conditional statement. for eg. if x_max>1024, replace the value with 1024, else retain the original value of xmax. This is ofcourse done for every value in the numpy array.

"""

train["x_max"] = np.where(x_max > 1024, 1024, x_max).tolist()

train["y_max"] = np.where(y_max > 1024, 1024, y_max).tolist()

 
extensions = []

for file in os.listdir(os.path.join(root_dir, 'train')):

    extension = file.split(".")[1]

    if extension not in extensions:

        extensions.append(extension)



if len(extensions) == 1:

    print(f"All images are of extension '{extensions[0]}'")

else:

    print(f"Looks like we have a problem. Not all the images have same extensions.")
# Sweet!, now we can rename the values

train["image_id"] = train['image_id'].apply(lambda x: x + '.jpg')

train.head()
# Let's save our csv

train.to_csv("wheat.csv", index=False)

wheat = pd.read_csv("wheat.csv")



#Selecting the first image from the csv file. The same image id appears multiple times, because there are multiple bounding boxes inside that image

chosen_image = cv2.imread(os.path.join(train_path, "b6ab77fd7.jpg"))

chosen_image_df = wheat.loc[wheat["image_id"]=="b6ab77fd7.jpg", ["x_min", "y_min", "x_max", "y_max"]]

bbox_array = np.array(chosen_image_df.values.tolist())

bbox_array.shape

for i in range(len(bbox_array)):

    # Apparenetly this cv2.rectangle function fails when we pass points which are floats. That's why we gotta convert our rectangle points to integers

    """

    pt 1 contains (x_min, y_min)

    pt 2 contains (x_max, y_max)

    """

    pt1 = (int(bbox_array[i][0]), int(bbox_array[i][1]))

    pt2 = (int(bbox_array[i][2]), int(bbox_array[i][3]))

    draw_chosen_image = cv2.rectangle(chosen_image, pt1, pt2, (0, 255, 0), 5)

plt.imshow(draw_chosen_image)
"""

Cool. Let's scale it for multiple images. We will use the generic function we defined above (plot_random_images()). 

We will draw the rectangles on images, and then pass the image matrix

"""



images_list = os.listdir(train_path)



def draw_images_with_bboxes(images_list, image_annotation_file, nrows, ncols, main_title=""):

    image_matrix = []

    random_list = np.random.choice(images_list, 12)

    for image in random_list:

        img = cv2.imread(os.path.join(train_path, image))

        img_df = image_annotation_file.loc[image_annotation_file['image_id'] == image, ["x_min", "y_min", "x_max", "y_max"]]

        bboxes = np.array(img_df.values.tolist())



        for i,bbox in enumerate(bboxes):

            pt1 = (int(bbox[0]), int(bbox[1]))

            pt2 = (int(bbox[2]), int(bbox[3]))

            img = cv2.rectangle(img, pt1, pt2, (255, 0, 0), 5)



        image_matrix.append(img)

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 15))

    fig.suptitle(main_title, fontsize=30)

    num = 0

    for i in range(nrows):

        for j in range(ncols):

            axes[i][j].imshow(image_matrix[num])

            axes[i][j].set_title(random_list[num], fontsize=14)

            num += 1

            

draw_images_with_bboxes(images_list, wheat, 3, 4, "Wheat heads with Bounding boxes")