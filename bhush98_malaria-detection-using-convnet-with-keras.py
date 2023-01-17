# Importing all the libraries we need



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import keras

from keras.layers import Conv2D , MaxPooling2D

from keras.layers import Dense , Flatten , Dropout

from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential

from sklearn.model_selection import train_test_split

from matplotlib.image import imread

from keras.preprocessing.image import ImageDataGenerator , load_img

import os



# Printing all the folder names 

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)



# Any results you write to the current directory are saved as output.
# Creting variables for storing the dir path so that it's easy when making dataset



# Uninfected

uninfected = os.listdir("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected")



#Infected

parasitized = os.listdir("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized")

filename = []

Category = []



#Looping through each file in the uninfected dir

for file in uninfected:

    

    #appending the file names to the list filename

    filename.append("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/"+file)

    

    #appending the Category of the image to the list Category

    Category.append("Uninfected")

    

# Looping through each file in the parasitized dir

for file in parasitized:

    

    # appending the file names to the list filename

    filename.append("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/"+file)

    

    #appending the Category of the image to the list Category

    Category.append("infected")

 

# Creating a dataframe from the lists we created 



df = pd.DataFrame({

    "File_name":filename,

    "Category":Category

})
# First ten instances of the DataFrame



df.head(10)
# Last ten instance of the Dataframe



df.tail(10)
# Getting the number of unique values and there counts



df['Category'].value_counts()
# plotting a bar graph of the unique values in the dataframe



df['Category'].value_counts().plot.bar()
# Shuffling the Dataframe to make it a little bit complex for the network



from sklearn.utils import shuffle

df = shuffle(df , random_state=0)
# First 10 instances of the shuffled data



df.head(10)
# Plotting the cell image of a Uninfected cell



img = load_img(df['File_name'][0])

plt.imshow(img)

plt.title("Uninfected from Malaria")

plt.show()
# Plotting the cell image of a Infected cell



img = load_img(df["File_name"][4])

plt.imshow(img)

plt.title("Infected from Malaria")

plt.show()
# Just to make clear that the data is perfectly shuffled with correct category tag



sample_data = df.head(20)

for index , row in sample_data.iterrows():

    print(row['File_name'] + " ------> " + row['Category'])
# Creating our Dataset from the dataframe



X = []

Y = []

for index , row in df.iterrows():

    

    try:

        # Reading the image from the dataframe

        img = cv2.imread(row['File_name'] , cv2.IMREAD_COLOR)

        

        # Resizing the image to our desired dimensions

        img = cv2.resize(img , (128,128))

        

        # Appending the resized image to X

        X.append(np.array(img))

        

        # Appending the Category to Y

        Y.append(row['Category'])

        

    except:

        pass

    

# Just to make sure everything is perfect

print(len(X))

print(len(Y))
# Verifyting if the images appended is of our desired dimesnsion or not



print(X[1].shape)
# Printing the first value of X i.e first image



print(X[0])
# First five instances of Y 



print(Y[0:5])
# Plotting some random images of uninfected and infected cell images to get a better idea, how the infected and uninfected cell looks like



import random



# Creating subplots

fig , ax = plt.subplots(2,10)

# Adjusting the space between the plots

plt.subplots_adjust(bottom=0.3,top=0.5, hspace=0)

# Setting the figure size

fig.set_size_inches(25,25)





# Looping thorugh 2 rows and 10 columns

for i in range(2):

    for j in range(10):

        

        # Genereating a random number per iteration

        l = random.randint(0,len(Y))

        # Plotting the image

        ax[i,j].imshow(X[l])

        # Setting the title of image

        ax[i,j].set_title(Y[l])

        # Giving aspect ratio

        ax[i,j].set_aspect('equal')

# Converting the list to numpy arrays for better computation practice



X = np.array(X)

Y = np.array(Y)
# Reassuring the type



print(type(X))

print(type(Y))
# Using sklearn's LabelEncoder to encode our categorical data



enc = LabelEncoder()

Y = enc.fit_transform(Y)
print(type(Y))
# Now the data is converted in form of numbers

# Where 0 is Uninfected and 1 is Infected



print(Y[0:5])
# Splitting the data into training and testing



X_train , x_test, Y_train ,  y_test = train_test_split(X , Y)
print(len(X_train))
print(len(x_test))
print(len(Y_train))
print(len(y_test))
# Creating our Convolutional Neural Network



model = Sequential()



# First Convolution layer

model.add(Conv2D(32 , (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



# Second Convolution layer

model.add(Conv2D(64 , (3,3) , activation = 'relu'))

model.add(MaxPooling2D( pool_size = (2,2)))



# Third Convolution layer

model.add(Conv2D(128 , (3,3) , activation = 'relu'))

model.add(MaxPooling2D( pool_size = (2,2)))



# Fourth Convolution layer

model.add(Conv2D( 256 , (3,3) , activation = 'relu') )

model.add(MaxPooling2D( pool_size = (2,2)))



# Converting the data to a one dimension format

model.add(Flatten())



# Hidden layer with 256 neurons

model.add(Dense(256 , activation = "relu"))



# 50% of the random neurons will shut down , to get rid of overfitting

model.add(Dropout(0.5))



model.add(Dense( 1 , activation = "sigmoid"))
# An essential step where we define what optimizer should be used by our model, what should be the loos function and based of what metrics model should learn



model.compile( optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
# Fitting our training data to our model



model.fit(X_train , Y_train , epochs = 20 , batch_size = 32)
# Evaluating our model on testing data



loss , accuracy = model.evaluate(x_test , y_test , batch_size = 32)



print('Test accuracy: {:2.2f}%'.format(accuracy*100))
y_pred = model.predict(x_test)
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix( y_test , y_pred.round() )

print(cm)

plt.matshow(cm)