!pip install face_recognition
# we will import the required libraries

import matplotlib.pyplot as plt #for ploting of image

from skimage.feature import hog # for extraction HOG features of image.

from skimage import data, exposure

import cv2 # for preprocessing on image



# Now we will read image from the disk

image = cv2.imread('../input/pictures/dhoni.jpeg')

# When we read image using OpenCV, it reads the image in BGR format

# So now we will convert the image into RGB format

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 



# Now we will show the image

plt.imshow(image)
# now lets perform the feature extraction.

# we are going to use the hog function from skimage to extrat the HOG(Histogram of Oriented Gradient) image from input image.



fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16,16),

                   cells_per_block=(1,1), visualize=True, multichannel=True)

# fd is feature discriptor which is used for representation of image

# hog_image is a HOG image extracted from input image



# now let's plot input image and hog image both so it's easy to compare

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12), sharex = True, sharey = True)



ax1.axis('off')

ax1.imshow(image, cmap=plt.cm.gray)

ax1.set_title('Input Image')



hog_rescaled_img = exposure.rescale_intensity(hog_image, in_range=(0, 10))



ax2.axis('off')

ax2.imshow(hog_rescaled_img, cmap=plt.cm.gray)

ax2.set_title('HOG Image')

plt.show()
# Let us import the important libraries for Face Detection purpose

import face_recognition # For Face Detection

from matplotlib.patches import Rectangle # To draw rectangles

import numpy as np # for mathematicle operations



# In Face Recognition library there is a function face_locations which detects the all faces in the image and 

# returns there locations

# let's use the function and detect the faces

face_locations = face_recognition.face_locations(image)



total_faces = len(face_locations)

# lets print out the number of faces in the image.

print('There are {} face(s) in the image'.format(total_faces))

def detect_the_faces(image_path):

    image = cv2.imread(image_path) # Read the image

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert BGR to RGB format

    plt.imshow(image)

    ax = plt.gca()

    #Detect the face locations

    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:

        return 'No Faces Detected'

    

    # Now we need draw rectangle for each faces found

    for face_location in face_locations:

        # Get the co-ordinates from the location

        y, w, h, x = face_location 

        # here y is for top, w for right width, h for bottom height and x for left

        

        # So we got the co-orinates, let's draw the rectangle

        rect = Rectangle((x,y), w-x, h-y, fill=False, color='red')

        ax.add_patch(rect)

    

    # Let's disply

    plt.show()

    
# Let us pass our input image to the function that we created above.

detect_the_faces('../input/pictures/dhoni.jpeg')
# We will use python dictionary as Database for known faces encodings

known_faces = {}



#Now we create a function to add the faces encoding into the data base.

def add_face(image_path, name):

    image = cv2.imread(image_path) # Read the image

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert BGR to RGB format

    

    # We will get a face encoding for the face in the image

    face_encoding = face_recognition.face_encodings(image)[0] # 0 because if there are more than one facec then it will take first detected face

    

    # We add the face encoding in thedatabase with corresponding names 

    known_faces[name] = face_encoding

    



    

    
# Let's add known faces in database, For now we will add 3 known faces

add_face('../input/pictures/dhoni.jpeg', 'Dhoni')

add_face('../input/pictures/hrithik.jpeg', 'Hrithik')

add_face('../input/pictures/Rohit.jpg', 'Rohit')



#Let's check total keys in database

print(list(known_faces.keys()))
from scipy.spatial import distance # This will be used for calculating the distance between two encodings



def recognize_face(image_path):

    # Let's read the image using cv2

    unknown_image = cv2.imread(image_path)

    unknown_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB) # Convert thr BGR to RGB format

    

    # Let's get encodings for the face in the image

    unknown_face_encodings = face_recognition.face_encodings(unknown_image) # Here we will not add [0] because we want all the faces to be recognized in the image.

    

    # Now we will check the euclidean distance between two faces

    found_faces=[]

    threshold = 0.6

    for unknown_face_encoding in unknown_face_encodings:

        for known_person in known_faces:

            # Calculate the Euclidean distance

            d = distance.euclidean(known_faces[known_person], unknown_face_encoding)

            # We will check that distance should be less than threshold

            if d <= threshold:

                found_faces.append(known_person)

    if len(found_faces) ==  0:

        found_faces.append('Unknown')

    

    return found_faces # We return a list on found images in the image

    

        

    
# Now Let's use above function that we created for recognition of person

# our function returns a list of found faces in the image



################### TEST NO 1

test_image_path = '../input/pictures/dhoni_test.jpg'

test_image = cv2.imread(test_image_path)

test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

plt.imshow(test_image)



faces_in_picture = recognize_face(test_image_path)

for face in faces_in_picture:

    print("There is {}'s face in image".format(face))
################### TEST NO 2

test_image_path = '../input/pictures/rohit_test.jpg'

test_image = cv2.imread(test_image_path)

test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

plt.imshow(test_image)



faces_in_picture = recognize_face(test_image_path)

for face in faces_in_picture:

    print("There is {}'s face in image".format(face))
################### TEST NO 3

test_image_path = '../input/pictures/hrithik_test.jpg'

test_image = cv2.imread(test_image_path)

test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

plt.imshow(test_image)



faces_in_picture = recognize_face(test_image_path)

for face in faces_in_picture:

    print("There is {}'s face in image".format(face))
################### TEST NO 4

test_image_path = '../input/pictures/Aftab.jpg'

test_image = cv2.imread(test_image_path)

test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

plt.imshow(test_image)



faces_in_picture = recognize_face(test_image_path)

for face in faces_in_picture:

    print("There is {}'s face in image".format(face))
################### TEST NO 5

test_image_path = '../input/pictures/rohit_dhoni_test.jpg'

test_image = cv2.imread(test_image_path)

test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

plt.imshow(test_image)



faces_in_picture = recognize_face(test_image_path)

for face in faces_in_picture:

    print("There is {}'s face in image".format(face))