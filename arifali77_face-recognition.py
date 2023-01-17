# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(20,20))
!pip install face_recognition
import face_recognition
import PIL.Image # Pillow

import PIL.ImageDraw # Pillow
# load the jpg file into a numpy array

image = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/people.jpg")
# find all the faces in the image

face_locations = face_recognition.face_locations(image)
number_of_faces  =  len(face_locations)

print("I found {} faces in this photograph".format(number_of_faces))
# Load the image into a Python Image Library object so that we can draw on top of image



pil_image = PIL.Image.fromarray(image)



for face_location in face_locations:

# prints the location of each face in this image. Each face is a list of co-ordinates

    top, right, bottom, left = face_location

    print("A face is located at pixel location Top: {}, Left : {}, Bottom : {}, Right {}".format(top,right,bottom,left))
# Lets draw a box around the face



draw = PIL.ImageDraw.Draw(pil_image)

draw.rectangle([top, right, bottom, left], outline = "red")



# Display the image on the screen

plt.imshow(pil_image) #pil_image.show()
# load the jpg file into a numpy array

image = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/people.jpg")
# find all the facial features in all the faces in the image

face_landmark_list = face_recognition.face_landmarks(image)
number_of_faces  =  len(face_locations)

print("I found {} faces in this photograph".format(number_of_faces))
# Load the image into a Python Image Library object so that we can draw on top of image

pil_image = PIL.Image.fromarray(image)
# loop over each face

for face_landmarks in face_landmark_list:

# Loop over each facial feature (eye, nose, mouth, lips,  etc)

    for name, list_of_points in face_landmarks.items():

# print the location of each facial feature in this image

        print("The {} in this face has the following points:{}". format(name, list_of_points))

# lets trace out each facial features in this image

        draw.line(list_of_points, fill='red', width =2)

plt.imshow(pil_image)
# face_recognition

# load the jpg files into numpy arrays

image = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/person.jpg")
# Generate the face encodings

face_encodings = face_recognition.face_encodings(image)
if len(face_encodings) == 0:

    # No faces found in the image

    print("No faces were found.")

else:

    # Grab the first face encoding

    first_face_encoding = face_encodings[0]

    # print the results

    print(first_face_encoding)
# face_recognition

# load the known images

image_of_person_1 = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/person_1.jpg")

image_of_person_2 = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/person_2.jpg")

image_of_person_3 = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/person_3.jpg")
# Get the face encoding of each person. This will fail if no one is found in the photo

person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0]

person_2_face_encoding = face_recognition.face_encodings(image_of_person_2)[0]

person_3_face_encoding = face_recognition.face_encodings(image_of_person_3)[0]
# Create a list of all known face encodings

known_face_encodings = [person_1_face_encoding,

                       person_2_face_encoding,

                       person_3_face_encoding]
# load the image we want to check

unknown_image = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/unknown_8.jpg")
# Get face encodings for any people in the picture

unknown_face_encodings = face_recognition.face_encodings(unknown_image)
# There might be more than one person in the photo, so we need to loop over each one

for unknown_face_encoding in unknown_face_encodings:

    # Test if this unknown face encoding matches any of the three people we know

    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encodings)

    

    name = "Unknown"

    if results[0]:

        name = "Person 1"

    elif results[1]:

        name = "Person 2"

    elif results[2]:

        name = "Person 3"

    print(f"Found {name} in the photo!")
# face_recognition

# load the known images

image_of_person_1 = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/person_1.jpg")

image_of_person_2 = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/person_2.jpg")

image_of_person_3 = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/person_3.jpg")
# Get the face encoding of each person. This will fail if no one is found in the photo

person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0]

person_2_face_encoding = face_recognition.face_encodings(image_of_person_2)[0]

person_3_face_encoding = face_recognition.face_encodings(image_of_person_3)[0]
# Create a list of all known face encodings

known_face_encodings = [person_1_face_encoding,

                       person_2_face_encoding,

                       person_3_face_encoding]
# load the image we want to check

unknown_image = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/unknown_2.jpg")
# Get face encodings for any people in the picture

unknown_face_encodings = face_recognition.face_encodings(unknown_image)
# There might be more than one person in the photo, so we need to loop over each one

for unknown_face_encoding in unknown_face_encodings:

    # Test if this unknown face encoding matches any of the three people we know

    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encodings)

    

    name = "Unknown"

    if results[0]:

        name = "Person 1"

    elif results[1]:

        name = "Person 2"

    elif results[2]:

        name = "Person 3"

    print(f"Found {name} in the photo!")
# face_recognition

# load the known images

image_of_person_1 = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/person_1.jpg")

image_of_person_2 = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/person_2.jpg")

image_of_person_3 = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/person_3.jpg")
# Create a list of all known face encodings

known_face_encodings = [person_1_face_encoding,

                       person_2_face_encoding,

                       person_3_face_encoding]
# load the image we want to check

unknown_image = face_recognition.load_image_file("/kaggle/input/facial-recognition-1/unknown_7.jpg")
# Get face encodings for any people in the picture (# for low resolution image)

face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=2) # increase image size 

unknown_face_encodings = face_recognition.face_encodings(unknown_image, known_face_locations=face_locations)
# There might be more than one person in the photo, so we need to loop over each one

for unknown_face_encoding in unknown_face_encodings:

    # Test if this unknown face encoding matches any of the three people we know

    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encodings)

    

    name = "Unknown"

    if results[0]:

        name = "Person 1"

    elif results[1]:

        name = "Person 2"

    elif results[2]:

        name = "Person 3"

    print(f"Found {name} in the photo!")