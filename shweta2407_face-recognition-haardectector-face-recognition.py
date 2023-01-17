import cv2 #OpenCv
from IPython.display import Image
image_path =r'../input/chris-hemsworth-image/group_girls.jpg'
cascade_path = r'../input/haar-cascade-xml-file/haar_cascade.xml'
# showing the image

Image(filename=image_path) 
# create the cascade classifier
cascade = cv2.CascadeClassifier(cascade_path)

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect faces in the image
faces = cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)
# print how many faces found
print("Total faces  ", len(faces))
# draw the rectangle around the faces in the image and render it

for (X, Y, width, height) in faces:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
cv2.imwrite('./group_girls.jpg', image)
Image(filename='./group_girls.jpg')
# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect faces in the image
faces = cascade.detectMultiScale(
    gray,
    
    # CHANGING THE SCALING FACTOR FROM 1.1 TO 1.2
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
) 
# print how many faces found
print("Total faces  ", len(faces))

# draw the rectangle around the faces in the image and render it

for (X, Y, width, height) in faces:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
    
cv2.imwrite('./group_girls.jpg', image)
Image(filename='./group_girls.jpg')
image_path =r'../input/chris-hemsworth-image/chris_1.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect faces in the image
faces = cascade.detectMultiScale(
    gray,
    
    # CHANGING THE SCALING FACTOR FROM 1.1 TO 1.2
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
) 

# print how many faces found
print("Total faces  ", len(faces))

# draw the rectangle around the faces in the image and render it

for (X, Y, width, height) in faces:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
    
cv2.imwrite('./chris_1.jpg', image)
Image(filename='./chris_1.jpg')
image_path =r'../input/chris-hemsworth-image/chris_4.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect faces in the image
faces = cascade.detectMultiScale(
    gray,
    
    # CHANGING THE SCALING FACTOR FROM 1.1 TO 1.2
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
) 

# print how many faces found
print("Total faces  ", len(faces))

# draw the rectangle around the faces in the image and render it

for (X, Y, width, height) in faces:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
    
cv2.imwrite('./chris_4.jpg', image)
Image(filename='./chris_4.jpg')
image_path =r'../input/chris-hemsworth-image/chris_4.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect faces in the image
faces = cascade.detectMultiScale(
    gray,
    
    # CHANGING THE SCALING FACTOR FROM 1.2 TO 1.5
    scaleFactor=1.5,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
) 

# print how many faces found
print("Total faces  ", len(faces))

# draw the rectangle around the faces in the image and render it

for (X, Y, width, height) in faces:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
    
cv2.imwrite('./chris_4.jpg', image)
Image(filename='./chris_4.jpg')
image_path =r'../input/chris-hemsworth-image/chris_6.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect faces in the image
faces = cascade.detectMultiScale(
    gray,
    
    scaleFactor=1.5,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
) 

# print how many faces found
print("Total faces  ", len(faces))

# draw the rectangle around the faces in the image and render it

for (X, Y, width, height) in faces:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
    
cv2.imwrite('./chris_6.jpg', image)
Image(filename='./chris_6.jpg')
image_path =r'../input/chris-hemsworth-image/chris_6.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect faces in the image
faces = cascade.detectMultiScale(
    gray,
    # scaling factor 1.2
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
) 

# print how many faces found
print("Total faces  ", len(faces))

# draw the rectangle around the faces in the image and render it

for (X, Y, width, height) in faces:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
    
cv2.imwrite('./chris_6.jpg', image)
Image(filename='./chris_6.jpg')
image_path =r'../input/chris-hemsworth-image/chris_7.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect faces in the image
faces = cascade.detectMultiScale(
    gray,
    
    # SCALING FACTOR 1.2
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
) 

# print how many faces found
print("Total faces  ", len(faces))

# draw the rectangle around the faces in the image and render it

for (X, Y, width, height) in faces:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
    
cv2.imwrite('./chris_7.jpg', image)
Image(filename='./chris_7.jpg')
# we take the photo containing both chris hemsworth and tom holland
image_path =r'../input/chris-hemsworth-image/chris_6.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# looping over the range 1.0 to 1.9

face_list =[]


for i in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
    
    # detect faces in the image
    faces = cascade.detectMultiScale(
    gray,
    scaleFactor= i,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    ) 
    
    face_list.append(len(faces))
face_list
image_path =r'../input/chris-hemsworth-image/chris_4.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# looping over the range 1.0 to 1.9

face_list =[]

for i in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
    
    # detect faces in the image
    faces = cascade.detectMultiScale(
    gray,
    scaleFactor= i,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    ) 
    
    face_list.append(len(faces))
face_list
import collections

#uploading chris_9 pic
image_path =r'../input/chris-hemsworth-image/chris_9.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# looping over the range 1.0 to 1.9

face_list =[]
face_values ={}

for i in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
    
    # detect faces in the image
    faces = cascade.detectMultiScale(
    gray,
    scaleFactor= i,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    ) 
    
    face_list.append(len(faces))
    face_values.update({len(faces):faces})
    
print("face_list is ", face_list)


# helper function to pic the max occuring value from the face_list

# variable to store the face value
fvalue =0

for key, value in collections.Counter(face_list).items():
    
    if value == max(collections.Counter(face_list).values()):
        fvalue = key
        break
        
for (X, Y, width, height) in face_values[fvalue]:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
    
cv2.imwrite('./chris_9.jpg', image)
Image(filename='./chris_9.jpg')
#uploading  pic
image_path =r'../input/chris-hemsworth-image/chris_11.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# looping over the range 1.0 to 1.9

face_list =[]
face_values ={}

for i in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
    
    # detect faces in the image
    faces = cascade.detectMultiScale(
    gray,
    scaleFactor= i,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    ) 
    
    face_list.append(len(faces))
    face_values.update({len(faces):faces})
    
print("face_list is ", face_list)


# helper function to pic the max occuring value from the face_list

# variable to store the face value
fvalue =0

for key, value in collections.Counter(face_list).items():
    
    if value == max(collections.Counter(face_list).values()):
        fvalue = key
        break
        
for (X, Y, width, height) in face_values[fvalue]:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
    
cv2.imwrite('./chris_11.jpg', image)
Image(filename='./chris_11.jpg')
#uploading pic
image_path =r'../input/chris-hemsworth-image/chris_12.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# looping over the range 1.0 to 1.9

face_list =[]
face_values ={}

for i in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
    
    # detect faces in the image
    faces = cascade.detectMultiScale(
    gray,
    scaleFactor= i,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    ) 
    
    face_list.append(len(faces))
    face_values.update({len(faces):faces})
    
print("face_list is ", face_list)


# helper function to pic the max occuring value from the face_list

# variable to store the face value
fvalue =0

for key, value in collections.Counter(face_list).items():
    
    if value == max(collections.Counter(face_list).values()):
        fvalue = key
        break
        
for (X, Y, width, height) in face_values[fvalue]:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
    
cv2.imwrite('./chris_12.jpg', image)
Image(filename='./chris_12.jpg')

#uploading chris_10 pic
image_path =r'../input/chris-hemsworth-image/chris_10.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# looping over the range 1.0 to 1.9

face_list =[]
face_values ={}

for i in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
    
    # detect faces in the image
    faces = cascade.detectMultiScale(
    gray,
    scaleFactor= i,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    ) 
    
    face_list.append(len(faces))
    face_values.update({len(faces):faces})
    
print("face_list is ", face_list)

# helper function to pic the max occuring value from the face_list

# variable to store the face value
fvalue =0

for key, value in collections.Counter(face_list).items():
    
    if value == max(collections.Counter(face_list).values()):
        fvalue = key
        break
        
for (X, Y, width, height) in face_values[fvalue]:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
    
cv2.imwrite('./chris_10.jpg', image)
Image(filename='./chris_10.jpg')
image_path =r'../input/chris-hemsworth-image/chris_10.jpg'

# read the image
image = cv2.imread(image_path)

# convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect faces in the image
faces = cascade.detectMultiScale(
    gray,
    
    # SCALING FACTOR 
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
) 

# print how many faces found
print("Total faces  ", len(faces))

# draw the rectangle around the faces in the image and render it

for (X, Y, width, height) in faces:
    
    # draw the rectangle from the given coordinates (X, Y) and width and height.
    cv2.rectangle(image,  (X,Y), (X+width, Y+height), (0, 0, 255), 2)
    
cv2.imwrite('./chris_10.jpg', image)
Image(filename='./chris_10.jpg')
# install the library
!pip install face_recognition
import face_recognition
Image(filename = '../input/chris-hemsworth-image/chris_12.jpg')
import tensorflow as tf

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
# load image
# dictionary to save image name mapped to their encodings
data = {}
enc =[]
image_name =[]
with strategy.scope():
    for i in [1, 4, 7]:
        image = cv2.imread('../input/chris-hemsworth-image/chris_{}.jpg'.format(i))
        # convert it into rgb
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # using 'cnn' model for encoding image 
        boxes = face_recognition.face_locations(image, model='cnn')
        encodings = face_recognition.face_encodings(image, boxes, num_jitters=5)

        enc.append(encodings)
        image_name.append('chris_{}'.format(i))
    
data = {'encodings': enc, 'image_name': image_name}
# load image
image = cv2.imread('../input/chris-hemsworth-image/chris_12.jpg')
# convert it into rgb
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(image, model='cnn')
encodings = face_recognition.face_encodings(image, boxes, num_jitters=5)
matches = face_recognition.compare_faces(data['encodings'][0], encodings[0], tolerance=0.45)
matches
if True in matches:

    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
    counts = {}
    
    for i in matchedIdxs:
        name = data["image_name"][i]
        counts[name] = counts.get(name, 0) + 1

        name = max(counts, key=counts.get)
name
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img 

plt.imshow(load_img('../input/chris-hemsworth-image/chris_12.jpg'))
plt.imshow(load_img('../input/chris-hemsworth-image/chris_1.jpg'))
plt.imshow(load_img('../input/chris-hemsworth-image/chris_4.jpg'))
plt.imshow(load_img('../input/chris-hemsworth-image/chris_7.jpg'))
