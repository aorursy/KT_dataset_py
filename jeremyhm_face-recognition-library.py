!pip install face_recognition
import os
print('Data:\n ',os.listdir('../input/'))
DIR = '../input'
X = []

from tqdm import tqdm
import cv2
for img in tqdm(os.listdir(DIR)):
    path = os.path.join(DIR,img)
    _, ftype = os.path.splitext(path)
    if ftype == ".jpg":
        img = face_recognition.load_image_file(path)
        X.append(img)
from PIL import Image, ImageDraw
from IPython.display import display

# The program we will be finding faces on the example below
pil_im = Image.open('../input/hmj.jpg')
display(pil_im)
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

hmj_image = face_recognition.load_image_file('../input/hmj.jpg')
hmj_face_encoding = face_recognition.face_encodings(hmj_image)[0]

raul_image = face_recognition.load_image_file('../input/raul.jpg')
raul_face_encoding = face_recognition.face_encodings(raul_image)[0]

panda_image = face_recognition.load_image_file('../input/panda.jpg')
panda_face_encoding = face_recognition.face_encodings(panda_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    hmj_face_encoding,
    raul_face_encoding,
    panda_face_encoding
]
known_face_names = [
    "Jeremy",
    "Raul",
    "Dany"
    
]
print('Learned encoding for', len(known_face_encodings), 'images.')
import matplotlib.pyplot as plt

fig,ax=plt.subplots(6,3)
fig.set_size_inches(15,15)
count = 0
for i in range(6):
    for j in range (3):
        ax[i,j].imshow(X[count])
        count+=1
        
plt.tight_layout()
def find_me(unknown_image):
    # Load an image with an unknown face
    #unknown_image = face_recognition.load_image_file(path)

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "?"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    display(pil_image)
for i in range(len(X)):
    find_me(X[i])