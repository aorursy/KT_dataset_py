from PIL import Image, ImageDraw
from IPython.display import display

# The program we will be finding faces on the example below
pil_im = Image.open('../input/MyFriends.jpg')
display(pil_im)
!pip install face_recognition
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
juned_image = face_recognition.load_image_file("../input/juned.jpg")
juned_face_encoding = face_recognition.face_encodings(juned_image)[0]

# Load a second sample picture and learn how to recognize it.
minesh_image = face_recognition.load_image_file("../input/Minesh.jpeg")
minesh_face_encoding = face_recognition.face_encodings(minesh_image)[0]

# Load a second sample picture and learn how to recognize it.
chirag_image = face_recognition.load_image_file("../input/Chirag.jpeg")
chirag_face_encoding = face_recognition.face_encodings(chirag_image)[0]

# Load a second sample picture and learn how to recognize it.
mitesh_image = face_recognition.load_image_file("../input/Mitesh.jpeg")
mitesh_face_encoding = face_recognition.face_encodings(mitesh_image)[0]

# Load a second sample picture and learn how to recognize it.
nimesh_image = face_recognition.load_image_file("../input/Nimesh.jpeg")
nimesh_face_encoding = face_recognition.face_encodings(nimesh_image)[0]

# Load a second sample picture and learn how to recognize it.
sandeep_image = face_recognition.load_image_file("../input/Sandeep.jpeg")
sandeep_face_encoding = face_recognition.face_encodings(sandeep_image)[0]



# Create arrays of known face encodings and their names
known_face_encodings = [
    juned_face_encoding,
    minesh_face_encoding,
    chirag_face_encoding,
    mitesh_face_encoding,
    nimesh_face_encoding,
    sandeep_face_encoding
]
known_face_names = [
    "Juned",
    "Minesh",
    "Chirag",
    "Mitesh",
    "Nimesh",
    "Sandeep"
]
print('Learned encoding for', len(known_face_encodings), 'images.')
# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("../input/MyFriends.jpg")

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

    name = "Unknown"

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