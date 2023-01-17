!pip install face_recognition
import PIL.Image

import PIL.ImageDraw

import face_recognition

from matplotlib.pyplot import imshow
input_image = face_recognition.load_image_file('/kaggle/input/billionaires.jpg')

imshow(input_image);
image_faces = face_recognition.face_locations(input_image)
print('Total number of faces in a image: ',len(image_faces))
output_image = PIL.Image.fromarray(input_image)

draw_shape = PIL.ImageDraw.Draw(output_image)
for faces in image_faces:

    top,left,bottom,right=faces

    draw_shape.rectangle([left,top,right,bottom],outline='red',width=20)
imshow(output_image);