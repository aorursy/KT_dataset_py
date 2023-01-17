import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
messi_path = '/kaggle/input/images_face_detect/messi.jpeg'

unkown_image_path = '/kaggle/input/images_face_detect/unknown.png'

known_image_path = '/kaggle/input/images_face_detect/known.jpg'





# Any results you write to the current directory are saved as output.

from IPython.display import display, Image
!pip install face-recognition
import face_recognition

known_image = face_recognition.load_image_file(known_image_path)

unknown_image = face_recognition.load_image_file(unkown_image_path)
display(Image(filename=known_image_path))
ronaldo_encoding = face_recognition.face_encodings(known_image)[0]

unknown_encoding = face_recognition.face_encodings(unknown_image)[0]



results = face_recognition.compare_faces([ronaldo_encoding], unknown_encoding)

print(results)
display(Image(filename=unkown_image_path))
messi_image = face_recognition.load_image_file(messi_path)

messi_encoding = face_recognition.face_encodings(messi_image)[0]



results = face_recognition.compare_faces([ronaldo_encoding], messi_encoding)

print(results)
display(Image(filename=messi_path))