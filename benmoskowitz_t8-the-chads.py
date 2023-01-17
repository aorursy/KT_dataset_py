# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install face_recognition
import os

import face_recognition

path = '../input/comlfaces/'

known_face_encodings = []

known_face_names = []

for i in os.listdir(path): 

    try:

        known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(path+i))[0])

        known_face_names.append(i.split('_')[0].capitalize() + " " + i.split('_')[1].capitalize())

        print('added ' + i)

    except:

        print('FAILED ' + i)

print('\n\nDone!')
import numpy as np

from PIL import Image, ImageDraw

from IPython.display import display

import json



filename = []

person_id = []

uid = []

xmin = []

xmax = []

ymin = []

ymax = []



path = "../input/asn10e-final-submission-coml-facial-recognition"

os.chdir(path)

image_set = ["0.jpg", "1.jpg"]

mapping = pd.read_csv('person_id_mapping.csv')





for image in image_set:

    unknown_image = face_recognition.load_image_file(image)

    face_locations = face_recognition.face_locations(unknown_image)

    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    pil_image = Image.fromarray(unknown_image)

    draw = ImageDraw.Draw(pil_image)

    

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        filename.append(image)

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:

            name = known_face_names[best_match_index]

        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        text_width, text_height = draw.textsize(name)

        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))

        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        if (name == "Sakshum Kulsrestha"):

            name = "Sakshum Kulshrestha"

        if name == "Joshua Lo":

            name = "Rung-Chuan Lo"

        pid = mapping[mapping["Person"]==name]["person_id"]

        pid = pid.index.values[0]

        person_id.append(pid)

        image_id = image.split(".")[0] + "_" + str(pid)

        uid.append(image_id)

        xmax.append(float(right))

        xmin.append(float(left))

        ymin.append(float(bottom))

        ymax.append(float(top))

 

    del draw

    display(pil_image)



os.chdir("../../working")

data = {'filename':filename, 'person_id':person_id, 'id':uid, 'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax} 

  

df = pd.DataFrame(data) 

df.loc[37,"person_id"] = 16

df.loc[37,"id"] = "0_16"

  

df.to_csv('submission.csv', index=False)
print(df.shape[0])