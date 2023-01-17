# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install face_recognition
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
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
import os
import pickle 
import cv2

def predict(img_dir):
    data = pickle.load(open("/kaggle/input/facialrecdata/data.p", "rb"))

    directory = "/kaggle/input/asn10e-final-submission-coml-facial-recognition/"
    known_face_encodings = data[0]
    known_face_names = data[1]

    
    face_locations = []
    face_encodings = []
    face_names = []
    
   
    
    frame = cv2.imread(directory+img_dir)
    
    
    big_frame = cv2.resize(frame, (0, 0), fx=2, fy=2)

    
    rgb_small_frame = big_frame[:, :, ::-1]

    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
        
    face_locations = [(top / 2, right / 2, bottom / 2, left / 2) for (top, right, bottom, left) in face_locations]
    
    
    big_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

    
    rgb_small_frame = big_frame[:, :, ::-1]

    
    face_locations_ = face_recognition.face_locations(rgb_small_frame)
    face_encodings_ = face_recognition.face_encodings(rgb_small_frame, face_locations_)

    c = 0
    for face_encoding in face_encodings_:
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

       

        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            if name not in face_names:
                face_names.append(name)
                face_locations.append(face_locations_[c])
                
        c += 1
            
    
    print ("displaying results")
    plt.rcParams["figure.figsize"]=20,20
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        left = int(left)

       
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 10)


        
        plt.text(left+10, bottom-10, name)

        
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()
    
    return face_locations, face_names
import pandas as pd
result_dict = {"filename": [],
             "person_id": [],
             "id": [],
             "xmin": [], "xmax": [],
             "ymin": [], "ymax": []}

name_to_id = pd.read_csv("/kaggle/input/asn10e-final-submission-coml-facial-recognition/person_id_mapping.csv")

print ("predicting on img 0.jpg")
locations, names = predict("0.jpg")
for name, location in zip(names, locations):
    name = " ".join(name.split("_"))
    if name != "Unknown":
        print (name)
        id_ = name_to_id[name_to_id["Person"] == name].values.tolist()[0][0]
        
        result_dict["filename"].append("0.jpg")
        result_dict["person_id"].append(id_)
        result_dict["id"].append("0_%s" % id_)
        result_dict["xmin"].append(location[3])
        result_dict["xmax"].append(location[1])
        result_dict["ymin"].append(location[2])
        result_dict["ymax"].append(location[0])
                   
print ("predicting on img 1.jpg")
locations, names = predict("1.jpg")
for name, location in zip(names, locations):
    name = " ".join(name.split("_"))
    print (name)
    if name == "Joshua Lo":
            name = "Rung-Chuan Lo"
    if name != "Unknown":
        id_ = name_to_id[name_to_id["Person"] == name].values.tolist()[0][0]
        
        result_dict["filename"].append("1.jpg")
        result_dict["person_id"].append(id_)
        result_dict["id"].append("1_%s" % id_)
        result_dict["xmin"].append(location[3])
        result_dict["xmax"].append(location[1])
        result_dict["ymin"].append(location[2])
        result_dict["ymax"].append(location[0])
import csv
print (result_dict)
with open('submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "person_id", "id", "xmin", "xmax", "ymin", 'ymax'])
    for i in range(len(result_dict["filename"])):
        row = [result_dict["filename"][i], result_dict["person_id"][i], result_dict["id"][i], result_dict["xmin"][i], 
              result_dict["xmax"][i], result_dict["ymin"][i], result_dict["ymax"][i]]
        writer.writerow(row)
    
print (len(result_dict["filename"]))