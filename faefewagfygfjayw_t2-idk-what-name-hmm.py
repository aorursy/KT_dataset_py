!pip install face_recognition

import face_recognition

import cv2

import numpy as np

import os, sys



filename = "../input/rawraw/raw"



def preprocess(filename):

    global known_face_names

    global known_face_encodings

    path = os.getcwd() + "/" + filename

    dirs = os.listdir(path)

    for items in dirs:

        if not os.path.isdir(path+"/"+items):

            continue

        dir2 = os.listdir(path+"/"+items)

        for item in dir2:

            count = 0

            itemz = items

            item = path+"/"+items + "/" + item

            if os.path.isfile(item):    

                print(item)       

                #print(" ".join(itemz.lower().split("_")[:2]))

                try:

                    image = face_recognition.load_image_file(item)

                    encoding = face_recognition.face_encodings(image)[0]

                    print(itemz.lower().split("_"))

                    print(" ".join(itemz.lower().split("_")[:2]))

                    known_face_names.append(" ".join(itemz.lower().split("_")[:2]))

                    

                    known_face_encodings.append(encoding)

                except Exception as e:

                    print(e)

                    continue









# Create arrays of known face encodings and their names

known_face_encodings = [

]

known_face_names = [

]

preprocess(filename)



# Initialize some variables

face_locations = []

face_encodings = []

face_names = []

process_this_frame = True



def preprocess2(filename):

    global known_face_names

    global known_face_encodings

    path = os.getcwd() + "/" + filename

    dirs = os.listdir(path)

    for items in dirs:

        if not os.path.isdir(path+"/"+items):

            continue

        dir2 = os.listdir(path+"/"+items)

        for item in dir2:

            count = 0

            itemz = items

            item = path+"/"+items + "/" + item

            if os.path.isfile(item):    

                print(item)       

                #print(" ".join(itemz.lower().split("_")[:2]))

                try:

                    image = face_recognition.load_image_file(item)

                    encoding = face_recognition.face_encodings(image)[0]

                    

                    known_face_names.append(" ".join(itemz.lower().split(" ")[:2]))

                    known_face_encodings.append(encoding)

                except Exception as e:

                    print(e)

                    continue

filename = "../input/comlcoml/COML Face Dataset"  

preprocess2(filename)


# Grab a single frame of video

path = "../input/rawraw/raw/david_baek"

dirs = os.listdir(path)

for items in dirs:

    image = path+"/"+items

    frame = cv2.imread(image)



    # ignore

    small_frame = frame



    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

    rgb_small_frame = small_frame[:, :, ::-1]



    # Only process every other frame of video to save time



    # Find all the faces and face encodings in the current frame of video

    face_locations = face_recognition.face_locations(rgb_small_frame)

    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)



    face_names = []

    for face_encoding in face_encodings:

        # See if the face is a match for the known face(s)

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"



        # # If a match was found in known_face_encodings, just use the first one.

        # if True in matches:

        #     first_match_index = matches.index(True)

        #     name = known_face_names[first_match_index]



        # Or instead, use the known face with the smallest distance to the new face

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:

            name = known_face_names[best_match_index]



        face_names.append(name)





    # Display the results

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # Draw a box around the face

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)



        # Draw a label with a name below the face

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        print(name.lower())

        print("top: " + str(top))

        print("right: " + str(right))

        print("bottom: " + str(bottom))

        print("left: " + str(left))

    cv2.imwrite("david.jpg", frame)

required = set()

file = "../input/asn10e-final-submission-coml-facial-recognition/sample_submission.csv"

with open(file, newline='') as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

        print(row['id'])

        required.add(str(row['id']))

print(required)

print(len(required))
import csv

file = "../input/asn10e-final-submission-coml-facial-recognition/person_id_mapping.csv"

person_to_id = dict()

with open(file, newline='') as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

        print(row['person_id'], row['Person'].lower())

        person_to_id[row['Person'].lower()] = int(row['person_id'])

print(person_to_id)

required = set()

file = "../input/asn10e-final-submission-coml-facial-recognition/sample_submission.csv"

with open(file, newline='') as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

        print(row['id'])

        required.add(str(row['id']))



# Grab a single frame of video

path = "../input/rawraw/raw/david_baek"

dirs = os.listdir(path)

for items in dirs:

    image = path+"/"+items

    frame = cv2.imread(image)



    # ignore

    small_frame = frame



    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

    rgb_small_frame = small_frame[:, :, ::-1]



    # Only process every other frame of video to save time



    # Find all the faces and face encodings in the current frame of video

    face_locations = face_recognition.face_locations(rgb_small_frame)

    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)



    face_names = []

    for face_encoding in face_encodings:

        # See if the face is a match for the known face(s)

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"



        # # If a match was found in known_face_encodings, just use the first one.

        # if True in matches:

        #     first_match_index = matches.index(True)

        #     name = known_face_names[first_match_index]



        # Or instead, use the known face with the smallest distance to the new face

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:

            name = known_face_names[best_match_index]



        face_names.append(name)





    # Display the results

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # Draw a box around the face

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)



        # Draw a label with a name below the face

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        print(person_to_id[name.lower()])

        print("top: " + str(top))

        print("right: " + str(right))

        print("bottom: " + str(bottom))

        print("left: " + str(left))

    cv2.imwrite("david.jpg", frame)



def write_to_csv(row_name, data):

    # row_name: the name of each row, save them as a string list

    # data: the data you want to store into csv

    

    f = open('res.csv','w', newline='')

    w = csv.writer(f)

    

    w.writerow(row_name)

    for i in range(0,len(data)-1):

        w.writerow(data[i])

    return
import pandas as pd

import matplotlib.pyplot as plt

labels = ['filename', 'person_id', 'id', 'xmin', 'xmax', 'ymin', 'ymax']



path = "../input/asn10e-final-submission-coml-facial-recognition"

dirs = os.listdir(path)

master = []

seen = set()



f = plt.figure(figsize=(90, 90))

count = 0

required = set()

file = "../input/asn10e-final-submission-coml-facial-recognition/sample_submission.csv"

with open(file, newline='') as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

        required.add(str(row['id']))

for items in dirs:

    if not items.endswith('jpg'):

        continue

    

    image = path+"/"+items

    frame = cv2.imread(image)



    # ignore

    small_frame = frame



    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

    rgb_small_frame = small_frame[:, :, ::-1]



    # Only process every other frame of video to save time



    # Find all the faces and face encodings in the current frame of video

    face_locations = face_recognition.face_locations(rgb_small_frame)

    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)



    face_names = []

    

    for face_encoding in face_encodings:

        

        

        # See if the face is a match for the known face(s)

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"



        # # If a match was found in known_face_encodings, just use the first one.

        # if True in matches:

        #     first_match_index = matches.index(True)

        #     name = known_face_names[first_match_index]



        # Or instead, use the known face with the smallest distance to the new face

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:

            name = known_face_names[best_match_index]



        face_names.append(name)





    # Display the results

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # Draw a box around the face

        result = []

        

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        

        # Draw a label with a name below the face

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if name.lower() in person_to_id:

            '''

            print(person_to_id[name.lower()])

            print("top: " + str(top))

            print("right: " + str(right))

            print("bottom: " + str(bottom))

            print("left: " + str(left))

            

            if person_to_id[name.lower()] in seen:

                continue

            '''

            if not items[0] + '_'+str(person_to_id[name.lower()])  in required:

                continue

            

            result.append(items)

            result.append(person_to_id[name.lower()])

            result.append(items[0] + '_'+str(person_to_id[name.lower()]))

            result.append(left)

            result.append(right)

            result.append(top)

            result.append(bottom)

            required.remove(items[0] + '_'+str(person_to_id[name.lower()]))

            master.append(result)

    count+=1

    f.add_subplot(2,1, count)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    plt.imshow(frame) 

#pad for 44 rows 

for x in required:

    result=[]

    split = x.split('_')

    result.append(split[0] + '.jpg')

    result.append(int(split[1]))

    result.append(x)

    result.append(0)

    result.append(0)

    result.append(0)

    result.append(0)

    master.append(result)





plt.show()

df = pd.DataFrame(master,columns=labels)

df.to_csv(index=False)
#pad to 44 rows

df.to_csv('IDK2.csv', encoding='utf-8', index=False)