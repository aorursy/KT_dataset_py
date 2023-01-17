!pip install face_recognition
# importing all the important libraries

import numpy as np 

import pandas as pd

import PIL.Image

import PIL.ImageDraw

import face_recognition

import matplotlib.pyplot as plt
# adding the image

manyPeople_img = face_recognition.load_image_file("../input/facerecognition/people.jpg")



plt.imshow(manyPeople_img)

plt.axis('off')

plt.show()

# the image is loaded as an array representing the pixels value in numpy array, printing the shape of array 

manyPeople_img.shape
# finding location of each face from the array, by default hog is used. We can choose cnn to

faceLocMany = face_recognition.face_locations(manyPeople_img, model="hog")

print("Number of faces: ",len(faceLocMany))

print("\nPosition of each face (top, right, bottom, left) : \n", faceLocMany)
# creates an image memory from numpy array

manyPeople_pil = PIL.Image.fromarray(manyPeople_img)
# drawing a ellipse around the faces in the image 

for faceLoc in faceLocMany:



    top, right, bottom, left = faceLoc   

    

    # creating an instance of Draw, to draw on the image

    draw = PIL.ImageDraw.Draw(manyPeople_pil)

    draw.ellipse([(left, top), (right, bottom)], outline="red", width=3)



plt.imshow(manyPeople_pil)

plt.axis('off')

plt.show()
# single person image

personImage = face_recognition.load_image_file("../input/facerecognition/person.jpg")



personFaceLoc = face_recognition.face_locations(personImage)



print("Face location:", personFaceLoc)



person_pil = PIL.Image.fromarray(personImage)



top, right, bottom, left = list(personFaceLoc[0])



# croping the face area

head = person_pil.crop((left, top, right, bottom))



plt.imshow(head)

plt.axis('off')

plt.show()
# finally pasting it on top of group image

manyPeopleCopy_pil = manyPeople_pil.copy()

for faceLoc in faceLocMany:



    top, right, bottom, left = faceLoc   

    

    head = head.resize((abs(right-left),abs(top-bottom)), PIL.Image.ANTIALIAS)

    manyPeopleCopy_pil.paste(head, [left, top])



plt.imshow(manyPeopleCopy_pil)

plt.axis('off')

plt.show()
# finding the landmarks for single person

faceLandmark_list = face_recognition.face_landmarks(personImage)

faceLandmark_list[0].keys()
# testing it for multiple people

test_faceLandmark_list = face_recognition.face_landmarks(manyPeople_img)

print("Number of faces" , len(test_faceLandmark_list))

print("Landmarks for each face", test_faceLandmark_list[0].keys())

# creating a copy first so that we can use the original again

personCopy_pil = person_pil.copy()



for faceLandmark in faceLandmark_list:



    # looping over each facial feature

    for name, list_of_points in faceLandmark.items():



        # printing the location of each facial feature

        print("The {} in this face has the following points: {}".format(name, list_of_points))



        # drawing a line on each facial feature

        draw = PIL.ImageDraw.Draw(personCopy_pil)

        draw.line(list_of_points, fill="red", width=2)



plt.imshow(personCopy_pil)

plt.axis('off')

plt.show()

# eyebrow co-ordinates

left_eyebrow =  faceLandmark_list[0].get('left_eyebrow')

right_eyebrow = faceLandmark_list[0].get('right_eyebrow')

print("Left eyebrow: ", left_eyebrow)

print("\nRight eyebrow: ", right_eyebrow)
# finding the extreme points

leftmost_point = left_eyebrow[0]

rightmost_point = right_eyebrow[len(right_eyebrow)-1]



print("Leftmost point: ", leftmost_point , " and Righmost point: ", rightmost_point)
# loading the person image and goggles image

# ( we have first load the person's image as an array, thus we need to load it again)

goggleImage = PIL.Image.open("../input/facerecognition/transparent_goggles.png")

image_person = PIL.Image.open("../input/facerecognition/person.jpg")



# resize the goggle's image in accordance with the eyebrows extreme point

goggleImage = goggleImage.resize((abs(rightmost_point[0]-leftmost_point[0]),int(0.5*abs(rightmost_point[0]-leftmost_point[0]))), PIL.Image.ANTIALIAS)

final1 = PIL.Image.new("RGBA", image_person.size)



# finally pasting the goggles on top of the face

final1.paste(image_person, (0,0))

final1.paste(goggleImage, (leftmost_point[0],leftmost_point[1]), goggleImage)



plt.imshow(final1)

plt.axis('off')

plt.show()
# we can see that image_person is an image not a numpy array like before

image_person
# finding face encodings

personEncoding = face_recognition.face_encodings(personImage)[0]

print(personEncoding)
unknownPerson1 = face_recognition.load_image_file("../input/facerecognition/unknown_1.jpg")

unknownPerson2 = face_recognition.load_image_file("../input/facerecognition/unknown_8.jpg")

unknownPerson3 = face_recognition.load_image_file("../input/facerecognition/unknown_6.jpg")

unknownPerson4 = face_recognition.load_image_file("../input/facerecognition/person_1.jpg")



fig,axs = plt.subplots(2,2, figsize = [10,10])



axs[0][0].imshow(unknownPerson1)

axs[0][0].set_title('Unknown Person 1')



axs[0][1].imshow(unknownPerson2)

axs[0][1].set_title('Unknown Person 2')



axs[1][0].imshow(unknownPerson3)

axs[1][0].set_title('Unknown Person 3')



axs[1][1].imshow(unknownPerson4)

axs[1][1].set_title('Unknown Person 4')



plt.suptitle("Unknown Persons")

plt.show()

# creating a list of unknown face encodings

unknownPersonList = [

    

    face_recognition.face_encodings(unknownPerson1)[0],

    face_recognition.face_encodings(unknownPerson2)[0],

    face_recognition.face_encodings(unknownPerson3)[0],

    face_recognition.face_encodings(unknownPerson4)[0]        

]



distance=[]

match=[]

# for each unknown face encoding, comapre it with the single person encoding

for (i,unknownPerson) in enumerate(unknownPersonList):



    result = face_recognition.compare_faces( [personEncoding], unknownPerson, tolerance=0.6)[0]

    print("Person", i, " is a match? ", result)



    face_distance = face_recognition.face_distance([personEncoding], unknownPerson)[0]

    print("Euclidian distance between faces for person", int(i)+1 ," :", face_distance)

    

    distance.append(face_distance)

    match.append(result)



    print()



fig,axs = plt.subplots(3,2, figsize = [10,10])



fig.tight_layout(pad = 5.0)



plt.suptitle("Comparision of faces", fontsize = 20)





axs[0][0].imshow(personImage)

axs[0][0].set_title("Known person")

axs[0][0].axis('off')



fig.delaxes(axs[0][1])



axs[1][0].imshow(unknownPerson1)

axs[1][0].set_title('1. Is a match: ' + str(match[0]) + '\n Face euclidean distance: ' + str("{:.2f}".format(distance[0])))

axs[1][0].axis('off')



axs[1][1].imshow(unknownPerson2)

axs[1][1].set_title('2. Is a match: ' + str(match[1]) + '\n Face euclidean distance: ' + str("{:.2f}".format(distance[1])))

axs[1][1].axis('off')



axs[2][0].imshow(unknownPerson3)

axs[2][0].set_title('3. Is a match: ' + str(match[2]) + '\n Face euclidean distance: ' + str("{:.2f}".format(distance[2])))

axs[2][0].axis('off')



axs[2][1].imshow(unknownPerson4)

axs[2][1].set_title('4. Is a match: ' + str(match[3]) + '\n Face euclidean distance: ' + str("{:.2f}".format(distance[3])))

axs[2][1].axis('off')





plt.show()

# image containing more than one clear face

groupOfUnknown = face_recognition.load_image_file('../input/facerecognition/unknown_2.jpg')



groupEncoding = face_recognition.face_encodings(groupOfUnknown)



print("Number of persons: ", len(groupEncoding))



for (i,encoding) in enumerate(groupEncoding):



    result = face_recognition.compare_faces( [personEncoding], encoding, tolerance=0.6)[0]

    

    if(result == True):

        print("Match found")

        break

    

    

plt.imshow(groupOfUnknown)

plt.show()
low_resolution = face_recognition.load_image_file('../input/facerecognition/unknown_7.jpg')



encoding = face_recognition.face_encodings(low_resolution)



print("Before : Number of persons: ", len(encoding) , " due to low resolution")



faceLoc = face_recognition.face_locations( low_resolution, number_of_times_to_upsample = 2 )

encoding = face_recognition.face_encodings(low_resolution, known_face_locations = faceLoc)



print("After : Number of persons: ", len(encoding))



result = face_recognition.compare_faces( personEncoding, encoding, tolerance=0.6)[0]

print("Match found ?", result)



face_distance = face_recognition.face_distance(personEncoding, encoding)[0]

print("Euclidian distance between faces: ", face_distance)

    

    

plt.imshow(low_resolution)

plt.show()