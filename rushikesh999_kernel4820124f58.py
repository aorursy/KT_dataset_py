
#importing library 
import cv2
import numpy as np

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('D:\\OpenCV\\opencv\\build\\etc\\Haarcascades\\haarcascade_frontalface_default.xml')
# crop the image into 20 by 20 px and change it to B&W
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face
#Asks for the total number of users
num=int(input("Enter no of user you want to add "))
user=1
# Initialize Webcam 
cap = cv2.VideoCapture(0)
#count the total number of faces
count = 0
print("Start Capturing the input Data Set  ")
d=input('Enter Face name')
# Collect 100 samples of your face from webcam input
while True:
    ret, frame = cap.read()
    print(type(frame))
    #condition to check if face is in the frame or not
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = './faces/user/' +str(d)+'-' +str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass
    
    #checks if ech user 100 images is captured
    if(count%100)==0 and count!= num*100 and count!=0:
        print("Place the new user Signature")
        cv2.waitKey()
        
    #checks if total images are captured
    if cv2.waitKey(1) == 13 or count == num*100: #13 is the Enter Key
        break

#closes all windows
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")
#import all the liabry
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = 'faces/user/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
model = cv2.face.LBPHFaceRecognizer_create()


# Let's train our model 
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")
#importing the liabrary
import cv2
import numpy as np

#Creating object of HAAR classifier
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    #reads images from webcam
    ret, frame = cap.read()
    
    image, face = face_detector(frame)

    try:
        #converts it into required format
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the sucess value
        results = model.predict(face)
        
        if results[1] < 500:
            sucess = int( 100 * (1 - (results[1])/400) )
            display_string = str(sucess) + '% sucess Authorised User'+ str(d)
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        #checks for matching characterstics
        if sucess > 90:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    #if no face is present in the frame
    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
#exit()
cap.release()
cv2.destroyAllWindows()  