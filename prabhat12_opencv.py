!pip install imutils
# importing all the libraries
import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
import imutils
# reading a sample image and printing it's shape
path = "../input/visual-yolo-opencv/"
img = cv2.imread(path+"vehicles.jpg")
print(img.shape)
fig,axs = plt.subplots(1,2, figsize=[15,15])
axs[0].imshow(img)
axs[0].set_title("Wrong color channel: RGB image, read in BGR", fontsize = 15)
axs[0].axis('off')

correct_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

axs[1].imshow(correct_img)
axs[1].set_title("Correct color channel: RGB image changed to BGR", fontsize = 15)
axs[1].axis('off')

plt.show()

rows = open('../input/visual-yolo-opencv/synset_words.txt').read().strip().split("\n")
classes = [ row [ row.find(" ") + 1:].split(",")[0] for row in rows]

print("Total number of classes are: ", len(classes))
print("Before : ", rows[:5])
print("After : ", classes[:5])
google_net = cv2.dnn.readNetFromCaffe('../input/visual-yolo-opencv/bvlc_googlenet.prototxt','../input/visual-yolo-opencv/bvlc_googlenet.caffemodel')
# cafemodel requires image dimension to be 224x224
# blob is a 4D tensor obtained from the image
# blob from image parameters: ( image = input image , scalefactor = if scaling the image, 
# size = size of output image, mean, swapRB = swapping RGB to BGR, crop, ddepth)
blob = cv2.dnn.blobFromImage(correct_img, 1, (224,224))

# feeding the blob as input to the network
google_net.setInput(blob)

# getting the 1000 probabilities
result = google_net.forward()

# printing the result for first 10 classes
print("Length of the result: ", len(result[0]))

# finding the max 3 probabilities after sorting all of them in descending order
index = np.argsort(result[0])[::-1][:3]

print("\n\nTop 3 probabilities index : ", index)

# based on the index, retrieve the classes from synset
print("\nTop 3 probabilties of classes based on retrieved index\n")
      
for (i,id) in enumerate(index):
    print("{}. {} : Probability {:.3}%".format(i+1, classes[id], result[0][id]*100) + "\n")

def display_match(image):
    
    # txt to store the results
    txt=""    
    
    blob = cv2.dnn.blobFromImage(image, 1, (224,224))
    
    google_net.setInput(blob)

    result = google_net.forward()

    index = np.argsort(result[0])[::-1][:3]
    
    for (i,id) in enumerate(index):
        txt += "{}. {} : Probability {:.3}%".format(i+1, classes[id], result[0][id]*100) + "\n"
            
    return txt        
result = display_match(correct_img)

plt.figure(figsize=[10,10])
plt.imshow(correct_img)
plt.title(result)
plt.axis('off')
plt.show()
img_list = ["beach.jpg","cycle.jpg","dog.jpg","elephant.jpg","tiger.jpg",'laptop.jpg']

for i in range(len(img_list)):
    img_list[i] = path + img_list[i]
fig,axs = plt.subplots(3,2, figsize=[15,15])
fig.subplots_adjust(hspace=.5)

count=0
for i in range(3):    
    for j in range(2):        
        new_img = cv2.imread(img_list[count])
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        txt = display_match(new_img)
        #cv2.putText(new_img, txt, (0, 25 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        
        axs[i][j].imshow(new_img)
        axs[i][j].set_title(txt, fontsize = 12)
        axs[i][j].axis('off')
        count+=1

plt.suptitle("Top 3 predictions shown in title", fontsize = 18)
plt.show()
img = cv2.imread(img_list[4])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

txt = display_match(img)
line = txt.split("\n")

for i in range(3):
    cv2.putText(img, str(line[i]), (10, 30 + 40*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)


plt.figure(figsize=[10,10])
plt.imshow(img)
plt.title("Writing probability on the image", fontsize = 15)
plt.axis('off')

plt.show()

vid = cv2.VideoCapture('../input/visual-yolo-opencv/computer.mp4')

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_computer.mp4',fourcc, 20.0, (640, 640))

if(vid.isOpened==False):
    print("Can't open the video file")
    
try:
    while(True):

        ret, frame = vid.read()
        if not ret:
            vid.release()
            out.release()
            print("Completed! Read all the frames.")
            break
            
        txt = display_match(frame)
        line = txt.split("\n")
        for i in range(3):
            cv2.putText(frame, str(line[i]), (10, 30 + 40*i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,0,0), 2)

        resized_frame = cv2.resize(frame,(640,640))
        out.write(resized_frame)
        plt.imshow(resized_frame)
        plt.show()
        
        clear_output(wait=True)

except KeyboardInterrupt:
    vid.release()
    print("Error in reading frames")
from IPython.display import YouTubeVideo
YouTubeVideo('EX7ULuSPpaY', width=800, height=450)
vid = cv2.VideoCapture('../input/visual-yolo-opencv/video.mp4')

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_study.mp4',fourcc, 20.0, (1800, 1800))

if(vid.isOpened==False):
    print("Can't open the video file")

try:
    while(True):
        
           
        ret, frame = vid.read()
        if not ret:
            vid.release()
            out.release()
            print("Completed! Read all the frames.")
            break
            
        txt = display_match(frame)
        line = txt.split("\n")
        for i in range(3):
            cv2.putText(frame, str(line[i]), (70, 300 + 250*i), cv2.FONT_HERSHEY_SIMPLEX, 8, (255,0,0), 25)

        resized_frame = cv2.resize(frame,(1800,1800))
        out.write(resized_frame)
        #plt.imshow(resized_frame)
        #plt.show()
        
        #clear_output(wait=True)

except KeyboardInterrupt:
    vid.release()
    print("Error in reading frames")
YouTubeVideo('liUmyzGuc70', width=800, height=450)
