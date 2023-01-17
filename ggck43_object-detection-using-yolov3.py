import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

## this notebook aims at detecting objects with YOLOv3
## the whole code was done using cv2 and other libraries. Deep learning parts were handled using cv2
#Load Yolo
net = cv2.dnn.readNet("../input/object-detection-with-yolov3/weights/yolov3.weights", 
                      "../input/object-detection-with-yolov3/yolov3.cfg")
classes = []
with open("../input/object-detection-with-yolov3/coco.names","r") as file:
    lines = file.readlines()  
# line is now a list with all the individual lines of the file. Each line contain a classname

for line in lines:
    classes.append(line.strip())

print(type(classes), len(classes))
print(classes)
## now the classes list contains the name of all of the classes(80).
## this function shows two images side-by-side
def plot_two_images(img1, img2, title1="", title2=""):
    fig = plt.figure(figsize=[15,15])
    ax1= fig.add_subplot(121)
    ax1.imshow(img1, cmap="gray")
    ax1.set(xticks=[], yticks=[], title=title1)
    
    ax2= fig.add_subplot(122)
    ax2.imshow(img2, cmap="gray")
    ax2.set(xticks=[], yticks=[], title=title2)
#loading image
img = cv2.imread("../input/object-detection-with-yolov3/images/person.jpg") ## BGR format
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plot_two_images(img,rgb_img, 'BGR image', 'RGB_image')
height, width, channels = img.shape
print(type(img),height, width, channels)
temp_image2 = rgb_img.copy() #for later usage
#Detecting objects
## the main image needs to be converted into BLOB image for input of YOLO model
blob = cv2.dnn.blobFromImage(rgb_img, 0.00392, (416, 416), (0,0,0), True, crop = False)
print(blob.shape, type(blob)) # (1,3,416,416)

fig, axes = plt.subplots(ncols=3, figsize = (11,11))
for b in blob:
    for n, img_blob in enumerate(b):
        axes[n].imshow(img_blob)
plt.show()
net.setInput(blob) # blob is passed into the network
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]  -1] for i in net.getUnconnectedOutLayers()]
outs = net.forward(output_layers)
## now 'outs' is the list containing information about all the detected objects

print(len(outs), type(outs))
print(outs[0].size)
# following lists will contain the classID, confidences and box-coordinates of detected classes.
confidences = []; class_ids = []; boxes = []

height, width, channels = img.shape
for out in outs:
    for detection in out:
        scores = detection[5:] # scores contain the confidence for each class on the detected object
        class_id=np.argmax(scores) # the maximum confidence is the most-probable class
        confidence = scores[class_id] # taking the ID of the most likely class
        if confidence >0.5:
            #object is detected
            ## finding the co-ordinates to bound-box
            center_x= int(detection[0]*width)
            center_y= int(detection[1]*height)
            w= int(detection[2]*width)
            h= int(detection[3]*height)
            ## rectangle coordinates
            x= int(center_x - w/2)
            y= int(center_y - h/2)
            
            cv2.rectangle(rgb_img, (x,y), (x+w, y+h), (0,255,0),2)
            cv2.circle(rgb_img, (center_x, center_y), 10, (255, 0, 0),5)
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
myPlt = plt.imshow(rgb_img)

fig = myPlt.get_figure()
fig.savefig("output_detection.png")

print('number of boxes ', len(boxes))
print(temp_image2.shape)

print(class_ids)
for id in class_ids:
    print(classes[id]) ## these classes have been detected
    
## we can see that some classes have been detected twice. That's why we can see more bounding boxes. 
# Lets remove redundant boxes.
plt.imshow(temp_image2) # we'll make out further analysis on the image that we copied before
## this block of codes could be used if we wanted a random color-box for each classes.
#colors = np.random.uniform(0, 255, size = (len(classes), 3))
#print(colors)
#help(cv2.dnn.NMSBoxes)
## removing redundant boxes

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

print(indexes) # these indexes are the unique ones

font = cv2.FONT_HERSHEY_PLAIN

for i in range (len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        #color= colors[i]
        cv2.rectangle(temp_image2, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.putText(temp_image2, label, (x, y + 30), font, 3, (255,0,0), 3)
        print(label) # just for analysis purpose
    
plot_two_images(rgb_img, temp_image2, 'Initial detection', 'After removing redundant boxes')
cv2.imwrite('output_detection.png', rgb_img)
cv2.imwrite('result.png', temp_image2)
result_image = cv2.imread('./result.png')
plt.imshow(result_image)