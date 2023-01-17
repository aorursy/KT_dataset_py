file_path="/kaggle/input/videotempfile/motorcycle racing-H264 75.mov"
print(file_path)
import cv2
# capture the video
cap = cv2.VideoCapture(file_path)

# check if capture was successful
if not cap.isOpened(): 
    print("Could not open!")
else:
    print("Video read successful!")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print('Total frames: ' + str(total_frames))
    print('width: ' + str(width))
    print('height: ' + str(height))
    print('fps: ' + str(fps))
#cap = cv2.VideoCapture(file_path)
#ret, frame = cap.read()
#while(1):
#    ret, frame = cap.read()
#    cv2.imshow('frame',frame)
#    if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
#        cap.release()
#        cv2.destroyAllWindows()
#        break
#    cv2.imshow('frame',frame)
## Creating Directory for storing the file . 
import os
if not os.path.exists('/kaggle/working/images'):
    os.makedirs('/kaggle/working/images')

## The length for Video  VID_20170801_192851240.mp4 is 104 Seconds and it creates 2833 images for it which is roughtly 27 image per second
import cv2
file_path="/kaggle/input/videotempfile/motorcycle racing-H264 75.mov"
vidcap = cv2.VideoCapture(file_path)
#vidcap = cv2.VideoCapture(0)
success,image = vidcap.read()

#################### Setting up parameters ################

seconds = 1
fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
#multiplier = fps * seconds

#################### Initiate Process ################

while success:
    frameId = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
    
    #cv2.imshow("myvideo",image.astype('uint8'))
    ## Every frame has 29 image so every second we are getting 5
    if frameId % 5 == 0:
        cv2.imwrite("/kaggle/working/images/Test_Image_%d.jpg" % frameId, image)
    #if cv2.waitKey(1) & 0xFF==25:
    #    break
    success, image = vidcap.read()
vidcap.release()
print ("Complete")
# Display the Folder 
!ls /kaggle/working/images |wc -l
import glob
import matplotlib.pyplot as plt
filenames = glob.glob("/kaggle/working/images/*.jpg")
#images = [cv2.imread(img) for img in filenames]
count=0
for file in filenames:
    imag1=cv2.imread(file)
    plt.imshow(imag1)
    plt.show()
