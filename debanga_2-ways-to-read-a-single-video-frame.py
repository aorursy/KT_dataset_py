import numpy as np

import matplotlib.pyplot as plt

import cv2

import time
""" Path to the sample video """

filename = "/kaggle/input/samplevideo/SampleVideo_1280x720_5mb.mp4"
# capture the video

cap = cv2.VideoCapture(filename)



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
start = time.time()

cap = cv2.VideoCapture(filename)

for i in range(total_frames):

    success = cap.grab()

    if (i == (total_frames-1)):

        ret, image = cap.retrieve()

        end = time.time()

        plt.figure(1)

        plt.imshow(image)

print("Total time taken: " + str(end-start) + " seconds") 
start = time.time()

cap = cv2.VideoCapture(filename)

cap.set(1,total_frames-1);

success = cap.grab()

ret, image = cap.retrieve()

end = time.time()

plt.figure(2)

plt.imshow(image)

print("Total time taken: " + str(end-start) + " seconds")  
# Approach 1

timer_approach1 = []

for loop in range(20):

    start = time.time()

    cap = cv2.VideoCapture(filename)

    for i in range(total_frames):

        success = cap.grab()

        if (i == (total_frames-1)):

            ret, image = cap.retrieve()

            end = time.time()

            timer_approach1.append(end - start)

            

# Approach 2

timer_approach2 = []

for loop in range(20):

    start = time.time()

    cap = cv2.VideoCapture(filename)

    cap.set(1,total_frames-1);

    success = cap.grab()

    ret, image = cap.retrieve()

    end = time.time()

    timer_approach2.append(end - start)

    

# Plot the results

plt.figure(3)

plt.plot(timer_approach1)

plt.title('Approach 1 in seconds')



plt.figure(4)

plt.plot(timer_approach2)

plt.title('Approach 2 in seconds')