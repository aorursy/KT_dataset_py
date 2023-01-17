!python -V

!python -m pip install --upgrade pip
!pip install mediapipe --use-feature=2020-resolver
import os

print(os.listdir('../input/shuffledance/ShuffleDance'))
datapath = '../input/shuffledance/ShuffleDance/'
import mediapipe as mp

import cv2

import matplotlib.pyplot as plt
print(cv2.__version__)
img = cv2.imread(datapath+'ShuffleDance0.jpg')

plt.imshow(img)
pose_tracker = mp.examples.UpperBodyPoseTracker()
input_file  = datapath+'ShuffleDance0.jpg'

output_file = './SuffleDance_pose.jpg'
pose_landmarks, _ = pose_tracker.run(input_file=input_file, output_file=output_file)
img = cv2.imread('./SuffleDance_pose.jpg')

plt.imshow(img)
pose_landmarks, annotated_image = pose_tracker.run(input_file=input_file)
# To print out the pose landmarks, you can simply do "print(pose_landmarks)".

# However, the data points can be more accessible with the following approach.

[print('x is', data_point.x, 'y is', data_point.y, 'z is', data_point.z, 'visibility is', data_point.visibility) for data_point in pose_landmarks.landmark]
#pose_tracker.run_live()
# Close the tracker

pose_tracker.close()