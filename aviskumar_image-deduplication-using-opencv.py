!pip install opencv-python==3.4.2.16
!pip install opencv-contrib-python==3.4.2.16
# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt

#For colab cv2 imshow
from google.colab.patches import cv2_imshow
img1=cv2.imread('/content/sample_data/IMG-1627.JPG')
img2=cv2.imread('/content/sample_data/IMG-1627 - Copy.JPG')

print(img1.shape,img2.shape)
plt.subplot(121)
plt.imshow(img1)

plt.subplot(122)
plt.imshow(img2)
# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None) 

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
print(len(matches))
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 25 matches.
img_matches = cv2.drawMatches(img1,kp1,img2,kp2,matches[:25],None,flags=2)
print(len(img_matches))
plt.imshow(img_matches)
plt.show()
# Create SIFT Object
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
print(matches)
# Apply ratio test
good = []
#Less Distance == Better Match
for match1,match2 in matches:
    if match1.distance < 0.75*match2.distance:
        good.append([match1])
print(len(matches),len(good))

#Since both are almost similar images, the difference between them both is nearly 50%
# cv2.drawMatchesKnn expects list of lists as matches.
sift_matches = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
plt.imshow(sift_matches)
plt.show()
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)  

flann = cv2.FlannBasedMatcher(index_params,search_params)

#K nearest matches
matches = flann.knnMatch(des1,des2,k=2)
#This block is for making green lines for similar features

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test
for i,(match1,match2) in enumerate(matches):
    if match1.distance < 0.7*match2.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

flann_matches = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(flann_matches)
plt.show()
#This block is for finding the score

good = []

# ratio test
for match1,match2 in matches:
    if match1.distance < 0.7*match2.distance:
        good.append([match1])
number_keypoints = 0
if len(kp1) >= len(kp2):
  print('img1')
  number_keypoints = len(kp1)
else:
  number_keypoints = len(kp2)
  print('img2')
print(number_keypoints,len(good))

percentage_similarity = len(good) / number_keypoints * 100
print("Similarity: " + str(int(percentage_similarity)) + "\n")

#Since img1 has more features, we can keep img1 and delete img2
print(len(matches),len(good))

#Since both are almost similar images, the difference between them both is nearly 50%

score=(len(good)/len(matches))*100
print(score)

if len(good) < (0.03*len(matches)):
  print("Both images are not same")
else:
  print('Both images are same')
import os
import cv2

folder_path='/content/sample_data/img_folder/'

# Create SIFT Object
sift = cv2.xfeatures2d.SIFT_create()
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)  

flann = cv2.FlannBasedMatcher(index_params,search_params)
for i in os.listdir(folder_path):
  i1_name=i
  i=folder_path+i
  img1_temp=cv2.imread(i)
  kp1_temp, des1_temp = sift.detectAndCompute(img1_temp,None)
  for j in os.listdir(folder_path):
    i2_name=j
    j=folder_path+j
    print(i1_name,i2_name)
    if i != j:
      img2_temp=cv2.imread(j)
      kp2_temp, des2_temp = sift.detectAndCompute(img2_temp,None)
      #K nearest matches
      matches_temp = flann.knnMatch(des1_temp,des2_temp,k=2)

      good_temp = []
      # ratio test
      for match1,match2 in matches_temp:
        if match1.distance < 0.7*match2.distance:
          good_temp.append([match1])
      number_keypoints_temp = 0

      if len(kp1_temp) >= len(kp2_temp):
        number_keypoints_temp = len(kp1_temp)
        del_img= i2_name
      else:
        number_keypoints_temp = len(kp2_temp)
        del_img= i1_name
      print(number_keypoints_temp,len(good_temp))

      percentage_similarity = len(good_temp) / number_keypoints_temp * 100
      print("Similarity: " + str(int(percentage_similarity)) )

      if int(percentage_similarity) > 3:
        print('Images ' +i1_name+ ' and ' +i2_name+ ' are duplicates and we can delete '+del_img + "\n" )
      else:
        print('Images ' +i1_name+ ' and ' +i2_name+ ' are not duplicates' + "\n")
