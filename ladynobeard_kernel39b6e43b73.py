# all algorithms and their use cases available in OpenCV in the field of image registration:



# 1:	keypoint detector and descriptor extractor

# 2:	only keypoint detector

# 3:	only keypoint descriptor

# 4:	descriptor extractor

# 5:	descriptor matcher



# o:	free to use

# x:	non free (patented)



# e:	experimental (v3.1.0)

# n:	non-experimental (v3.1.0)



# only caring about label 1s

# b:	blob

# c:	corner

# e:	edge



# Algorithm	Use cases	License	status	

# AKAZE	1	o	        n	            

# MSER	2	o	        n	            

# BRISK	1	o	        n	            

# ORB	1	o	        n	            

# KAZE	1	o	        n	            

# FAST	2	o	        n	            

# SURF	1	x	        n	            

# SIFT	1	x	        n	            

# FREAK	3	o	        e	            

# BRIEF	4	o	        e	            

# DAISY	3	o	        e	            

# LATCH	3	o	        e	            

# LUCID	3	o	        e	            

# STAR	2	o	        e	            

# MSD	2	o	        e	            

# BF	5	o	        n	            

# FLANN	5	o	        n	 

import cv2 as cv



def lowe_ratio_test(matches, loweTh):

    """ ratio test as per Lowe's paper """

    matchesMask = []

    for m, n in matches:

        if m.distance < loweTh * n.distance:

            matchesMask.append(m)

    return matchesMask



kaze = cv.KAZE_create(nOctaveLayers = 9, diffusivity = cv.KAZE_DIFF_PM_G2)

imgSrc = cv.imread(img1)

imgDst = cv.imread(img2)



# 1. extracting key points and description

kp1, des1 = kaze.detectAndCompute(imgSrc, None)

kp2, des2 = kaze.detectAndCompute(imgDst, None)



# 2. match the descriptions of the key points

bf = cv.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)



# 3. if you're using brute force k nearest neight (bf.knnMatch), 

# lowe's ratio can be used to get rid of bad matches (https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

matches = lowe_ratio_test(matches, 0.6)



# 4. intrinsic camera matrix (if you choose to extract the essential matrix)

# can be set as [image width,            0, image width  / 2]

#               [          0, image height, image height / 2]

#               [          0,            0,                1]

K = np.array([[595, 0, 297.5],[0, 32, 16], [0, 0, 1]])



# 5. gathering points for H=omography matrix

ptsSrc = np.zeros((len(matches), 2), dtype=np.float32)

ptsDst = np.zeros((len(matches), 2), dtype=np.float32)

    

for i, mat in enumerate(matches):

    ptsSrc[i, :] = kp1[mat.queryIdx].pt

    ptsDst[i, :] = kp2[mat.trainIdx].pt



# 6. find your homography matrix

# We need at least 4 corresponding points. 

h, mask = cv.findHomography(ptsSrc, ptsDst)

# e, mask = cv.findEssentialMat(ptsSrc, ptsDst, K, cv.RANSAC)



# 7. The calculated homography can be used to warp the source image to destination. 

imgDst = cv.warpPerspective(imgSrc, h, size(imgDst))