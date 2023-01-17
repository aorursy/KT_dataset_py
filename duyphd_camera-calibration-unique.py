import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
def showGrid(grid, img):
    '''
    input:  grid is array (numpy array format)
            img is numpy array format
    output: show gird by plot
    '''
    if grid is not None:
        x = []
        y = []
        if grid.ndim == 2:
            for ar in grid:
                x.append(ar[0])
                y.append(ar[1])
        elif grid.ndim == 3:
            for ar in grid:
                x.append(ar[0][0])
                y.append(ar[0][1])
        x = np.asarray(x, dtype='float32')
        y = np.asarray(y, dtype='float32')
        plt.figure()
        plt.plot(x, y, marker='.', color='r', linestyle='none')
        if img is not None:
            plt.imshow(img)
def show2Img(img1, img2):
    '''
    input: 2 img np array format
    output: display 2 img
    '''
    plt.figure()
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(img1)
    axes[1].imshow(img2)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30, 0.001)
# prepare object points, like (0, 0, 0), (1, 0, 0),
# (2, 0, 0)....(6, 5, 0)
# 3D points, point in chessboard (points in real world space)
objp = np.zeros((6*7, 3), np.float32)
objp
# assume that z = 0
# point: (x, y, z)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
objp
# show grid points in plot
showGrid(objp[:, :2], None)
# Arrays to store objects and image points from all the images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane
# join all file .jpg in sample_chessboard dir to list
images = glob.glob('/kaggle/input/sample-chessboard/*.jpg')
images
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find the chess board corners
    '''
    chess board corners, that is, points where
    the black squares touch each other
    input: img, patternSize(w(cols),h(rows)),...)
    output: ret, corners
    ret: True if find gird in chessboard ? False
    corners: matrix corner points
    '''
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    # If found, add obj points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        # cornerSubPix is used to increase accurancy for corner points
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11),
                                    (-1, -1), criteria)
        imgpoints.append(corners2)
        
#         Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                                objpoints, imgpoints, gray.shape[::-1],
                                None, None,
                                flags=(cv2.CALIB_FIX_PRINCIPAL_POINT))
img = cv2.imread('sample_chessboard/left08.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist,
                                                  (w, h), 1, (w, h))
# undistrort
#c1
# dst_non_crop = cv2.undistort(img, mtx, dist, None, newcameramtx)

#c2
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx,
                                         (w,h),5)
dst_non_crop = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
show2Img(img, dst_non_crop)
# roi is rectangles inside images where all the pixels are valid
x, y, w, h = roi
dst_non_crop = cv2.rectangle(dst_non_crop,(x, y),(x + w, y + h),(0,255,0),3)
# show2Img(img, dst_non_crop)
plt.figure()
plt.imshow(dst_non_crop)
# crop the image
x, y, w, h = roi
dst = dst_non_crop[y:y + h, x:x+w]
show2Img(img, dst)
