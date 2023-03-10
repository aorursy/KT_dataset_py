# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#!/usr/bin/python3

#!--*-- coding: utf-8 --*--

### OpenPose using OpenCV DNN

from __future__ import division

import cv2

import time

import numpy as np

import matplotlib.pyplot as plt

import os



class general_pose_model(object):

    def __init__(self, modelpath, mode="BODY25"):

        # 指定采用的模型

        #   Body25: 25 points

        #   COCO:   18 points

        #   MPI:    15 points

        self.inWidth = 368

        self.inHeight = 368

        self.threshold = 0.1

        if mode == "BODY25":

            self.pose_net = self.general_body25_model(modelpath)

        elif mode == "COCO":

            self.pose_net = self.general_coco_model(modelpath)

        elif mode == "MPI":

            self.pose_net = self.get_mpi_model(modelpath)

			

    def get_mpi_model(self, modelpath):

        self.points_name = { 

            "Head": 0, "Neck": 1, 

            "RShoulder": 2, "RElbow": 3, "RWrist": 4,

            "LShoulder": 5, "LElbow": 6, "LWrist": 

            7, "RHip": 8, "RKnee": 9, "RAnkle": 10, 

            "LHip": 11, "LKnee": 12, "LAnkle": 13, 

            "Chest": 14, "Background": 15 }

        self.num_points = 15

        self.point_pairs = [[0, 1], [1, 2], [2, 3], 

                            [3, 4], [1, 5], [5, 6], 

                            [6, 7], [1, 14],[14, 8], 

                            [8, 9], [9, 10], [14, 11], 

                            [11, 12], [12, 13]

                            ]

        prototxt = os.path.join(

            modelpath,

            "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt")

        caffemodel = os.path.join(

            modelpath, 

            "pose/mpi/pose_iter_160000.caffemodel")

        mpi_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)



        return mpi_model

    def general_coco_model(self, modelpath):

        self.points_name = {

            "Nose": 0, "Neck": 1, 

            "RShoulder": 2, "RElbow": 3, "RWrist": 4,

            "LShoulder": 5, "LElbow": 6, "LWrist": 7, 

            "RHip": 8, "RKnee": 9, "RAnkle": 10, 

            "LHip": 11, "LKnee": 12, "LAnkle": 13, 

            "REye": 14, "LEye": 15, 

            "REar": 16, "LEar": 17, 

            "Background": 18}

        self.num_points = 18

        self.point_pairs = [[1, 0], [1, 2], [1, 5], 

                            [2, 3], [3, 4], [5, 6], 

                            [6, 7], [1, 8], [8, 9],

                            [9, 10], [1, 11], [11, 12], 

                            [12, 13], [0, 14], [0, 15], 

                            [14, 16], [15, 17]]

        prototxt   = os.path.join(

            modelpath, 

            "pose/coco/pose_deploy_linevec.prototxt")

        caffemodel = os.path.join(

            modelpath, 

            "pose/coco/pose_iter_440000.caffemodel")

        coco_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)



        return coco_model



    def general_body25_model(self, modelpath):

        self.num_points = 25

        self.point_pairs = [[1, 0], [1, 2], [1, 5], 

                            [2, 3], [3, 4], [5, 6], 

                            [6, 7], [0, 15], [15, 17], 

                            [0, 16], [16, 18], [1, 8],

                            [8, 9], [9, 10], [10, 11], 

                            [11, 22], [22, 23], [11, 24],

                            [8, 12], [12, 13], [13, 14], 

                            [14, 19], [19, 20], [14, 21]]

        prototxt   = os.path.join(

            modelpath, 

            "pose/body_25/pose_deploy.prototxt")

        caffemodel = os.path.join(

            modelpath, 

            "pose/body_25/pose_iter_584000.caffemodel")

        coco_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)



        return coco_model

		

    def predict(self, img_cv2):

#    def predict(self, imgfile):

#        img_cv2 = cv2.imread(imgfile)

        img_height, img_width, _ = img_cv2.shape

        inpBlob = cv2.dnn.blobFromImage(img_cv2, 

                                        1.0 / 255, 

                                        (self.inWidth, self.inHeight),

                                        (0, 0, 0), 

                                        swapRB=False, 

                                        crop=False)

        self.pose_net.setInput(inpBlob)

        self.pose_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        self.pose_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)



        output = self.pose_net.forward()



        H = output.shape[2]

        W = output.shape[3]

#        print(output.shape)



        # vis heatmaps

        #self.vis_heatmaps(img_file, output)

        #

        points = []

        for idx in range(self.num_points):

            probMap = output[0, idx, :, :] # confidence map.



            # Find global maxima of the probMap.

            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)



            # Scale the point to fit on the original image

            x = (img_width * point[0]) / W

            y = (img_height * point[1]) / H



            if prob > self.threshold:

                points.append((int(x), int(y)))

            else:

                points.append(None)



        return points



    def vis_heatmaps(self, img_cv2, net_outputs):

#    def vis_heatmaps(self, imgfile, net_outputs):

#        img_cv2 = cv2.imread(imgfile)

        plt.figure(figsize=[10, 10])



        for pdx in range(self.num_points):

            probMap = net_outputs[0, pdx, :, :]

            probMap = cv2.resize(

                probMap, 

                (img_cv2.shape[1], img_cv2.shape[0])

            )

            plt.subplot(5, 5, pdx+1)

            plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

            plt.imshow(probMap, alpha=0.6)

            plt.colorbar()

            plt.axis("off")

        plt.show()



    def vis_pose(self, img_cv2, points):

#    def vis_pose(self, imgfile, points):

#        img_cv2 = cv2.imread(imgfile)

        img_cv2_copy = np.copy(img_cv2)

#        for idx in range(len(points)):

#            if points[idx]:

#                cv2.circle(img_cv2_copy, 

#                           points[idx], 

#                           8, 

#                           (0, 255, 255), 

#                           thickness=-1,

#                           lineType=cv2.FILLED)

#                cv2.putText(img_cv2_copy, 

#                            "{}".format(idx), 

#                            points[idx], 

#                            cv2.FONT_HERSHEY_SIMPLEX,

#                            1, 

#                            (0, 0, 255), 

#                            2, 

#                            lineType=cv2.LINE_AA)



        # Draw Skeleton

        for pair in self.point_pairs:

            partA = pair[0]

            partB = pair[1]



            if points[partA] and points[partB]:

                cv2.line(img_cv2, 

                         points[partA], 

                         points[partB], 

                         (0, 255, 255), 2)

                cv2.circle(img_cv2, 

                           points[partA], 

                           3, 

                           (0, 0, 255), 

                           thickness=-1, 

                           lineType=cv2.FILLED)					   

        cv2.imshow('image', img_cv2) # display image

        cv2.waitKey(1)               # wait for vision

#        plt.figure(figsize=[10, 10])

#        plt.subplot(1, 2, 1)

#        plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

#        plt.axis("off")

#        plt.subplot(1, 2, 2)

#        plt.imshow(cv2.cvtColor(img_cv2_copy, cv2.COLOR_BGR2RGB))

#        plt.axis("off")

#        plt.show()



if __name__ == '__main__':

    print("[INFO]Pose estimation.")



    video_file = "fall-01-cam0.mp4"

    #

    start = time.time()

    modelpath = "./models"

    pose_model = general_pose_model(modelpath, mode="BODY25")

    print("[INFO]Model loads time: ", time.time() - start)

	

    cap = cv2.VideoCapture(video_file)

    _, frame = cap.read()

    vid_H, vid_W, vid_ch = frame.shape	

    print("video = %d x %d" % (vid_W, vid_H))

	

    framecount =0 

    while True:

#        start = time.time()	

        _, frame = cap.read()

        frame=frame[0:vid_H, int(vid_W/2):vid_W] # cut the left part

        res_points = pose_model.predict(frame)

#        print("[INFO]Model predicts time: ", time.time() - start)

        pose_model.vis_pose(frame, res_points)

        print('frame count: ', framecount)

        framecount+=1			