# External input from full train dfdc_train_part_0

!ls -l /kaggle/input/strange-video
import argparse

import sys

import os

import cv2

import json

import numpy as np

import matplotlib.pyplot as plt





INPUT_ROOT = "/kaggle/input/strange-video"



# Strange sample

FAKE_NAME = "owxbbpjpch.mp4"

SAMPLE_FAKE1 = INPUT_ROOT + os.sep + FAKE_NAME

SAMPLE_ORG1 = INPUT_ROOT + os.sep + "wynotylpnm.mp4"
# Expand white area above threshold

def enhance(fgmask, ek, dk):

    kernel = np.ones((ek, ek), np.uint8)

    fgmask = cv2.erode(fgmask, kernel, iterations=1)

    kernel = np.ones((dk, dk), np.uint8)

    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    return fgmask
# Differences between Fake and Real by Frame

def _get_background_subtraction(image1, image2):

    fgbg = cv2.createBackgroundSubtractorMOG2()

    fgbg.apply(image1)

    fgmask = fgbg.apply(image2)

    fgmask = enhance(fgmask, 3, 11)

    fgmask = enhance(fgmask, 22, 31)

    return fgmask
# Differences between Fake and Real by Video

def video_diff(org, fake, out):

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    corg = cv2.VideoCapture(org)

    cfake = cv2.VideoCapture(fake)

    writer = None

    mask_writer = None

    while True:

        cret, forg = corg.read()

        fret, ffake = cfake.read()

        if not cret or not fret:

            print("end")

            break

        diff = _get_background_subtraction(forg, ffake)

        cimg = cv2.hconcat([forg, ffake, cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)])

        cimg = cv2.resize(cimg, (int(cimg.shape[1]/4), int(cimg.shape[0]/4)))

        if writer is None:

            writer = cv2.VideoWriter("blend_{}.mp4".format(out), fourcc, 30, (cimg.shape[1], cimg.shape[0]))

            mask_writer = cv2.VideoWriter("mask_{}.MOV".format(out), fourcc, 30, (diff.shape[1], diff.shape[0]))

        writer.write(cimg)

        mask_writer.write(cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB))
video_diff(SAMPLE_ORG1, SAMPLE_FAKE1, FAKE_NAME.split(".")[0])
!ls -l /kaggle/working
blend = cv2.VideoCapture("/kaggle/working/blend_owxbbpjpch.mp4")

blend.set(0,5*1000)

_, f = blend.read()

plt.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

plt.show()

blend.set(0,6*1000)

_, f = blend.read()

plt.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

plt.show()

blend.set(0,7*1000)

_, f = blend.read()

plt.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

plt.show()

blend.set(0,8*1000)

_, f = blend.read()

plt.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

plt.show()