import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
folder = "/kaggle/input/charlie/"

my_api_key = "cea06635-af88-4a86-bb5e-a0ea0bf5e501"

colorize_url = "https://api.deepai.org/api/colorizer"

sr_url = "https://api.deepai.org/api/torch-srgan"
from PIL import Image

import math

import requests

from io import BytesIO

from tqdm import tqdm as tq

import numpy as np

import urllib

import base64

import datetime

import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

print(cv2.__version__)
def url_to_image(url):

    resp = urllib.request.urlopen(url)

    image = np.asarray(bytearray(resp.read()), dtype="uint8")

    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image
def colorize_image(image):

    retval, imgByteArr = cv2.imencode('.jpg', image)

    r = requests.post(colorize_url, files={'image': imgByteArr}, headers={'api-key': my_api_key})

    colored_bytes = BytesIO(requests.get(r.json()['output_url']).content)

    r = requests.post(sr_url, files={'image': colored_bytes}, headers={'api-key': my_api_key})

    return url_to_image(r.json()['output_url'])
def colorize_video(video_name, out_name, cnt_limit):

    vidcap = cv2.VideoCapture(folder+video_name+".mp4")

    fps = vidcap.get(cv2.CAP_PROP_FPS)

    success,image = vidcap.read()

    count = 0

    size = (3200,1600)

    #images = []

    video = cv2.VideoWriter(out_name+'.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    while success:

        colored = colorize_image(image)

        video.write(colored)

        #images.append(colored)

        count += 1

#         if count % max(1,int(cnt_limit/100)) == 0:

#             print(count)

        if count == cnt_limit:

            break

        success, image = vidcap.read()

  

    #height, width, channel = images[0].shape

    #size = (width, height)

    

    #for image in images:

    #    video.write(image)



    cv2.destroyAllWindows()

    video.release()
st = datetime.datetime.now()

colorize_video('charlie','project',25000)

et = datetime.datetime.now()

print((et - st).seconds)