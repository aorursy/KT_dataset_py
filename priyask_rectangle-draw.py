import numpy as np 

import pandas as pd

import math

import os

import cv2



eyeleftright = pd.read_csv('../input/eyeleftright/eyelr.csv')
def drawrectangle(df, rows=49):

    for i in range(rows):

        df = eyeleftright

        image_path = df.loc[i,'id_code']

        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')

        x=int(df.loc[i,'x'])

        y=int(df.loc[i,'y'])

        x1= int(df.loc[i,'left']*x)-5

        y1= int(df.loc[i,'top']*y)-5

        x2=int(df.loc[i,'left']*x+ df.loc[i,'width']*x)+5

        y2=int(df.loc[i,'top']*y+df.loc[i,'height']*y)+5

        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

        cv2.imwrite(image_path+'.png',img)                     



drawrectangle(eyeleftright)



           

                        

         

                          

       