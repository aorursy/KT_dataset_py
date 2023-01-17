# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#bottom left and  top right corner is given





def is_overlap(b1,b2):

    #rectange on left and right

    if(b2[0]>b1[2] or b1[0]>b2[2]):

        return False

    if(b1[1]>b2[3] or b2[1]>b2[3] ):

        return False

    return 1

def iou(b1,b2):

    #Bottom left point

    poi1_X=max(b1[0],b2[0])

    poi1_Y=max(b1[1],b2[1])

    #top right point

    poi2_X=min(b1[2],b2[2])

    poi2_Y=min(b1[3],b2[3])

    l=poi1_X-poi2_X

    b=poi1_Y-poi2_Y

    area=l*b

    return area

if __name__ == '__main__':

    #b ={x1,y1,x2,y2}

    #bottom left point(x1,y1) 

    #top right point(x2,y2)

    b1=[0,0,5,3]

    b2=[1,1,6,4]

   # x=is_overlap(b1,b2)

   # print(x)

    area_of_rectangle1=(b1[2]-b1[0])*(b1[3]-b1[1])

    area_of_rectangle2=(b2[2]-b2[0])*(b2[3]-b2[1])

    #print(area_of_rectangle2)

    if(is_overlap(b1,b2)):

        intersection_area=iou(b1,b2)

        print("area of intersection",intersection_area)

        area_not_in_the_intersection=area_of_rectangle1-intersection_area

        print(" area apart from intersection",area_not_in_the_intersection)

        union_area=area_of_rectangle1+area_of_rectangle2-intersection_area

        iou_ratio=intersection_area/union_area

        print("union_area",union_area)

        print("iou_ratio",iou_ratio)
