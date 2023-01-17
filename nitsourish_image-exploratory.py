# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from subprocess import check_output 

check_output(["ls", "../input/images_sample/6812223"]).decode("utf8").strip().split('\n')
def process_image(path):

    path = '../input/images_sample/'+path[0:7]+'/'+path

    im = np.array(Image.open(path))



    #get dims

    width = im.shape[1]

    height = im.shape[0]

    

    #flatten image

    im = im.transpose(2,0,1).reshape(3,-1)

   

    

    #brightness is simple, assign 1 if zero to avoid divide

    brg = np.amax(im,axis=0)

    brg[brg==0] = 1

    

    #hue, same, assign 1 if zero, not working atm due to arccos

    denom = np.sqrt((im[0]-im[1])**2-(im[0]-im[2])*(im[1]-im[2]))

    denom[denom==0] = 1

    #hue = np.arccos(0.5*(2*im[0]-im[1]-im[2])/denom)

    

    #saturation

    sat = (brg - np.amin(im,axis=0))/brg

    

    #return mean values

    return width,height,np.mean(brg),np.mean(sat)
#second helper function - process a row of a dataset

#return mean of each property for all images

def process_row(row):

    images = check_output(["ls", "../input/images_sample/"+str(row.listing_id)]).decode("utf8").strip().split('\n')

    res = np.array([process_image(x) for x in images])

    res = np.mean(res,axis=0)

    row['img_width'] = res[0]

    row['img_height'] = res[1]

    row['img_brightness'] = res[2]

    row['img_saturation'] = res[3]

    return row