import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

sdss = pd.read_csv('../input/SDSS DATA 2019.csv')

sdss['brightness'] = 1./sdss['g'] #this is just a proxy...

sdss.head()
import skimage

def getImStamp(ra,dec,width=300,height=300):

    url="http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx?ra="+str(ra)

    url+="&dec="+str(dec)+"&width="+str(width)

    url+="&height="+str(height)

    img=skimage.io.imread(url)

    return img