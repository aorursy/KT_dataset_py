import numpy as np

import os

import pydicom

pts = ["ID00011637202177653955184","ID00052637202186188008618"]

paths = ["../input/osic-pulmonary-fibrosis-progression","../input/osic-pulmonary-fibrosis-progression-decompressed"]
for s in os.listdir(paths[0]+"/train/"+pts[0]+'/'):

    sl = pydicom.read_file(paths[0]+"/train/"+pts[0]+'/'+s)

    pixel_array = np.zeros((sl.Rows,sl.Columns))

    pixel_array = sl.pixel_array
for s in os.listdir(paths[0]+"/train/"+pts[1]+'/'):

    sl = pydicom.read_file(paths[0]+"/train/"+pts[1]+'/'+s)

    pixel_array = np.zeros((sl.Rows,sl.Columns))

    pixel_array = sl.pixel_array
for pt in pts:

    for s in os.listdir(paths[1]+"/train/"+pt+'/'):

        sl = pydicom.read_file(paths[1]+"/train/"+pt+'/'+s)

        pixel_array = np.zeros((sl.Rows,sl.Columns))

        pixel_array = sl.pixel_array

print("Fin.")