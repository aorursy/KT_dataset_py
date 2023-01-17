import pandas as pd
import numpy as np
import os
from zipfile import ZipFile
import cv2
import numpy as np
import pandas as pd
from dask import bag, threaded
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
df = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

df_train = df
df_test = test
# get filenames
zipped = ZipFile('../input/train_jpg.zip')
filenames = zipped.namelist()[1:] # exclude the initial directory listing
print(len(filenames))


#get blurrness score

def get_blurrness(file):
    exfile = zipped.read(file)
    arr = np.frombuffer(exfile, np.uint8)
    if arr.size > 0:   # exclude dirs and blanks
        imz = cv2.imdecode(arr, flags=cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(imz, cv2.CV_64F).var()
    else: 
        fm = -1
    return fm

blurrness = []
iteration = filenames[600000:750000]
for i in range(0, len(iteration)):
    print(i)
    blurrness.append(get_blurrness(iteration[i]))
    
frame = pd.DataFrame({"File" : filenames[600000:750000], "Score": blurrness})
frame.to_csv("7.csv", index = False)