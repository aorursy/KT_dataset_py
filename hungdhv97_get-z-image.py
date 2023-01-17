import os

os.getcwd()

os.chdir("../input/crawl-image/JPNotebook/")

!ls
!pip install imutils dlib
import numpy as np 

import pandas as pd

import model 

import align_face

from os import listdir

from os.path import isfile, join

from PIL import Image
files = listdir("../DSNTT-AlignFace/")

files_path = [join("../DSNTT-AlignFace/", f) for f in files]

files
i = 0

def get_z(file_path):

    try:

        print(file_path)

        img = align_face.align(file_path)

        img = np.reshape(img, [1,256,256,3])

        eps = model.encode(img)

#         print(eps.tolist())

        return eps.tolist()

    except:

        print("remove ", file_path)

        files.remove(os.path.basename(file_path))

    



z = []

for f in files_path:

    i = i + 1

    print(i, end = " ")

    tmp = get_z(f)

    if tmp is not None:

        z.append(tmp)
os.chdir("../../../working/")

os.getcwd()
df = pd.DataFrame()

df['File_name'] = files

df.to_csv("filename.csv")

df
z_new = np.array(z)

np.save("z.npy", np.reshape(z_new, (1078, 196608)))
os.listdir(".")