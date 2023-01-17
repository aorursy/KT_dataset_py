import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from subprocess import check_output

classes = check_output(["ls", "../input/train"]).decode("utf8").strip().split("\n")
classes
dir_list = []

for c in classes:

    files = check_output(["ls", "../input/train/%s" % c]).decode("utf8").strip().split("\n")

    dir_list.append(files)

    files = check_output(["ls", "-l", "../input/train/%s" % c]).decode("utf8").strip().split("\n")    
df = pd.DataFrame({"n_images": [len(x) for x in dir_list]}, index=classes)
df.plot(kind="bar", figsize=(15,10))
from PIL import Image
images = []

im_class = []

im_height = []

im_width = []

for c, files in zip(classes, dir_list):

    for img in files:

        im = Image.open("../input/train/%s/%s" % (c, img))

        images.append(img)

        im_class.append(c)

        im_height.append(im.height)

        im_width.append(im.width)



df_all = pd.DataFrame({"class": im_class, "height": im_height, "width": im_width}, index=images)

        
df_all.head()
a_ratio = df_all.height.astype("float") / df_all.width.astype("float")

print(a_ratio.mean(), a_ratio.std())
df_all["a_ratio"] = a_ratio
sns.distplot(df_all.height)
sns.distplot(df_all.a_ratio)