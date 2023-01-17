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
print(check_output(["ls", "../input/train"]).decode("utf8"))

print(check_output(["ls", "../input/additional"]).decode("utf8"))
# Tanks for Philipp Schmidt in https://www.kaggle.com/philschmidt/intel-mobileodt-cervical-cancer-screening/cervix-eda

from glob import glob

basepath = '../input/train/'



all_cervix_images = []



for path in glob(basepath + "*"):

    cervix_type = path.split("/")[-1]

    cervix_images = glob(basepath + cervix_type + "/*")

    all_cervix_images = all_cervix_images + cervix_images



all_cervix_images = pd.DataFrame({'imagepath': all_cervix_images})

all_cervix_images['filetype'] = all_cervix_images.apply(lambda row: row.imagepath.split(".")[-1], axis=1)

all_cervix_images['type'] = all_cervix_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)

all_cervix_images.head()
all_cervix_images.shape