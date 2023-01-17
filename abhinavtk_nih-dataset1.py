# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation','Edema', 'Effusion', 'Emphysema','Fibrosis', 'Infiltration', 'Hernia', 'Mass','Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
for label in labels:

    os.makedirs('/kaggle/working/'+label)
df = pd.read_csv("../input/data/Data_Entry_2017.csv")

df1 = pd.DataFrame()

df1["Image Index"] = df["Image Index"]

df1["Finding Labels"] = df["Finding Labels"]

df1.head()
list1 = os.listdir('/kaggle/input/data/images_010/images')

len(list1)
list1[0]
for file in list1:

    #base = os.path.basename('/kaggle/input/data/images_010/images/'+file)

    label = df.loc[df['Image Index'] == file, 'Finding Labels'].iloc()[0]

    if label == 'Atelectasis': 

        im = cv2.imread("/kaggle/input/data/images_010/images/"+file)

        os.chdir('/kaggle/working/'+label)

        cv2.imwrite(file, im)

        os.chdir('/kaggle/working')

    