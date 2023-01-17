import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
! pip install simple_image_download
from simple_image_download import simple_image_download as simp 

response = simp.simple_image_download



lst=['Sachin Tendulkar', 'Rahul Dravid', 'Virat Kolhi','Rohit Sharma','AB de Villiers','Shane Warne','Brian Lara']



for rep in lst:

    response().download(rep +' face' , 300)
rep='Other Cricketer'

response().download(rep +' face' , 50)
lst.append(rep)

lst
import os

for rep in lst:

    os.rename('/kaggle/working/simple_images/'+rep+' face/', '/kaggle/working/simple_images/'+rep)



os.rename('/kaggle/working/simple_images/', '/kaggle/working/train')