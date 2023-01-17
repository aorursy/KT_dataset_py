import json

import os

import sys

import numpy as np

import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt

import cv2
f, axarr = plt.subplots(2,2)

img1 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_0.jpg')

img2 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_1.jpg')

img3 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_2.jpg')

img4 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_3.jpg')

axarr[0,0].imshow(img1)

axarr[0,1].imshow(img2)

axarr[1,0].imshow(img3)

axarr[1,1].imshow(img4)
f, axarr = plt.subplots(2,2)

img1 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Test_10.jpg')

img2 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Test_1005.jpg')

img3 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Test_101.jpg')

img4 = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Test_1.jpg')

axarr[0,0].imshow(img1)

axarr[0,1].imshow(img2)

axarr[1,0].imshow(img3)

axarr[1,1].imshow(img4)
data_train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
data_train.head()
data_train_new = pd.DataFrame(columns = ['image_id', 'Category']) 
data_train_new.head()
for index,row in data_train.iterrows():

    #print(row)

    #print(index)

    pathname = str(row['image_id'])+'.jpg'

    data_train_new.loc[index,'image_id']=pathname

    if(row['healthy']==1):

        cat = 'healthy'

    elif(row['multiple_diseases']==1):

        cat = 'multiple_diseases'

    elif(row['rust']==1):

        cat = 'rust'

    else:

        cat = 'scab'

    

    data_train_new.loc[index,'Category']=cat
data_train_new.head()
data_train_new.to_csv("trainWithext.csv", index=False)
!git clone https://github.com/Tessellate-Imaging/monk_v1.git
!cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt
# Monk

import os

import sys

sys.path.append("monk_v1/monk/");
#Using pytorch backend 

from pytorch_prototype import prototype
gtf = prototype(verbose=1);

gtf.Prototype("PlantPathology2020", "Using_Pytorch_Backend");
gtf.Default(dataset_path="/kaggle/input/plant-pathology-2020-fgvc7/images/",

            path_to_csv="trainWithext.csv", # updated csv file 

            model_name="resnet18", 

            freeze_base_network=False,

            num_epochs=20); 
gtf.EDA(check_corrupt=True)
gtf.List_Models();
#Start Training

gtf.Train();

#Read the training summary generated once you run the cell and training is completed
gtf = prototype(verbose=0);

gtf.Prototype("PlantPathology2020", "Using_Pytorch_Backend", eval_infer=True);
from IPython.display import Image

Image(filename="workspace/PlantPathology2020/Using_Pytorch_Backend/output/logs/train_val_accuracy.png") 
from IPython.display import Image

Image(filename="workspace/PlantPathology2020/Using_Pytorch_Backend/output/logs/train_val_loss.png") 
img_name = "/kaggle/input/plant-pathology-2020-fgvc7/images/Test_0.jpg";

predictions = gtf.Infer(img_name=img_name);



#Display 

from IPython.display import Image

Image(filename=img_name)
img_name = "/kaggle/input/plant-pathology-2020-fgvc7/images/Test_1004.jpg";

predictions = gtf.Infer(img_name=img_name);



#Display 

from IPython.display import Image

Image(filename=img_name)
img_name = "/kaggle/input/plant-pathology-2020-fgvc7/images/Test_10.jpg";

predictions = gtf.Infer(img_name=img_name);



#Display 

from IPython.display import Image

Image(filename=img_name)
import pandas as pd

from tqdm import tqdm_notebook as tqdm

from scipy.special import softmax

#np.set_printoptions(precision=2)

df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")
img_name = "/kaggle/input/plant-pathology-2020-fgvc7/images/Test_10.jpg";

predictions = gtf.Infer(img_name=img_name,return_raw=True);



type(predictions)



predictions.keys()



print(predictions["raw"])



print(" Predictions in terms of probabilities")

print(softmax(predictions["raw"]))



#Display 

from IPython.display import Image

Image(filename=img_name)
for i in tqdm(range(len(df))):

    img_name = "/kaggle/input/plant-pathology-2020-fgvc7/images/" + df["image_id"][i] + ".jpg";

    

    #Invoking Monk's nferencing engine inside a loop

    predictions = gtf.Infer(img_name=img_name, return_raw=True);

    x = predictions["raw"]

    out = softmax(x)

    df["healthy"][i] = out[0];

    df["multiple_diseases"][i] = out[1];

    df["rust"][i] = out[2];

    df["scab"][i] = out[3];
df.head()
df.to_csv("submission.csv", index=False);
! rm -r monk_v1
! rm -r workspace
! rm pylg.log trainWithext.csv