# Just checking if we have a GPU

!nvidia-smi
# Cloning the monk repository as we are going to use the MonkAI Library

!git clone https://github.com/Tessellate-Imaging/monk_v1.git
# Installing the dependencies for Kaggle required by Monk

!pip install -r monk_v1/installation/Misc/requirements_kaggle.txt
! rm -r monk_v1/.git

! rm -r monk_v1/installation

! rm -r monk_v1/study_roadmaps

! rm -r monk_v1/webinars_lectures
# Appending the Monk repo to our working directory

import sys

sys.path.append("/kaggle/working/monk_v1/monk/")
import os



import pandas as pd

df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")



combined = [];

from tqdm.notebook import tqdm

for i in tqdm(range(len(df))):

    img_name = df["image_name"][i] + ".jpg";

    if(df["benign_malignant"][i] == 'benign'):

        label = "0";

    elif(df["benign_malignant"][i] == 'malignant'):

        label = "1"; 

    combined.append([img_name, label]);

    

df2 = pd.DataFrame(combined, columns = ['ID', 'Label']) 

df2.to_csv("train.csv", index=False);

# Using mxnet backend

from gluon_prototype import prototype
# Defining path for training and validation dataset

train_path = '../input/siim-isic-melanoma-classification/jpeg/train'

csv_train = 'train.csv'
# Initialize the protoype model and setup project directory

gtf=prototype(verbose=1)

gtf.Prototype("Melanoma-Detection", "Hyperparameter-Analyser")
# Define the prototype with default parameters

gtf.Default(dataset_path=train_path,

            path_to_csv=csv_train,

           model_name="se_resnext101_64x4d",

           freeze_base_network=False,

           num_epochs=5)
gtf.Train()
! rm pylg.log train.csv
gtf = prototype(verbose=1)

gtf.Prototype("Melanoma-Detection", "Hyperparameter-Analyser",eval_infer = True)
import pandas as pd

from tqdm.notebook import tqdm

from scipy.special import softmax

df = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")
import numpy as np

import os

from IPython.display import FileLink

for i in tqdm(range(len(df))):

    img_name = "../input/siim-isic-melanoma-classification/jpeg/test/" + df['image_name'][i] + ".jpg";

    

    #Invoking Monk's inferencing engine inside a loop

    predictions = gtf.Infer(img_name=img_name, return_raw=True);

    out = predictions['raw']

    prob_mal = ((np.exp(out[1]))/(np.exp(out[0])+np.exp(out[1])))

    df['target'][i] = str(prob_mal)

    print("Probability: ", df['target'][i])



os.chdir(r'kaggle/working')

df.to_csv("submission.csv", index=False)

FileLink(r'submission.csv')
df.to_csv("submission.csv", index=False)
! rm -r monk_v1

! rm -r workspace