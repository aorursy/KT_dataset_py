# Convert CSV data to images

import pandas as pd

import numpy as np

import cv2

import os

from tqdm import tqdm_notebook as tqdm



os.mkdir("images");

os.mkdir("images/train");

os.mkdir("images/test");
#Training data

dv = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv");

for i in tqdm(range(60000)):

    x = dv.iloc[i].to_numpy();

    label = str(x[0])

    img = x[1:]

    img = np.reshape(img, (28, 28));

    img = np.expand_dims(img, axis=0);

    img = np.vstack((img, img, img))

    img = np.swapaxes(img, 0, 1);

    img = np.swapaxes(img, 1, 2);

    if(not os.path.isdir("images/train/" + label)):

        os.mkdir("images/train/" + label);

    

    cv2.imwrite("images/train/" + label + "/" + str(i) + ".jpg", img);

    

    

#Testing data

dv = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv");

for i in tqdm(range(5000)):

    x = dv.iloc[i].to_numpy();

    id_ = str(x[0])

    img = x[1:]

    img = np.reshape(img, (28, 28));

    img = np.expand_dims(img, axis=0);

    img = np.vstack((img, img, img))

    img = np.swapaxes(img, 0, 1);

    img = np.swapaxes(img, 1, 2);

    cv2.imwrite("images/test/" + str(id_) + ".jpg", img);
!git clone https://github.com/Tessellate-Imaging/monk_v1.git
!pip install -r /kaggle/working/monk_v1/installation/requirements_kaggle.txt
import sys

sys.path.append("/kaggle/working/monk_v1/monk/")
# Step 0 - Using Pytorch

from pytorch_prototype import prototype
# Step 1 - Create experiment

ptf = prototype(verbose=1);

ptf.Prototype("sample-project-1", "sample-experiment-1");
# Step 2

ptf.Default(dataset_path="images/train/",

           model_name="resnet50", freeze_base_network=True,

           num_epochs=25);
# Additional Step

ptf.update_save_intermediate_models(False); 

ptf.update_display_progress_realtime(False);

ptf.update_batch_size(32);

ptf.Reload();
# Step - 3

ptf.Train();
# Step 0 - Using Gluon

from gluon_prototype import prototype
# Step 1 - Create experiment

ptf = prototype(verbose=1);

ptf.Prototype("sample-project-1", "sample-experiment-2");
# Step 2 - Invoke Quick Prototype Default mode

ptf.Default(dataset_path="images/train/",

           model_name="resnet50_v2", freeze_base_network=True,

           num_epochs=25);
# Additional Step

ptf.update_save_intermediate_models(False); 

ptf.update_display_progress_realtime(False);

ptf.update_batch_size(32);

ptf.Reload();
# Step 3 - Train

ptf.Train();
# Step 0 - Using Keras

from keras_prototype import prototype
# Step 1 - Create experiment

ptf = prototype(verbose=1);

ptf.Prototype("sample-project-1", "sample-experiment-3");
# Step 2 - Invoke Quick Prototype Default mode

ptf.Default(dataset_path="images/train/",

           model_name="resnet50_v2", freeze_base_network=True,

           num_epochs=25);
# Additional Step

ptf.update_save_intermediate_models(False); 

ptf.update_display_progress_realtime(False);

ptf.update_batch_size(32);

ptf.Reload();
ptf.Train();
from compare_prototype import compare



ctf = compare(verbose=1);

ctf.Comparison("Sample-Comparison-1");

ctf.Add_Experiment("sample-project-1", "sample-experiment-1");

ctf.Add_Experiment("sample-project-1", "sample-experiment-2");

ctf.Add_Experiment("sample-project-1", "sample-experiment-3");



ctf.Generate_Statistics();
from IPython.display import Image

from IPython.display import display
os.listdir("/kaggle/working/workspace/comparison/Sample-Comparison-1/")
x = Image(filename='/kaggle/working/workspace/comparison/Sample-Comparison-1/train_accuracy.png') 

y = Image(filename='/kaggle/working/workspace/comparison/Sample-Comparison-1/train_loss.png') 

display(x, y)
x = Image(filename='/kaggle/working/workspace/comparison/Sample-Comparison-1/val_accuracy.png') 

y = Image(filename='/kaggle/working/workspace/comparison/Sample-Comparison-1/val_loss.png') 

display(x, y)
x = Image(filename='/kaggle/working/workspace/comparison/Sample-Comparison-1/stats_training_time.png') 

y = Image(filename='/kaggle/working/workspace/comparison/Sample-Comparison-1/stats_max_gpu_usage.png') 

display(x, y)
from pytorch_prototype import prototype

ptf = prototype(verbose=1);

ptf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True);
inference_dataset = "images/test/";

output = ptf.Infer(img_dir=inference_dataset);
# Create submission

import pandas as pd

sub = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv");

for i in range(len(output)):

    index = int(sub[sub['id']==int(output[i]['img_name'].split(".")[0])].index[0])

    #print(output[i]);

    #print(index);

    sub['label'][index] = int(output[i]['predicted_class'])

    #print(sub.iloc[index:index+1])

    #break

sub.to_csv("submission.csv", index=False);
sub.iloc[0:5]
!rm -r /kaggle/working/monk_v1/
!rm -r images