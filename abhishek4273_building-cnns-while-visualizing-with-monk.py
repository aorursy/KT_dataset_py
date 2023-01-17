# Since Monk supports Image as inputs
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
# Since the kernel cannot use internet, the installation got a little lengthy
import os

os.listdir("/kaggle/input/monk-kaggle/kaggle/")
import os

os.listdir("/kaggle/input/monk-kaggle/kaggle/installs")
!cd /kaggle/input/monk-kaggle/kaggle/installs/ && pip install PyLg-1.3.3-py3-none-any.whl
!cd /kaggle/input/monk-kaggle/kaggle/installs/ && pip install blessings-1.7-py3-none-any.whl
!cd /kaggle/input/monk-kaggle/kaggle/installs/ && pip install netron-3.7.3-py2.py3-none-any.whl
!cd /kaggle/input/monk-kaggle/kaggle/installs/ && pip install gpustat-0.6.0-py3-none-any.whl
import sys

sys.path.append("/kaggle/input/monk-kaggle/kaggle/monk_v1/monk/")

sys.path.append("/kaggle/input/monk-kaggle/kaggle/installs/")
from gluon_prototype import prototype
gtf = prototype(verbose=1);

gtf.Prototype("sample-project", "sample-experiment-1");
gtf.Dataset_Params(dataset_path="images/train/", 

                   split=0.8, input_size=28, 

                batch_size=16, shuffle_data=True, num_processors=3);



# Transform

gtf.apply_random_horizontal_flip(train=True, val=True);

gtf.apply_normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=True, val=True, test=True);



# Set Dataset

gtf.Dataset();
classes = gtf.system_dict["dataset"]["params"]["classes"];

classes
network = [];

network.append(gtf.convolution(output_channels=64));

network.append(gtf.batch_normalization());

network.append(gtf.relu());

network.append(gtf.convolution(output_channels=64));

network.append(gtf.batch_normalization());

network.append(gtf.relu());

network.append(gtf.max_pooling());
gtf.debug_custom_model_design(network);
subnetwork = [];

branch1 = [];

branch1.append(gtf.convolution(output_channels=64));

branch1.append(gtf.batch_normalization());

branch1.append(gtf.convolution(output_channels=64));

branch1.append(gtf.batch_normalization());



branch2 = [];

branch2.append(gtf.convolution(output_channels=64));

branch2.append(gtf.batch_normalization());



branch3 = [];

branch3.append(gtf.identity())



subnetwork.append(branch1);

subnetwork.append(branch2);

subnetwork.append(branch3);

subnetwork.append(gtf.concatenate());



network.append(subnetwork);
gtf.debug_custom_model_design(network);
network.append(gtf.convolution(output_channels=128));

network.append(gtf.batch_normalization());

network.append(gtf.relu());

network.append(gtf.max_pooling());
gtf.debug_custom_model_design(network);
subnetwork = [];

branch1 = [];

branch1.append(gtf.convolution(output_channels=128));

branch1.append(gtf.batch_normalization());

branch1.append(gtf.convolution(output_channels=128));

branch1.append(gtf.batch_normalization());



branch2 = [];

branch2.append(gtf.convolution(output_channels=128));

branch2.append(gtf.batch_normalization());



branch3 = [];

branch3.append(gtf.identity())



subnetwork.append(branch1);

subnetwork.append(branch2);

subnetwork.append(branch3);

subnetwork.append(gtf.add());



network.append(subnetwork);
gtf.debug_custom_model_design(network);
network.append(gtf.convolution(output_channels=256));

network.append(gtf.batch_normalization());

network.append(gtf.relu());

network.append(gtf.max_pooling());
gtf.debug_custom_model_design(network);
network.append(gtf.flatten());

network.append(gtf.fully_connected(units=1024));

network.append(gtf.dropout(drop_probability=0.2));

network.append(gtf.fully_connected(units=len(classes)));
gtf.Compile_Network(network, data_shape=(3, 28, 28));
gtf.Training_Params(num_epochs=10, display_progress=True, display_progress_realtime=True, 

        save_intermediate_models=False, intermediate_model_prefix="intermediate_model_", save_training_logs=True);



gtf.optimizer_sgd(0.01);

gtf.lr_fixed();

gtf.loss_softmax_crossentropy();
gtf.Train();
inference_dataset = "images/test/";

output = gtf.Infer(img_dir=inference_dataset);
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
sub[:10]
!rm -r images