# Clone the library
! git clone https://github.com/Tessellate-Imaging/monk_v1.git
# Install the pre-requisistes
!pip install -r monk_v1/installation/requirements_kaggle.txt
# Add to path
import sys

sys.path.append("/kaggle/working/monk_v1/monk/")
# Unzip datasets
!unzip -q /kaggle/input/aerial-cactus-identification/train.zip
!unzip -q /kaggle/input/aerial-cactus-identification/test.zip
# Import prototype
from pytorch_prototype import prototype
# Set Dataset
gtf = prototype(verbose=1);

gtf.Prototype("sample-project-1", "sample-experiment-1");



gtf.Dataset_Params(dataset_path="train/",

           path_to_csv="/kaggle/input/aerial-cactus-identification/train.csv",

        input_size=(32, 32), batch_size=16, shuffle_data=True, num_processors=3);



gtf.apply_random_horizontal_flip(train=True, val=True);

gtf.apply_normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=True, val=True, test=True);



gtf.Dataset();
# Create model and debug simultaneously
network = [];

network.append(gtf.convolution(output_channels=16));

network.append(gtf.batch_normalization());

network.append(gtf.relu());

network.append(gtf.convolution(output_channels=16));

network.append(gtf.batch_normalization());

network.append(gtf.relu());

network.append(gtf.max_pooling());

gtf.debug_custom_model_design(network);

subnetwork = [];

branch1 = [];

branch1.append(gtf.convolution(output_channels=16));

branch1.append(gtf.batch_normalization());

branch1.append(gtf.convolution(output_channels=16));

branch1.append(gtf.batch_normalization());



branch2 = [];

branch2.append(gtf.convolution(output_channels=16));

branch2.append(gtf.batch_normalization());



branch3 = [];

branch3.append(gtf.identity())



subnetwork.append(branch1);

subnetwork.append(branch2);

subnetwork.append(branch3);

subnetwork.append(gtf.concatenate());





network.append(subnetwork);

gtf.debug_custom_model_design(network);





network.append(gtf.convolution(output_channels=32));

network.append(gtf.batch_normalization());

network.append(gtf.relu());

network.append(gtf.max_pooling());

gtf.debug_custom_model_design(network);



subnetwork = [];

branch1 = [];

branch1.append(gtf.convolution(output_channels=32));

branch1.append(gtf.batch_normalization());

branch1.append(gtf.convolution(output_channels=32));

branch1.append(gtf.batch_normalization());



branch2 = [];

branch2.append(gtf.convolution(output_channels=32));

branch2.append(gtf.batch_normalization());



branch3 = [];

branch3.append(gtf.identity())



subnetwork.append(branch1);

subnetwork.append(branch2);

subnetwork.append(branch3);

subnetwork.append(gtf.add());





network.append(subnetwork);

gtf.debug_custom_model_design(network);





network.append(gtf.convolution(output_channels=32));

network.append(gtf.batch_normalization());

network.append(gtf.relu());

network.append(gtf.max_pooling());

gtf.debug_custom_model_design(network);







network.append(gtf.flatten());

network.append(gtf.fully_connected(units=1024));

network.append(gtf.dropout(drop_probability=0.2));

network.append(gtf.fully_connected(units=2));

gtf.Compile_Network(network, data_shape=(3, 32, 32));
# Set training parameters
gtf.Training_Params(num_epochs=50, display_progress=True, display_progress_realtime=True, 

        save_intermediate_models=False, save_training_logs=True);





gtf.optimizer_sgd(0.001);

gtf.lr_fixed();

gtf.loss_softmax_crossentropy();
gtf.Train();
# Step 0 - Using Pytorch

from pytorch_prototype import prototype



# Step 1 - Load experiment in evaluation mode

ptf = prototype(verbose=1);

ptf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True)





# Step 2 - Run inference on dataset

output = ptf.Infer(img_dir="test/");
num_0 = 0;

num_1 = 1;
# Create submission

import pandas as pd

sub = pd.read_csv("/kaggle/input/aerial-cactus-identification/sample_submission.csv");

for i in range(len(output)):

    index = int(sub[sub['id']==output[i]['img_name']].index[0])

    if(int(output[i]['predicted_class']) == 0):

        num_0 += 1;

    else:

        num_1 += 1;

    sub['has_cactus'][index] = int(output[i]['predicted_class'])

sub.to_csv("submission.csv", index=False);
num_0, num_1
!rm -r /kaggle/working/monk_v1/
!rm -r test
!rm -r train