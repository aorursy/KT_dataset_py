!git clone https://github.com/Tessellate-Imaging/monk_v1.git
!pip install -r /kaggle/working/monk_v1/installation/requirements_kaggle.txt
import sys

sys.path.append("/kaggle/working/monk_v1/monk/")
# Step 0 - Using Gluon

from gluon_prototype import prototype

gtf = prototype(verbose=1);
gtf.List_Optimizers();
# Step 1 - Create experiment

gtf = prototype(verbose=1);

gtf.Prototype("Project-Testing-Optimizers", "SGD");



# Step 2 - Invoke Quick Prototype Default mode

gtf.Default(dataset_path="/kaggle/input/aerial-cactus-identification/train/train",

           path_to_csv="/kaggle/input/aerial-cactus-identification/train.csv",

           model_name="resnet18_v1", freeze_base_network=True,

           num_epochs=20);



gtf.optimizer_sgd(0.001);

gtf.update_save_intermediate_models(False); 

gtf.update_display_progress_realtime(False);

gtf.update_batch_size(32);

gtf.Reload();



# Step 3 - Train

gtf.Train();
# Step 1 - Create experiment

gtf = prototype(verbose=1);

gtf.Prototype("Project-Testing-Optimizers", "NAG");



# Step 2 - Invoke Quick Prototype Default mode

gtf.Default(dataset_path="/kaggle/input/aerial-cactus-identification/train/train",

           path_to_csv="/kaggle/input/aerial-cactus-identification/train.csv",

           model_name="resnet18_v1", freeze_base_network=True,

           num_epochs=20);



gtf.optimizer_nag(0.001);

gtf.update_save_intermediate_models(False); 

gtf.update_display_progress_realtime(False);

gtf.update_batch_size(32);

gtf.Reload();



# Step 3 - Train

gtf.Train();
# Step 1 - Create experiment

gtf = prototype(verbose=1);

gtf.Prototype("Project-Testing-Optimizers", "RMSProp");



# Step 2 - Invoke Quick Prototype Default mode

gtf.Default(dataset_path="/kaggle/input/aerial-cactus-identification/train/train",

           path_to_csv="/kaggle/input/aerial-cactus-identification/train.csv",

           model_name="resnet18_v1", freeze_base_network=True,

           num_epochs=20);



gtf.optimizer_rmsprop(0.001);

gtf.update_save_intermediate_models(False); 

gtf.update_display_progress_realtime(False);

gtf.update_batch_size(32);

gtf.Reload();



# Step 3 - Train

gtf.Train();
# Step 1 - Create experiment

gtf = prototype(verbose=1);

gtf.Prototype("Project-Testing-Optimizers", "ADAM");



# Step 2 - Invoke Quick Prototype Default mode

gtf.Default(dataset_path="/kaggle/input/aerial-cactus-identification/train/train",

           path_to_csv="/kaggle/input/aerial-cactus-identification/train.csv",

           model_name="resnet18_v1", freeze_base_network=True,

           num_epochs=20);



gtf.optimizer_adam(0.001);

gtf.update_save_intermediate_models(False); 

gtf.update_display_progress_realtime(False);

gtf.update_batch_size(32);

gtf.Reload();



# Step 3 - Train

gtf.Train();
# Step 1 - Create experiment

gtf = prototype(verbose=1);

gtf.Prototype("Project-Testing-Optimizers", "ADAGrad");



# Step 2 - Invoke Quick Prototype Default mode

gtf.Default(dataset_path="/kaggle/input/aerial-cactus-identification/train/train",

           path_to_csv="/kaggle/input/aerial-cactus-identification/train.csv",

           model_name="resnet18_v1", freeze_base_network=True,

           num_epochs=20);



gtf.optimizer_adagrad(0.001);

gtf.update_save_intermediate_models(False); 

gtf.update_display_progress_realtime(False);

gtf.update_batch_size(32);

gtf.Reload();



# Step 3 - Train

gtf.Train();
# Step 1 - Create experiment

gtf = prototype(verbose=1);

gtf.Prototype("Project-Testing-Optimizers", "ADADelta");



# Step 2 - Invoke Quick Prototype Default mode

gtf.Default(dataset_path="/kaggle/input/aerial-cactus-identification/train/train",

           path_to_csv="/kaggle/input/aerial-cactus-identification/train.csv",

           model_name="resnet18_v1", freeze_base_network=True,

           num_epochs=20);



gtf.optimizer_adadelta(0.001);

gtf.update_save_intermediate_models(False); 

gtf.update_display_progress_realtime(False);

gtf.update_batch_size(32);

gtf.Reload();



# Step 3 - Train

gtf.Train();
# Step 1 - Create experiment

gtf = prototype(verbose=1);

gtf.Prototype("Project-Testing-Optimizers", "ADAMax");



# Step 2 - Invoke Quick Prototype Default mode

gtf.Default(dataset_path="/kaggle/input/aerial-cactus-identification/train/train",

           path_to_csv="/kaggle/input/aerial-cactus-identification/train.csv",

           model_name="resnet18_v1", freeze_base_network=True,

           num_epochs=20);



gtf.optimizer_adamax(0.001);

gtf.update_save_intermediate_models(False); 

gtf.update_display_progress_realtime(False);

gtf.update_batch_size(32);

gtf.Reload();



# Step 3 - Train

gtf.Train();
# Step 1 - Create experiment

gtf = prototype(verbose=1);

gtf.Prototype("Project-Testing-Optimizers", "NADAM");



# Step 2 - Invoke Quick Prototype Default mode

gtf.Default(dataset_path="/kaggle/input/aerial-cactus-identification/train/train",

           path_to_csv="/kaggle/input/aerial-cactus-identification/train.csv",

           model_name="resnet18_v1", freeze_base_network=True,

           num_epochs=20);



gtf.optimizer_nadam(0.001);

gtf.update_save_intermediate_models(False); 

gtf.update_display_progress_realtime(False);

gtf.update_batch_size(32);

gtf.Reload();



# Step 3 - Train

gtf.Train();
# Step 0

from compare_prototype import compare

ctf = compare(verbose=1);

ctf.Comparison("Sample-Comparison-1")
# Step 1 - Add experiment

ctf.Add_Experiment("Project-Testing-Optimizers", "SGD");

ctf.Add_Experiment("Project-Testing-Optimizers", "NAG");

ctf.Add_Experiment("Project-Testing-Optimizers", "RMSProp");

ctf.Add_Experiment("Project-Testing-Optimizers", "ADAM");

ctf.Add_Experiment("Project-Testing-Optimizers", "ADAGrad");

ctf.Add_Experiment("Project-Testing-Optimizers", "ADADelta");

ctf.Add_Experiment("Project-Testing-Optimizers", "ADAMax");

ctf.Add_Experiment("Project-Testing-Optimizers", "NADAM");
# Step 2 - Compare

ctf.Generate_Statistics();
import os

os.listdir("/kaggle/working/workspace/comparison/Sample-Comparison-1/")
from IPython.display import Image

from IPython.display import display
x = Image(filename='/kaggle/working/workspace/comparison/Sample-Comparison-1/train_accuracy.png') 

y = Image(filename='/kaggle/working/workspace/comparison/Sample-Comparison-1/train_loss.png') 

display(x, y)
x = Image(filename='/kaggle/working/workspace/comparison/Sample-Comparison-1/val_accuracy.png') 

y = Image(filename='/kaggle/working/workspace/comparison/Sample-Comparison-1/val_loss.png') 

display(x, y)
## Visualize comparisons - Training time and gpu usages
x = Image(filename='/kaggle/working/workspace/comparison/Sample-Comparison-1/stats_training_time.png') 

y = Image(filename='/kaggle/working/workspace/comparison/Sample-Comparison-1/stats_max_gpu_usage.png') 

display(x, y)
ptf = prototype(verbose=1);

ptf.Prototype("Project-Testing-Optimizers", "NADAM", eval_infer=True);
# Step 2 - Run inference on dataset

output = ptf.Infer(img_dir="/kaggle/input/aerial-cactus-identification/test/test/");
# Create submission

import pandas as pd

sub = pd.read_csv("/kaggle/input/aerial-cactus-identification/sample_submission.csv");

for i in range(len(output)):

    index = int(sub[sub['id']==output[i]['img_name']].index[0])

    sub['has_cactus'][index] = int(output[i]['predicted_class'])

sub.to_csv("submission.csv", index=False);
sub.iloc[0:5]
!rm -r /kaggle/working/monk_v1/