# Just checking if we have a GPU

!nvidia-smi
# Cloning the monk repository as we are going to use the MonkAI Library

!git clone https://github.com/Tessellate-Imaging/monk_v1.git
# Installing the dependencies for Kaggle required by Monk

!pip install -r monk_v1/installation/Misc/requirements_kaggle.txt
# Appending the Monk repo to our working directory

import sys

sys.path.append("/kaggle/working/monk_v1/monk/")
# Using mxnet backend

from gluon_prototype import prototype
# Defining path for training and validation dataset

train_path = '../input/kermany2018/OCT2017 /train'

val_path = '../input/kermany2018/OCT2017 /val' 

test_path = '../input/kermany2018/OCT2017 /test' 
# Initialize the protoype model and setup project directory

gtf=prototype(verbose=1)

gtf.Prototype("Retina-OCT", "Hyperparameter-Analyser")
# Define the prototype with default parameters

gtf.Default(dataset_path=train_path,

           model_name="densenet121",

           freeze_base_network=False,

           num_epochs=5)
# Analysis Project Name

analysis_name = "analyse_hyperparameters"
lrs = [0.1, 0.05, 0.01, 0.005, 0.0001] # learning rates

batch_sizes = [2, 4, 8, 12] # Batch sizes

models = [["densenet121", False, True], ["densenet169", False, True], ["densenet201", False, True]] # models

optimizers = ["sgd", "adam", "adagrad"] # optimizers

epochs=10 # number of epochs

percent_data=5 # percent of data to use
# keep_none state to delete all sub-experiments created

# Analysis of learning rates

analysis = gtf.Analyse_Learning_Rates(analysis_name, lrs, percent_data, 

                                      num_epochs=epochs, state="keep_none");
# Update prototype with the best learning rate

gtf.update_learning_rate(0.01)



# Very important to reload post updates

gtf.Reload()
# keep_none state to delete all sub-experiments created

# Analysis of Batch Sizes

analysis = gtf.Analyse_Batch_Sizes(analysis_name, batch_sizes, percent_data, 

                                      num_epochs=epochs, state="keep_none");
# Update prototype with optimum batch size

gtf.update_batch_size(12)



gtf.Reload()
# keep_none state to delete all sub-experiments created

# Analysis of Models

analysis = gtf.Analyse_Models(analysis_name, models, percent_data, 

                                      num_epochs=epochs, state="keep_none");
# Update prototype with best model

gtf.update_model_name("densenet169")

gtf.update_freeze_base_network(False)

gtf.update_use_pretrained(True)



gtf.Reload()
# keep_none state to delete all sub-experiments created

# Analysis of Optimizers

analysis = gtf.Analyse_Optimizers(analysis_name, optimizers, percent_data, 

                                      num_epochs=epochs, state="keep_none");
#Update prototype with optimizer

gtf.optimizer_sgd(0.01)



gtf.Reload()
gtf.Train()
# Set flag eval_infer as True

gtf=prototype(verbose=1)

gtf.Prototype("Retina-OCT", "Hyperparameter-Analyser", eval_infer = True)
# Load the validation dataset

gtf.Dataset_Params(dataset_path=val_path)

gtf.Dataset()
# Run validation

accuracy, class_based_accuracy = gtf.Evaluate()
# Set flag eval_infer as True

gtf=prototype(verbose=1)

gtf.Prototype("Retina-OCT", "Hyperparameter-Analyser", eval_infer = True)
# Running sample inference

img_name = test_path +"/DRUSEN/DRUSEN-1786810-3.jpeg"

predictions = gtf.Infer(img_name=img_name)



#Display 

from IPython.display import Image

Image(filename=img_name) 
# Load the validation dataset

gtf.Dataset_Params(dataset_path=test_path)

gtf.Dataset()
# Run inference on test data

accuracy, class_based_accuracy = gtf.Evaluate()