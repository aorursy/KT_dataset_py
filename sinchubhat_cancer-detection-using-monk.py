!git clone https://github.com/Tessellate-Imaging/monk_v1.git
!cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt
! pip install pillow==5.4.1
# Monk
import os
import sys
sys.path.append("monk_v1/monk/");
#Using keras backend 
from keras_prototype import prototype
gtf = prototype(verbose=1);
gtf.Prototype("Cancer-Detection-Using-MONK", "Using-Keras-Backend");
gtf.List_Models()
gtf.Default(dataset_path="/kaggle/input/cancer/train", 
            model_name="resnet152_v2", 
            num_epochs=10);

#Read the summary generated once you run this cell.
gtf.update_save_intermediate_models(False);
gtf.Reload();
#Start Training
gtf.Train();
#Read the training summary generated once you run the cell and training is completed
gtf = prototype(verbose=1);
gtf.Prototype("Cancer-Detection-Using-MONK", "Using-Keras-Backend", eval_infer=True);
gtf.Dataset_Params(dataset_path="/kaggle/input/cancer/validation");
gtf.Dataset();
accuracy, class_based_accuracy = gtf.Evaluate();
gtf = prototype(verbose=1);
gtf.Prototype("Cancer-Detection-Using-MONK", "Using-Keras-Backend", eval_infer=True);
img_name = "/kaggle/input/cancer/test/c2 (10004).jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)
img_name = "/kaggle/input/cancer/test/c2 (10).jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)
img_name = "/kaggle/input/cancer/test/c2 (10014).jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)
from IPython.display import Image
Image(filename="workspace/Cancer-Detection-Using-MONK/Using-Keras-Backend/output/logs/train_val_accuracy.png") 
from IPython.display import Image
Image(filename="workspace/Cancer-Detection-Using-MONK/Using-Keras-Backend/output/logs/train_val_loss.png") 
#Using pytorch backend 
from pytorch_prototype import prototype
gtf = prototype(verbose=1);
gtf.Prototype("Cancer-Detection-Using-MONK", "Using-Pytorch-Backend");
gtf.List_Models();
gtf.Default(dataset_path="/kaggle/input/cancer/train", 
            model_name="densenet201",
            num_epochs=10);

#Read the summary generated once you run this cell.
# Need not save intermediate epoch weights
gtf.update_save_intermediate_models(False);
gtf.Reload();
#Start Training
gtf.Train();

#Read the training summary generated once you run the cell and training is completed
# Compare experiments
# Invoke the comparison class
from compare_prototype import compare
# Create a project 
gtf = compare(verbose=1);
gtf.Comparison("Campare-backends");
gtf.Add_Experiment("Cancer-Detection-Using-MONK", "Using-Keras-Backend");
gtf.Add_Experiment("Cancer-Detection-Using-MONK", "Using-Pytorch-Backend");
gtf.Generate_Statistics();
from IPython.display import Image
Image(filename="workspace/comparison/Campare-backends/train_accuracy.png") 
from IPython.display import Image
Image(filename="workspace/comparison/Campare-backends/train_loss.png") 
from IPython.display import Image
Image(filename="workspace/comparison/Campare-backends/val_accuracy.png") 
from IPython.display import Image
Image(filename="workspace/comparison/Campare-backends/val_loss.png")
from IPython.display import Image
Image(filename="workspace/comparison/Campare-backends/stats_training_time.png") 
from IPython.display import Image
Image(filename="workspace/comparison/Campare-backends/stats_best_val_acc.png") 
! rm -r monk_v1
! rm -r workspace
! rm pylg.log