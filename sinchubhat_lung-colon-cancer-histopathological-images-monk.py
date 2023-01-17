!git clone https://github.com/Tessellate-Imaging/monk_v1.git
!cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt
# Monk
import os
import sys
sys.path.append("monk_v1/monk/");
#Using pytorch backend 
from pytorch_prototype import prototype
gtfL = prototype(verbose=1);
gtfL.Prototype("Lung-cancer", "Using-pytorch-backend");
gtfL.List_Models()
gtfL.Default(dataset_path="/kaggle/input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/lung_image_sets", 
            model_name="resnet18", 
            num_epochs=5);

#Read the summary generated once you run this cell.
#Start Training
gtfL.Train();

#Read the training summary generated once you run the cell and training is completed
from IPython.display import Image
img_name = "/kaggle/working/workspace/Lung-cancer/Using-pytorch-backend/output/logs/train_val_loss.png"
Image(filename=img_name)
from IPython.display import Image
img_name = "/kaggle/working/workspace/Lung-cancer/Using-pytorch-backend/output/logs/train_val_accuracy.png"
Image(filename=img_name)
gtfC = prototype(verbose=1);
gtfC.Prototype("Colon-cancer", "Using-pytorch-backend");
gtfC.Default(dataset_path="/kaggle/input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/colon_image_sets", 
            model_name="resnet152", 
            num_epochs=5);

#Read the summary generated once you run this cell.", 
#Start Training
gtfC.Train();

#Read the training summary generated once you run the cell and training is completed
from IPython.display import Image
img_name = "/kaggle/working/workspace/Colon-cancer/Using-pytorch-backend/output/logs/train_val_accuracy.png"
Image(filename=img_name)
from IPython.display import Image
img_name = "/kaggle/working/workspace/Colon-cancer/Using-pytorch-backend/output/logs/train_val_loss.png"
Image(filename=img_name)