import cv2
import matplotlib.pyplot as plt
f, axarr = plt.subplots(2,2)
img1 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1008_virus_1691.jpeg')
img2 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person100_virus_184.jpeg')
img3 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0131-0001.jpeg')
img4 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0152-0001.jpeg')
axarr[0,0].imshow(img1)
axarr[0,1].imshow(img2)
axarr[1,0].imshow(img3)
axarr[1,1].imshow(img4)
!git clone https://github.com/Tessellate-Imaging/monk_v1.git
!cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt
! pip install pillow==5.4.1
# Monk
import os
import sys
sys.path.append("monk_v1/monk/");
#Using pytorch backend 
from pytorch_prototype import prototype
gtf = prototype(verbose=1);
gtf.Prototype("PneumoniaClassificationMONK", "UsingPytorchBackend");
gtf.List_Models()
gtf.Default(dataset_path="/kaggle/input/chest-xray-pneumonia/chest_xray/train", 
            model_name="resnet50", 
            freeze_base_network=False,
            num_epochs=25); 
#Start Training
gtf.Train();
#Read the training summary generated once you run the cell and training is completed
gtf = prototype(verbose=1);
gtf.Prototype("PneumoniaClassificationMONK", "UsingPytorchBackend",eval_infer=True);
gtf.Dataset_Params(dataset_path="/kaggle/input/chest-xray-pneumonia/chest_xray/val");
gtf.Dataset();
accuracy, class_based_accuracy = gtf.Evaluate();
gtf = prototype(verbose=1);
gtf.Prototype("PneumoniaClassificationMONK", "UsingPytorchBackend",eval_infer=True);
img_name = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0005-0001.jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)
img_name = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person103_bacteria_489.jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)
img_name = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person108_bacteria_506.jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)
from IPython.display import Image
Image(filename="/kaggle/working/workspace/PneumoniaClassificationMONK/UsingPytorchBackend/output/logs/train_val_accuracy.png") 
from IPython.display import Image
Image(filename="/kaggle/working/workspace/PneumoniaClassificationMONK/UsingPytorchBackend/output/logs/train_val_loss.png") 