! git clone https://github.com/Tessellate-Imaging/Monk_Object_Detection.git
! cd Monk_Object_Detection/7_yolov3/installation && cat requirements_colab.txt | xargs -n 1 -L 1 pip install
! mkdir '/kaggle/working/drone/'
! mkdir '/kaggle/working/drone/images/'
! mkdir '/kaggle/working/drone/labels/'
! cp /kaggle/input/drone-dataset-uav/drone_dataset_yolo/dataset_txt/*.jpg /kaggle/working/drone/images/
! cp /kaggle/input/drone-dataset-uav/drone_dataset_yolo/dataset_txt/*.txt /kaggle/working/drone/labels/
! mv /kaggle/working/drone/labels/classes.txt /kaggle/working/drone/classes.txt
# ! diff -bur '/kaggle/working/drone/images/' '/kaggle/working/drone/labels/'
import os
import sys
root_dir = '/kaggle/working/drone/';
img_dir = '/kaggle/working/drone/images/';

labels_dir = '/kaggle/working/drone/labels/';
classes_file = '/kaggle/working/drone/classes.txt';
sys.path.append("/kaggle/working/Monk_Object_Detection/7_yolov3/lib");
from train_detector import Detector
gtf = Detector();
gtf.set_train_dataset(img_dir, labels_dir, classes_file, batch_size=2)
gtf.set_val_dataset(img_dir, labels_dir)
gtf.set_model(model_name="yolov3");
gtf.set_hyperparams(optimizer="sgd", lr=0.00579, multi_scale=True, evolve=False);
gtf.Train(num_epochs=5);
from IPython.display import Image
from infer_detector import Infer
gtf = Infer();
f = open("/kaggle/working/drone/classes.txt");
class_list = f.readlines();
f.close();
model_name = "yolov3";
weights = "/kaggle/working/weights/last.pt";
gtf.Model(model_name, classes_file, weights, use_gpu=True, input_size=416);
img_path = "/kaggle/working/drone/images/0001.jpg";
gtf.Predict(img_path, conf_thres=0.4, iou_thres=0.5);
weights
type(weights)
