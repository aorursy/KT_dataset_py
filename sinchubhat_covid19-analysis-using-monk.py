!git clone https://github.com/Tessellate-Imaging/monk_v1.git
!cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt
! pip install pillow==5.4.1
# Monk
import os
import sys
sys.path.append("monk_v1/monk/");
#Using keras-gluon backend 
from keras_prototype import prototype
gtf = prototype(verbose=1);
gtf.Prototype("Covid19", "Using-keras-backend");
gtf.List_Models()
gtf.Default(dataset_path="/kaggle/input/covid19-image-dataset/Covid19-dataset/train", 
            model_name="densenet201", 
            num_epochs=25);

#Read the summary generated once you run this cell.
#Start Training
gtf.Train();

#Read the training summary generated once you run the cell and training is completed
gtf = prototype(verbose=1);
gtf.Prototype("Covid19", "Using-keras-backend", eval_infer=True);
gtf.Dataset_Params(dataset_path="/kaggle/input/covid19-image-dataset/Covid19-dataset/test");
gtf.Dataset();
accuracy, class_based_accuracy = gtf.Evaluate();
gtf = prototype(verbose=1);
gtf.Prototype("Covid19", "Using-keras-backend", eval_infer=True);
img_name = "/kaggle/input/covid19-image-dataset/Covid19-dataset/test/Covid/0108.jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)
img_name = "/kaggle/input/covid19-image-dataset/Covid19-dataset/test/Normal/0121.jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)
img_name = "/kaggle/input/covid19-image-dataset/Covid19-dataset/test/Viral Pneumonia/0115.jpeg";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name)
