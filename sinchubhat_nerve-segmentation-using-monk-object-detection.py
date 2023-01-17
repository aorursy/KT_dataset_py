! mkdir trainJPEG testJPEG
! mkdir trainJPEG/trainimg trainJPEG/trainmask testJPEG/testimg testJPEG/testmask
! pwd
import os, sys
from PIL import Image
for infile in os.listdir("/kaggle/input/ultrasound-nerve-segmentation/train/"):
    #print("file : " + infile)
    
    if infile[-3:] == "tif":
        
        if infile[-8:-4] == "mask":
            file = "/kaggle/input/ultrasound-nerve-segmentation/train/" + infile
            outfile = "/kaggle/working/trainJPEG/trainmask/"+ infile[:-9] + ".jpeg"
            im = Image.open(file)
            out = im.convert("RGB")
            out.save(outfile, "JPEG", quality=100)
        else:
            fileImg = "/kaggle/input/ultrasound-nerve-segmentation/train/" + infile
            outfileImg = "/kaggle/working/trainJPEG/trainimg/"+ infile[:-3] + "jpeg"
            imImg = Image.open(fileImg)
            outImg = imImg.convert("RGB")
            outImg.save(outfileImg, "JPEG", quality=100)
for infile in os.listdir("/kaggle/input/ultrasound-nerve-segmentation/test/"):
    #print("file : " + infile)
    
    if infile[-3:] == "tif":
        
        if infile[-8:-4] == "mask":
            file = "/kaggle/input/ultrasound-nerve-segmentation/test/" + infile
            outfile = "/kaggle/working/testJPEG/testmask/"+ infile[:-9] + ".jpeg"
            im = Image.open(file)
            out = im.convert("RGB")
            out.save(outfile, "JPEG", quality=100)
        else:
            fileImg = "/kaggle/input/ultrasound-nerve-segmentation/test/" + infile
            outfileImg = "/kaggle/working/testJPEG/testimg/"+ infile[:-3] + "jpeg"
            imImg = Image.open(fileImg)
            outImg = imImg.convert("RGB")
            outImg.save(outfileImg, "JPEG", quality=100)  
import cv2
import matplotlib.pyplot as plt
f, axarr = plt.subplots(2,4)

img1 = cv2.imread('/kaggle/working/trainJPEG/trainimg/10_103.jpeg')
img2 = cv2.imread('/kaggle/working/trainJPEG/trainmask/10_103.jpeg')
img3 = cv2.imread('/kaggle/working/trainJPEG/trainimg/10_104.jpeg')
img4 = cv2.imread('/kaggle/working/trainJPEG/trainmask/10_104.jpeg')

img5 = cv2.imread('/kaggle/working/trainJPEG/trainimg/10_109.jpeg')
img6 = cv2.imread('/kaggle/working/trainJPEG/trainmask/10_109.jpeg')
img7 = cv2.imread('/kaggle/working/trainJPEG/trainimg/10_112.jpeg')
img8 = cv2.imread('/kaggle/working/trainJPEG/trainmask/10_112.jpeg')

axarr[0,0].imshow(img1)
axarr[0,1].imshow(img2)
axarr[1,0].imshow(img3)
axarr[1,1].imshow(img4)

axarr[0,2].imshow(img5)
axarr[0,3].imshow(img6)
axarr[1,2].imshow(img7)
axarr[1,3].imshow(img8)
! git clone https://github.com/Tessellate-Imaging/Monk_Object_Detection.git
! cd Monk_Object_Detection/9_segmentation_models/installation && cat requirements_colab.txt | xargs -n 1 -L 1 pip install
DATA_DIR = '/kaggle/working/'
import os
import sys
sys.path.append("Monk_Object_Detection/9_segmentation_models/lib/");
from train_segmentation import Segmenter
gtf = Segmenter();
img_dir = "/kaggle/working/trainJPEG/trainimg/";
mask_dir = "/kaggle/working/trainJPEG/trainmask/";
classes_dict = {
    'background': 0, 
    'nerves': 1,
};
classes_to_train = ['background', 'nerves'];
gtf.Train_Dataset(img_dir, mask_dir, classes_dict, classes_to_train)
img_dir = "/kaggle/working/testJPEG/testimg/";
mask_dir = "/kaggle/working/testJPEG/testmask/";
gtf.Val_Dataset(img_dir, mask_dir)
gtf.List_Backbones();
gtf.Data_Params(batch_size=2, backbone="efficientnetb3", image_shape=[580, 420])
gtf.List_Models();
gtf.Model_Params(model="Linknet")
gtf.Train_Params(lr=0.001)
gtf.Setup();
gtf.Train(num_epochs=5);
gtf.Visualize_Training_History();
from infer_segmentation import Infer
gtf = Infer();
classes_dict = {
    'background': 0, 
    'nerves': 1,
};
classes_to_train = ['nerves'];
gtf.Data_Params(classes_dict, classes_to_train, image_shape=[580, 420])
gtf.Model_Params(model="Linknet", backbone="efficientnetb3", path_to_model='best_model.h5')
gtf.Setup();
gtf.Predict("/kaggle/working/trainJPEG/trainimg/10_103.jpeg", vis=True);
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/kaggle/working/trainJPEG/trainmask/10_103.jpeg", 0)
cv2.imwrite("tmp.jpg", img)

from IPython.display import Image
Image(filename="tmp.jpg")