import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
sample_submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
train.head()
test.head()
X_train=train.drop('label',axis=1)
Y_train=train.label
X_train.head()
Y_train.head()
test=test.drop('id',axis=1)
test.head()
X_train=X_train/255
test=test/255
X_train=X_train.values.reshape((-1,28,28,1))
test=test.values.reshape((-1,28,28,1))
X_train.shape
# Number of rows in X_train
X_train.shape[0]
test.shape
# print(X_train[0][:,:,0])
plt.imshow(X_train[0][:,:,0])
plt.title(Y_train[0])
plt.imshow(X_train[1][:,:,0])
plt.title(Y_train[1])
! pwd
! ls
! mkdir trainIm testIm
! ls
array = X_train[0][:,:,0].astype(np.uint8)
print(array)
lengthX_train = X_train.shape[0]
for i in range(lengthX_train):
    array = X_train[i][:,:,0].astype(np.uint8)
    img = Image.fromarray(array)
    img = img.convert("L")
    fn = "/kaggle/working/trainIm/TrainImg{}.png".format(i)
    img.save(fn)
lengthtest = test.shape[0]
for i in range(lengthtest):
    array = test[i][:,:,0].astype(np.uint8)
    img = Image.fromarray(array)
    img = img.convert("L")
    fn = "/kaggle/working/testIm/TestImg{}.png".format(i)
    img.save(fn)
train_label = pd.DataFrame(columns = ['image_id_path', 'Label']) 
train_label.head()
Y_train.head()
for index,row in Y_train.iteritems():
    #print(row)
    #print(index)
    pathname = "TrainImg{}.png".format(index)
    train_label.loc[index,'image_id_path']=pathname
    train_label.loc[index,'Label']=row
train_label.head()
train_label.to_csv("train.csv", index=False)
! pwd
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
gtf.Prototype("Kannada-MNIST", "Using_Pytorch_Backend");
gtf.List_Models()
gtf.Default(dataset_path="/kaggle/working/trainIm/",
            path_to_csv="/kaggle/working/train.csv", # updated csv file 
            model_name="resnet50", 
            freeze_base_network=False,
            num_epochs=20); 
#Start Training
gtf.Train();
#Read the training summary generated once you run the cell and training is completed
gtf = prototype(verbose=0);
gtf.Prototype("Kannada-MNIST", "Using_Pytorch_Backend", eval_infer=True);
img_name = "/kaggle/working/testIm/TestImg0.png";
predictions = gtf.Infer(img_name=img_name);

#Display 
from IPython.display import Image
Image(filename=img_name,width=200,height=300)
from tqdm import tqdm_notebook as tqdm
from scipy.special import softmax
img_name = "/kaggle/working/testIm/TestImg1.png";
predictions = gtf.Infer(img_name=img_name);
print(predictions)
print(predictions['predicted_class'])

#Display 
from IPython.display import Image
Image(filename=img_name,width=200,height=300)
for i in tqdm(range(len(sample_submission))):
    img_name = "/kaggle/working/testIm/TestImg{}.png".format(i)
    
    #Invoking Monk's nferencing engine inside a loop
    predictions = gtf.Infer(img_name=img_name, return_raw=True);
    x = predictions['predicted_class']
    sample_submission["id"][i] = i;
    sample_submission["label"][i] = x;
   
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False);
! rm -r monk_v1
! rm -r workspace