%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
from fastai.widgets import *

import os
import sys
import cv2
import shutil  
import tarfile
import numpy as np
# Set the path to the dataset directory (needs to be movings to kaggle/working to be extracted because the input folder is read-only)
path = '/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge'

os.chdir(path)

print(f"Before moving file, file path is:\n{os.getcwd()}\n\nThe directory contains:\n{os.listdir(path)} \n")  

# Destination path  
destination = '/kaggle/working'

if not os.path.isdir('/kaggle/working/challenges-in-representation-learning-facial-expression-recognition-challenge'):
    try:
        # Lets move fer2013.tar.gz to working
        dest = shutil.move(path, destination)
    except OSError:
        print(sys.exc_info())

    
# Remove files
# shutil.rmtree("/kaggle/working/challenges-in-representation-learning-facial-expression-recognition-challenge")
# Let's rename the folder name since it's too long
if not os.path.isdir('/kaggle/working/fer-challenge'):
    os.rename("/kaggle/working/challenges-in-representation-learning-facial-expression-recognition-challenge", "/kaggle/working/fer-challenge")

# Set path to where we moved the dataset in output/working
os.chdir("/kaggle/working/fer-challenge")

# Extract fer2013tar.gz 
tf = tarfile.open("fer2013.tar.gz")
tf.extractall()
output_path =  "/kaggle/working/fer-challenge/images"

if os.path.exists(output_path):
    os.system('rm -rf {}'.format(output_path))

os.system('mkdir {}'.format(output_path))

label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

data = np.genfromtxt('fer2013/fer2013.csv',delimiter=',',dtype=None, encoding=None)
labels = data[1:,0].astype(np.int32)
image_buffer = data[1:,1]
images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in image_buffer])
usage = data[1:,2]
dataset = zip(labels, images, usage)
usage_path = ""
for i, d in enumerate(dataset):
    if(d[-1] == "Training" or d[-1] == "PrivateTest"):
        usage_path = os.path.join(output_path, "Training")
    else:
        usage_path = os.path.join(output_path, d[-1])

    label_path = os.path.join(usage_path, label_names[d[0]])
    img = d[1].reshape((48,48))
    img_name = '%08d.jpg' % i
    img_path = os.path.join(label_path, img_name)
    if not os.path.exists(usage_path):
        os.system('mkdir {}'.format(usage_path))
    if not os.path.exists(label_path):
        os.system('mkdir {}'.format(label_path))
    cv2.imwrite(img_path, img)

    #     print('Write {}'.format(img_path))
# Copy cleaned.csv file to working folder
path = '/kaggle/input/cleaned/cleaned.csv'
destination = '/kaggle/working/fer-challenge/images/cleaned.csv'
shutil.copyfile(path, destination)
# Change path to where we formed our images
path = "/kaggle/working/fer-challenge/images"
df = pd.read_csv(path+'/cleaned.csv', header='infer')
np.random.seed(42)
data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
ds_tfms=get_transforms(), size=224, num_workers=8).normalize(imagenet_stats)
# # change path to where we formed our images
# path = "/kaggle/working/fer-challenge/images"
# os.chdir(path)

# # bs = 64
# tfms = get_transforms(do_flip=False)
# data = ImageDataBunch.from_folder(path, train = "Training", valid_pct=0.2, ds_tfms=tfms, size=26, num_workers=0, bs = 64)
print(f"Classes in our data: {data.classes}\nNumber of classes: {data.c}\nTraining Dataset Length: {len(data.train_ds)}\nValidation Dataset Length: {len(data.valid_ds)}")

data.show_batch(rows=3, columns = 5, figsize=(5,5))
learn = cnn_learner(data, models.resnet34, metrics=[accuracy,error_rate])
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.load('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(3e-6,3e-3))
learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
db = (ImageList.from_folder("/kaggle/working/fer-challenge/images/Training")
                   .split_none()
                   .label_from_folder()
                   .transform(get_transforms(), size=224)
                   .databunch()
     )
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)

learn_cln.load('/kaggle/working/fer-challenge/images/models/stage-2');
ds, idxs = DatasetFormatter().from_toplosses(learn_cln)
ImageCleaner(ds, idxs, path, batch_size=6)
learn.load('stage-2')
# ds_tfms=get_transforms(), size=224, num_workers=8
data_test = (ImageList.from_folder(path)
            .split_by_folder(train='Training', valid='PublicTest')
            .label_from_folder()
            .transform(tfms=get_transforms(), size=224)
            .databunch()
            .normalize()
        )

loss, acc, err_r = learn.validate(data_test.valid_dl)
loss = str(np.round(loss, 3))
print(f"Our final model's training loss: {loss}, with Accuracy: {round(acc.item(), 3)} and Error Rate: {round(err_r.item(), 3)}")