from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import train_test_split

import tensorflow as tf

import os

from pathlib import Path

import shutil

from tqdm import tqdm

import matplotlib.pyplot as plt
GCS_PATH = KaggleDatasets().get_gcs_path()

DATA_PATH = '/kaggle/input/gan-getting-started'

NEW_DATA_PATH = '/kaggle/new_scheme/gan-getting-started'



MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_jpg/*.jpg'))

print('Monet .jpg Files:', len(MONET_FILENAMES))



PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_jpg/*.jpg'))

print('Photo .jpg Files:', len(PHOTO_FILENAMES))
TEST_RATIO = 0.00000000001

photo_train_paths, photo_test_paths = train_test_split(PHOTO_FILENAMES, test_size=TEST_RATIO, random_state=42)

monet_train_paths, monet_test_paths = train_test_split(MONET_FILENAMES, test_size=TEST_RATIO, random_state=42)
os.chdir(DATA_PATH)
dir_names = ['trainA', 'trainB', 'testA', 'testB']

all_image_paths = [photo_train_paths, monet_train_paths, photo_test_paths, monet_test_paths]



# sample cutter for debugging purposes

SAMPLE = 1

N = int(SAMPLE)

all_image_paths = [x[:int(SAMPLE*len(x))] for x in all_image_paths] 

NEW_DATA_PATH = '/kaggle/new_scheme/gan-getting-started' + '_' + str(SAMPLE)
for dir_name, single_image_paths in zip(dir_names, all_image_paths):

    new_dir_path = Path(NEW_DATA_PATH) / dir_name

    old_dir_path = Path(DATA_PATH) 

    

    new_dir_path.mkdir(parents=True, exist_ok=True)

    #old_dir_path.mkdir(parents=True, exist_ok=True)

    

    for single_image_path in tqdm(single_image_paths):

        shutil.copy(old_dir_path / Path(single_image_path).parent.name / Path(single_image_path).name, new_dir_path / Path(single_image_path).name)
DATA_PATH
NEW_DATA_PATH
WORKING_PATH = '/kaggle/working'

os.chdir(WORKING_PATH)

!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

import os

os.chdir('pytorch-CycleGAN-and-pix2pix/')

!pip install -r requirements.txt
! python train.py --dataroot "/kaggle/new_scheme/gan-getting-started_0.01" --name photo2monet --model cycle_gan --n_epochs 10 --n_epochs_decay 10
MODEL_PATH = Path("/kaggle/working/pytorch-CycleGAN-and-pix2pix/checkpoints/photo2monet")

os.listdir(MODEL_PATH)

shutil.copy(MODEL_PATH / 'latest_net_G_A.pth', MODEL_PATH / 'latest_net_G.pth')
os.listdir('/kaggle/working/pytorch-CycleGAN-and-pix2pix/checkpoints')
#! python test.py --dataroot "/kaggle/new_scheme/gan-getting-started_0.01/testA" --name photo2monet002 --model test --no_dropout

! python test.py --dataroot "/kaggle/input/gan-getting-started/photo_jpg/" --name photo2monet --model test --no_dropout --num_test 7010

RESULTS_PATH = Path('/kaggle/working/pytorch-CycleGAN-and-pix2pix/results/photo2monet/test_latest/images')
result_imgs = os.listdir(RESULTS_PATH)
result_imgs_real = sorted([RESULTS_PATH / x for x in result_imgs if "real" in x])

result_imgs_fake = sorted([RESULTS_PATH / x for x in result_imgs if "fake" in x])
sorted(['a','c', 'b'])
plt.imshow(plt.imread(result_imgs_fake[0]))
plt.imshow(plt.imread(result_imgs_real[0]))
results_path = Path('/kaggle/working/results')

results_path.mkdir(parents=True, exist_ok=True)



submission_path = Path('/kaggle/working/images')

submission_path.mkdir(parents=True, exist_ok=True)



for img_path in result_imgs_fake:

    shutil.copy(img_path, results_path / os.path.basename(img_path))
import shutil

shutil.make_archive(submission_path, 'zip', results_path)
plt.imshow(plt.imread(results_path / os.listdir(results_path)[0]))
len(os.listdir(results_path))
! rm -r "/kaggle/working/pytorch-CycleGAN-and-pix2pix"
! rm -r "/kaggle/working/results"