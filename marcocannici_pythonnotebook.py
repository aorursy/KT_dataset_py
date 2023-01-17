# Visualize Tensorboard within the notebook

%load_ext tensorboard.notebook

%tensorboard --logdir logs
# Path to dataset folder (the zip), containing npy files

DATA_PATH = "../input/classwork_dataset"

# Path to images

IMG_PATH = "../input/classwork_dataset/dataset"

FOLDER_IMG = "./folders_dataset"

# Path where to save Tensorboard checkpoints

TB_PATH = "./logs"
# If you want to inspect local files and download them run

# this cell, then click on links

from IPython.display import FileLink, FileLinks

FileLinks('.')
import os

import numpy as np

import shutil



def move_files(imgs, src, dst, lbl=None):

    for i, img in enumerate(imgs):

        src_img = os.path.join(src, img + '.JPG')

        if lbl is not None:

            dst_img = os.path.join(dst, str(lbl[i]), img + '.JPG')

        else:

            dst_img = os.path.join(dst, img + '.JPG')



        os.makedirs(os.path.dirname(dst_img), exist_ok=True)

        shutil.copy(src=src_img, dst=dst_img)



# Split images in folders

test_imgs = np.load(os.path.join(DATA_PATH, "test_inputs.npy"))

train_imgs = np.load(os.path.join(DATA_PATH, "training_inputs.npy"))

train_lbls = np.load(os.path.join(DATA_PATH, "training_targets.npy"))

val_imgs = np.load(os.path.join(DATA_PATH, "validation_inputs.npy"))

val_lbls = np.load(os.path.join(DATA_PATH, "validation_targets.npy"))



#move_files(test_imgs, "../input/classwork_dataset/dataset", "./folders_dataset/test")

#move_files(train_imgs, "../input/classwork_dataset/dataset", "./folders_dataset/train", lbl=train_lbls)

#move_files(val_imgs, "../input/classwork_dataset/dataset", "./folders_dataset/validation", lbl=val_lbls)

import os

import base64

import numpy as np

from datetime import datetime

from IPython.display import HTML



# A notebook version of the same function you find in utils.py

# Once executed, this function displays a link that you can use to

# download your csv file!

def create_csv(results, results_dir='./'):



    csv_fname = 'results_'

    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'



    content = 'Id,Category\n'

    for key, value in results.items():

        content += key + ',' + str(value) + '\n'



    b64 = base64.b64encode(content.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">Download result csv file</a>'

    html = html.format(payload=payload,filename=csv_fname)

    

    file_path = os.path.join(results_dir, csv_fname)

    with open(file_path, 'w') as f:

        f.write(content)

    print("Results file create at: ", file_path)

    print("Click the link below to download!")

    

    return HTML(html)
# Train loop.....

# ....



from PIL import Image

results = {}

for img_name in test_imgs:

    img = Image.open(os.path.join("./folders_dataset/test", img_name + ".JPG"))

    img = np.array(img, dtype=np.float32).reshape(1, 100, 100, 3)

    lbl = model.predict(img)[0]

    cls = np.argmax(lbl)

    results[img_name] = cls
# Run create_csv on a separate cell to create a download link

create_csv(results)