from fastai import *

from fastai.vision import *

import os

from os import listdir

%reload_ext autoreload

%autoreload 2

%matplotlib inline

path = "../input/plantvillage/PlantVillage/"

os.listdir(path)
path = Path(path); path
directory_root = '../input/plantvillage/'

image_list, label_list = [], []

try:

    print("[INFO] Loading images ...")

    root_dir = listdir(directory_root)

    for directory in root_dir :

        # remove .DS_Store from list

        if directory == ".DS_Store" :

            root_dir.remove(directory)



    for plant_folder in root_dir :

        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")

        

        for disease_folder in plant_disease_folder_list :

            # remove .DS_Store from list

            if disease_folder == ".DS_Store" :

                plant_disease_folder_list.remove(disease_folder)



        for plant_disease_folder in plant_disease_folder_list:

            print(f"[INFO] Processing {plant_disease_folder} ...")

            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

                

            for single_plant_disease_image in plant_disease_image_list :

                if single_plant_disease_image == ".DS_Store" :

                    plant_disease_image_list.remove(single_plant_disease_image)



            for image in plant_disease_image_list[:200]:

                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"

                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:

                    image_list.append(image_directory)

                    label_list.append(plant_disease_folder)

    print("[INFO] Image loading completed")  

except Exception as e:

    print(f"Error : {e}")
tfms = get_transforms(flip_vert=True, max_warp=0., max_zoom=0., max_rotate=0.)
file_path = '../input/plantvillage/PlantVillage/Potato___Early_blight/0faca7fe-7254-4dfa-8388-bbc776338c9c___RS_Early.B 7929.JPG'
dir_name = os.path.dirname(file_path)
dir_length = len(dir_name.split("/"))

dir_name.split("/")
dir_name.split("/")[dir_length - 1]
def get_labels(file_path): 

    dir_name = os.path.dirname(file_path)

    split_dir_name = dir_name.split("/")

    dir_length = len(split_dir_name)

    label  = split_dir_name[dir_length - 1]

    return(label)
data = ImageDataBunch.from_name_func(path, image_list, label_func=get_labels,  size=224, 

                                     bs=64,num_workers=2,ds_tfms=tfms)

data = data.normalize()
data.show_batch(rows=3, figsize=(15,11))
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir='/tmp/models/')
learn.fit_one_cycle(10)
interpretation = ClassificationInterpretation.from_learner(learn)

losses, indices = interpretation.top_losses()

interpretation.plot_top_losses(4, figsize=(15,11))
interpretation.plot_confusion_matrix(figsize=(12,12), dpi=60)
interpretation.most_confused(min_val=2)
learn.save('classification-1')
learn.unfreeze()

learn.fit_one_cycle(1)
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-3))
interpretation = ClassificationInterpretation.from_learner(learn)

losses, indices = interpretation.top_losses()

interpretation.plot_top_losses(4, figsize=(15,11))
interpretation.most_confused(min_val=2)
learn.save('resnet34-classifier.pkl')
learn.recorder.plot_losses()
os.chdir("/tmp/models/")