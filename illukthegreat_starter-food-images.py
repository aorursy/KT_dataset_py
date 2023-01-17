from pathlib import Path

from fastai.vision import *

from fastai.metrics import error_rate
bs = 128   # batch size

arch = models.resnet50
path = Path(os.path.join('..', 'input', 'food41'))

path_h5 = path

path_img = path/'images'

path_meta = path/'meta/meta'

path_working = '/kaggle/working/'

path_last_model = Path(os.path.join('..', 'input', 'starter-food-images'))



!ls {path}
!ls {path_last_model}
# Modify the from folder function in fast.ai to use the dictionary mapping from folder to space seperated labels

def label_from_folder_map(class_to_label_map):

    return  lambda o: class_to_label_map[(o.parts if isinstance(o, Path) else o.split(os.path.sep))[-2]]
# Develop dictionary mapping from classes to labels

classes = pd.read_csv(path_meta/'classes.txt', header=None, index_col=0,)

labels = pd.read_csv(path_meta/'labels.txt', header=None)

classes['map'] = labels[0].values

classes_to_labels_map = classes['map'].to_dict()

label_from_folder_food_func = label_from_folder_map(classes_to_labels_map)
# Setup the training ImageList for the DataBunch

train_df = pd.read_csv(path_meta/'train.txt', header=None).apply(lambda x : x + '.jpg')

train_image_list = ImageList.from_df(train_df, path_img)



# Setup the validation ImageList for the DataBunch

valid_df = pd.read_csv(path_meta/'test.txt', header=None).apply(lambda x : x + '.jpg')

valid_image_list = ImageList.from_df(valid_df, path_img)
def get_data(bs, size):

    """Function to return DataBunch with different batch and image sizes."""

    # combine training and validation image lists into one ImageList

    data = (train_image_list.split_by_list(train_image_list, valid_image_list))

    

    tfms = get_transforms() # get all transformations



    # label with function defined above using the mapping from folder name to labels

    # perform transformations and turn into a DataBunch

    data = data.label_from_func(label_from_folder_food_func).transform(

        tfms, size=size).databunch(bs=bs,  num_workers = 0).normalize(

        imagenet_stats)

    return data



data = get_data(bs, 64)
# show a batch to get an idea of the images and labels

data.show_batch(rows=4, figsize=(10,9),)
# print all labels in the dataset

print(data.classes)
# setup data, model architecture, and metrics

learn = cnn_learner(data, arch, metrics=error_rate)
# model_dir is set to the path where DataBunch is located

# Kaggle this needs to be set to '/kaggle/working/'

learn.model_dir = path_working
learn.data = get_data(bs//4, 412) #bs = 32

learn.purge();
learn.load(path_last_model/'stage-2-50-412');
learn.validate(learn.data.valid_dl)
learn.TTA()