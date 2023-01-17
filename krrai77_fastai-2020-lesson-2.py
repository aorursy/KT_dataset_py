#You have to install torch 1.6, Fastai >=2.0.0 version.



!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html



#Upgrade kornia and allennlp version since current version does not support torch 1.6



!pip install --upgrade kornia

!pip install allennlp==1.1.0.rc4



#Install/upgrade fastai package



!pip install --upgrade fastai

#Load the libraries and verify the versions



import torch

print(torch.__version__)

print(torch.cuda.is_available())



import fastai

print(fastai.__version__)



from fastai.vision.all import *
key = os.environ.get('AZURE_SEARCH_KEY', 'e7a1d32f56364353b6e9e376b5ba20cd') # save the api key in a variable
from utils_bing_image_search import * #importing the utils file.
search_images_bing
results = search_images_bing(key, 'grizzly bear') #search for grizzly bear using the key in bing image search

ims = results.attrgot('content_url') #download the urls of all the images matching the given criteria

len(ims)
ims
bear_types = 'grizzly','black','teddy' #define the labels/categories for the images to be classified

path = Path('bears')
#all the matching image urls that fall into the defined lables/categories are grouped into seperate folders

if not path.exists():

    path.mkdir()

    for o in bear_types:

        dest = (path/o)

        dest.mkdir(exist_ok=True)

        results = search_images_bing(key, f'{o} bear')

        download_images(dest, urls=results.attrgot('content_url'))
fns = get_image_files(path) #view the grouped images

fns
failed = verify_images(fns) #verify images that failed to open

failed
failed.map(Path.unlink) #fast.ai provides a simple method to delete the path of all the failed files
#preparation for loading the data using dataloaders function. 2 blocks for independent and dependent variables are created.

#here independent are the images, dependent are lables defined.

#all images are resized to a common square size.

bears = DataBlock(

    blocks=(ImageBlock, CategoryBlock), 

    get_items=get_image_files, 

    splitter=RandomSplitter(valid_pct=0.2, seed=42),

    get_y=parent_label,

    item_tfms=Resize(128))
#bears template is used with the dataloaders function. imgage path has to be given as input.

dls = bears.dataloaders(path)
#see a sample of the validation images. 

dls.valid.show_batch(max_n=6, nrows=1)
#try image transformation techniques like squishing and padding and view the output

bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))

dls = bears.dataloaders(path)

dls.valid.show_batch(max_n=4, nrows=1)
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))

dls = bears.dataloaders(path)

dls.valid.show_batch(max_n=4, nrows=1)
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3)) #30% of the image area is zoomed by specifying 0.3.

dls = bears.dataloaders(path)

dls.train.show_batch(max_n=4, nrows=1,unique=True)
#prepare the data for training

bears = bears.new(

    item_tfms=RandomResizedCrop(224, min_scale=0.5),

    batch_tfms=aug_transforms())

dls = bears.dataloaders(path)
#train the model using resnet18 architecture that is pretrained.

learn = cnn_learner(dls, resnet18, metrics=error_rate)

learn.fine_tune(4)
#validate the model performance using confusion matrix. Diagonals (dark blue) indicate correct predictions for each class.

#other cells indicate the number of wrong predictions.

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
#use this awesome fast.ai function to see the wrong predictions based on the highest loss rate

#first lable indicates predicted, second indicates target label, next is the loss rate and fourth value is the probability

#high probability indicates high confidence level by the model. It ranges between 0 and 1.

#high loss rate indicates how bad the model performace is.

interp.plot_top_losses(5, nrows=3)
#clean the fautly images and lables.

from fastai.vision.widgets import *

cleaner = ImageClassifierCleaner(learn)

cleaner
#delete unwanted images by removing the links.

for idx in cleaner.delete(): cleaner.fns[idx].unlink()
#update the modified lables in the path folder.

for idx,cat in cleaner.change(): 

    real_dst = os.path.join(path/cat, cleaner.fns[idx].name)

    if os.path.exists(real_dst):

        old_file_path = cleaner.fns[idx]

        old_cat = old_file_path.parent.stem

        new_file_path = f'{path/cat/old_cat}_{str(old_file_path.name.replace(" ","").lower())}'

        shutil.move(str(cleaner.fns[idx]), new_file_path)

    else:

        shutil.move(str(cleaner.fns[idx]), path/cat)
#export your entire model along with the dataloader info.

learn.export()
#fast.ai saves the exported model as .pkl file that can be used for production.

path = Path()

path.ls(file_exts='.pkl')
#create inference learner from the exported file.

learn_inf = load_learner(path/'export.pkl')
#sample images used for prediction

from PIL import Image



imagebear = Image.open("../input/bearimage/becca-_r6w0R6SueQ-unsplash.jpg")

imagegriz = Image.open("../input/blackbear/zdenek-machacek-_QG2C0q6J-s-unsplash.jpg")

imageblack = Image.open("../input/blackclose/marc-olivier-jodoin-sI2Dz2dacGI-unsplash.jpg")
#while doing inference, we're getting predictions for one image at a time

learn_inf.predict("../input/blackclose/marc-olivier-jodoin-sI2Dz2dacGI-unsplash.jpg")
learn_inf.dls.vocab