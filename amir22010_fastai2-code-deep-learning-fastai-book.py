!pip install fastai2
#The first line imports all of the fastai.vision library. This gives us all of the functions and classes we will need to create a wide variety of computer vision models.
from fastai2.vision.all import *
#The second line downloads a standard dataset from the fast.ai datasets collection 
path = untar_data(URLs.PETS)/'images'
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(path, get_image_files(path), valid_pct=0.2, seed=42,label_func=is_cat, item_tfms=Resize(224))
#Tine tells fastai to create a convolutional neural network (CNN), and selects what architecture to use
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
img = PILImage.create(uploader.data[0])
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}; Probability it's a cat: {probs[1].item():.6f}")
img = PILImage.create(uploader.data[0])
img.to_thumb(192)
img = PILImage.create(uploader.data[0])
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}; Probability it's a cat: {probs[1].item():.6f}")
path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np.loadtxt(path/'codes.txt', dtype=str))
#unet
learn = unet_learner(dls, resnet34)
learn.fine_tune(10)
learn.show_results(max_n=8, figsize=(7,8))
from fastai2.text.all import *
dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)
learn.predict("I really liked that movie!")
#check documentation
doc(learn.predict)
from fastai2.tabular.all import *
path = untar_data(URLs.ADULT_SAMPLE)
!pip install nbdev
from fastai2.collab import *
path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv')
learn = collab_learner(dls, y_range=(0.5,5.5))
learn.fine_tune(10)
learn.show_results()
