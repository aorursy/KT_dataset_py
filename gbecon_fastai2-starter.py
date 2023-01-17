!pip install fastai2
!pip install nbdev azure-cognitiveservices-search-imagesearch sentencepiece
!rm -rf course-v4

!git clone https://github.com/fastai/course-v4.git
!ls course-v4/nbs
from utils import *

from fastai2.vision.all import *
path = untar_data(URLs.PETS)/'images'
!ls /root/.fastai/data
def is_cat(x): return x[0].isupper()

dls = ImageDataLoaders.from_name_func(

    path, get_image_files(path), valid_pct=0.2, seed=42,

    label_func=is_cat, item_tfms=Resize(224))



learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fine_tune(1)
img = PILImage.create('course-v4/nbs/images/chapter1_cat_example.jpg')

img.to_thumb(192)
# from fastai2.vision import  widgets



# uploader = widgets.FileUpload()

# uploader
# PILImage.create(uploader.data[0])
uploader = SimpleNamespace(data = ['course-v4/nbs/images/chapter1_cat_example.jpg'])

img = PILImage.create(uploader.data[0])

is_cat,_,probs = learn.predict(img)

print(f"Is this a cat?: {is_cat}.")

print(f"Probability it's a cat: {probs[1].item():.6f}")
path = untar_data(URLs.CAMVID_TINY)

dls = SegmentationDataLoaders.from_label_func(

    path, bs=8, fnames = get_image_files(path/"images"),

    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',

    codes = np.loadtxt(path/'codes.txt', dtype=str)

)



learn = unet_learner(dls, resnet34)

learn.fine_tune(8)
learn.show_results(max_n=6, figsize=(7,8))
from fastai2.tabular.all import *

path = untar_data(URLs.ADULT_SAMPLE)



dls = TabularDataLoaders.from_csv(path/'adult.csv', path, y_names="salary",

    cat_names = ['workclass', 'education', 'marital-status', 'occupation',

                 'relationship', 'race'],

    cont_names = ['age', 'fnlwgt', 'education-num'],

    procs = [Categorify, FillMissing, Normalize])



learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(5)