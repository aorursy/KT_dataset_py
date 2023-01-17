#hide

from utils import *
#id first_training

#caption Results from the first training

# CLICK ME

from fastai2.vision.all import *

path = untar_data(URLs.PETS)/'images'



def is_cat(x): return x[0].isupper()

dls = ImageDataLoaders.from_name_func(

    path, get_image_files(path), valid_pct=0.2, seed=42,

    label_func=is_cat, item_tfms=Resize(224))



learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fine_tune(1)
1+1
img = PILImage.create('images/chapter1_cat_example.jpg')

img.to_thumb(192)
#hide_output

uploader = widgets.FileUpload()

uploader
#hide

# For the book, we can't actually click an upload button, so we fake it

uploader = SimpleNamespace(data = ['images/chapter1_cat_example.jpg'])
img = PILImage.create(uploader.data[0])

is_cat,_,probs = learn.predict(img)

print(f"Is this a cat?: {is_cat}.")

print(f"Probability it's a cat: {probs[1].item():.6f}")
#hide_input

#caption A traditional program

#id basic_program

#alt Pipeline inputs, program, results

gv('''program[shape=box3d width=1 height=0.7]

inputs->program->results''')
#hide_input

#caption A program using weight assignment

#id weight_assignment

gv('''model[shape=box3d width=1 height=0.7]

inputs->model->results; weights->model''')
#hide_input

#caption Training a machine learning model

#id training_loop

#alt The basic training loop

gv('''ordering=in

model[shape=box3d width=1 height=0.7]

inputs->model->results; weights->model; results->performance

performance->weights[constraint=false label=update]''')
#hide_input

#caption Using a trained model as a program

#id using_model

gv('''model[shape=box3d width=1 height=0.7]

inputs->model->results''')
#hide_input

#caption Detailed training loop

#id detailed_loop

gv('''ordering=in

model[shape=box3d width=1 height=0.7 label=architecture]

inputs->model->predictions; parameters->model; labels->loss; predictions->loss

loss->parameters[constraint=false label=update]''')
path = untar_data(URLs.CAMVID_TINY)

dls = SegmentationDataLoaders.from_label_func(

    path, bs=8, fnames = get_image_files(path/"images"),

    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',

    codes = np.loadtxt(path/'codes.txt', dtype=str)

)



learn = unet_learner(dls, resnet34)

learn.fine_tune(8)
learn.show_results(max_n=6, figsize=(7,8))
from fastai2.text.all import *



dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')

learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

learn.fine_tune(4, 1e-2)
learn.predict("I really liked that movie!")
from fastai2.tabular.all import *

path = untar_data(URLs.ADULT_SAMPLE)



dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",

    cat_names = ['workclass', 'education', 'marital-status', 'occupation',

                 'relationship', 'race'],

    cont_names = ['age', 'fnlwgt', 'education-num'],

    procs = [Categorify, FillMissing, Normalize])



learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(3)
from fastai2.collab import *

path = untar_data(URLs.ML_SAMPLE)

dls = CollabDataLoaders.from_csv(path/'ratings.csv')

learn = collab_learner(dls, y_range=(0.5,5.5))

learn.fine_tune(10)
learn.show_results()