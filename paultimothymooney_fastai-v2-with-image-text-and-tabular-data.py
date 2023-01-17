import torch

import fastai

from fastai.tabular.all import *

from fastai.text.all import *

from fastai.vision.all import *

from fastai.medical.imaging import *

from fastai import *



import time

from datetime import datetime



print(f'Notebook last run on {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')

print('Using fastai version ',fastai.__version__)

print('And torch version ',torch.__version__)
def plot_fastai_results(learn):

    '''

    Plots sensitivity, speficificty, prevalence, accuracy, and confusion matrix for a fastai model named "learn".

    Some portions are adapted from https://github.com/fastai/fastai/blob/master/nbs/61_tutorial.medical_imaging.ipynb

    '''

    interp = Interpretation.from_learner(learn)

    interp = ClassificationInterpretation.from_learner(learn)

    interp.plot_confusion_matrix(figsize=(7,7))

    losses,idxs = interp.top_losses()

    len(dls.valid_ds)==len(losses)==len(idxs)

    upp, low = interp.confusion_matrix()

    tn, fp = upp[0], upp[1]

    fn, tp = low[0], low[1]

    sensitivity = tp/(tp + fn)

    print('Sensitivity: ',sensitivity)

    specificity = tn/(fp + tn)

    print('Specificity: ',specificity)

    #val = dls.valid_ds.cat

    prevalance = 15/50

    print('Prevalance: ',prevalance)

    accuracy = (sensitivity * prevalance) + (specificity * (1 - prevalance))

    print('Accuracy: ',accuracy)
pneumothorax_source = untar_data(URLs.SIIM_SMALL)

items = get_dicom_files(pneumothorax_source/f"train/")

trn,val = RandomSplitter()(items)

df = pd.read_csv(pneumothorax_source/f"labels.csv")

pneumothorax = DataBlock(blocks=(ImageBlock(cls=PILDicom), CategoryBlock),

                   get_x=lambda x:pneumothorax_source/f"{x[0]}",

                   get_y=lambda x:x[1],

                   batch_tfms=aug_transforms(size=224))

dls = pneumothorax.dataloaders(df.values)

dls.show_batch(max_n=16)
learn = cnn_learner(dls, resnet34, metrics=accuracy, model_dir='/kaggle/tmp/model/')

learn.lr_find()

learn.fine_tune(5)

learn.show_results()
plot_fastai_results(learn=learn)
path = Path('/kaggle/input/chest-xray-pneumonia/chest_xray/')

dls = ImageDataLoaders.from_folder(path, train='train',

                                   item_tfms=Resize(224),valid_pct=0.2,

                                   bs=64,seed=0)

dls.show_batch()
learn = cnn_learner(dls, resnet34, metrics=accuracy, model_dir='/kaggle/tmp/model/')

learn.lr_find()

learn.fine_tune(5)

learn.show_results()
plot_fastai_results(learn=learn)
path = untar_data(URLs.ADULT_SAMPLE)

df = pd.read_csv(path/'adult.csv')

dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",

    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],

    cont_names = ['age', 'fnlwgt', 'education-num'],

    procs = [Categorify, FillMissing, Normalize])

splits = RandomSplitter(valid_pct=0.2)(range_of(df))

to = TabularPandas(df, procs=[Categorify, FillMissing,Normalize],

                   cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],

                   cont_names = ['age', 'fnlwgt', 'education-num'],

                   y_names='salary',

                   splits=splits)

dls = to.dataloaders(bs=64)

dls.show_batch()
learn = tabular_learner(dls, metrics=accuracy)

learn.lr_find()

learn.fine_tune(5)

learn.show_results()
plot_fastai_results(learn=learn)
df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv', skipinitialspace=True)

dls = TabularDataLoaders.from_df(df=df, path=path, y_names="income",

    cat_names = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race'],

    cont_names = ['age', 'fnlwgt', 'education.num'],

    procs = [Categorify, FillMissing, Normalize])

splits = RandomSplitter(valid_pct=0.2)(range_of(df))

to = TabularPandas(df, procs=[Categorify, FillMissing,Normalize],

                   cat_names = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race'],

                   cont_names = ['age', 'fnlwgt', 'education.num'],

                   y_names='income',

                   splits=splits)

dls = to.dataloaders(bs=64)

dls.show_batch()
learn = tabular_learner(dls, metrics=accuracy)

learn.lr_find()

learn.fine_tune(5)

learn.show_results()
plot_fastai_results(learn=learn)
path = untar_data(URLs.IMDB)

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')

dls.show_batch(max_n=3) # investigate https://forums.fast.ai/t/most-of-the-items-in-show-batch-is-xxpad-strings/78989
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

learn.lr_find()

learn.fine_tune(7)

learn.show_results(max_n=3)
plot_fastai_results(learn=learn)
df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

dls = TextDataLoaders.from_df(df=df,model_dir='/kaggle/tmp/model/')

#dls.show_batch() # investigate https://forums.fast.ai/t/most-of-the-items-in-show-batch-is-xxpad-strings/78989/5

df.head(15)
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

learn.lr_find()

learn.fine_tune(5)

learn.show_results(max_n=3) # investigate https://forums.fast.ai/t/most-of-the-items-in-show-batch-is-xxpad-strings/78989/2
plot_fastai_results(learn=learn)
!mkdir /kaggle/working/docker/

!pip freeze > '../working/docker/requirements.txt'