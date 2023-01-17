!pip install fastai2 -q
#Load the dependancies
from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *
from fastai2.medical.imaging import *

import pydicom
import seaborn as sns

import numpy as np
import pandas as pd
import os
source = Path("../input/siim-isic-melanoma-classification")
files = os.listdir(source)
print(files)
train = source/'train'
train_files = get_dicom_files(train)
train_files
patient1 = train_files[7]
dimg = dcmread(patient1)
dimg
def show_one_patient(file):
    """ function to view patient image and choosen tags within the head of the DICOM"""
    pat = dcmread(file)
    print(f'patient Name: {pat.PatientName}')
    print(f'Patient ID: {pat.PatientID}')
    print(f'Patient age: {pat.PatientAge}')
    print(f'Patient Sex: {pat.PatientSex}')
    print(f'Body part: {pat.BodyPartExamined}')
    trans = Transform(Resize(256))
    dicom_create = PILDicom.create(file)
    dicom_transform = trans(dicom_create)
    return show_image(dicom_transform)
show_one_patient(patient1)
from pydicom.pixel_data_handlers.util import convert_color_space
arr = dimg.pixel_array
convert = convert_color_space(arr, 'YBR_FULL_422', 'RGB')
show_image(convert)
px = dimg.pixels.flatten()
plt.hist(px, color='c')
df = pd.read_csv(source/'train.csv')
df.head()
#Plot 3 comparisons
def plot_comparison3(df, feature, feature1, feature2):
    "Plot 3 comparisons from a dataframe"
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (16, 4))
    s1 = sns.countplot(df[feature], ax=ax1)
    s1.set_title(feature)
    s2 = sns.countplot(df[feature1], ax=ax2)
    s2.set_title(feature1)
    s3 = sns.countplot(df[feature2], ax=ax3)
    s3.set_title(feature2)
    plt.show()
plot_comparison3(df, 'sex', 'age_approx', 'benign_malignant')
#Plot 1 comparisons
def plot_comparison1(df, feature):
    "Plot 1 comparisons from a dataframe"
    fig, (ax1) = plt.subplots(1,1, figsize = (16, 4))
    s1 = sns.countplot(df[feature], ax=ax1)
    s1.set_title(feature)
    plt.show()
plot_comparison1(df, 'diagnosis')
plot_comparison1(df, 'target')
plot_comparison1(df, 'anatom_site_general_challenge')
eda_df = df[['sex','age_approx','anatom_site_general_challenge','diagnosis','target']]
eda_df.head()
len(eda_df)
sex_count = eda_df['sex'].isna().sum(); age_count = eda_df['age_approx'].isna().sum(); anatom_count = eda_df['anatom_site_general_challenge'].isna().sum()
print(f'Nan values in sex column: {sex_count}, age column: {age_count}, anatom count: {anatom_count}')
df_drop = eda_df.dropna()
len(df_drop)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
edaa_df = eda_df.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
edaa_df.head()
sns.set(style="whitegrid")
sns.set_context("paper")
sns.pairplot(eda_df, hue="target", height=5, aspect=2, palette='gist_rainbow_r')
sns.pairplot(eda_df, hue="age_approx", height=6, aspect=3, diag_kws={'bw':'0.05'})
sns.set(style="whitegrid")
sns.set_context("poster")
sns.pairplot(edaa_df, hue="target", height=6, palette='gist_rainbow', diag_kws={'bw':'0.05'})
get_x = lambda x:source/'train'/f'{x[0]}.dcm'
get_y=ColReader('target')
batch_tfms = aug_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
class PILDicom2(PILBase):
    _open_args,_tensor_cls,_show_args = {},TensorDicom,TensorDicom._show_args
    @classmethod
    def create(cls, fn:(Path,str,bytes), mode=None)->None:
        "Open a `DICOM file` from path `fn` or bytes `fn` and load it as a `PIL Image`"
        dimg = dcmread(fn)
        arr = dimg.pixel_array; convert = convert_color_space(arr,'YBR_FULL_422', 'RGB')
        im = Image.fromarray(convert)
        im.load()
        im = im._new(im.im)
        return cls(im.convert(mode) if mode else im)
blocks = (ImageBlock(cls=PILDicom2), CategoryBlock)
melanoma = DataBlock(blocks=blocks,
                   get_x=get_x,
                   splitter=RandomSplitter(),
                   item_tfms=Resize(128),
                   get_y=ColReader('target'),
                   batch_tfms=batch_tfms)
dls = melanoma.dataloaders(df.sample(100), bs=2)
dls = dls.cuda()
dls.show_batch(max_n=12, nrows=2, ncols=6)
roc = RocAuc()
dls.c
model = xresnet18_deeper(n_out=dls.c)
set_seed(77)
learn = Learner(dls, model, 
                opt_func=ranger,
                loss_func=LabelSmoothingCrossEntropy(),
                metrics=[accuracy, roc],
                cbs = ShowGraphCallback())
learn.freeze()
learn.fit_one_cycle(1, 5e-2)
learn.save('xresnet18_stg1')
learn.unfreeze()
learn.fit_flat_cos(2,slice(1e-6,1e-4))
learn.save('xresnet18_stg2')
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(12)
tst = source/'test'
test_set = get_dicom_files(tst)
test_set
test_set = test_set[:100]
test_set
test_patient = test_set[1]
test_patient
learn.load('xresnet18_stg2')
_ = learn.predict(test_patient)
_
_ = learn.predict(test_patient)
print(_[2][1])
sample_sub = pd.read_csv(source/'sample_submission.csv')
sample_sub = sample_sub[:100]
sample_sub
del sample_sub['target']
sample_list = []
for i in test_set:
    pre = learn.predict(i)
    l = float(pre[2][1])
    sample_list.append(l)
sample_list
sub = sample_sub.assign(target=sample_list)
sub.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv')
sub