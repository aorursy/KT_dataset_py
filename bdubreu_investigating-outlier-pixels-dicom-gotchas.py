!pip install torch torchvision feather-format pyarrow --upgrade   > /dev/null

!pip install git+https://github.com/arraiyopensource/kornia       > /dev/null

!pip install git+https://github.com/fastai/fastai_dev             > /dev/null



    

from fastai2.basics           import *

from fastai2.medical.imaging  import *

from tqdm import tqdm_notebook

np.set_printoptions(linewidth=120)
path_inp = Path('../input')

path = path_inp/'rsna-intracranial-hemorrhage-detection'

path_trn = path/'stage_1_train_images'

path_tst = path/'stage_1_test_images'
# To understand what's going on here, please refer to https://www.kaggle.com/jhoward/some-dicom-gotchas-to-be-aware-of-fastai

path_df = path_inp/'creating-a-metadata-dataframe'



df_lbls = pd.read_feather(path_df/'labels.fth')

df_tst = pd.read_feather(path_df/'df_tst.fth')

df_trn = pd.read_feather(path_df/'df_trn.fth')



comb = df_trn.join(df_lbls.set_index('ID'), 'SOPInstanceUID')

assert not len(comb[comb['any'].isna()])



# This is where you find the weird min/max values

repr_flds = ['BitsStored','PixelRepresentation']

comb.pivot_table(values=['img_mean','img_max','img_min','PatientID','any'], index=repr_flds,

                   aggfunc={'img_mean':'mean','img_max':'max','img_min':'min','PatientID':'count','any':'mean'})
# we've got 256 images in the training set with such high values

comb[comb['img_max'] > 30000].shape
#as seen two cells above, we've got at least an image with a max px value of 32767

max_val = comb['img_max'].max() #32767

#Actually, we've got two of them

comb[comb['img_max'] == max_val]['fname']
# We can use fastaiv2 fantastic functionalities to quickly plot this pic

dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_d6a09aba5.dcm').show(figsize=(6,6))
# floor_value is the value above which we consider pixel values to be curious (the cases we want to investigate)

# based on the above table, I put it at 3000, but feel free to experiment

floor_value = 3000

f_name = comb[comb['img_max'] > floor_value].sample()['fname'].values[0]

print(f_name)

dcmread(f_name).show(figsize=(6,6))
dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b914af0ba.dcm').show(figsize=(6,6))
# PS: not all of them are shiny objects. Some other are just plainly screwed images, like this one: 

dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b79eed528.dcm').show(figsize=(6,6))

# I wonder how we could come up with a systematized way to find them out ? I'll keep looking and update this kernel if I come up with something interesting.

# In the meantime, thank you for you attention !
