%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import pandas as pd

from fastai.utils.mem import *
path = Path('/kaggle/input/iwildcam-2019-fgvc6')



debug =1

if debug:

    train_pct=0.04

else:

    train_pct=0.5
# Load train dataframe

train_df = pd.read_csv(path/'train.csv')

train_df = pd.concat([train_df['id'],train_df['category_id']],axis=1,keys=['id','category_id'])

train_df.head()
# Load sample submission

test_df = pd.read_csv(path/'test.csv')

test_df = pd.DataFrame(test_df['id'])

test_df['predicted'] = 0

test_df.head()

free = gpu_mem_get_free_no_cache()

# the max size of bs depends on the available GPU RAM

if free > 8200: bs=64

else:           bs=32

print(f"using bs={bs}, have {free}MB of GPU RAM free")



tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,

                      p_affine=1., p_lighting=1.)

# app_train = app_train.append(app_test).reset_index()
train, test = [ImageList.from_df(df, path=path, cols='id', folder=folder, suffix='.jpg') 

               for df, folder in zip([train_df, test_df], ['train_images', 'test_images'])]

if debug:

    src= train.split_subsets(train_size=train_pct, valid_size= train_pct*2)

#     test=test[:1000]

else:

    src= train.split_subsets(train_size=train_pct, valid_size=0.2, seed=2)

#     src= train.split_by_rand_pct(0.2, seed=2)



print(src)

    

def get_data(size, bs, padding_mode='reflection'):

    return (src.label_from_df(cols='category_id')

           .add_test(test)

           .transform(tfms, size=size, padding_mode=padding_mode)

           .databunch(bs=bs).normalize(imagenet_stats))    

    

# data = (train.split_by_rand_pct(0.2, seed=2)

#         .label_from_df(cols='category_id')

#         .add_test(test)

#         .transform(get_transforms(), size=32)

#         .databunch(path=Path('.'), bs=64).normalize())
data = get_data(224, bs, 'zeros')
def _plot(i,j,ax):

    x,y = data.train_ds[3]

    x.show(ax, y=y)



plot_multi(_plot, 3, 3, figsize=(8,8))
# learn = cnn_learner(data, base_arch=models.densenet121, metrics=[FBeta(),accuracy], wd=1e-5).mixup()

gc.collect()

# wd=1e-2

wd=1e-1

learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True, wd=wd )

learn.model_dir= '/kaggle/working/'
# lr=1e-2

# learn.fit_one_cycle(3, slice(lr), pct_start=0.8)
# learn.save('223')
# learn.unfreeze()

# learn.lr_find()

# learn.recorder.plot(suggestion=True)
# # lr = 2e-2

# # learn.fit_one_cycle(2, slice(lr))



# learn.fit_one_cycle(6, max_lr=slice(5.75E-06,lr/5), pct_start=0.8)
# learn.save('224')
!cp /kaggle/input/fastai-starter-iwildcam-2019-ad561b/224.pth /kaggle/working

learn.load('224')
data = get_data(352,bs)

learn.data = data

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))

learn.save('352')
# !cp /kaggle/input/352pth/352.pth /kaggle/working

# learn.load('352')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
lr = 1e-3

learn.fit_one_cycle(8, slice(lr/100, lr))
learn.save('stage-2-sz32')
# !cp /kaggle/input/stage-2-sz32.pth /kaggle/working

# learn.load('stage-2-sz32')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
test_preds = learn.get_preds(DatasetType.Test)

test_df['predicted'] = test_preds[0].argmax(dim=1)
test_df.shape
csv_path ='/kaggle/working/submission.csv'

test_df.to_csv(csv_path, index=False)
# # import the modules we'll need

# from IPython.display import HTML

# import base64





# # function that takes in a dataframe and creates a text link to  

# # download it (will only work for files < 2MB or so)

# def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

#     csv = df.to_csv()

#     b64 = base64.b64encode(csv.encode())

#     payload = b64.decode()

#     html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

#     html = html.format(payload=payload,title=title,filename=filename)

#     return HTML(html)







# # create a link to download the dataframe

# create_download_link(test_df[90000:120000])
