from fastai import *

from fastai.vision import *
classes = ['bus','sedan','truck']
busfolder = 'bus'

sedanfolder = 'sedan'

truckfolder = 'truck'

busfile = 'bus.txt'

sedanfile = 'sedan.txt'

truckfile = 'truck.txt'



path = Path('data')

busdest = path/busfolder

busdest.mkdir(parents=True, exist_ok=True)

truckdest = path/truckfolder

truckdest.mkdir(parents=True, exist_ok=True)

sedandest = path/sedanfolder

sedandest.mkdir(parents=True, exist_ok=True)
!cp ../input/*/* {path}/
download_images(path/busfile, busdest, max_pics=700)

download_images(path/sedanfile, sedandest, max_pics=700)

download_images(path/truckfile, truckdest, max_pics=700)
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=700)
def random_seed(seed_value, use_cuda):

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    random.seed(seed_value) # Python

    if use_cuda: 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False

#Remember to use num_workers=0 when creating the DataBunch.
random_seed(123,True)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, ds_tfms=get_transforms(), size=256, num_workers=0).normalize(imagenet_stats)
data
data.show_batch(rows=3, figsize=(7,7))
learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
lrf = learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
learn.load('stage-1')

interp.plot_top_losses(9, figsize=(15,11), heatmap=False)

#ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)

#ImageCleaner(ds, idxs, path)
## import the modules we'll need

#from IPython.display import HTML

#import pandas as pd

#import numpy as np

#import base64



## function that takes in a dataframe and creates a text link to  

## download it (will only work for files < 2MB or so)

#def create_download_link(df, title = "Download CSV file", filename = "cleaned.csv"):  

#    csv = df.to_csv()

#    b64 = base64.b64encode(csv.encode())

#    payload = b64.decode()

#    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

#    html = html.format(payload=payload,title=title,filename=filename)

#    return HTML(html)



## create a link to download the dataframe

#create_download_link(df)



## ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 
df = pd.read_csv(path/'modifiedcsv.csv', index_col=0)

df.head()
for index, row in df.iterrows():

    file_path = path/row['name']

    if not file_path.is_file():

        print("Can't find file... dropping " + str(row['name']))

        df.drop(index, axis=0, inplace=True)
random_seed(123,True)

data2 = ImageDataBunch.from_df(df=df, path=path, valid_pct=0.2, size=256)
learn = cnn_learner(data2, models.resnet50, metrics=error_rate)

learn.fit_one_cycle(4)
learn.save('clean-stage-1')
lrf = learn.lr_find()

learn.recorder.plot()
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize=(15,11), heatmap=False)
interp.plot_top_losses(9, figsize=(15,11), heatmap=True)
import fastai

defaults.device = torch.device('cpu')
bus_img = open_image(busdest/'00000399.jpg')

bus_img.show()
classes = ['bus', 'sedan', 'truck']
datafinal = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=256).normalize(imagenet_stats)

learn = cnn_learner(datafinal, models.resnet50).load('stage-1')
pred_class,pred_idx,outputs = learn.predict(bus_img)

pred_class
learn.export('vehicle_classifier_resnet50_256.pkl')
import shutil

shutil.rmtree(busdest)

shutil.rmtree(truckdest)

shutil.rmtree(sedandest)