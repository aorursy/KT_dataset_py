%reload_ext autoreload

%autoreload 2

%matplotlib inline



import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')



from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.metrics import cohen_kappa_score
# Función para calcular el coeficiene utilizado de comparación

def quadratic_kappa(y_hat, y):

    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')
import os

os.listdir('../input')

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 999

seed_everything(SEED)
base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')

train_dir = os.path.join(base_image_dir,'train_images/')

df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))

df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))

df = df.drop(columns=['id_code'])

df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe

df.head
len_df = len(df)

print(len_df)
f, ax = plt.subplots(figsize=(10, 6))

sns.countplot(df['diagnosis'])

plt.show()
from PIL import Image

im = Image.open(df['path'][1])

plt.imshow(np.asarray(im))
width, height = im.size

print("Dimensiones %s, %s" % (width,height)) 
batch = 64

dimension = 224

trans = get_transforms(do_flip=True, flip_vert=True, max_rotate=360, max_warp=0, max_zoom=1.1, max_lighting=0.1, p_lighting=0.5)

src = (ImageList.from_df(df=df,path='./',cols='path').split_by_rand_pct(0.2).label_from_df(cols='diagnosis', label_cls=FloatList))

data = (src.transform(trans, size=dimension, resize_method=ResizeMethod.SQUISH, padding_mode='zeros').databunch(bs=batch, num_workers=4).normalize(imagenet_stats))
simple_model = cnn_learner(data, base_arch=models.resnet50, metrics = [quadratic_kappa])
simple_model.lr_find()

simple_model.recorder.plot(suggestion=True)
simple_model.fit_one_cycle(4,max_lr = 1e-2)