# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from fastai import *

from fastai.vision import *



import torchvision



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from time import time

import os

print(os.listdir("../input"))



from IPython.display import HTML

import base64



import PIL



# Any results you write to the current directory are saved as output.
# Put these at the top of every notebook, to get automatic reloading and inline plotting

%reload_ext autoreload

%autoreload 2

%matplotlib inline
def imagify(tensor):

    reshaped = tensor.reshape(-1, 28, 28)

    print(reshaped.shape)

    reshaped = np.stack((reshaped,) *3, axis = 1)

    print(reshaped.shape)

    image_arr = []



    for idx, current_image in enumerate(reshaped):

        img = torch.tensor(current_image, dtype=torch.float) / 255.

        img = vision.image.Image(img)

        image_arr.append(img)

    return image_arr



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# Unused. Old implementation

def save_images(pathBase, images, labels=None, is_test=False):

    for idx in range(len(images)):

        fPathBase = f'/tmp/{pathBase}'

        label = None

        base = None

        if not is_test:

            label = labels[idx]

            fPathBase = f'{fPathBase}/{label}'

            base = time() * 1000

        else:

            base = '{0:05d}'.format(idx + 1)

        image = images[idx]

        image = torch.tensor(image)

        image = vision.image.Image(image)

        #image.show()

        if not os.path.exists(fPathBase):

            os.makedirs(fPathBase)

        image.save(f'{fPathBase}/{base}.png')



    Path(fPathBase).ls()



def split_data(data, labels, pct=0.8):

    train_xl = []

    train_yl = []

    valid_xl = []

    valid_yl = []



    for img, label in zip(data, labels):

        if random.random() >= pct:

            valid_xl.append(img)

            valid_yl.append(label)

        else:

            train_xl.append(img)

            train_yl.append(label)

    

    return train_xl, train_yl, valid_xl, valid_yl



def create_label_lists(train_xl, train_yl, valid_xl, valid_yl):

    train_xl = TensorImageList(train_xl)

    train_yl = CategoryList(train_yl, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    valid_xl = TensorImageList(valid_xl)

    valid_yl = CategoryList(valid_yl, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    

    train_ll = LabelList(train_xl, train_yl)

    valid_ll = LabelList(valid_xl, valid_yl)

    

    return LabelLists(Path('.'), train_ll, valid_ll)



class TensorImageList(ImageList):

    def get(self, i):

        img = self.items[i]

        self.sizes[i] = img.size

        return img
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
labels = df_train.iloc[:,0].values.flatten()

images = imagify(df_train.iloc[:,1:].values)
train_xl, train_yl, valid_xl, valid_yl = split_data(images, labels, 0.9)
test_xl = imagify(df_test.values)

test_xl = TensorImageList(test_xl)
lls = create_label_lists(train_xl, train_yl, valid_xl, valid_yl)
#tfms = get_transforms(do_flip=False, max_rotate=20, xtra_tfms=rand_pad(2, 28))
#tfms = (tfms[0], [])

tfms = (rand_pad(2, 28), [])
mnist_data = ImageDataBunch.create_from_ll(lls, ds_tfms=tfms)
mnist_data.add_test(test_xl)
mnist_data.show_batch()
arch = models.resnet50 # because why not?

learner = cnn_learner(mnist_data, arch, metrics=[accuracy])
learner.lr_find()

learner.recorder.plot()
lr = 1e-3
learner.fit_one_cycle(10, lr)
# Accuracy Plot:

learner.recorder.plot_metrics()
# Losses Plot

learner.recorder.plot_losses()
learner.save('mnist-1') #0.994268
learner.lr_find()

learner.recorder.plot()
learner.unfreeze()
learner.fit_one_cycle(10, slice(1e-6, lr/5))
learner.save('mnist-2') #0.995119
c_interpret = ClassificationInterpretation.from_learner(learner)

c_interpret.plot_top_losses(12)
preds = learner.get_preds(ds_type=DatasetType.Test)
pred_values = preds[0].argmax(1).numpy()
submission = DataFrame({'ImageId': list(range(1, len(pred_values) + 1)), 'Label': pred_values})

submission.head()
#create_download_link(submission)