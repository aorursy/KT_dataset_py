# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fastai.vision import *

from tqdm import tqdm_notebook as tqdm

import gc
# original size of images

HEIGHT = 137

WIDTH = 236
class ComponentMetric(Callback):

    def __init__(self, start, end):

        super().__init__()

        self.start = start

        self.end = end

        self.tp = 0

        self.total = 0

        self.batch_num = 0

    def on_epoch_begin(self, **kwargs):

        self.tp = 0

        self.total = 0



    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):

        assert last_output.shape == last_target.shape # sanity check

        last_output = last_output[:, self.start: self.end]

        last_target = last_target[:, self.start: self.end]

        preds = last_output.argmax(1).cpu()

        targs = last_target.argmax(1).cpu()

        self.tp += (preds == targs).sum()

#         print('batch num: {}, sum: {}, tp: {}, total: {}'.format(self.batch_num, (preds == targs).sum(), self.tp, self.total))

        self.total += last_output.shape[0]

        self.batch_num += 1

#         return (preds == targs).sum().item() # return value for testing only

    

    def on_epoch_end(self, last_metrics, **kwargs):     

        print('{}, tp: {}, total: {}'.format(last_metrics, self.tp, self.total))

        return add_metrics(last_metrics, float(self.tp)/self.total) # integer divide yields zero

    

    

class TotalMetric(Callback):

    def __init__(self):

        super().__init__()

        self.grapheme = ComponentMetric(0, 168)

        self.vowel = ComponentMetric(168, 179)

        self.consonant = ComponentMetric(179, 187)

        

    def on_epoch_begin(self, **kwargs):

        self.grapheme.on_epoch_begin(**kwargs)

        self.vowel.on_epoch_begin(**kwargs)

        self.consonant.on_epoch_begin(**kwargs)

    

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):

        self.grapheme.on_batch_end(last_output, last_target, **kwargs)

        self.vowel.on_batch_end(last_output, last_target, **kwargs)

        self.consonant.on_batch_end(last_output, last_target, **kwargs)

        

    def on_epoch_end(self, last_metrics, **kwargs): 

        return add_metrics(last_metrics, 0.5*self.grapheme._recall() +

                0.25*self.vowel._recall() + 0.25*self.consonant._recall())
learn = load_learner('/kaggle/input/bengali-fast-ai-training/')
def parse_prediction(pred_string):

    pred_string = pred_string.split(';')

    pred_dict = {'g': None, 'v': None, 'c': None} # ensure that we only get at most one value per field

    for p in pred_string:

        p = p.split('_')

        if pred_dict[p[0]] is None:

            pred_dict[p[0]] = p[1]

    return [int(x) if x is not None else 0 for x in pred_dict.values()]
# unit test

parse_prediction('g_15;v_0;c_3')
count = 0

row_id = []

target= []



for i in range(4):

    test = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i))

    data = test.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

    # hacky way of converting a numpy array img into a fastai Image object.

    for img in data:

        img = np.stack((img,)*3, axis=-1)

        img = pil2tensor(img,np.float32).div_(255)

        # gets the Multicategory portion of the prediction and converts to string

        # format looks something like: g_3;v_0;c_0

        pred_string = str(learn.predict(Image(img))[0])

        prediction = parse_prediction(pred_string)

        row_id += ['Test_{}_grapheme_root'.format(count), 

                   'Test_{}_vowel_diacritic'.format(count), 

                   'Test_{}_consonant_diacritic'.format(count)]

        target += prediction

        count += 1
sub_df = pd.DataFrame({'row_id': row_id, 'target': target})

sub_df.to_csv('submission.csv', index=False)

sub_df