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
%reload_ext autoreload

%autoreload 2

%matplotlib inline
!cp -r /kaggle/input/aerial-cactus-identification /kaggle/working/
!unzip /kaggle/working/aerial-cactus-identification/train.zip -d /kaggle/working/aerial-cactus-identification/train
from fastai.vision import *

from fastai.metrics import error_rate
bs = 64
data = ImageList.from_csv("/kaggle/working/aerial-cactus-identification","train.csv",folder="train/train").split_by_rand_pct().label_from_df().databunch()
data.show_batch(rows=3, figsize=(7,6))
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.save('stage-0')
learn.fit_one_cycle(4)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.save('stage-2')
learn.load('stage-1');
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
learn.save("stage-3")
learn.load("stage-2")
!unzip /kaggle/working/aerial-cactus-identification/test.zip -d /kaggle/working/aerial-cactus-identification/test
import tqdm
predictions = []
for filename in tqdm.tqdm(os.listdir("/kaggle/working/aerial-cactus-identification/test/test")):

    img = open_image(f"/kaggle/working/aerial-cactus-identification/test/test/{filename}")

    pred = int(learn.predict(img)[1].item())

    predictions.append({"id":filename, "has_cactus":pred})

    
predictions
import csv



csv_columns = ['id','has_cactus']



csv_file = "submission.csv"



with open(csv_file, 'w') as csvfile:

    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)

    writer.writeheader()

    for data in predictions:

        writer.writerow(data)
!rm -r /kaggle/working/aerial-cactus-identification/