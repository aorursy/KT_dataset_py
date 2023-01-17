# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from fastai.vision import *



#data=ImageDataBunch.from_folder(path='/kaggle/input/turkish-lira-banknote-dataset/',size=256)

data = (ImageList.from_folder('/kaggle/input/turkish-lira-banknote-dataset/').

        random_split_by_pct().label_from_folder().transform(get_transforms(), size=122).databunch())
data.show_batch()
learner=cnn_learner(data,models.resnet18,metrics=accuracy)

learner.fit(5)
interp = ClassificationInterpretation.from_learner(learner)

interp.plot_top_losses(9, figsize=(10,10))
interp.plot_confusion_matrix()
interp.most_confused()
interp.plot_top_losses(k=8)
learner2=cnn_learner(data,models.resnet152,metrics=accuracy)

learner2.fit(3)
interp2 = ClassificationInterpretation.from_learner(learner2)

interp2.plot_top_losses(9, figsize=(10,10))
interp2.plot_confusion_matrix()
interp2.most_confused()
interp2.plot_top_losses(k=8)
learner3=cnn_learner(data,models.densenet201,metrics=accuracy)

learner3.fit(3)
interp3 = ClassificationInterpretation.from_learner(learner3)

interp3.plot_top_losses(9, figsize=(10,10))
interp3.plot_confusion_matrix()
interp3.most_confused()
interp3.plot_top_losses(k=8)
learner4=cnn_learner(data,models.alexnet,metrics=accuracy)

learner4.fit(7)
interp4 = ClassificationInterpretation.from_learner(learner4)

interp4.plot_top_losses(9, figsize=(10,10))
interp4.plot_confusion_matrix()
interp4.most_confused()