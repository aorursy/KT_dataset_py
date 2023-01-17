# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai.tabular import *



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

path = Path('../input/')

df = pd.read_csv(path/'train.csv')

df = df.assign(SurvivedBool = lambda x: x['Survived'] > 0.5)
dep_var = 'SurvivedBool'

# PClass: Ticket Class

# SibSp: Number of Siblings / Spouses aboard

# Parch: Number of parents / children aboard

cat_names = ['Embarked','Sex']

cont_names = ['Pclass','SibSp','Parch','Age','Fare']

procs = [FillMissing, Categorify, Normalize]
data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_rand_pct(0.2)

                           .label_from_df(cols=dep_var)

                           .databunch())
data.show_batch(rows=10)
learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
random_seed(111,True)

learn.fit(10, 1e-2)
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn,preds,y,losses)

interp.plot_confusion_matrix()
losses,idxs = interp.top_losses()

idxs[:10]
df.iloc[14]
learn.predict(df.iloc[14])