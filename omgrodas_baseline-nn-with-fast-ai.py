import pandas as pd

import numpy as np

from datetime import datetime

from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from fastai.tabular import *
path=Path("../input/")

train=pd.read_csv(path/"train.csv").drop("ID_code",axis=1)

test=pd.read_csv(path/"test.csv").drop("ID_code",axis=1)
batch_size = 2048

s={'layer1': 9998,

 'layer2': 683,

 'learning_rate1': 0.020649894930894752,

 'learning_rate2': 0.002500911574914439,

 'learning_rate3': 0.0003185017075228141,

 'nepoch1': 4,

 'nepoch2': 1,

 'nepoch3': 5,

 'ps1': 0.7558348036793733,

 'ps2': 0.9502223751560093,

 'wd1': 0.019226758779243965,

 'wd2': 0.03410326162786878,

 'wd3': 0.2332441129291334}
procs = [Normalize]

result=np.zeros(test.shape[0])

counter=0

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for train_index, valid_index in skf.split(train,train.target):

    data = TabularDataBunch.from_df("", train, "target", valid_idx=valid_index, procs=procs,bs=batch_size,test_df=test)

    learn = tabular_learner(data, layers=[s["layer1"],s["layer2"]],ps=[s["ps1"],s["ps2"]])

    learn.fit_one_cycle(s["nepoch1"], s["learning_rate1"],wd=s["wd1"])

    learn.fit_one_cycle(s["nepoch2"], s["learning_rate2"],wd=s["wd2"])

    learn.fit_one_cycle(s["nepoch3"], s["learning_rate3"]/100,wd=s["wd3"])  

    test_predicts, _ = learn.get_preds(ds_type=DatasetType.Test)

    result+=np.array(test_predicts[:,1])

    counter += 1
submission = pd.read_csv(path/'sample_submission.csv')

submission['target'] = result

filename="{:%Y-%m-%d_%H_%M}_sub.csv".format(datetime.now())

submission.to_csv(filename, index=False)