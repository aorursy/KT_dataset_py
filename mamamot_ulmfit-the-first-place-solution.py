import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
import fastai

fastai.__version__
from fastai.text import *

from fastai.callbacks import *
data_train = pd.read_csv("../input/train.csv")

data_valid = pd.read_csv("../input/valid.csv")

data_test = pd.read_csv("../input/test.csv")
data_train.fillna("xxempty", inplace=True)

data_valid.fillna("xxempty", inplace=True)

data_test.fillna("xxempty", inplace=True)
data_train["full"] = data_train["text"].apply(lambda x: x + " xxtitle ") + data_train["title"]

data_valid["full"] = data_valid["text"].apply(lambda x: x + " xxtitle ") + data_valid["title"]

data_test["full"] = data_test["text"].apply(lambda x: x + " xxtitle ") + data_test["title"]
data_train["is_valid"] = False

data_valid["is_valid"] = True
data_lm = (TextList.from_df(pd.concat([data_train, data_valid]), cols=["full"])

           .split_from_df("is_valid")

           .label_for_lm()

           .databunch())
lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3, pretrained=True)
lm.lr_find()
lm.recorder.plot(suggestion=True)
lm.fit_one_cycle(1, 4e-3, callbacks=[SaveModelCallback(lm, name="best_lm")], moms=(0.8,0.7))
lm.unfreeze()
lm.fit_one_cycle(5, 4e-3, callbacks=[SaveModelCallback(lm, name="best_lm")], moms=(0.8,0.7))
lm.load("best_lm")
lm.save_encoder("enc")
data_clf = (TextList.from_df(pd.concat([data_train, data_valid]), vocab=data_lm.vocab, cols=["full"]).

           split_from_df("is_valid").

           label_from_df("label").

           add_test(data_test["full"]).

           databunch()

          )
clf = text_classifier_learner(data_clf, AWD_LSTM, drop_mult=0.3)
del lm

torch.cuda.empty_cache()
clf.load_encoder("enc")
clf.lr_find()
clf.recorder.plot()
clf.fit(3, 2e-3, callbacks=[SaveModelCallback(clf, name="best_clf")])
clf.load("best_clf")
clf.unfreeze()
clf.fit(1, 3e-4, callbacks=[SaveModelCallback(clf, name="best_clf_ft1")])
clf.fit(1, 3e-4, callbacks=[SaveModelCallback(clf, name="best_clf_ft2")])
pred_val = clf.get_preds(DatasetType.Valid, ordered=True)
pred_val_l = pred_val[0].argmax(1)
from sklearn.metrics import classification_report
print(classification_report(pred_val[1], pred_val_l))
pred_test, label_test = clf.get_preds(DatasetType.Test, ordered=True)
pred_test_ = pred_test.argmax(1)

pred_test_l = [data_clf.train_ds.y.classes[n] for n in pred_test_]
res = pd.Series(pred_test_l, index=data_test.index, name="label")
res.index.name = "id"
pd.DataFrame(res).to_csv("submission.csv")