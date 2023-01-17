import os, gc

from fastai.text import *

import pandas as pd

from fastai import *

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
plt.hist(train.question_text.apply(lambda x: len(x)), density = False, bins = 40)

#Length of questions asked
np.random.seed(42)

train_small = train.iloc[train.sample(frac=0.99).index]

train = train_small  #Comment out this cell later, taking smaller dataset in order to run through process

plt.bar(["False",'True'], train.groupby('target').count().qid)
train.shape
sample_size = 0.2

train_df =train.sample(frac=(1-sample_size))

valid_df = train[~train.index.isin(train_df)]
%%time

data_lm = TextLMDataBunch.from_df(path = '.',

                            train_df = train_df,

                            valid_df = valid_df,

                            test_df = test,

                            text_cols = 'question_text',

                            label_cols = 'target',

                            max_vocab = 20000)

print(len(data_lm.vocab.itos))

data_lm.save()
data_lm.show_batch()
data_lm.vocab.itos[100:105]
%%time

data_class = TextClasDataBunch.from_df(path = '.',

                                       train_df = train_df,

                                       valid_df = valid_df,

                                       test_df = test,

                                       text_cols = 'question_text',

                                       label_cols = 'target',

                                       max_vocab = 20000,

                                       vocab=data_lm.vocab)
data_class.show_batch()
path = Path("../")

model_path = path/'models'

model_path.mkdir(exist_ok=True)

url = 'http://files.fast.ai/models/wt103_v1/'

download_url(f'{url}lstm_wt103.pth', model_path/'lstm_wt103.pth')

download_url(f'{url}itos_wt103.pkl', model_path/'itos_wt103.pkl')
learn = language_model_learner(data_lm, pretrained_fnames=['lstm_wt103', 'itos_wt103'], drop_mult=0.3, arch = AWD_LSTM, model_dir=model_path)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 1e-1)
learn.unfreeze()

learn.fit_one_cycle(1, 1e-1)
learn.save_encoder('ft_enc')
data_lm.vocab.itos[10:20]
learn.predict('Why in the earth', 10)

#So this looks great, ha
def f1_score(y_pred, targets):

    epsilon = 1e-07

    

    y_pred = y_pred.argmax(dim = -1)

    #targets = targets.argmax(dim=-1)



    tp = (y_pred*targets).float().sum(dim=0)

    tn = ((1-targets)*(1-y_pred)).float().sum(dim=0)

    fp = ((1-targets)*y_pred).float().sum(dim=0)

    fn = (targets*(1-y_pred)).sum(dim=0)



    p = tp / (tp + fp + epsilon)

    r = tp / (tp + fn + epsilon)



    f1 = 2*p*r / (p+r+epsilon)

    f1 = torch.where(f1!=f1, torch.zeros_like(f1), f1)

    return f1.mean()
%%time

learn_class = text_classifier_learner(data_class, drop_mult = 0.5, 

                                      arch = AWD_LSTM, model_dir=model_path, 

                                     metrics = [accuracy, f1_score])



learn_class.load_encoder('ft_enc')
learn = None

gc.collect()
learn_class.lr_find()

learn_class.recorder.plot_lr()
learn_class.fit_one_cycle(3, 1e-3)
learn_class.freeze_to(-2)

learn_class.fit_one_cycle(4, slice(1e-3, 1e-1))
from sklearn.metrics import roc_curve, precision_recall_curve

def threshold_search(y_true, y_proba, plot=False):

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    thresholds = np.append(thresholds, 1.001) 

    F = 2 / (1/precision + 1/recall)

    best_score = np.max(F)

    best_th = thresholds[np.argmax(F)]

    if plot:

        plt.plot(thresholds, F, '-b')

        plt.plot([best_th], [best_score], '*r')

        plt.show()

    search_result = {'threshold': best_th , 'f1': best_score}

    return search_result 
gc.collect()
preds = learn_class.get_preds(DatasetType.Valid)

proba = to_np(preds[0][:,1])

ytrue = to_np(preds[1])
thr = threshold_search(ytrue, proba, plot=True); thr
probs, _ = learn_class.get_preds(DatasetType.Test)

preds = np.argmax(probs, axis=1)



submission = pd.DataFrame(test['qid'])

submission['prediction'] = preds 

submission.to_csv('submission.csv',index=False)

submission.head()