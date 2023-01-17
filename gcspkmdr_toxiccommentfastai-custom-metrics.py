import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")



from fastai.text import *

from fastai import *

from sklearn.metrics import roc_auc_score,accuracy_score

import seaborn as sns

sns.set_style('darkgrid')

%matplotlib inline

mpl.rcParams['figure.figsize'] = (15, 15)

mpl.rcParams['axes.grid'] = True
fnames=['/kaggle/input/pretrained-models/lstm_fwd','/kaggle/input/pretrained-models/itos_wt103']
train_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

test_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

train_data.shape
train_data.head()
test_data.head()
test_data.shape
train_data['neutral'] =[0] * train_data.shape[0]

for idx, row in train_data.iterrows():

    label = []

    if row['toxic'] == 0 and row['severe_toxic'] == 0 and row['obscene'] == 0 and row['insult'] == 0 and row['identity_hate'] == 0:

      train_data['neutral'][idx] = 1
train_data.head()
toxic = train_data.toxic.value_counts()[1]

severe_toxic = train_data.severe_toxic.value_counts()[1]

obscene = train_data.obscene.value_counts()[1]

threat = train_data.threat.value_counts()[1]

insult = train_data.insult.value_counts()[1]

identity_hate = train_data.identity_hate.value_counts()[1]

neutral = train_data.neutral.value_counts()[1]
class_rep = pd.DataFrame(columns=['count'],index = ['toxic','severe_toxic','obscene','threat','insult','identity_hate','neutral'])

class_rep['count']['toxic'] =toxic

class_rep['count']['severe_toxic'] =severe_toxic

class_rep['count']['obscene'] =obscene

class_rep['count']['threat'] =threat

class_rep['count']['insult'] =insult

class_rep['count']['identity_hate'] =identity_hate

class_rep['count']['neutral'] =neutral

plt.title("Class Representation")

sns.barplot(y = class_rep['count'] ,x = class_rep.index)
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']



def clean_text(x):

    x = str(x)

    for punct in puncts:

        if punct in x:

            x = x.replace(punct, ' ')

    return x





train_data['comment_text'] = train_data['comment_text'].apply(lambda x : clean_text(x))
train_data_lm = (TextList.from_df(df=train_data,cols='comment_text').split_by_rand_pct(0.1).label_for_lm().databunch(bs=48))
train_data_lm.save('train_data_lm.pkl')
train_data_lm.vocab.itos[:10]
train_data_lm.train_ds[0][0]
train_data_lm = load_data('', 'train_data_lm.pkl', bs=48)
train_data_lm.show_batch()
languageModel = language_model_learner(train_data_lm, arch=AWD_LSTM, pretrained_fnames=fnames, drop_mult=0.3)
languageModel.lr_find()
languageModel.recorder.plot(suggestion = True)
min_grad_lr = languageModel.recorder.min_grad_lr

min_grad_lr
languageModel.fit_one_cycle(1, min_grad_lr,moms=(0.8,0.7))
languageModel.save_encoder('fine_tuned_enc1')
#languageModel.lr_find()
#languageModel.recorder.plot(suggestion = True)
#min_grad_lr = languageModel.recorder.min_grad_lr

#min_grad_lr
#languageModel.fit_one_cycle(1, min_grad_lr)
#languageModel.save_encoder('fine_tuned_enc2')
#languageModel.lr_find()
#languageModel.recorder.plot(suggestion = True)
#min_grad_lr = languageModel.recorder.min_grad_lr

#min_grad_lr
#languageModel.fit_one_cycle(1, min_grad_lr)
#languageModel.save_encoder('fine_tuned_enc3')
label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate','neutral']
data_classifier = (TextList.from_df(df=train_data,cols='comment_text', vocab=train_data_lm.vocab)

                     .split_by_rand_pct(0.1)

                     .label_from_df(label_cols)

                     .add_test(test_data)

                     .databunch(bs=48))
data_classifier.save('data_classifier.pkl')
data_classifier = load_data('','data_classifier.pkl',bs=48)
data_classifier.show_batch()
class AUCROC(Callback):

    _order = -20 #is crucial - without it the custom columns will not be added - it tells the callback system to run this callback before the recorder system.



    def __init__(self, learn, **kwargs): 

      self.learn = learn

      self.output, self.target = [], []

        

    def on_train_begin(self, **kwargs): 

      self.learn.recorder.add_metric_names(['AUROC'])

        

    def on_epoch_begin(self, **kwargs): 

      self.output, self.target = [], []

    

    def on_batch_end(self, last_target, last_output, train, **kwargs):

        if not train:

          self.output.append(last_output)

          self.target.append(last_target)

                

    def on_epoch_end(self, last_metrics, **kwargs):

      if len(self.output) > 0:

            output = torch.cat(self.output)

            target = torch.cat(self.target)

            preds = F.softmax(output, dim=1)

            metric = roc_auc_score(target.cpu().numpy(), preds.cpu().numpy(),average='macro')

            return add_metrics(last_metrics, [metric])

      else:

            return
class AccPerClass(Callback):

    _order = -20 



    def __init__(self, learn, **kwargs): 

      self.learn = learn

      self.output, self.target = [], []

        

    def on_train_begin(self, **kwargs): 

      self.learn.recorder.add_metric_names(['toxic','severe_toxic','obscene','threat','insult','identity_hate','neutral'])

        

    def on_epoch_begin(self, **kwargs): 

      self.output, self.target = [], []

    

    def on_batch_end(self, last_target, last_output, train, **kwargs):

        if not train:

          self.output.append(last_output)

          self.target.append(last_target)

                

    def on_epoch_end(self, last_metrics, **kwargs):

      if len(self.output) > 0:

            output = torch.cat(self.output)

            target = torch.cat(self.target)

            preds = F.softmax(output, dim=1)

            metric = []

            for i in range(0,target.shape[1]):

              metric.append(accuracy_score(target.cpu().numpy()[...,i].flatten(), (preds[...,i] >0.2).byte().cpu().numpy().flatten()))

            return add_metrics(last_metrics, metric)

      else:

            return
acc = partial(accuracy_thresh, thresh=0.2) 

fbetaScore = partial(fbeta, thresh=0.2)
classifierModel = text_classifier_learner(data_classifier , arch=AWD_LSTM,

                                          drop_mult=0.5,metrics = [acc], callback_fns = [AUCROC,AccPerClass] )

classifierModel.load_encoder('fine_tuned_enc1')

classifierModel.freeze()
classifierModel.summary()
classifierModel.lr_find()
classifierModel.recorder.plot(suggestion = True)
min_grad_lr = classifierModel.recorder.min_grad_lr

min_grad_lr
classifierModel.fit_one_cycle(1, 2e-2,moms=(0.8,0.7))
classifierModel.save('classifierModel1')
classifierModel.show_results()
classifierModel.load('classifierModel1')
#classifierModel.unfreeze()

#classifierModel.freeze_to(-2)

#classifierModel.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2),moms=(0.8,0.7))
#classifierModel.save('classifierModel2')

#classifierModel.load('classifierModel2')
#classifierModel.unfreeze()

#classifierModel.freeze_to(-3)

#classifierModel.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3),moms=(0.8,0.7))
preds = classifierModel.get_preds(DatasetType.Test)
submission_final = pd.DataFrame({'id': test_data['id']})

submission_final = pd.concat([submission_final, pd.DataFrame(preds[0].numpy()[...,:-1], columns = label_cols[:-1])], axis=1)



submission_final.to_csv('submission.csv', index=False)

submission_final.head()