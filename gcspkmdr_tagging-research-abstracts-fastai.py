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

from sklearn.metrics import f1_score,accuracy_score

import seaborn as sns

sns.set_style('darkgrid')

%matplotlib inline

mpl.rcParams['figure.figsize'] = (12, 12)

mpl.rcParams['axes.grid'] = True
fnames=['/kaggle/input/pretrained-models/lstm_fwd','/kaggle/input/pretrained-models/itos_wt103']
train_data = pd.read_csv('/kaggle/input/researchtopictags/train.csv')

print(train_data.shape)

train_data.head()
test_data = pd.read_csv('/kaggle/input/researchtopictags/test.csv')

print(test_data.shape)

test_data.head()
com_sc = train_data['Computer Science'].value_counts()[1]

phy = train_data['Physics'].value_counts()[1]

mat = train_data['Mathematics'].value_counts()[1]

stats = train_data['Statistics'].value_counts()[1]

bio = train_data['Quantitative Biology'].value_counts()[1]

fin = train_data['Quantitative Finance'].value_counts()[1]



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

topics = ['Computer Science','Physics','Mathematics', 'Statistics','Quantitative Biology','Quantitative Finance']

counts = [com_sc,phy,mat,stats,bio,fin]

ax.bar(topics,counts)

plt.show()



train_data['combined_text'] = train_data['TITLE'] + "<join>" + train_data['ABSTRACT']

test_data['combined_text'] = test_data['TITLE'] + "<join>" + test_data['ABSTRACT']
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





train_data['combined_text'] = train_data['combined_text'].apply(lambda x : clean_text(x))

test_data['combined_text'] = test_data['combined_text'].apply(lambda x : clean_text(x))
train_data_lm = (TextList.from_df(df=train_data,cols='combined_text').split_by_rand_pct(0.3).label_for_lm().databunch(bs=48))
train_data_lm.save('train_data_lm.pkl')
train_data_lm.vocab.itos[:10]
train_data_lm.train_ds[0][0]
train_data_lm = load_data('', 'train_data_lm.pkl', bs=48)
train_data_lm.show_batch()
languageModel = language_model_learner(train_data_lm, arch=AWD_LSTM, pretrained_fnames=fnames, drop_mult=0.3)
languageModel.lr_find()
languageModel.recorder.plot(suggestion = True)
min_grad_lr = languageModel.recorder.min_grad_lr

print(min_grad_lr)
languageModel.fit_one_cycle(5, 0.045)
languageModel.save_encoder('fine_tuned_enc1')
languageModel.lr_find()
languageModel.recorder.plot(suggestion = True)
min_grad_lr = languageModel.recorder.min_grad_lr

print(min_grad_lr)
languageModel.fit_one_cycle(1, 1e-2)
languageModel.save_encoder('fine_tuned_enc2')
languageModel.lr_find()
languageModel.recorder.plot(suggestion = True)
min_grad_lr = languageModel.recorder.min_grad_lr

print(min_grad_lr)
languageModel.fit_one_cycle(1, 1e-2)
languageModel.save_encoder('fine_tuned_enc3')
label_cols = topics
data_classifier = (TextList.from_df(df=train_data,cols='combined_text', vocab=train_data_lm.vocab)

                     .split_by_rand_pct(0.3)

                     .label_from_df(label_cols)

                     .add_test(test_data)

                     .databunch(bs=48))
data_classifier.save('data_classifier.pkl')
data_classifier = load_data('','data_classifier.pkl',bs=48)
data_classifier.show_batch()
threshold = 0.2
class MicroF1(Callback):



    _order = -20 #is crucial - without it the custom columns will not be added - it tells the callback system to run this callback before the recorder system.



    def __init__(self,learn,thresh,eps = 1e-15, sigmoid = True,**kwargs):

        self.learn = learn

        self.thresh = thresh

        self.eps = eps

        self.sigmoid = sigmoid



    def on_train_begin(self, **kwargs): 

        self.learn.recorder.add_metric_names(['MicroF1'])

    

    def on_epoch_begin(self, **kwargs):

        self.tp = 0

        self.total_pred = 0

        self.total_targ = 0

    

    def on_batch_end(self, last_output, last_target, **kwargs):

        pred, targ = ((last_output.sigmoid() if self.sigmoid else last_output) > self.thresh).byte(), last_target.byte()

        if torch.equal(torch.tensor(pred.shape),torch.tensor(targ.shape)):

            

            m = pred*targ

            self.tp += m.sum(0).float()

            self.total_pred += pred.sum(0).float()

            self.total_targ += targ.sum(0).float()

    

    def fbeta_score(self, precision, recall):

        return 2*(precision*recall)/((precision + recall) + self.eps)



    def on_epoch_end(self, last_metrics, **kwargs):

        self.total_pred += self.eps

        self.total_targ += self.eps

        precision, recall = self.tp.sum() / self.total_pred.sum(), self.tp.sum() / self.total_targ.sum()

        res = self.fbeta_score(precision, recall)        

        return add_metrics(last_metrics, res)
class AccPerClass(Callback):

    _order = -20 



    def __init__(self, learn, **kwargs): 

        self.learn = learn

        self.output, self.target = [], []

        

    def on_train_begin(self, **kwargs): 

        self.learn.recorder.add_metric_names(topics)

        

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
microF1 = partial(MicroF1,thresh = threshold)
classifierModel = text_classifier_learner(data_classifier , arch=AWD_LSTM,drop_mult=0.4, callback_fns = [microF1,AccPerClass] )

classifierModel.load_encoder('fine_tuned_enc1')

classifierModel.freeze()
classifierModel.summary()
classifierModel.lr_find()
classifierModel.recorder.plot(suggestion = True)
min_grad_lr = classifierModel.recorder.min_grad_lr

print(min_grad_lr)
classifierModel.fit_one_cycle(20, 1e-1)
classifierModel.save('classifierModel1')
classifierModel.show_results()
classifierModel.load('classifierModel1')
classifierModel.lr_find()
classifierModel.recorder.plot(suggestion = True)
min_grad_lr = classifierModel.recorder.min_grad_lr

print(min_grad_lr)
for i in range(2,7):

    classifierModel.freeze_to(-i)

    classifierModel.fit_one_cycle(1,slice((1*10**-(i+1))/(2.6**4),1*10**-(i+1)))

    print ('')
classifierModel.save('classifierModel2')
preds = classifierModel.get_preds(DatasetType.Test)
submission = pd.read_csv('/kaggle/input/researchtopictags/sample.csv')

submission.iloc[:,1:] =  (preds[0]>0.3).byte().numpy()

submission.to_csv('submission.csv', index=False)

submission.head()