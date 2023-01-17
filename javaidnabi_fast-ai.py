from fastai.text import *

import html

import pandas as pd
df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv',delimiter='\t',encoding='utf-8')
df.head()
df_test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv',delimiter='\t',encoding='utf-8')
df_test.head()

df_test2 = df_test.drop(["PhraseId", "SentenceId"], axis=1)
df_test2.head(5)
trn_texts = df['Phrase'].values

trn_labels = df['Sentiment'].values
np.random.seed(42)

trn_idx = np.random.permutation(len(trn_texts))

trn_texts = trn_texts[trn_idx]

trn_labels = trn_labels[trn_idx]
from sklearn.model_selection import train_test_split

# create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(trn_texts, trn_labels, test_size=0.1)

print (X_train.shape)

print(y_train.shape)

print (X_test.shape)

print(y_test.shape)
col_names = ['labels','text']

df_trn = pd.DataFrame({'text':X_train, 'labels':y_train}, columns=col_names)

df_val = pd.DataFrame({'text':X_test, 'labels':y_test}, columns=col_names)
df_trn['labels'].value_counts()
df_val['labels'].value_counts()
df_trn.shape
# Language model data

data_lm = TextLMDataBunch.from_df('./', train_df=df_trn, valid_df=df_val)
em_sz,nl = 400,3
learn = language_model_learner(data_lm, emb_sz=em_sz, nl=nl, drop_mult=0.1)

learn = LanguageLearner(data_lm, learn.model, bptt=70)

learn.load_pretrained('../input/wiki103/lstm_wt103.pth', '../input/wiki103/itos_wt103.pkl')
learn.metrics = [accuracy]

learn.freeze_to(-1)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, 1e-2, moms=(.8,.7))
learn.unfreeze()

learn.fit_one_cycle(1, 1e-3)
learn.predict("this is a review about", n_words =10)
# Classifier model data

data_clas = TextClasDataBunch.from_df('./', train_df=df_trn, valid_df=df_val, vocab=data_lm.train_ds.vocab, bs=32)
learn.save_encoder('fine_enc')
# Classifier

classifier = text_classifier_learner(data_clas, drop_mult=0.5)

classifier.load_encoder('fine_enc')

classifier.crit = F.cross_entropy
classifier.lr_find()
classifier.recorder.plot()
classifier.fit_one_cycle(1, 1e-2, moms=(.8,.7))
classifier.freeze_to(-2)

classifier.fit_one_cycle(1, 1e-3, moms=(.8,.7))
classifier.freeze_to(-3)

classifier.fit_one_cycle(1, 1e-4, moms=(.8,.7))
classifier.unfreeze()

classifier.fit_one_cycle(5, 50e-5, moms=(.8,.7))
classifier.predict("This is not a  good movie")
preds = classifier.get_preds()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
if torch.cuda.is_available():

    test_txt_list = (TextList.from_df(df_test2, './',  processor=[TokenizeProcessor(), NumericalizeProcessor(vocab=data_lm.vocab)])).process()

    classifier.model = classifier.model.to(device)

    classifier.model = classifier.model.eval()



predicted = []

with torch.no_grad():

    for i, doc in enumerate(test_txt_list.items):

        if i % 10000 == 0: print("Evaluating...",i) 



        doc = torch.LongTensor(doc).to(device)

        pred, _, _ = classifier.model(doc.unsqueeze(0))

        pred = pred.detach().cpu().numpy()

        predicted_labels = np.argmax(pred.squeeze())

        predicted.append(predicted_labels.item())

       
df_test['Predicted'] = predicted
df_test.head()
my_submission = pd.DataFrame({'PhraseId': df_test.PhraseId, 'Sentiment': df_test.Predicted})

my_submission.to_csv('submission.csv', index= False)
my_submission.head()