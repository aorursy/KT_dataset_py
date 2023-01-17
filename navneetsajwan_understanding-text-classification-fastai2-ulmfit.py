# Fastai 2 works on Pytorch 1.6
# Since kaggle still uses PyTorch 1.5.x we need to manually upgrade to 1.6
!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
import torch
torch.__version__
!pip install fastai==2.0.13
import fastai
torch.__version__, fastai.__version__
import fastai
from fastai.text.all import *
path = Path('../input/nlp-getting-started')
path.ls()
train_df = pd.read_csv(path/'train.csv')
train_df.head(5)
test_df = pd.read_csv(path/'test.csv')
test_df.head(5)
import seaborn as sns
plt.figure(figsize = (14,6))
plt.title("Classes histogram")
sns.distplot(a=train_df['target'], kde=False)
plt.xlabel("Class")
plt.xlabel("Frequency")
#we can use both the train and test data to train lamguage model
op = pd.concat([train_df['text'], test_df['text']])
op = pd.DataFrame(op)
# op.head()
len(op), len(train_df)+len(test_df)
dls_lm = TextDataLoaders.from_df(op, path=path, text_col='text', is_lm=True, valid_col=None)
dls_lm.show_batch(max_n=3)
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], path='/kaggle/working/', wd=0.1).to_fp16()
torch.cuda.is_available()
learn.fit_one_cycle(1, 1e-2)
learn.save('1epoch')
learn = learn.load('1epoch')
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3)
learn.save_encoder('finetuned')
!cp ../input/nlp-getting-started/train.csv /kaggle/working
dls = TextDataLoaders.from_csv(path='/kaggle/working/', csv_fname='train.csv', text_col='text', label_col='target', bs = 8, text_vocab=dls_lm.vocab)
dls.show_batch(max_n=8)
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics = accuracy).to_fp16()
learn = learn.load_encoder('finetuned')
learn.fit_one_cycle(4, 2e-2)
learn.freeze_to(-2)
learn.fit_one_cycle(4, slice(1e-2/(2.6**4),1e-2))
# learn.freeze_to(-3)
# learn.fit_one_cycle(4, slice(5e-3/(2.6**4),5e-3))
# learn.unfreeze()
# learn.fit_one_cycle(4, slice(1e-3/(2.6**4),1e-3))
test_df.head(5)
test_df.iloc[:10,3]# row first, column second
id_ , target_  = [], [] 
for i, text in zip(test_df.iloc[:,0], test_df.iloc[:,3]):
    id_.append(i)
    target_.append(int(learn.predict(text)[0]))
id_, target_
submission = pd.DataFrame(list(zip(id_, target_)), 
               columns =['id', 'target']) 
submission.head()
submission.to_csv('/kaggle/working/submission.csv', index = False)
