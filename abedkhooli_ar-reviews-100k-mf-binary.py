# Arabic Reviews 100k - binary classification using MULTIFiT
import time 

print(f'Started: {time.ctime()}')
# copy files 

#     spm.vocab

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16sj4YC-TQJs7qUPL9SM31scUylSb82tT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16sj4YC-TQJs7qUPL9SM31scUylSb82tT" -O spm.vocab && rm -rf /tmp/cookies.txt 

#    spm.model 

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1EYgmceJU0d5VQL-UKD974tzE-7mTI6sQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EYgmceJU0d5VQL-UKD974tzE-7mTI6sQ" -O spm.model && rm -rf /tmp/cookies.txt 

#  ar_wt_sp15_multifit.pth 

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uK560hj5qtdB9ImoePdmvIWrWqeaDXtj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uK560hj5qtdB9ImoePdmvIWrWqeaDXtj" -O ar_wt_sp15_multifit.pth && rm -rf /tmp/cookies.txt 

#  ar_wt_vocab_sp15_multifit.pkl

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rQJDjYjivyKLSZMw8n7Pg07q2PQCRQHS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rQJDjYjivyKLSZMw8n7Pg07q2PQCRQHS" -O ar_wt_vocab_sp15_multifit.pkl && rm -rf /tmp/cookies.txt 

!pip uninstall fastai --y

!pip install ninja

!pip install sentencepiece

!pip install fastai==1.0.57 
# fastai version 

from fastai import *

from fastai.text import *

from fastai.callbacks import *



import matplotlib.pyplot as plt



%reload_ext autoreload

%autoreload 2

%matplotlib inline
df = pd.read_csv('/kaggle/input/arabic-100k-reviews/ar_reviews_100k.tsv',sep='\t')

print(df['label'].value_counts(),'\n')

df = df.sample(frac=1).reset_index(drop=True)

df.head()
dfm = df[df['label']=='Mixed']

df = df.drop(dfm.index)

len(df)
import matplotlib.cm as cm

from sklearn.metrics import f1_score



@np_func

def f1(inp,targ): return f1_score(targ, np.argmax(inp, axis=-1), average='weighted')



import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "on
config = awd_lstm_lm_config.copy()

config['qrnn'] = True

config['n_hid'] = 1550 #default 1152

config['n_layers'] = 4 #default 3
!mkdir -p /kaggle/working/models/ 

# need to move vocab and model here for fastai to work

!mkdir -p /root/.fastai/data/arwiki/corpus2_100/tmp/ 

!cp /kaggle/working/spm.model /root/.fastai/data/arwiki/corpus2_100/tmp/spm.model

!cp /kaggle/working/spm.vocab /root/.fastai/data/arwiki/corpus2_100/tmp/spm.vocab
dest = '/root/.fastai/data/arwiki/corpus2_100/'

path = '/kaggle/working/'

# columns names

reviews = "text"

label = "label"

bs=32 
%%time

# Databunch <================================= use dfa for all (production)

data_lm = (TextList.from_df(df, path, cols=reviews, processor=SPProcessor.load(dest))

    .split_by_rand_pct(0.15, seed=42)

    .label_for_lm()           

    .databunch(bs=bs, num_workers=1))



data_lm.save(f'/kaggle/working/models/ar_databunch_lm_rev_sp15_multifit')
path_ds = '/kaggle/working/models/'

data_lm = load_data(path_ds, 'ar_databunch_lm_rev_sp15_multifit', bs=bs)
config = awd_lstm_lm_config.copy()

config['qrnn'] = True

config['n_hid'] = 1550 #default 1152

config['n_layers'] = 4 #default 3
!cp /kaggle/working/ar_wt_sp15_multifit.pth  /kaggle/working/models/ar_wt_sp15_multifit.pth

!cp /kaggle/working/ar_wt_vocab_sp15_multifit.pkl  /kaggle/working/models/ar_wt_vocab_sp15_multifit.pkl
lm_ar = ['/kaggle/working/models/ar_wt_sp15_multifit', '/kaggle/working/models/ar_wt_vocab_sp15_multifit']
%%time

perplexity = Perplexity()

learn_lm = language_model_learner(data_lm, AWD_LSTM, config=config, 

                                  pretrained_fnames=lm_ar, drop_mult=0.5, # 1., also 0.3 used for FR

                                  metrics=[error_rate, accuracy, perplexity]).to_fp16()
learn_lm.lr_find()

learn_lm.recorder.plot(suggestion=True)
lr = 1e-2

#lr *= bs/48

wd = 0.1 
learn_lm.fit_one_cycle(5, lr*10, wd=wd, moms=(0.8,0.7)) 
# saving optional

learn_lm.save(f'ar_fine_tuned1_reviews_sp15_multifit')

learn_lm.save_encoder(f'ar_fine_tuned1_enc_reviews_sp15_multifit')
learn_lm.unfreeze() # 16 min/cycle (400k)

learn_lm.fit_one_cycle(18, lr/2., wd=wd, moms=(0.8,0.7), callbacks=[ShowGraph(learn_lm)])
# best model and encoder      

learn_lm.save(f'ar_fine_tuned_reviews_sp15_multifit')

learn_lm.save_encoder(f'ar_fine_tuned_enc_reviews_sp15_multifit')
%%time

bs = 16

data_lm = load_data(path_ds, 'ar_databunch_lm_rev_sp15_multifit', bs=bs)
%%time

data_clas = (TextList.from_df(df, path, vocab=data_lm.vocab, cols=reviews, processor=SPProcessor.load(dest))

    .split_by_rand_pct(0.15, seed=42)

    .label_from_df(cols=label)

    .databunch(bs=bs, num_workers=0)) 
%%time

data_clas.save(f'/kaggle/working/models/ar_textlist_class_reviews_sp15_multifit')

data_clas = load_data(path_ds, f'ar_textlist_class_reviews_sp15_multifit', bs=bs, num_workers=1)
num_trn = len(data_clas.train_ds.x)

num_val = len(data_clas.valid_ds.x)

print(num_trn, num_val, num_trn+num_val)

trn_LabelCounts = np.unique(data_clas.train_ds.y.items, return_counts=True)[1]

val_LabelCounts = np.unique(data_clas.valid_ds.y.items, return_counts=True)[1]

print(trn_LabelCounts, val_LabelCounts)

trn_weights = [1 - count/num_trn for count in trn_LabelCounts]

val_weights = [1 - count/num_val for count in val_LabelCounts]

print(trn_weights, val_weights)
%%time

data_clas = load_data(path_ds, f'ar_textlist_class_reviews_sp15_multifit', bs=bs, num_workers=1)

config = awd_lstm_clas_config.copy()

config['qrnn'] = True

config['n_hid'] = 1550 #default 1152

config['n_layers'] = 4 #default 3

learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False, 

                                  drop_mult=0.5,  # 0.3 was 0.5 for FR

                                  metrics=[accuracy,f1]).to_fp16()
learn_c.load_encoder(f'ar_fine_tuned_enc_reviews_sp15_multifit');
# train 

learn_c.freeze()

learn_c.lr_find()

learn_c.recorder.plot()
lr = 1e-3 # was 2e-1 for pt

#lr *= bs/48

wd = 0.1 
learn_c.fit_one_cycle(5, lr, wd=wd, moms=(0.8,0.7)) # 2 min/cycle (66k)

learn_c.save(f'ar_clas_reviews_sp15_multifit')
learn_c.freeze_to(-2)

learn_c.fit_one_cycle(4, slice(lr/(2.6**4),lr), wd=wd, moms=(0.8,0.7))

learn_c.save(f'ar_clas_reviews_sp15_multifit')
learn_c.freeze_to(-3)

learn_c.fit_one_cycle(4, slice(lr/2/(2.6**4),lr/2), wd=wd, moms=(0.8,0.7))
# unfreeze and tune (was 4 cycles) - tried 8 but last 2 were bad so 6 is best

learn_c.unfreeze()

learn_c.fit_one_cycle(4, slice(lr/10/(2.6**4),lr/10), wd=wd, moms=(0.8,0.7)) 
learn_c.save(f'ar_classifier_reviews100_sp15_multifit_nows_save')

# stop fine-tuning here and go to confusion matrix?

learn_c.export(f'ar_classifier_reviews100_sp15_multifit_nows_exp.pkl')

learn_c.to_fp32().export(f'ar_classifier_reviews100_sp15_multifit_nows_2fp_exp.pkl') 
%%time

#========================================================

# confusion matrix





data_clas = load_data(path_ds, f'ar_textlist_class_reviews_sp15_multifit', bs=bs, num_workers=1);



config = awd_lstm_clas_config.copy()

config['qrnn'] = True

config['n_hid'] = 1550 #default 1152

config['n_layers'] = 4 #default 3



learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False)

learn_c.load(f'ar_clas_reviews_sp15_multifit', purge=False);



preds,y,losses = learn_c.get_preds(with_loss=True)

predictions = np.argmax(preds, axis = 1)



interp = ClassificationInterpretation(learn_c, preds, y, losses)

interp.plot_confusion_matrix()



from sklearn.metrics import confusion_matrix, f1_score

cm = confusion_matrix(np.array(y), np.array(predictions))

print(cm)



## acc

#print(f'accuracy global: {(cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/(cm.sum())}')

print(f'accuracy global: {(cm[0,0]+cm[1,1])/(cm.sum())}') # 3 classes



# acc neg, acc pos

print(f'accuracy on class 0: {cm[0,0]/(cm.sum(1)[0])*100}') 

print(f'accuracy on class 1: {cm[1,1]/(cm.sum(1)[1])*100}')



print ('F1 score:', f1_score(y, predictions,average='weighted'))



preds,targs = learn_c.get_preds(ordered=True)

accuracy(preds,targs),f1(preds,targs)
# The darker the word-shading in the below example, the more it contributes to the classification. 

test_text  ="موقع الفندق جميل لكن وجبة الفطور سيئة جدا لا انصح به ابدا"

txt_ci = TextClassificationInterpretation.from_learner(learn_c)

txt_ci.show_intrinsic_attention(test_text,cmap=plt.cm.Purples)
test_text  ="وجبة الفطور سيئة جدا لكن موقع الفندق جميل لا انصح به ابدا "

pred = learn_c.predict(test_text)

print(pred)

txt_ci.show_intrinsic_attention(test_text,cmap=plt.cm.Purples)
print(f'Finished: {time.ctime()}')