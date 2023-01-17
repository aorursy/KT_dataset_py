from fastai import *
from fastai.text import * 

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
        
random_seed(42, True)
Path('data/hard/').mkdir(parents=True, exist_ok=True)
!cp '../input/balanced-reviews/balanced-reviews-utf8.tsv' data/hard/balanced-reviews.tsv
path=Path('data/hard/')
df_ar = pd.read_csv(path/'balanced-reviews.tsv', delimiter='\t')
df_ar.head() # the first review sounds positive but rating is 2 (-ve). Reviewer's choice!
df_ar = df_ar[['rating', 'review']] # we are interested in rating and review only
# code rating as +ve if > 3, -ve if less, no 3s in dataset 
df_ar['rating'] = df_ar['rating'].apply(lambda x: -1 if x < 3 else 1)
# rename columns to fit default constructor in fastai
df_ar.columns = ['label', 'text']
df_ar.head()
df_valid = df_ar.sample(21140, replace = False) # 20% for validation
df_valid['is_valid'] = True
df_train = df_ar.drop(df_valid.index)
df_train['is_valid'] = False

df_all = pd.concat([df_train, df_valid])
df_all.head()
# write to csv (overwrites by default)
df_all.to_csv(path/'hard_text.csv', index=False)
df = pd.read_csv(path/'hard_text.csv')
df.head()
Path('models/').mkdir(parents=True, exist_ok=True)
!cp -a '../input/model45_30_4/lm_best.pth' models/
!cp '../input/model45_30_4/itos.pkl' models/
Path('models/').absolute() # get absolute path od model files
pretrained_fnames=['/kaggle/working/models/lm_best','/kaggle/working/models/itos']
# Language model data
data_lm = TextLMDataBunch.from_csv(path, 'hard_text.csv')
# Classifier model data
data_clas = TextClasDataBunch.from_csv(path, 'hard_text.csv', vocab=data_lm.train_ds.vocab, bs=64, num_workers=0)
data_lm.save()
data_clas.save()
data_lm = TextLMDataBunch.load(path)
data_clas = TextClasDataBunch.load(path, bs=64)
learn = language_model_learner(data_lm, pretrained_fnames=pretrained_fnames, drop_mult=0.1) # was .5
learn.lr_find(start_lr = slice(10e-7,10e-5),end_lr=slice(0.1,10))#start_lr = slice(10e-7,10e-5),end_lr=slice(0.1,10))
learn.recorder.plot(skip_end=10)
learn.fit_one_cycle(1, 2e-2)
learn.unfreeze()
learn.fit_one_cycle(3, 2e-2)
learn.predict("كان الاستقبال في الفندق", n_words=10)
# first amount of words (here 10), the next 10 target words (actual) and the ones predicted.
#learn.show_results(max_len = 10)
learn.save_encoder('ft_enc')
learn.recorder.plot_losses()
#classifier
learn_clas = text_classifier_learner(data_clas, drop_mult=0.5)
learn_clas.load_encoder('ft_enc')
data_clas.show_batch(2)
learn_clas.lr_find(start_lr = slice(10e-7,10e-5),end_lr=slice(0.1,10))
learn_clas.recorder.plot(skip_end=10)
learn_clas.fit_one_cycle(1, 2e-3)
learn_clas.freeze_to(-2)
learn_clas.fit_one_cycle(1, slice(1e-4, 1e-2))
learn_clas.unfreeze()
learn_clas.fit_one_cycle(3, slice(1e-4, 1e-2))
print(learn_clas.validate())
learn_clas.recorder.plot_losses()
learn_clas.show_results(rows=20)
