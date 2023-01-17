%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.text import *
path = untar_data(URLs.IMDB_SAMPLE)

path.ls()
df = pd.read_csv(path/'texts.csv')

df.head()
df['text'][1]
data_lm = TextDataBunch.from_csv(path, 'texts.csv')
data_lm.save()
data = load_data(path)
data = TextClasDataBunch.from_csv(path, 'texts.csv')

data.show_batch()
data.vocab.itos[:10]

#len(data.vocab.itos)

data.train_ds[0][0]

data.train_ds[0][0].data[:10]
data = (TextList.from_csv(path, 'texts.csv', cols='text')

                .split_from_df(col=2)

                .label_from_df(cols=0)

                .databunch(num_workers=0))
bs=48
path = untar_data(URLs.IMDB)

path.ls()
(path/'train').ls()
data_lm = (TextList.from_folder(path)

           #Inputs: all the text files in path

            .filter_by_folder(include=['train', 'test', 'unsup']) 

           #We may have other temp folders that contain text files so we only keep what's in train and test

            .split_by_rand_pct(0.1)

           #We randomly split and keep 10% (10,000 reviews) for validation

            .label_for_lm()           

           #We want to do a language model so we label accordingly

            .databunch(bs=bs))

data_lm.save('data_lm.pkl')
data_lm = load_data(path, 'data_lm.pkl', bs=bs)
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

learn.lr_find()
learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn.save('fit_head')
learn.load('fit_head');
learn.unfreeze()
# commented out because the training time didn't fit in a single Kernel session

# learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')
learn.load('fine_tuned');
TEXT = "i liked this movie because"

N_WORDS = 40

N_SENTENCES = 2
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
learn.save_encoder('fine_tuned_enc')
path = untar_data(URLs.IMDB)
data_clas = (TextList.from_folder(path, vocab=data_lm.vocab)

             #grab all the text files in path

             .split_by_folder(valid='test')

             #split by train and valid folder (that only keeps 'train' and 'test' so no need to filter)

             .label_from_folder(classes=['neg', 'pos'])

             #label them all with their folders

             .databunch(bs=bs))



data_clas.save('data_clas.pkl')
data_clas = load_data(path, 'data_clas.pkl', bs=bs)
data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('fine_tuned_enc')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
learn.save('first')
learn.load('first');
learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn.save('second')
learn.load('second');
learn.freeze_to(-3)

learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn.save('third')
learn.load('third');
learn.unfreeze()

learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn.predict("I really loved that movie, it was awesome!")
learn.predict("The movie was disappointing")