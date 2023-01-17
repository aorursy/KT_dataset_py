%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.text import *
# import fastai.utils.collect_env



# fastai.utils.collect_env.show_install()
# bs=48

# bs=24

bs=192
torch.cuda.set_device(2)
path = untar_data(URLs.IMDB_SAMPLE)

path.ls()
data_lm = TextDataBunch.from_csv(path, 'texts.csv')
data_lm.vocab.itos[:10]
data_lm.train_ds[0][0]
data_lm.train_ds[0][0].data[:10]
data = (TextList.from_csv(path, 'texts.csv', cols='text')

                .split_from_df(col=2)

                .label_from_df(cols=0)

                .databunch())
path = untar_data(URLs.IMDB)

path.ls()
(path/'train').ls()
path.ls()
data_lm = (TextList.from_folder(path)

           #Inputs: all the text files in path

            .filter_by_folder(include=['train', 'test', 'unsup']) 

           #We may have other temp folders that contain text files so we only keep what's in train and test

            .split_by_rand_pct(0.1, seed=42)

           #We randomly split and keep 10% (10,000 reviews) for validation

            .label_for_lm()           

           #We want to do a language model so we label accordingly

            .databunch(bs=bs, num_workers=1))
len(data_lm.vocab.itos),len(data_lm.train_ds)
data_lm.show_batch()
data_lm.save('lm_databunch')
data_lm = load_data(path, 'lm_databunch', bs=bs)
learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
wiki_itos = pickle.load(open(Config().model_path()/'wt103-1/itos_wt103.pkl', 'rb'))
wiki_itos[:10]
vocab = data_lm.vocab
vocab.stoi["stingray"]
vocab.itos[vocab.stoi["stingray"]]
vocab.itos[vocab.stoi["mobula"]]
awd = learn_lm.model[0]
from scipy.spatial.distance import cosine as dist
enc = learn_lm.model[0].encoder
enc.weight.size()
len(wiki_itos)
len(vocab.itos)
i, unks = 0, []

while len(unks) < 50:

    if data_lm.vocab.itos[i] not in wiki_itos: unks.append((i,data_lm.vocab.itos[i]))

    i += 1
wiki_words = set(wiki_itos)
imdb_words = set(vocab.itos)
wiki_not_imbdb = wiki_words.difference(imdb_words)
imdb_not_wiki = imdb_words.difference(wiki_words)
wiki_not_imdb_list = []



for i in range(100):

    word = wiki_not_imbdb.pop()

    wiki_not_imdb_list.append(word)

    wiki_not_imbdb.add(word)
wiki_not_imdb_list[:15]
imdb_not_wiki_list = []



for i in range(100):

    word = imdb_not_wiki.pop()

    imdb_not_wiki_list.append(word)

    imdb_not_wiki.add(word)
imdb_not_wiki_list[:15]
vocab.stoi["modernisation"]
"modernisation" in wiki_words
vocab.stoi["30-something"]
"30-something" in wiki_words, "30-something" in imdb_words
vocab.stoi["linklater"]
"linklater" in wiki_words, "linklater" in imdb_words
"house" in wiki_words, "house" in imdb_words
np.allclose(enc.weight[vocab.stoi["30-something"], :], 

            enc.weight[vocab.stoi["linklater"], :])
np.allclose(enc.weight[vocab.stoi["30-something"], :], 

            enc.weight[vocab.stoi["house"], :])
new_word_vec = enc.weight[vocab.stoi["linklater"], :]
TEXT = "The color of the sky is"

N_WORDS = 40

N_SENTENCES = 2
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
TEXT = "I hated this movie"

N_WORDS = 30

N_SENTENCES = 2
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
doc(LanguageLearner.predict)
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.10) for _ in range(N_SENTENCES)))
doc(LanguageLearner.predict)
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.10) for _ in range(N_SENTENCES)))
learn_lm.lr_find()
learn_lm.recorder.plot(skip_end=15)
lr = 1e-3

lr *= bs/48
learn_lm.to_fp16();
learn_lm.fit_one_cycle(1, lr*10, moms=(0.8,0.7))
learn_lm.save('fit_1')
learn_lm.load('fit_1');
learn_lm.unfreeze()
learn_lm.fit_one_cycle(10, lr, moms=(0.8,0.7))
learn_lm.save('fine_tuned')
learn_lm.save_encoder('fine_tuned_enc')
learn_lm.load('fine_tuned');
enc = learn_lm.model[0].encoder
np.allclose(enc.weight[vocab.stoi["30-something"], :], 

            enc.weight[vocab.stoi["linklater"], :])
np.allclose(enc.weight[vocab.stoi["30-something"], :], new_word_vec)
TEXT = "i liked this movie because"

N_WORDS = 40

N_SENTENCES = 2
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
TEXT = "This movie was"

N_WORDS = 30

N_SENTENCES = 2
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
TEXT = "I hated this movie"

N_WORDS = 40

N_SENTENCES = 2
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
bs=48
data_clas = (TextList.from_folder(path, vocab=data_lm.vocab)

             #grab all the text files in path

             .split_by_folder(valid='test')

             #split by train and valid folder (that only keeps 'train' and 'test' so no need to filter)

             .label_from_folder(classes=['neg', 'pos'])

             #label them all with their folders

             .databunch(bs=bs, num_workers=1))
data_clas.save('imdb_textlist_class')
data_clas = load_data(path, 'imdb_textlist_class', bs=bs, num_workers=1)
data_clas.show_batch()
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3) #.to_fp16()

learn_c.load_encoder('fine_tuned_enc')

learn_c.freeze()
learn_c.lr_find()
learn_c.recorder.plot()
learn_c.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
learn_c.save('first')
learn_c.load('first');
learn_c.freeze_to(-2)

learn_c.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn_c.save('2nd')
learn_c.freeze_to(-3)

learn_c.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn_c.save('3rd')
learn_c.unfreeze()

learn_c.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn_c.save('clas')
learn_c.predict("I really loved that movie, it was awesome!")
learn_c.predict("I didn't really love that movie, and I didn't think it was awesome.")