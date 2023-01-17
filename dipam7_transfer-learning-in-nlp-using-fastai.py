from fastai.text import *
path = "../input/review_polarity/txt_sentoken/"
bs = 48
np.random.seed(42)

data = (TextList.from_folder(path)

                        .split_by_rand_pct(0.1)

                        .label_from_folder()

                        .databunch(bs=bs))
data.save("/kaggle/working/data.pkl")
data.show_batch()
data.vocab.itos[:10]
data.train_ds[0][0]
data.train_ds[0][0].data[:10]
np.random.seed(42)

data_lm = (TextList.from_folder(path)

                        .split_by_rand_pct(0.1)

                        .label_for_lm()

                        .databunch(bs=bs))
data_lm.show_batch()
learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.3, model_dir="/tmp/model/")
learn.lr_find()
learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(1, 3e-02, moms=(0.8, 0.7))
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(10, 3e-04, moms=(0.8, 0.7))
TEXT = "Avengers is a "

N_WORDS = 30

N_SENTENCES = 2



print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

TEXT = "My sister watched that movie because"

N_WORDS = 10

N_SENTENCES = 2



print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

learn.save('/kaggle/working/fine_tuned')
learn.save_encoder('fine_tuned_enc')
learn = text_classifier_learner(data, arch=AWD_LSTM, drop_mult=0.3, model_dir="/tmp/model/")
learn.load_encoder('fine_tuned_enc')

learn.freeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, 2e-2, moms=(0.8,0.7))
learn.save('/kaggle/working/first')
learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn.save('/kaggle/working/second')
learn.freeze_to(-3)

learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn.save('/kaggle/working/third')
learn.unfreeze()

learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn.save('/kaggle/working/fourth')
learn.predict("I really loved that movie, it was awesome!")
learn.predict("The movie is disappointing")
learn.predict("I would not recommend this movie to anyone")
learn.predict("The movie was okay")
learn.predict("I would feel sorry for anyone watching this movie")
learn.predict("The direction was horrible")
learn.predict("One would not be disappointed watching the movie.")
learn.predict("One would not be disappointed watching the movie. I especially loved the part when the protagonist goes for a retreat and finds all his answers")