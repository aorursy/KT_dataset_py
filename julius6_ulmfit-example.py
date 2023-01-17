from fastai.callbacks import *

from fastai.text import *



import pandas as pd
train = pd.read_csv("../input/train.csv")

val = pd.read_csv("../input/val.csv")



train = train.drop(["id", "Unnamed: 0"], axis=1)

val = val.drop(["id", "Unnamed: 0"], axis=1)
data_lm = TextLMDataBunch.from_df(path="", train_df=train, valid_df=val, text_cols="comment_text", bs=64)
data_class = TextClasDataBunch.from_df(path="", train_df=train, valid_df=val, vocab=data_lm.train_ds.vocab, bs=64,

                                       text_cols="comment_text", label_cols="toxic")
config_lm = awd_lstm_lm_config.copy()

config_class = awd_lstm_clas_config.copy()



print(config_lm)

print(config_class)
learn = language_model_learner(data=data_lm, arch=AWD_LSTM, config=config_lm)
#learn.freeze_to(-1)

#learn.lr_find()

#learn.recorder.plot(suggestion=True)
learn.unfreeze()

callbacks = [EarlyStoppingCallback(learn, patience=5)]

learn.fit_one_cycle(25, max_lr=[.1/81, .1/27, .1/9, .1/3], callbacks=callbacks)

learn.save_encoder("lm_enc")
learn = text_classifier_learner(data=data_class, arch=AWD_LSTM, model_dir="../working/models")

learn.load_encoder("lm_enc")
#learn.freeze_to(-1)

#learn.lr_find()

#learn.recorder.plot()
callbacks = [EarlyStoppingCallback(learn, patience=5)]



learn.freeze_to(-1)

learn.fit_one_cycle(1, max_lr=[.1/81, .1/27, .1/9, .1/3, .1])

learn.recorder.plot_losses()



learn.freeze_to(-2)

learn.fit_one_cycle(1, max_lr=[.1/81, .1/27, .1/9, .1/3, .1])

learn.recorder.plot_losses()

"""

learn.freeze_to(-3)

learn.fit_one_cycle(1, max_lr=[.1/81, .1/27, .1/9, .1/3, .1])

learn.recorder.plot_losses()



learn.freeze_to(-4)

learn.fit_one_cycle(1, max_lr=[.1/81, .1/27, .1/9, .1/3, .1])

learn.recorder.plot_losses()

"""
learn.unfreeze()

learn.fit_one_cycle(25, max_lr=[.01/81, .01/27, .01/9, .01/3, .01], callbacks=callbacks)

learn.recorder.plot_losses()
print(learn.predict("A group of boys on bicycles gather on a dusty side-street in eastern Saudi Arabia."))

print(learn.predict("Shit."))