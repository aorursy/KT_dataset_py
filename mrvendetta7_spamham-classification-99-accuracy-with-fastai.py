from fastai.text import *

import pandas as pd

import seaborn as sns
!mkdir spamham

!cp /kaggle/input/spam-text-message-classification/* /kaggle/working/spamham
path = Path('/kaggle/working/spamham')

df = pd.read_csv(path/'SPAM text message 20170820 - Data.csv')

df.head()
df.isna().sum()
sns.countplot(df['Category'])
data_lm = (TextList

    .from_csv(path, 'SPAM text message 20170820 - Data.csv', cols=1)

    .split_by_rand_pct(0.1)

    .label_for_lm()

    .databunch(bs=64)

)
learn_lm = language_model_learner(data_lm, AWD_LSTM)

learn_lm.lr_find()

learn_lm.recorder.plot()
learn_lm.fit_one_cycle(7, slice(0.05))
learn_lm.unfreeze()

learn_lm.lr_find()

learn_lm.recorder.plot()
learn_lm.fit_one_cycle(15, slice(1e-3, 0.01))
learn_lm.predict('Hi', 15)
learn_lm.save_encoder('enc')
datacls = (TextList

    .from_csv(path, 'SPAM text message 20170820 - Data.csv', cols=1, vocab=data_lm.vocab)

    .split_by_rand_pct(0.33, seed=42)

    .label_from_df(0)

    .databunch(bs=64)

)
learn = text_classifier_learner(datacls, AWD_LSTM)

learn.load_encoder('enc')

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8, slice(1e-3/2))
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, slice(1e-2, 1e-3/10))
interp = TextClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
cmx = interp.confusion_matrix()

print(f'Sensitivity: {cmx[1,1]/cmx[1].sum()}')

print(f'Specificity: {cmx[0,0]/cmx[0].sum()}')
interp.show_intrinsic_attention("Would you like to buy this amazing product?")
interp.show_top_losses(3)