from fastai import *

from fastai.text import * 

from fastai.gen_doc.nbdoc import *

from fastai.datasets import * 

from fastai.datasets import Config

from pathlib import Path

import pandas as pd
path = Path('../input/')
df_test = pd.read_csv(path/'test.csv')

df_test.head()
df = pd.read_csv(path/'train.csv')#.sample(frac=0.05)

df.head()
data_lm = (TextList.from_csv(path, 'train.csv', cols='question_text')

            .random_split_by_pct(.2)

            .label_for_lm()

            .add_test(TextList.from_csv(path, 'test.csv', cols='question_text'))

            .databunch())
data_clas = (TextList.from_csv(path, 'train.csv', cols='question_text', vocab=data_lm.vocab)

            #.split_from_df(col='target')

             .random_split_by_pct(.2)

            .label_from_df(cols='target')

            #.add_test(TextList.from_csv(path, 'test.csv', cols='question_text'))

            .databunch(bs=42))
data_clas.show_batch()
MODEL_PATH = "/tmp/model/"
learn = text_classifier_learner(data_clas,model_dir=MODEL_PATH)

#learn.load_encoder('mini_train_encoder')

learn.fit_one_cycle(2, slice(1e-3,1e-2))

#learn.save('mini_train_clas')
learn.show_results()
# Language model data

#data_lm = TextLMDataBunch.from_csv(path, 'train.csv', cols='question_text')

# Classifier model data

#data_clas = TextClasDataBunch.from_csv(path, 'train.csv', vocab=data_lm.train_ds.vocab, bs=4) #32



#data_lm = TextLMDataBunch.from_df(path, df)

#data_clas = TextClasDataBunch.from_df(path, df,  vocab=data_lm.train_ds.vocab)
data_lm.show_batch()
#??language_model_learner()
learn = language_model_learner(data_lm, drop_mult=0.5, model_dir=MODEL_PATH) #bptt=65, emb_sz=400, pretrained_model=URLs.WT103
acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)



learn.metrics = metrics=[accuracy,acc_02, f_score]
learn.fit_one_cycle(1, 1e-2)
learn.unfreeze()

learn.fit_one_cycle(1, 1e-3)
learn.predict("This is a review about", n_words=10)
learn.save_encoder('ft_enc')
learn = text_classifier_learner(data_clas, drop_mult=0.5)

learn.load_encoder('ft_enc')
data_clas.show_batch()
learn.fit_one_cycle(1, 1e-2)
learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

learn.unfreeze()

learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
learn.predict("This was a great movie!")
learn.show_results()
probs, _ = learn.get_preds(DatasetType.Test)
preds = np.argmax(probs, axis=1)
ids = df_test["qid"].copy()
submission = pd.DataFrame(data={

    "qid": ids,

    "prediction": preds

})

submission.to_csv("submission.csv", index=False)

submission.head(n=50)