from fastai.text import *

import html

import json

from sklearn.model_selection import train_test_split



BOS = 'xbos'  # beginning-of-sentence tag

FLD = 'xfld'  # data field tag



PATH=Path('/kaggle/input/lolol/lolol')
LM_PATH=Path('/temp')

LM_PATH.mkdir(exist_ok=True)



LANG_FILENAMES = [str(f) for f in PATH.rglob("*/*")]

print(len(LANG_FILENAMES))

LANG_FILENAMES[0:5]
LANG_TEXT = []

for i in LANG_FILENAMES:

    for line in open(i):

        LANG_TEXT.append(json.loads(line))

        

LANG_TEXT = pd.DataFrame(LANG_TEXT)
LANG_TEXT.to_csv(f"{LM_PATH}/wiki_bangla_corpus.csv", index=False)

LANG_TEXT = pd.read_csv(f"{LM_PATH}/wiki_bangla_corpus.csv")
data_lm = TextLMDataBunch.from_csv(LM_PATH,'wiki_bangla_corpus.csv',text_cols='text')
data_lm.show_batch()
learner=language_model_learner(data_lm,AWD_LSTM,pretrained=False,metrics=accuracy)
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(15,2e-2)
learner.recorder.plot_losses()
learner.recorder.plot_metrics()
learner.save('/kaggle/working/gen1')
learner.save_encoder('/kaggle/working/gen1enc')
data_lm.save('/kaggle/working/data.pkl')
torch.save(learner.model.state_dict(),'/kaggle/working/model_state.h5')