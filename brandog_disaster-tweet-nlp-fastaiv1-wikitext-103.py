from fastai.text import *



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = "/kaggle/input/nlp-getting-started/"

base_path="../output"

text_columns=['text']

label_columns=['target']

BATCH_SIZE=128
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df.head()
test_df.head()
list(train_df[train_df["target"] == 1]["text"].values[0:10])
list(train_df[train_df["target"] == 0]["text"].values[0:10])
from nltk.tokenize import TweetTokenizer

twt = TweetTokenizer(strip_handles=True)

def tweets(r):

    s = ' '.join(twt.tokenize(r['text']))

    s = re.sub(r'http\S+', '', s)

    s = re.sub(r'https\S+', '', s)    

    return s
train_df['ptext'] = train_df.apply(tweets, axis=1)

test_df['ptext'] = test_df.apply(tweets, axis=1)
# use the new preprocessed text column

text_columns = ['ptext']
tweets = pd.concat([train_df[text_columns], test_df[text_columns]])

print(tweets.shape)
tweets.head()
data_lm = (TextList.from_df(tweets)

           #Inputs: all the text files in path

            .split_by_rand_pct(0.15)

           #We randomly split and keep 10% for validation

            .label_for_lm()           

           #We want to do a language model so we label accordingly

            .databunch(bs=BATCH_SIZE))

data_lm.save('tmp_lm')

data_lm.show_batch()
data = TextClasDataBunch.from_csv(path, csv_name='train.csv', valid_pct=0.2, test='test.csv', text_cols='text', label_cols='target')

data.show_batch()
data.vocab.itos[:15]
data.train_ds[0][0]
data.train_ds[0][0].data[:10]
data_lm.show_batch()
data_lm.train_ds
data_lm.valid_ds
# no test set

data_lm.test_ds
# download a model

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))

learn.save('fit_head')
learn.load('fit_head');
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')
learn.load('fine_tuned');
TEXT = "There is a fire "

N_WORDS = 40

N_SENTENCES = 2



print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
TEXT = "I love this movie "

N_WORDS = 40

N_SENTENCES = 2



print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
learn.save_encoder('fine_tuned_enc')
data_clas = (TextList.from_df(train_df, cols=text_columns, vocab= data_lm.vocab)

             # create a vaildation set from 15% of the training df

            .split_by_rand_pct(0.15)

             # identify the label columns

            .label_from_df(label_columns)

             # add the test df

            .add_test(test_df[text_columns])

            .databunch(bs=BATCH_SIZE))

data_clas.save('tmp_clas')

data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('fine_tuned_enc')
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(1, 3e-2, moms=(0.8,0.7))
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

learn.fit_one_cycle(3, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
#learn.export()

learn.save('final')
learn.predict("There is a fire on the street")
learn.predict("I love hot wings, they are fire")
from fastai.vision import ClassificationInterpretation



interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(6,6), dpi=60)
interp = TextClassificationInterpretation.from_learner(learn)

interp.show_top_losses(10)
# check some actual predictions

for i in range(10):

    print(test_df.loc[i,'text'])

    print(learn.predict(test_df.loc[i,'text']))

    print(' ')
## get test set predictions and ids

preds, _ = learn.get_preds(ds_type=DatasetType.Test,  ordered=True)

preds = preds.argmax(dim=-1)



id = test_df['id']
my_submission = pd.DataFrame({'id': id, 'target': preds})

my_submission.to_csv('submission.csv', index=False)
my_submission['target'].value_counts()