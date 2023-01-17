from fastai import *

from fastai.text import *

from fastai.metrics import Precision, Recall, FBeta

import random

import re

random.seed(42) # set the random seed
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

irony_data = pd.read_csv('/kaggle/input/ironic-corpus/irony-labeled.csv')

irony_data.head()
imdb_path = untar_data(URLs.IMDB_SAMPLE)

imdb_path.ls()

imdb = pd.read_csv(imdb_path/'texts.csv')
imdb.head()
combined = imdb.append(irony_data.rename(columns={'comment_text':'text'}),sort=False)

combined.columns
bs = 48

data_lm = (TextList.from_df(df=combined, cols='text')

            .split_by_rand_pct(0.1)

            .label_for_lm()           

            .databunch(bs=bs))

data_lm.save('data_lm.pkl')
data_lm.show_batch()
bs=48

path = "."

data_lm = load_data(path, 'data_lm.pkl', bs=bs)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(4, 1e-2, moms=(0.8,0.7))
learn.save('fit_head')
learn.load('fit_head');
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, slice(1e-2/2,1e-3), moms=(0.8,0.7))
learn.save('fine_tuned')
learn.load('fine_tuned');
TEXT = "I think that"

N_WORDS = 25

N_SENTENCES = 2
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
learn.save_encoder('fine_tuned_enc')
imdb = imdb[['text','label']]
imdb_clas = (TextList.from_df(df=imdb,cols='text',vocab=data_lm.vocab)

             .split_by_rand_pct(.2)

             #split by random 20% 

             .label_from_df(cols='label')

             #label from the csv file

             .databunch(bs=bs))



imdb_clas.save('imdb_clas.pkl')
imdb_clas = load_data(path, 'imdb_clas.pkl', bs=bs)
imdb_clas.show_batch()
learn = text_classifier_learner(imdb_clas, AWD_LSTM, drop_mult=0.2)

learn.load_encoder('fine_tuned_enc');
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(4, 1e-3, moms=(0.8,0.7))
learn.save('froze_imdb')
learn.load('froze_imdb');
bs = 24 # was previously 48

imdb_clas = load_data(path, 'imdb_clas.pkl', bs=bs)

learn = text_classifier_learner(imdb_clas, AWD_LSTM, drop_mult=0.5)

learn.load('froze_imdb');
learn.unfreeze()

learn.fit_one_cycle(1, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn.save('unfroze_imdb')
from sklearn.model_selection import KFold # import KFold



irony_data.head()

X = irony_data['comment_text']

y = irony_data['label']

kf = KFold(n_splits=5)

kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf)
trains = list()

tests = list()

for train_index, test_index in kf.split(X):

    trains.append(train_index)

    tests.append(test_index)
def create_validation(valnum):



    train = {'comment_text': X[trains[valnum]], 'label': y[trains[valnum]]}

    dftrain = pd.DataFrame(data=train)

    

    valid = {'comment_text': X[tests[valnum]], 'label': y[tests[valnum]]}

    dfvalid = pd.DataFrame(data=valid)

    

    return dftrain, dfvalid
fold1_train, fold1_valid = create_validation(0)

fold2_train, fold2_valid = create_validation(1)

fold3_train, fold3_valid = create_validation(2)

fold4_train, fold4_valid = create_validation(3)

fold5_train, fold5_valid = create_validation(4)
bs=48

path = "."

data_lm = load_data(path, 'data_lm.pkl', bs=bs)



trains = [fold1_train, fold2_train, fold3_train, fold4_train, fold5_train]

valids = [fold1_valid, fold2_valid, fold3_valid, fold4_valid, fold5_valid]

n_reps = 1

# to hold precision, recall and f1 values across reps

metrics = np.zeros([len(trains),n_reps,3]) 
weights = [1., 3.]

class_weights=torch.FloatTensor(weights).cuda()
foldx = TextDataBunch.from_df(".",fold1_train,fold1_valid,text_cols=0,label_cols=1,vocab=data_lm.vocab,bs=bs)

learn = text_classifier_learner(foldx, AWD_LSTM, drop_mult=0.2,

                                loss_func = nn.CrossEntropyLoss(weight=class_weights))

learn.load('unfroze_imdb');

learn.lr_find()

learn.recorder.plot(suggestion=True)
for reps in range(n_reps):

    for fold in range(0,len(trains)):

        foldx = TextDataBunch.from_df(".",trains[fold],valids[fold],text_cols=0,label_cols=1,vocab=data_lm.vocab,bs=bs)

        learn = text_classifier_learner(foldx, AWD_LSTM, drop_mult=0.2,metrics=[Precision(),Recall(),FBeta(beta=1)],

                                       loss_func = nn.CrossEntropyLoss(weight=class_weights))

        learn.load('unfroze_imdb');

        learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))

        metrics[fold,reps:] = learn.recorder.metrics

    
avg_per_fold = np.mean(metrics,axis=1);avg_per_fold
def format_scores(avg_metrics):

    def print_line(name,arr):

        print(name,':',format(np.mean(arr), '.3f'), '(range ', np.min(arr), ' - ',np.max(arr))

    

    print_line('F1 score',avg_metrics[:,2])

    print_line('recall',avg_metrics[:,1])

    print_line('precision',avg_metrics[:,0])

    
format_scores(avg_per_fold)