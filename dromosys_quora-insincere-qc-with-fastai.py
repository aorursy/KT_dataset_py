import os, gc

from fastai.text import *

from tqdm import tqdm_notebook as tqdm

print(os.listdir("../input"))
# make training deterministic/reproducible

def seed_everything(seed=2018):

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True

seed_everything()



def f1_score(y_pred, targets):

    epsilon = 1e-07

    

    y_pred = y_pred.argmax(dim=1)

    targets = targets.argmax(dim=1)



    tp = (y_pred*targets).float().sum(dim=0)

    tn = ((1-targets)*(1-y_pred)).float().sum(dim=0)

    fp = ((1-targets)*y_pred).float().sum(dim=0)

    fn = (targets*(1-y_pred)).sum(dim=0)



    p = tp / (tp + fp + epsilon)

    r = tp / (tp + fn + epsilon)



    f1 = 2*p*r / (p+r+epsilon)

    f1 = torch.where(f1!=f1, torch.zeros_like(f1), f1)

    return f1.mean()
EMBED_SIZE = 50

MAX_FEATURES = 60000

MAX_LENGTH = 100

EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
# df = pd.read_csv('../input/train.csv')



# insincere_df = df[df.target==1]

# sincere_df = df[df.target==0]



# sincere_df = sincere_df.iloc[np.random.permutation(len(sincere_df))]

# sincere_df = sincere_df[:int(len(insincere_df)*5)]



# del df



# df = pd.concat([insincere_df, sincere_df])

# df = df.iloc[np.random.permutation(len(df))]



# del insincere_df

# del sincere_df

# gc.collect()
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.head()
def truncate(df):

    df['question_text'] = df.question_text.apply(lambda x: x[:MAX_LENGTH])

    

truncate(train_df)

truncate(test_df)
train_df = train_df.iloc[np.random.permutation(len(train_df))]

cut = int(0.2 * len(train_df)) + 1

train_df, valid_df = train_df[cut:], train_df[:cut]
%%time

data = TextDataBunch.from_df(path='.',

                             train_df=train_df, 

                             valid_df=valid_df,

                             test_df=test_df,

                             text_cols='question_text', 

                             label_cols='target',

                             max_vocab=MAX_FEATURES)

print(len(data.vocab.itos))

data.save()

del train_df

del valid_df 

del test_df 

del data

gc.collect()
%%time

data = TextLMDataBunch.load(path='.', bs=32)

data.show_batch()
gc.collect()
learner = language_model_learner(data, drop_mult=0.7, pretrained_model=URLs.WT103) #emb_sz=EMBED_SIZE
#learner.lr_find()

#learner.recorder.plot(skip_start=25)
learner.fit_one_cycle(1, 5e-2, moms=(0.8,0.7))
learner.unfreeze()
learner.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learner.save_encoder('fine_tuned_enc')
data = TextClasDataBunch.load(path='.', bs=32)

data.show_batch()
learner = text_classifier_learner(data, drop_mult=0.3) #emb_sz=EMBED_SIZE

learner.load_encoder('fine_tuned_enc')

learner.freeze()
#learner.lr_find()

#learner.recorder.plot()
learner.fit_one_cycle(1, 5e-2, moms=(0.8,0.7))
#learner.freeze_to(-2)

#learner.fit_one_cycle(1, slice(1e-3,1e-1), moms=(0.8,0.7))
learner.unfreeze()

learner.fit_one_cycle(1, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
#learner.fit_one_cycle(1, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
preds, targets = learner.get_preds()



predictions = np.argmax(preds, axis = 1)
%matplotlib inline

from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=2)

#predictions = model.predict(X_test, batch_size=1000)



LABELS = ['Normal','Insincere'] 



confusion_matrix = metrics.confusion_matrix(targets, predictions)



plt.figure(figsize=(5, 5))

sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", annot_kws={"size": 20});

plt.title("Confusion matrix", fontsize=20)

plt.ylabel('True label', fontsize=20)

plt.xlabel('Predicted label', fontsize=20)

plt.show()
%time learner.predict("How much does a tutor earn in Bangalore?")
#preds = learner.get_preds(ds_type=DatasetType.Test)
#preds = preds[0].argmax(dim=1)

#preds.sum()
test_df = pd.read_csv('../input/test.csv')
#test_df.drop(['question_text'], axis=1, inplace=True)

#test_df['prediction'] = preds.numpy()
#test_df.to_csv("submission.csv", index=False)
probs, _ = learner.get_preds(DatasetType.Test)

preds = np.argmax(probs, axis=1)



submission = pd.DataFrame(test_df['qid'])

submission['prediction'] = preds 

submission.to_csv('submission.csv',index=False)

submission.head()