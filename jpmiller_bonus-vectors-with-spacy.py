import numpy as np
import pandas as pd
import spacy as sp

nlp_lg = sp.load('en_core_web_lg')
nlp_lg
# get spacy vector
lgword = nlp_lg("and")
lgvec =   ",".join(lgword.vector[0:10].round(5).astype(str))

# get glove vector
glv = pd.read_csv('../input/embeddings/glove.840B.300d/glove.840B.300d.txt', header=None, sep=' ', skiprows=2, nrows=5, index_col=[0])
glvec = glv.loc['and', 0:10].round(5).astype(str).str.cat(sep=' ')

print(lgword.vector.shape[0], "\n",
      lgvec, "\n",
      glv.shape[1], "\n",
      glvec)
nlp_sm = sp.load('en_core_web_sm')
smword = nlp_sm("and")
smvec = ",".join(smword.vector[0:10].round(5).astype(str))

print(smword.vector.shape[0], "\n",
       smvec)
e = nlp_lg('Why are aliens so smart?')

print(e.vector.shape, "\n",
       e.vector[0:10])
c = nlp_sm('What capital city is the prettiest?') 
d = nlp_sm('Which country has the nicest people?')
e = nlp_sm('Why are aliens so smart?')

print("\n", c.similarity(d),
        c.similarity(e))
df = pd.DataFrame({"text": [tokens.text for tokens in d], 
                   "lemmatized": [tokens.lemma_ for tokens in d],
                   "part of speech": [tokens.pos_ for tokens in d],
                  "stop word": [tokens.is_stop for tokens in d]})
display(df)                 
sp.displacy.render(d, style='dep', jupyter=True, options={'compact':60})
#%% import
import time
import numpy as np
import pandas as pd
import spacy as sp
nlp_lg = sp.load('en_core_web_lg')
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


# get train data
train = pd.read_csv('../input/train.csv', nrows=30_000)  #limiting the data for time's sake
train['question_text'] = train.question_text.str.replace('?', ' ?')
train['question_text'] = train.question_text.str.replace('.', ' .')
tstacked = pd.DataFrame(train.question_text.str.split(expand=True).stack(), 
                columns=['token'])

tlist = tstacked.token.unique().tolist()
vlist = [nlp_lg(str).vector for str in tqdm(tlist)]
lookup = dict(zip(tlist, vlist))

tstacked['vec'] = tstacked.token.map(lookup)

colnames = ['t'+str(i) for i in range(300)]
tstacked[colnames] = pd.DataFrame(tstacked.vec.values.tolist(), 
                            index=tstacked.index)
tstacked.drop(['token', 'vec'], axis=1, inplace=True)

del tlist
del vlist
del lookup
tagg = tstacked.groupby(level=0).apply(np.mean)
del tstacked

X_vecs = tagg.values
y = train.target.values
del tagg
# Logistic Regression
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=911)
train_pred = np.zeros(train.shape[0])
for train_idx, val_idx in skf.split(X_vecs, y):
    X_train, y_train  = X_vecs[train_idx], y[train_idx]
    X_val, y_val = X_vecs[val_idx], y[val_idx]
    model = LogisticRegression(solver='saga', class_weight='balanced', 
                                    C=0.5, max_iter=250, verbose=1, n_jobs=-1) #seed not set
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)
    train_pred[val_idx] = val_pred[:,1]
    

print("finding best threshold")
best_thresh = 0.0
best_score = 0.0
for thresh in np.arange(0, 1, 0.01):
    score = f1_score(y, train_pred > thresh)
    if score > best_score:
        best_thresh = thresh
        best_score = score
print(best_thresh, best_score)
print(best_thresh, best_score)
# predict on test set
test = pd.read_csv('../input/test.csv', index_col=['qid'])
test.head()
X_test = test.question_text.tolist()
X_testvecs = np.array([nlp_lg(text).vector for text in tqdm(X_test)])

trounds = 3
preds_test = np.zeros(len(X_test))
for i in range(trounds):
    model = LogisticRegression(solver='saga', class_weight='balanced', 
                                    C=0.5, max_iter=250, verbose=1, n_jobs=-1, random_state=40*i)
    model.fit(X_vecs, y)
    preds_test += lgr.predict_proba(X_testvecs)[:, 1] / trounds

    
# submit
sub = pd.read_csv('../input/sample_submission.csv', index_col=['qid'])
sub['prediction'] = preds_test > best_thresh
sub.to_csv('submission.csv')
#%% import
import numpy as np
import pandas as pd
import spacy as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# get data and make json format for spacy
train = pd.read_csv('../input/train.csv', nrows=10_000)  ## using part of the data again
texts = train.question_text.tolist()
cats = train.target.apply(lambda t: {'cats': {'Insincere': t == 1}}).tolist()
train_texts, dev_texts, train_cats, dev_cats = train_test_split(texts, cats, 
        test_size=0.2, random_state=90)
train_data = list(zip(train_texts, train_cats))
print("Example format \n", train_data[0:10])


#%% set up the pipeline
nlp_bl = sp.blank('en') 
nlp_bl.vocab.vectors.name = 'spacy_pretrained_vectors'
textcat = nlp_bl.create_pipe('textcat')
nlp_bl.add_pipe(textcat, last=True)
textcat.add_label('Insincere')


# train
n_iter = 10
other_pipes = [pipe for pipe in nlp_bl.pipe_names if pipe != 'textcat']
with nlp_bl.disable_pipes(*other_pipes):  #only train textcat
    optimizer = nlp_bl.begin_training()
    print("Training the model...")
    for i in range(n_iter):
        losses = {}
        batches = sp.util.minibatch(train_data, size=sp.util.compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp_bl.update(texts, annotations, sgd=optimizer, drop=0.2,
                        losses=losses)
        print("iter {} loss: {:4f}".format(i, losses['textcat']))

        
# evaluate model
preds = []
docs = (nlp_bl(text) for text in dev_texts)
for doc in docs:
    pred = doc.cats['Insincere']
    preds.append(pred)
    
truths = [val['Insincere'] for val in [dc['cats'] for dc in dev_cats]]

#%% find best threshold
best_thresh = 0.0
best_score = 0.0
for thresh in np.arange(0, 1, 0.01):
    score = f1_score(truths, preds > thresh)
    if score > best_score:
        best_thresh = thresh
        best_score = score
print(best_thresh, best_score)