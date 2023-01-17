%matplotlib inline
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA = pd.read_csv('../input/Speed Dating Data.csv', encoding='ISO-8859-1')
DATA.head()
# print('features: ', list(DATA))
print('samples: ', len(DATA.iloc[:,0]))
# DATA.isnull().sum()
DATA.drop(['id', 'iid'], axis=1, inplace=True)
pd.crosstab(DATA['match'], 'count')
filter_1 = ['gender', 'match', 'int_corr', 'age_o', 'pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha', 'dec_o', 'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o', 'met_o', 'age', 'imprace', 'imprelig', 'income', 'goal', 'date', 'go_out', 'career', 'career_c', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'exphappy', 'expnum', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1', 'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1', 'attr3_1', 'sinc3_1', 'fun3_1', 'intel3_1', 'amb3_1', 'attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1', 'dec', 'attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob', 'met', 'match_es', 'attr1_s', 'sinc1_s', 'intel1_s', 'fun1_s', 'amb1_s', 'shar1_s', 'attr3_s', 'sinc3_s', 'intel3_s', 'fun3_s', 'amb3_s', 'satis_2', 'length', 'numdat_2', 'attr7_2', 'sinc7_2', 'intel7_2', 'fun7_2', 'amb7_2', 'shar7_2', 'attr1_2', 'sinc1_2', 'intel1_2', 'fun1_2', 'amb1_2', 'shar1_2', 'attr4_2', 'sinc4_2', 'intel4_2', 'fun4_2', 'amb4_2', 'shar4_2', 'attr2_2', 'sinc2_2', 'intel2_2', 'fun2_2', 'amb2_2', 'shar2_2', 'attr3_2', 'sinc3_2', 'intel3_2', 'fun3_2', 'amb3_2', 'attr5_2', 'sinc5_2', 'intel5_2', 'fun5_2', 'amb5_2', 'you_call', 'them_cal', 'date_3', 'numdat_3', 'num_in_3', 'attr1_3', 'sinc1_3', 'intel1_3', 'fun1_3', 'amb1_3', 'shar1_3', 'attr7_3', 'sinc7_3', 'intel7_3', 'fun7_3', 'amb7_3', 'shar7_3', 'attr4_3', 'sinc4_3', 'intel4_3', 'fun4_3', 'amb4_3', 'shar4_3', 'attr2_3', 'sinc2_3', 'intel2_3', 'fun2_3', 'amb2_3', 'shar2_3', 'attr3_3', 'sinc3_3', 'intel3_3', 'fun3_3', 'amb3_3', 'attr5_3', 'sinc5_3', 'intel5_3', 'fun5_3', 'amb5_3']
DATA = DATA[filter_1]
for f in sorted(list(DATA)):
    print(f)
ambs = '''amb
amb1_1
amb1_2
amb1_3
amb1_s
amb2_1
amb2_2
amb2_3
amb3_1
amb3_2
amb3_3
amb3_s
amb4_1
amb4_2
amb4_3
amb5_1
amb5_2
amb5_3
amb7_2
amb7_3
amb_o'''.split()

attrs = '''attr
attr1_1
attr1_2
attr1_3
attr1_s
attr2_1
attr2_2
attr2_3
attr3_1
attr3_2
attr3_3
attr3_s
attr4_1
attr4_2
attr4_3
attr5_1
attr5_2
attr5_3
attr7_2
attr7_3
attr_o'''.split()

funs = '''fun
fun1_1
fun1_2
fun1_3
fun1_s
fun2_1
fun2_2
fun2_3
fun3_1
fun3_2
fun3_3
fun3_s
fun4_1
fun4_2
fun4_3
fun5_1
fun5_2
fun5_3
fun7_2
fun7_3
fun_o'''.split()

intels = '''intel
intel1_1
intel1_2
intel1_3
intel1_s
intel2_1
intel2_2
intel2_3
intel3_1
intel3_2
intel3_3
intel3_s
intel4_1
intel4_2
intel4_3
intel5_1
intel5_2
intel5_3
intel7_2
intel7_3
intel_o'''.split()

shars = '''shar
shar1_1
shar1_2
shar1_3
shar1_s
shar2_1
shar2_2
shar2_3
shar4_1
shar4_2
shar4_3
shar7_2
shar7_3
shar_o'''.split()

sincs = '''sinc
sinc1_1
sinc1_2
sinc1_3
sinc1_s
sinc2_1
sinc2_2
sinc2_3
sinc3_1
sinc3_2
sinc3_3
sinc3_s
sinc4_1
sinc4_2
sinc4_3
sinc5_1
sinc5_2
sinc5_3
sinc7_2
sinc7_3
sinc_o'''.split()
intersted_field = [ambs, attrs, funs, intels, shars, sincs]
for f in intersted_field:
    fig, ax = plt.subplots(dpi=90)
    corr = DATA[f].corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, ax=ax)
    plt.show()
others = '''age
age_o
art
career_c
clubbing
concerts
date
date_3
dec
dec_o
dining
exercise
exphappy
expnum
gaming
gender
go_out
goal
hiking
imprace
imprelig
income
int_corr
length
like
like_o
match_es
met
met_o
movies
museums
music
num_in_3
numdat_2
numdat_3
pf_o_amb
pf_o_att
pf_o_fun
pf_o_int
pf_o_sha
pf_o_sin
prob
prob_o
reading
satis_2
shopping
sports
theater
them_cal
tv
tvsports
yoga
you_call'''.split()

filter_2 = ['match', 'amb', 'amb1_1', 'amb2_1', 'amb3_1', 'amb4_1', 'amb5_1', 'amb7_2', 'amb_o', 
'attr', 'attr1_1', 'attr2_1', 'attr3_1', 'attr4_1', 'attr5_1', 'attr7_2', 'attr_o',
'fun', 'fun1_1', 'fun2_1', 'fun3_1', 'fun4_1', 'fun5_1', 'fun7_2', 'fun_o',
'intel', 'intel1_1', 'intel2_1', 'intel2_3', 'intel3_1', 'intel4_1', 'intel5_1', 'intel7_2', 'intel_o',
'sinc', 'sinc1_1', 'sinc2_1', 'sinc2_3', 'sinc3_1', 'sinc4_1', 'sinc5_1', 'sinc7_2', 'sinc_o'] + shars + others
DATA = DATA[filter_2]
DATA.isnull().sum()
DATA = DATA.iloc[:, np.asarray(DATA.isnull().sum()<1000, dtype=np.bool)]
DATA.isnull().sum()
# corrlations with match
corr = DATA.corrwith(DATA['match'])
corr.sort_values(ascending=False)
neg = np.abs(corr)<0.01
black_list = list(corr[neg].keys())
black_list
### We don't know potential parter's decision in real world. 

black_list += ['dec_o'] 
black_list += ['career_c', 'length'] # not interested in career and length of night event
DATA.drop(black_list, axis=1, inplace=True)
for key, val in DATA.dtypes.items():
    print('{:>10}: {:s}'.format(str(key), str(val)))
DATA.dropna(inplace=True)
DATA.isnull().sum()
X, Y = np.array(DATA.iloc[:, 1:], dtype=np.float32), np.array(DATA.iloc[:, 0], dtype=np.int16)
Y = Y*2-1
print(X.shape)
print(Y.shape)
kfold = StratifiedKFold(n_splits=5, shuffle=True) # 5-fold cross validation
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
### Use simple PLA model ###
class PLA(object):
    def __init__(self, x_dim, eta=1.0, shuffle=False, verbose=False):
        self.shuffle = shuffle
        self.verbose = verbose
        self.eta = eta
        self.Wxb = np.random.normal(0, np.sqrt(2/(x_dim+1)), size=(1,x_dim+1)) # initialize Wxb using he-normal
    def predict(self, x, pocket=False):
        X = np.append(x, [1], axis=-1)[...,np.newaxis]
        pred = np.squeeze(self.Wxb @ X)
        return -1 if pred<=0 else 1
    def train(self, Xs, Ys):
        updates = 0
        correct_cnt = 0
        i = 0
        while correct_cnt<len(Xs): # cyclic method
            if self.shuffle and correct_cnt==0:
                idx = np.random.permutation(len(Xs))
                Xs, Ys = Xs[idx], Ys[idx] # faster
                i = 0
            x, y = Xs[i], Ys[i]
            p = self.predict(x)
            if p!=y: # wrong
                self.Wxb = self.Wxb + (self.eta*y*np.append(x, [1], axis=-1))[np.newaxis]
                updates += 1
                if self.verbose:
                    print('iteration {:d}: '.format(updates), self.Wxb)
                correct_cnt = 0
            else:
                correct_cnt += 1
            i = (i+1)%len(Xs)
        return updates

class PocketPLA(PLA):
    def __init__(self, x_dim, eta=1.0, pocket_maxiter=None, shuffle=False, verbose=False):
        super(PocketPLA, self).__init__(x_dim, eta=eta, shuffle=shuffle, verbose=verbose)
        self.pocket_maxiter = pocket_maxiter
        self.Wxb_pocket = np.zeros_like(self.Wxb, dtype=np.float32) # (1, 4)
    def predict(self, x, pocket=False):
        W = self.Wxb_pocket if pocket else self.Wxb
        X = np.append(x, [1], axis=-1)[...,np.newaxis]
        pred = np.squeeze(W @ X)
        return -1 if pred<=0 else 1
    def train(self, Xs, Ys):
        updates = 0
        last_errors = np.inf
        while True:
            if self.shuffle: # precomputed random order; else: naive cyclic
                idx = np.random.permutation(len(Xs))
                Xs, Ys = Xs[idx], Ys[idx] # faster
            for x, y in zip(Xs, Ys):
                p = self.predict(x)
                if p!=y: # wrong
                    self.Wxb = self.Wxb + (self.eta*y*np.append(x, [1], axis=-1))[np.newaxis]
                    updates += 1
                    break
            errors = 0
            for x, y in zip(Xs, Ys):
                p = self.predict(x)
                errors += 1 if p!=y else 0
            if errors < last_errors:
                last_errors = errors
                self.Wxb_pocket = self.Wxb.copy()
                if self.verbose:
                    print('iteration {:d}: update pocket weights: err: {:.2f}'.format(updates, errors/len(Xs)))
            if updates>=self.pocket_maxiter or last_errors==0:
                return last_errors
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
max_iteration = 3000
eta = 0.5
accs = []
precs = []
recs = []
f1s = []

X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)
print('{:d} samples for train/val, {:d} samples for testing.'.format(len(X), len(X_test)))

for train, valid in kfold.split(X, Y):
    print('{:d} samples for training, {:d} samples for validation.'.format(len(train), len(valid)))
    X_train, Y_train = X[train], Y[train]
    X_valid, Y_valid = X[valid], Y[valid]
    X_train = scaler.fit_transform(X_train) # only fit on training set
    X_valid = scaler.transform(X_valid)
    pocket_pla = PocketPLA(X_train.shape[-1], eta=eta, pocket_maxiter=max_iteration, shuffle=True)
    pocket_pla.train(X_train, Y_train) # apply pla
    preds = np.asarray([pocket_pla.predict(x) for x in X_valid], dtype=np.int16) # prediction
    acc = accuracy_score(Y_valid, preds) # evaluations
    precision = precision_score(Y_valid, preds)
    recall = recall_score(Y_valid, preds)
    f1 = f1_score(Y_valid, preds)
    accs.append(acc)
    precs.append(precision)
    recs.append(recall)
    f1s.append(f1)
    print('acc: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}'.format(acc,precision,recall, f1))
am, pm, rm, fm = np.mean(accs), np.mean(precs), np.mean(recs), np.mean(f1s)
ad, pd_, rd, fd = np.std(accs)*2, np.std(precs)*2, np.std(recs)*2, np.std(f1s)*2
print('acc: {:.2f}+/-{:.2f}, precision: {:.2f}+/-{:.2f}, recall: {:.2f}+/-{:.2f}, f1: {:.2f}+/-{:.2f}'.format(am, ad, pm, pd_, rm, rd, fm, fd))
print('Testing set performance: ')
preds = np.asarray([pocket_pla.predict(x) for x in scaler.transform(X_test)], dtype=np.int16) # prediction
acc = accuracy_score(Y_test, preds) # evaluations
precision = precision_score(Y_test, preds)
recall = recall_score(Y_test, preds)
f1 = f1_score(Y_test, preds)
print('acc: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}'.format(acc, precision, recall, f1))
W_inspect = pocket_pla.Wxb_pocket.flatten() # Check weights of perceptron to acquire knowledge of dating? ;)
features_key = np.array(list(DATA.iloc[:, 1:]) + ['w0 (+1)'])
order = np.argsort(-W_inspect)
weights, keys = W_inspect[order], features_key[order]
for w, k in zip(weights, keys):
    print('{:>10}: {:.4f}'.format(k, w))