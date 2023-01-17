%matplotlib inline
import random
random.seed(10)
import numpy as np 
np.random.seed(10)
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import multiprocessing
cpn_cnt = multiprocessing.cpu_count()

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
print('samples: ', len(DATA.iloc[:,0]))
formula = 'match ~ amb*amb_o + attr*attr_o + fun*fun_o + intel*intel_o + sinc*sinc_o + C(shar1_1) + age*age_o + clubbing + date + dining + go_out + sports + int_corr + like*like_o + met*met_o + movies + museums + music + numdat_2 + prob*prob_o + reading + satis_2 + tv + yoga + gaming + goal + C(met)*C(met_o)' # 觀察變數
lm_model = ols(formula, DATA).fit()
aov_table = sm.stats.anova_lm(lm_model, typ=2)
significat_fators = aov_table['PR(>F)']<0.05
aov_table['significant'] = np.where(significat_fators, '*', ' ')
display(aov_table)
display(aov_table[significat_fators]) # 
X, Y = np.array(DATA.iloc[:, 1:], dtype=np.float32), np.array(DATA.iloc[:, 0], dtype=np.int16)
print(X.shape)
print(Y.shape)
random.seed(10) # 固定變數，讓結果可以重現
np.random.seed(10)
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from IPython.display import display
parameters = {'C': [0.1, 1, 10], 'max_iter': [500, 1000], 'solver': ['lbfgs', 'liblinear']}
X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)
print('{:d} samples for train/val, {:d} samples for testing.'.format(len(X), len(X_test)))

lr = GridSearchCV(LogisticRegression(), param_grid=parameters, cv=10, scoring='accuracy', n_jobs=max(1, cpn_cnt-1))
lr.fit(X, Y)
display(lr.cv_results_)
display(lr.best_params_)
print('Testing set performance: ')
preds = lr.predict(X_test) # prediction
acc = accuracy_score(Y_test, preds) # evaluations
precision = precision_score(Y_test, preds)
recall = recall_score(Y_test, preds)
f1 = f1_score(Y_test, preds)
print('acc: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}'.format(acc, precision, recall, f1))
best_lr = lr.best_estimator_
W_inspect = np.append(best_lr.coef_.flatten(), best_lr.intercept_.flatten(), axis=-1) # Check weights of perceptron to acquire knowledge of dating? ;)
features_key = np.array(list(DATA.iloc[:, 1:]) + ['w0 (+1)'])
order = np.argsort(-W_inspect)
weights, keys = W_inspect[order], features_key[order]
for w, k in zip(weights, keys):
    print('{:>10}: {:.4f}'.format(k, w))