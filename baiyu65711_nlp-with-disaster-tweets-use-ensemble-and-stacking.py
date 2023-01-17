# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as ms

from tqdm import tqdm 

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
DATA_PATE = '/kaggle/input/nlp-getting-started/'

SEED = 23

train = pd.read_csv(os.path.join(DATA_PATE,"train.csv"))

test = pd.read_csv(os.path.join(DATA_PATE,"test.csv"))



print('train.shape:',train.shape)

print('test.shepe:',test.shape)



print('train.columns:',train.columns)

print('test.columns:',test.columns)
print(train.head())
print("keyword value count")

print('train:')

print(train.keyword.value_counts())

print('test:')

print(test.keyword.value_counts())



print('1:Disaster:',len(train[train.target==1]))

print('0:Not Disaster:',len(train[train.target==0]))


countplt = sns.countplot(x = 'target', data = train, hue = train['target'])

countplt.set_xticklabels(['0: Not Disaster (4342)', '1: Disaster (3271)'])

plt.title('target')

plt.show()
print('train data Count NaN:')

print(train.isnull().sum())

ms.bar(train)

ms.matrix(train)

plt.show()

print('test data Count NaN:')

print(test.isnull().sum())

ms.bar(test)

ms.matrix(test)

plt.show()
train['target_mean'] = train.groupby('keyword')['target'].transform('mean')



fig = plt.figure(figsize=(10, 96), dpi=100)



sns.countplot(y=train.sort_values(by='target_mean', ascending=False)['keyword'],

              hue=train.sort_values(by='target_mean', ascending=False)['target'])



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=12)

# plt.legend(loc=1)

plt.title('Target Distribution in Keywords')



plt.show()



train.drop(columns=['target_mean'], inplace=True)
#把 keyword 和text合并

train.loc[train['keyword'].notnull(), 'text'] = train['keyword'] + ' ' + train['text']

test.loc[test['keyword'].notnull(), 'text'] = test['keyword'] + ' ' + test['text']



# view

train.head()
import string

from nltk.corpus import stopwords

# stop_words = set(stopwords.words('english'))

# print(stop_words)

stop_words = {}.fromkeys([ line.rstrip() for line in open('/kaggle/input/stopwords/stopwords.txt')])

# print(list(stopwords.keys()))



# NLTK Tweet Tokenizer for now

from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer(strip_handles=True)



corpus = []

#去掉特殊字符

# clean up text

def clean_text(text):

    """

    Copied from other notebooks

    """

    # expand acronyms

    

    # special characters

    text = re.sub(r"\x89Û_", "", text)

    text = re.sub(r"\x89ÛÒ", "", text)

    text = re.sub(r"\x89ÛÓ", "", text)

    text = re.sub(r"\x89ÛÏWhen", "When", text)

    text = re.sub(r"\x89ÛÏ", "", text)

    text = re.sub(r"China\x89Ûªs", "China's", text)

    text = re.sub(r"let\x89Ûªs", "let's", text)

    text = re.sub(r"\x89Û÷", "", text)

    text = re.sub(r"\x89Ûª", "", text)

    text = re.sub(r"\x89Û\x9d", "", text)

    text = re.sub(r"å_", "", text)

    text = re.sub(r"\x89Û¢", "", text)

    text = re.sub(r"\x89Û¢åÊ", "", text)

    text = re.sub(r"fromåÊwounds", "from wounds", text)

    text = re.sub(r"åÊ", "", text)

    text = re.sub(r"åÈ", "", text)

    text = re.sub(r"JapÌ_n", "Japan", text)    

    text = re.sub(r"Ì©", "e", text)

    text = re.sub(r"å¨", "", text)

    text = re.sub(r"SuruÌ¤", "Suruc", text)

    text = re.sub(r"åÇ", "", text)

    text = re.sub(r"å£3million", "3 million", text)

    text = re.sub(r"åÀ", "", text)

    

    # emojis

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

    

    

    """

    Our Stuff

    """

    # remove numbers

    text = re.sub(r'[0-9]', '', text)

    

    # remove punctuation and special chars (keep '!')

    for p in string.punctuation.replace('!', ''):

        text = text.replace(p, '')

        

    # remove urls

    text = re.sub(r'http\S+', '', text)

    

    # tokenize

    text = tknzr.tokenize(text)

    

    # remove stopwords

    text = [w.lower() for w in text if not w in stop_words]

    corpus.append(text)

    

    # join back

    text = ' '.join(text)

    

    return text
%%time

train['text'] = train['text'].apply(lambda s: clean_text(s))

test['text'] = test['text'].apply(lambda s: clean_text(s))



# see some cleaned data

train.sample(10)
train[train['keyword'].notnull()].head(20)
train.drop(['id','keyword','location'],axis = 1,inplace=True)

print(train.head())

print(test.head())
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression



vectorizer = CountVectorizer()

x_train_vectorized = vectorizer.fit_transform(train['text'])



# print vocabulary

print(vectorizer.get_feature_names()[2500:2600])

test.head()

test.drop(['id','keyword','location'],axis = 1,inplace=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.text,train.target,test_size = 0.2, random_state = SEED)

X_train = vectorizer.transform(X_train)

X_test = vectorizer.transform(X_test)
import itertools

def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
LR_model = LogisticRegression()

LR_model = LR_model.fit(X_train, y_train)

lr_y_pred = LR_model.predict(X_test)

lr_cnf_matrix = confusion_matrix(y_test,lr_y_pred)



print("LogisticRegression Recall metric in the testing dataset: ", lr_cnf_matrix[1,1]/(lr_cnf_matrix[1,0]+lr_cnf_matrix[1,1]))



print("LogisticRegression accuracy metric in the testing dataset: ", (lr_cnf_matrix[1,1]+lr_cnf_matrix[0,0])/(lr_cnf_matrix[0,0]+lr_cnf_matrix[1,1]+lr_cnf_matrix[1,0]+lr_cnf_matrix[0,1]))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(lr_cnf_matrix

                      , classes=class_names

                      , title='LogisticRegression Confusion matrix')

plt.show()
x_train, x_test, y_train, y_test = train_test_split(train.text,train.target,test_size = 0.2, random_state = SEED)

x_train = vectorizer.transform(x_train)

x_test = vectorizer.transform(x_test)
from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.kernel_approximation import Nystroem, RBFSampler

from sklearn.pipeline import make_pipeline



import matplotlib.colors as colors

import matplotlib.cm as cmx



# 评估

from sklearn.metrics import roc_auc_score

# 画各个model的相关率

from mlens.visualization import corrmat

# 画roc曲线

from sklearn.metrics import roc_curve

def get_models():

    nb = MultinomialNB()

    svc = SVC(C=100, probability=True)

    knn = KNeighborsClassifier(n_neighbors=300)

    lr = LogisticRegression(C=1, random_state=SEED)

    nn = MLPClassifier((200, 10), early_stopping=False, random_state=SEED)

    gb = GradientBoostingClassifier(n_estimators=200, random_state=SEED)

    rf = RandomForestClassifier(n_estimators=40, max_depth=10, random_state=SEED)



    models = {

        'svm': svc,

        'knn': knn,

        'naive_bayes': nb,

        'mlp-nn': nn,

        'random forest': rf,

        'gbm': gb,

        'logistic': lr

    }



    return models





def train_predict(model_list):

    P = np.zeros((y_test.shape[0], len(model_list)))

    P = pd.DataFrame(P)



    print("让每个模型开始训练")

    cols = list()

    for i, (name, m) in enumerate(model_list.items()):

        print("%s ..." % name, end=" ", flush=False)

#         if(name == "naive_bayes"): #有的模型肯要toarray

#             m.fit(x_train.toarray(), y_train)

#             P.iloc[:, i] = m.predict_proba(x_test.toarray())[:, 1]

#         else:

        m.fit(x_train, y_train)

        P.iloc[:, i] = m.predict_proba(x_test)[:, 1]

        cols.append(name)

        print("down")

        print()

    P.columns = cols

    print("train全部完成")

    return P





def score_models(P, y):

    # y.index = range(len(y))

    print("所有model的roc_auc.")

    for m in P.columns:

        # print("m:::::::":m)

        # print(P.loc[:m])

        score = roc_auc_score(y, P.loc[:, m], multi_class="ovr", average="weighted")

        print("%-26s: %.4f" % (m, score))

    print("score全部完成")





def plot_roc_curve(y_test, P_base_learners, P_ensemble, labels, end_label):

    # 画roc曲线

    plt.figure(figsize=(20,10))

    plt.plot([0, 1], [0, 1], 'k--')

    # plt.cm.

    values = range(P_base_learners.shape[1] + 1)

    jet = cm = plt.get_cmap('jet')

    cNorm = colors.Normalize(vmin=0, vmax=values[-1])

    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    # cm = [plt.cm.get_cmap("rainbow").colors[i] for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]



    for i in range(P_base_learners.shape[1]):

        colorVal = scalarMap.to_rgba(values[i + 1])

        p = P_base_learners[:, i]

        fpr, tpr, _ = roc_curve(y_test, p)

        plt.plot(fpr, tpr, label=labels[i], c=colorVal)



    fpr, tpr, _ = roc_curve(y_test, P_ensemble)

    # fig, ax = plt.subplots(figsize=(20, 10))



    plt.plot(fpr, tpr, label=end_label, c=scalarMap.to_rgba(values[i + 1]))



    plt.xlabel('FP rate')

    plt.ylabel("TP rate")

    plt.title("ROC curve")

    plt.legend(frameon=False)

#     plt.figure(figsize=(10,10))

    plt.show()





%%time

models = get_models()

P = train_predict(models)

score_models(P, y_test)

print("集成后 ROC_AUC score：%4f" % roc_auc_score(y_test, P.mean(axis=1)))

# corrmat(P.corr(), inflate=False, figsize=(20, 10))

corrmat(P.corr(), inflate=False)

plt.show()

plot_roc_curve(y_test, P.values, P.mean(axis=1), list(P.columns), "ensemble")

p = P.apply(lambda x: 1*(x>=0.5).value_counts(normalize=True))

p.index=['Not Disaster', 'Disaster']

p.loc['Disaster',:].sort_values().plot(kind='bar')

disaster_percent = 3271/(3271+4342)

plt.axhline(disaster_percent,color='k',linewidth=0.5)

plt.text(0.,disaster_percent,'True Disaster percent value')

plt.show()



p.loc['Not Disaster',:].sort_values().plot(kind='bar')

not_disaster_percent = 4342/(3271+4342)

plt.axhline(not_disaster_percent,color='k',linewidth=0.5)

plt.text(0.,not_disaster_percent,'True Not Disaster percent value')

plt.show()
include = [c for c in P.columns if c not in ['knn','random forest','gbm']]

print("ensemble ROC-AUC score: %.5f" %roc_auc_score(y_test, P.mean(axis=1)))

print("truncate ensemble ROC-AUC score: %.5f" %roc_auc_score(y_test, P.loc[:, include].mean(axis=1)))
def get_better_models():

    nb = MultinomialNB()

    svc = SVC(C=100, probability=True)

#     knn = KNeighborsClassifier(n_neighbors=300)

    lr = LogisticRegression(C=1, random_state=SEED)

    nn = MLPClassifier((200, 10), early_stopping=False, random_state=SEED)

#     gb = GradientBoostingClassifier(n_estimators=200, random_state=SEED)

#     rf = RandomForestClassifier(n_estimators=40, max_depth=10, random_state=SEED)



    models = {

        'svm': svc,

#         'knn': knn,

        'naive_bayes': nb,

        'mlp-nn': nn,

#         'random forest': rf,

#         'gbm': gb,

        'logistic': lr

    }



    return models

base_learners = get_better_models()
# meta_learner = GradientBoostingClassifier(

#     n_estimators=3000,

#     random_state=SEED,

#     loss="exponential",

#     max_features=4,

#     max_depth=20,

# #     subsample=0.5,

#     learning_rate=0.005

# )

meta_learner = lr = LogisticRegression(C=1, random_state=SEED)
xtrain_base,xpred_base,ytrain_base,ypred_base=train_test_split(

    x_train,y_train,train_size=0.5,random_state=SEED)

# xtrain_base,xpred_base,ytrain_base,ypred_base=train_test_split(

#     train.text,train.target,train_size=0.5,random_state=SEED)

# xtrain_base = vectorizer.transform(xtrain_base)

# xpred_base = vectorizer.transform(xpred_base)
def train_base_learners(base_learners,inp,out,varbose=True):

    if varbose: print("Fitting models.")

    for i, (name,m) in enumerate(base_learners.items()):

        if varbose: print("%s..." % name,end=" ",flush=False)

        m.fit(inp,out)

        if varbose: print("done.")
train_base_learners(base_learners, xtrain_base,ytrain_base)
def predict_base_learners(pred_base_learners,inp,varbose=True):

    P = np.zeros((inp.shape[0],len(pred_base_learners)))

    

    if varbose: print("Base Learner Predictions.")

    for i, (name,m) in enumerate(pred_base_learners.items()):

        if varbose: print("%s..." % name,end=" ",flush=False)

        p = m.predict_proba(inp)

        P[:, i] = p[:, 1]

        if varbose: print("done.")

    return P
P_base = predict_base_learners(base_learners, xpred_base)
P_base.shape
P_base[0]
meta_learner.fit(P_base, ypred_base)
def ensemble_predict(base_learners, meta_learner, inp):

    P_pred = predict_base_learners(base_learners,inp)

    return P_pred, meta_learner.predict_proba(P_pred)[:, 1], meta_learner.predict(P_pred)
P_pred, predict_proba ,predict= ensemble_predict(base_learners,meta_learner,x_test)

print("Ensemble ROC-AUC score: %.4f"% roc_auc_score(y_test,predict_proba))

print(predict)
P_pred = predict_base_learners(base_learners,vectorizer.transform(test.text))

predict = meta_learner.predict(P_pred)
print(len(predict))

print(len(test))

predict_data = pd.DataFrame(predict)

predict_data.info()
sample_submission = pd.read_csv(os.path.join(DATA_PATE,"sample_submission.csv"))

sample_submission['target']=predict_data

print(sample_submission.head())

sample_submission.to_csv('submission_data_stacking_2.csv',index=False)

print('submission_data saved')
from mlens.ensemble import SuperLearner



sl = SuperLearner(

    folds=10,

    random_state=SEED,

    verbose=2,

    backend="multiprocessing"

)



sl.add(list(base_learners.values()),proba=True)

sl.add_meta(meta_learner,proba=True)



sl.fit(x_train,y_train)



p_sl = sl.predict_proba(x_test)

print("Ensemble ROC-AUC score: %.4f"% roc_auc_score(y_test,p_sl[:,1]))

print("Ensemble ROC-AUC score: %.4f"% roc_auc_score(y_test,p_sl[:,1]))


prodect_mlens = sl.predict(vectorizer.transform(test.text))

predict_data_mlens = pd.DataFrame(prodect_mlens)

predict_data_mlens.info()

predict_data_mlens.loc[:,1]

predict_data_f = []

for s in predict_data_mlens.loc[:,1]:

    if s >=0.5:

        predict_data_f.append(1)

    else:

        predict_data_f.append(0)
sample_submission = pd.read_csv(os.path.join(DATA_PATE,"sample_submission.csv"))

sample_submission['target']=predict_data_f

print(sample_submission.head())

sample_submission.to_csv('submission_data_mlens.csv',index=False)

print('submission_data saved')
# #用BernoulliNB，效果好

# bnb = BernoulliNB()

# bnb.fit(vectorizer.transform(train.text),train.target)

# predict = bnb.predict(vectorizer.transform(test.text))
# len(test)
# len(predict)
# predict_data = pd.DataFrame(predict)

# predict_data.info()
# predict_data.head()
# sample_submission = pd.read_csv(os.path.join(DATA_PATE,"sample_submission.csv"))

# sample_submission['target']=predict_data

# sample_submission.head()
# sample_submission.info()

# sample_submission.to_csv('submission_data_1.csv',index=False)