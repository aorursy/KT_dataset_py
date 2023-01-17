import fastai; fastai.__version__
%reload_ext autoreload

%autoreload 2

%matplotlib inline



import pandas as pd

import re

import numpy as np

from fastai import * # notebook was run with fastai 1.0.51

from fastai.text import *

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from sklearn.metrics import classification_report



pd.set_option('display.max_colwidth', -1)



# by setting a random seed number, we'll ensure that when doing language model, same training-validation split is used.

np.random.seed(42) 

path = Path('../input')
df = pd.read_csv(path/"trainset.csv")

df_test = pd.read_csv(path/"testset_w_refs.csv")

df_dev = pd.read_csv(path/"devset.csv")

print(df.shape)

print(df_dev.shape)

print(df_test.shape)

df.head()
import unicodedata

def strip_accents(s):

   return ''.join(c for c in unicodedata.normalize('NFD', s)

                  if unicodedata.category(c) != 'Mn')
def delexicalize(attribute,value,new_value,new_row,row):

    new_row["ref"] = re.sub(value,new_value,new_row["ref"])

    new_row["ref"] = re.sub(value.lower(),new_value.lower(),new_row["ref"])

    new_row["ref"] = re.sub(strip_accents(value.lower()),new_value.lower(),new_row["ref"])

    new_row["ref"] = re.sub(strip_accents(value),new_value,new_row["ref"])

    value0=value[0]+value[1:].lower()

    new_row["ref"] = re.sub(value0,new_value,new_row["ref"])

    new_row["ref"] = re.sub(strip_accents(value0),new_value,new_row["ref"])

    value0=value[0].lower()+value[1:]

    new_row["ref"] = re.sub(value0,new_value,new_row["ref"])

    new_row["ref"] = re.sub(strip_accents(value0),new_value,new_row["ref"])

    return new_row
from nltk import sent_tokenize

def process_features(df):

    rows = []

    for i,row in df.iterrows():

        row0 = row.to_dict()

        row0["old_mr"] = row0["mr"]

        row0["mr"] = re.sub("  +"," ",row0["mr"])

        name = re.sub(r"^.*name\[([^\]]+)\].*$",r"\1",row0["mr"].strip())

        near = re.sub(r"^.*near\[([^\]]+)\].*$",r"\1",row0["mr"].strip())

        name = re.sub("  +"," ",name)

        near = re.sub("  +"," ",near)

        row0 = delexicalize("name",name,"XXX",row0,row)

        row0 = delexicalize("near",near,"YYY",row0,row)

        row0["mr"] = re.sub(r"name\[[^\]]+\](, *| *$)","",row0["mr"].strip())

        row0["mr"] = re.sub(r"near\[[^\]]+\](, *| *$)",r"near[yes]\1",row0["mr"].strip())

        row0["mr"] = re.sub(r", *$","",row0["mr"].strip())

        row0["mr"] = re.sub(r" *, *",",",row0["mr"].strip())

        row0["mr"] = row0["mr"].strip()

        if row["ref"]==row0["ref"]:

            continue

        rows.append(row0)

    return pd.DataFrame(rows)
df=process_features(df)

df_dev=process_features(df_dev)

df_test=process_features(df_test)

print(df.shape)

print(df_dev.shape)

print(df_test.shape)

df.head()
from nltk.tokenize import sent_tokenize

rows=[]

for i,row in df.iterrows():

    mrs = row["mr"].split(",")

    sents = sent_tokenize(row["ref"])

    for mr in mrs:

        row[mr]=1

        if not mr.startswith("near") and not mr.startswith("name"):

            feature_name = re.sub(r"^([^\[]+)\[.*$",r"\1",mr.strip())

            row[feature_name]=1

    row["num_mrs"]=len(mrs)

    row["num_sents"]=len(sents)

    rows.append(row)
df_stats = pd.DataFrame(rows)

df_stats = df_stats.fillna(0)

df_stats.head(5)
stats = {}

df_sample = df_stats

rows =[]

for col in df_sample.columns:

    row={}

    if df_stats[col].dtype == np.float64:

        if "[" not in col:

            row["feature"]="_"+col

        else:

            row["feature"]=col

        row["num"]=df_sample[col].sum()

        row["mean"]=df_sample[col].mean()

        row["std"]=df_sample[col].std()

        rows.append(row)

    elif df_sample[col].dtype == np.int64:

        row["feature"]="__"+col

        row["num_1"] = (df_sample.loc[df_sample[col]==1]).shape[0]

        row["num"]=df_sample[col].sum()

        row["mean"]=df_sample[col].mean()

        row["min"]=df_sample[col].min()

        row["max"]=df_sample[col].max()

        row["std"]=df_sample[col].std()

        row["median"]=df_sample[col].median()

        rows.append(row)

df_stats0 = pd.DataFrame(rows)

df_stats0 = df_stats0.sort_values(by="feature")

df_stats0
df_all = pd.concat([df, df_dev,df_test], ignore_index=True)

df_all.shape
bs = 56
df_all.sample(5)
data_lm = (TextList.from_df(df_all, ".", cols='ref')

                .split_by_rand_pct(0.1)

                .label_for_lm()

                .databunch(bs=bs))
data_lm.export('data_lm.pkl')
data_lm.show_batch()
learn_lm = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=1e-7)
learn_lm.freeze()
learn_lm.lr_find()

learn_lm.recorder.plot()
learn_lm.fit_one_cycle(1, 5e-02, moms=(0.8,0.7))
learn_lm.unfreeze()
learn_lm.lr_find()

learn_lm.recorder.plot(suggestion=True)
learn_lm.fit_one_cycle(4, 1e-03, moms=(0.8,0.7),wd=0.3)
learn_lm.recorder.plot_losses()
learn_lm.save('fine_tuned')

learn_lm.save_encoder('fine_tuned_enc')
TEXT = "Near"

N_WORDS = 50

N_TEXTS = 2
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_TEXTS)))
with open('vocab.pkl', 'wb') as f:

    pickle.dump(data_lm.vocab, f)
bs = 56
def precision(log_preds, targs, thresh=0.5, epsilon=1e-8):

    pred_pos = (log_preds > thresh).float()

    tpos = torch.mul((targs == pred_pos).float(), targs.float())

    return (tpos.sum()/(pred_pos.sum() + epsilon))#.item()
def recall(log_preds, targs, thresh=0.5, epsilon=1e-8):

    pred_pos = (log_preds > thresh).float()

    tpos = torch.mul((targs == pred_pos).float(), targs.float())

    return (tpos.sum()/(targs.sum() + epsilon))
data_clas = TextClasDataBunch.from_df(".", train_df=df, valid_df=df_dev, 

                                  vocab=data_lm.vocab, 

                                  text_cols='ref', 

                                  label_cols='mr',

                                  label_delim=',',

                                  bs=bs)
data_clas.show_batch()
print(len(data_clas.valid_ds.classes))

data_clas.valid_ds.classes
learn = text_classifier_learner(data_clas, arch=AWD_LSTM,drop_mult=1e-7)

learn.metrics = [accuracy_thresh, precision, recall]

learn.load_encoder('fine_tuned_enc')
learn.freeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, 6.92E-02, moms=(0.8,0.7),wd=1e-6)
learn.recorder.plot_losses()
learn.save("stage1")
learn = text_classifier_learner(data_clas, arch=AWD_LSTM,drop_mult=1e-7)

learn = learn.load("stage1")

learn.metrics = [accuracy_thresh, precision, recall]
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, slice(1E-03/(2.6**4),1E-03), moms=(0.8,0.7), wd=0.5)
learn.save("classifier_model",return_path=True, with_opt=True)
from IPython.display import FileLinks

FileLinks('.') # input argument is specified folder
learn.recorder.plot_losses()
with open('vocab.pkl', 'rb') as fp:

    vocab = pickle.load(fp)
def make_predictions(path,voc,model_name,df_train,df_valid,vocab,bs):

    data_clas = TextClasDataBunch.from_df(path, train_df=df_train, valid_df=df_valid, 

                                          vocab=voc,

                                      text_cols='ref', 

                                      label_cols='mr',

                                      label_delim=',',

                                      bs=bs)

    learn = text_classifier_learner(data_clas, arch=AWD_LSTM)

    learn.load(model_name)

    learn.data = data_clas

    preds, y = learn.get_preds(ordered=True)

    return learn,preds,y
path="."
learn_train,preds_train,y_train = make_predictions(path,vocab,"classifier_model",df,df,None,bs)

learn_valid,preds_valid,y_valid = make_predictions(path,vocab,"classifier_model",df,df_dev,None,bs)

learn_valid,preds_test,y_test = make_predictions(path,vocab,"classifier_model",df,df_test,None,bs)
f1_train = f1_score(y_train, preds_train>0.5, average='micro')

f1_valid = f1_score(y_valid, preds_valid>0.5, average='micro')

f1_test = f1_score(y_test, preds_test>0.5, average='micro')

f1_train,f1_valid,f1_test
y_true_train = y_train.numpy()

scores_train = preds_train.numpy()

report = classification_report(y_true_train, scores_train>0.5, target_names=data_clas.valid_ds.classes)

print(report)
y_true_valid = y_valid.numpy()

scores_valid = preds_valid.numpy()

report = classification_report(y_true_valid, scores_valid>0.5, target_names=data_clas.valid_ds.classes)

print(report)
y_true_test = y_test.numpy()

scores_test = preds_test.numpy()

report = classification_report(y_true_test, scores_test>0.5, target_names=data_clas.valid_ds.classes)

print(report)
learn,preds,y = make_predictions(path,vocab,"classifier_model",df,df,None,bs)
f1_score(y, preds>0.5, average='micro')
def set_row_metrics(row,true_mrs,predicted_mrs):

        tp=0

        fp=0

        tn=0

        fn=0

        for mr in predicted_mrs:

            if mr in true_mrs:

                tp+=1

            else:

                fp+=1

        for mr in true_mrs:

            if mr not in predicted_mrs:

                fn+=1

            else:

                tn+=1

        row["tp"]=tp

        row["fp"]=fp

        row["fn"]=fn

        row["tn"]=tn

        row["precision"]=0

        row["recall"]=0

        row["fscore"]=0

        if tp+fp>0:

            row["precision"]=float(tp)/(tp+fp)

        if tp+fn>0:

            row["recall"]=float(tp)/(tp+fn)

        if row["precision"]+row["recall"]>0:

            row["fscore"]= 2*((row["precision"]*row["recall"])/(row["precision"]+row["recall"]))

        return row
def set_labels(df,preds,classes):

    preds_true = (preds>0.5)

    counter=0

    rows=[]

    for i,row in df.iterrows():

        row_preds = preds[counter]

        indices = [j for j in range(len(preds_true[counter])) if preds_true[counter][j]==True]

        row_labels = [classes[j] for j in indices]

        row["mr_predict"]=",".join(sorted(row_labels))

        predicted_mrs = row["mr"].split(",")

        row["mr"]=",".join(sorted(predicted_mrs))

        row = set_row_metrics(row,row_labels,predicted_mrs)

        rows.append(row)

        counter=counter+1

    return pd.DataFrame(rows)
learn.data.valid_ds.classes
df_preds = set_labels(df,preds,learn.data.valid_ds.classes)
df_preds[df_preds["fscore"]==1].shape[0]/df_preds.shape[0]
df_preds = df_preds.sort_values(by=["fscore","precision","recall"],ascending=True)
df_preds.to_csv("training_preds.csv",sep="\t",index=False)
df_preds[df_preds["fscore"]<1][df_preds["fscore"]>0].head(5)
def convert_to_dict(features):

    d = {}

    features = features.split(",")

    for f in features:

        name = (re.sub(r"^([^\[]+)\[([^\]]+)\]$",r"\1",f)).strip()

        value = (re.sub(r"^([^\[]+)\[([^\]]+)\]$",r"\2",f)).strip()

        if name not in d.keys():

            d[name]=set()

        d[name].add(value)

    return d
rows=[]

for i,row in df_preds.iterrows():

    row0=row

    mrs = convert_to_dict(row["mr"])

    mrs_predict = convert_to_dict(row["mr_predict"])

    missing=0

    mismatch=0

    added=0

    for feature in mrs.keys():

        if feature not in mrs_predict.keys():

            missing+=1

        else:

            for value in mrs[feature]:

                if value not in mrs_predict[feature]:

                    mismatch+=1

                    break

    for feature in mrs_predict.keys():

        if feature not in mrs.keys():

            added+=1

    row0["missing"]=missing

    row0["mismatch"]=mismatch

    row0["added"]=added

    rows.append(row0)

    

pd_preds0 = pd.DataFrame(rows)

pd_preds0.sample(5)