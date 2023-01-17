import os

import numpy as np 

import pandas as pd 



from scipy.special import softmax
path = "../input/jigsaw-multilingual-toxic-comment-classification/"

record = "../input/buffer/"



base = record + "submission-9462.csv"

monos = ["submission-it-9467.csv",

         "submission-pt-9470.csv",

         "submission-es-9467.csv",

         "submission-tr-9470.csv",

         "submission-fr-9473.csv",]



get_lang = lambda x: x.split('-')[1]
test = pd.read_csv(path + "test.csv")

dic_ids = {k:v.id for k,v in test.groupby(["lang"])}
sub = pd.read_csv(base)

for m in monos:

    res = pd.read_csv(record + m)

    ids = dic_ids[get_lang(m)]

    sub.loc[ids,"toxic"] = res.toxic[ids]



sub.head()
adj = {

    "fr":1.04,

    "es":1.06,

    "pt":.96,

    "it":.97,

    "tr":.98,

}

for l,v in adj.items():

    ids = dic_ids[l]

    sub.loc[ids,"toxic"] *= v



sub.head()
weight = lambda x: softmax(1/(1-x))



def mix_result(subs,pbs):

    toxics = np.array([df.toxic.values for df in subs])

    w = weight(np.array(pbs))

    print(["{:.3f}".format(i) for i in w])

    return toxics.T@w
sub1 = pd.read_csv(record+"submission-public-mix-9482.csv")

sub["toxic"] = mix_result([sub,sub1],[.9508,.9482])

sub.head()
sub1 = pd.read_csv(record+"submission-1st-place-9550.csv")

sub2 = pd.read_csv(record+"submission-2nd-place-9522.csv")

sub["toxic"] = mix_result([sub,sub2,sub1],[.9514,.9522,.9550])

sub.head()
sub.to_csv('submission.csv', index=False)