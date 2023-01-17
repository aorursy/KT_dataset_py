import os

import numpy as np

import pandas as pd

from surprise import Reader, Dataset

import scipy.io

from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
ds_dir = '../input/pda2019'
train = pd.read_csv(os.path.join(ds_dir,"train-PDA2019.csv"))

test = pd.read_csv(os.path.join(ds_dir,"test-PDA2019.csv"))
train
itemIDs = train.itemID.unique()
test
train.drop('timeStamp',inplace=True,axis=1)
userIDs = train.userID.unique()
watchedList = train.groupby('userID')['itemID'].apply(list)
reader = Reader(rating_scale=(1,5))

data = Dataset.load_from_df(train, reader)

trainset = data.build_full_trainset()
def predict(user):

    pred = []

    for x in itemIDs:

        if x in watchedList[user.userID]:

            continue

        pred.append((x,model.predict(user.userID,x).est))

    pred = sorted(pred, key = lambda x: x[1], reverse=True)[:10]

    pred = [i[0] for i in pred]

    pred = ' '.join(map(str, pred)) 

    return pred
def get_top10(row):

    if row.userID in userIDs:

        pred = grouped.get_group(row.userID).sort_values(by=['prediction'],ascending=False)

        pred = pred[~pred.itemID.isin()]

        pred = ' '.join(map(str, pred.head(10).itemID.values))

        return pred

    else:

        pred = grouped.get_group(row.userID).sort_values(by=['prediction'],ascending=False)

        pred = ' '.join(map(str, pred.head(10).itemID.values))

        return pred
%%time

model = SVD(verbose=1,random_state=0).fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_SVD.csv',index=False)
%%time

model = SVDpp(verbose=1,random_state=0).fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_SVDpp.csv',index=False)
%%time

model = SlopeOne().fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_SlopeOne.csv',index=False)
%%time

model = NMF(verbose=1,random_state=0).fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_NMF.csv',index=False)
%%time

model = NormalPredictor().fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_NormalPredictor.csv',index=False)
%%time

model = KNNBaseline(verbose=1,bsl_options={'method':'als'}).fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_KNNBaseline_ALS.csv',index=False)
%%time

model = KNNBaseline(verbose=1,bsl_options={'method':'sgd'}).fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_KNNBaseline_SGD.csv',index=False)
%%time

model = KNNBasic(verbose=1).fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_KNNBasic.csv',index=False)
%%time

model = KNNWithMeans(verbose=1).fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_KNNWithMeans.csv',index=False)
%%time

model = KNNWithZScore(verbose=1).fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_KNNWithZScore.csv',index=False)
%%time

model = BaselineOnly(verbose=1,bsl_options={'method':'als'}).fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_BaselineOnly_ALS.csv',index=False)
%%time

model = BaselineOnly(verbose=1,bsl_options={'method':'sgd'}).fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_BaselineOnly_SGD.csv',index=False)
%%time

model = CoClustering(verbose=1,random_state=0).fit(trainset)
%%time

test['recommended_itemIDs'] = test.apply(predict, axis=1)

test.to_csv('sub_pda_surprise_CoClustering.csv',index=False)