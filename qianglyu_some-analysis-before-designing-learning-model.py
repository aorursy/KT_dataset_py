import pandas as pd

dfTrain = pd.read_csv('../input/clicks_train.csv')
print('there are ', len(dfTrain), ' rows in the original train set.')
gpTrain = dfTrain.groupby('display_id')
print('there are ', len(gpTrain), ' displays in the train set.')

dfTest = pd.read_csv('../input/clicks_test.csv')
print('there are ', len(dfTest), ' rows in the original test set.')
gpTest = dfTest.groupby('display_id')
print('there are ', len(gpTest), ' displays in the test set.')
countTestInst = 0
for disp in gpTest:
    countTestInst += len(disp[1]) - 1
print('there are ', countTestInst, ' test instances all together.')
import numpy as np
# the original platform fields mixed with char datatype, cast to int64
dfEvent = pd.read_csv('../input/events.csv', index_col='display_id')
print(dfEvent.dtypes)
dfEvent.head(5)
print(len(dfEvent.index.unique()))
print(len(dfEvent))
print('the number of docs defined in events ', len(dfEvent['document_id'].unique()))
print('the number of uuid defined in events ', len(dfEvent['uuid'].unique()))
dfDocMeta = pd.read_csv('../input/documents_meta.csv')
print(dfDocMeta.dtypes)
print('the unique doc_id number ', len(dfDocMeta['document_id'].unique()))
print('the rows of doc_meta ', len(dfDocMeta))
# if the above two are equal, we are safe to make doc_id as index
dfDocMeta.set_index(dfDocMeta.document_id, inplace=True)
dfDocMeta.head(5)
dfDocTopic = pd.read_csv('../input/documents_topics.csv')
print(dfDocTopic.dtypes)
dfDocTopic.head(5)
print('the number of unique topic_id', len(dfDocTopic['topic_id'].unique()))
print('the number of unique doc_id', len(dfDocTopic['document_id'].unique()))
gpDocTopic = dfDocTopic.groupby('document_id')
print(gpDocTopic['topic_id'].count().describe())
dfDocEnt = pd.read_csv('../input/documents_entities.csv')
print(dfDocEnt.dtypes)
dfDocEnt.head(5)
gpDocEnt = dfDocEnt.groupby('document_id')
print(gpDocEnt['entity_id'].count().describe())
dfDocCat = pd.read_csv('../input/documents_categories.csv')
print(dfDocCat.dtypes)
print(dfDocCat.head(5))
gpDocCat = dfDocCat.groupby('document_id')
print(gpDocCat['category_id'].count().describe())
dfPageView = pd.read_csv('../input/page_views_sample.csv')
print(dfPageView.dtypes)
dfPageView.head(5)
for i in range(5):
    pv = dfPageView.iloc[i]
    print(pv[0])
    print(dfEvent[(dfEvent.uuid==pv[0]) & (dfEvent.document_id==pv[1])])
    #print(timestamp,platform,geo)
countUnmatched = 0
countMatched = 0
count = 0
for i in range(len(dfEvent)):
    ev = dfEvent.iloc[i]
    matchedPV = dfPageView[(dfPageView.uuid==ev[0]) & (dfPageView.document_id==ev[1]) & (dfPageView.timestamp==ev[2]) & (dfPageView.platform==ev[3]) & (dfPageView.geo_location==ev[4])]
    if len(matchedPV) == 0:
        # this should not happen for page_view.csv, the complete set
        countUnmatched += 1
        continue
    elif len(matchedPV) > 1:
        print('multiple matched found in pageview!')
        print(i, ev)
        print(matchedPV.head(10))
        break
    countMatched += 1
    #print(i,ev)
    #print(matchedPV.head(0))
    if countMatched == 5 :
        break
print('we have found matched pv ', countMatched)
print('and unmatched pv in sample_pageview ', countUnmatched, ' in the first ',i+1,' rows of events.csv.')
dfPromotedCont = pd.read_csv('../input/promoted_content.csv')
print(dfPromotedCont.dtypes)
print(dfPromotedCont.head(5))
gpAdid = dfPromotedCont.groupby('ad_id')
print('group number of ad_id ', len(gpAdid))
print('unique number of promoted_content ', len(dfPromotedCont['ad_id'].unique()))
dfPromotedCont = pd.read_csv('../input/promoted_content.csv', index_col=0)
print(dfPromotedCont.dtypes)
print(dfPromotedCont.head(5))