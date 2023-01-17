import pandas as pd
key = pd.read_csv('../input/key_1.csv')

train = pd.read_csv('../input/train_1.csv')
key.Page[0]
key.head(3)
key = key.set_index('Id')
train.head(6)
train = train.set_index('Page')
submission = pd.read_csv('../input/sample_submission_1.csv')
submission.head(3)
train.at['2NE1_zh.wikipedia.org_all-access_spider','2016-03-01']
def idFinder(page='2NE1_zh.wikipedia.org_all-access_spider',year='2017',month='01',day='01'):

    return key.loc[key.Page == page+'_'+year+'-'+month+'-'+day].index[0]
idFinder()
def dayFinder(id_='ff8c1aade3de'):

    page_name = key.at[id_,'Page']

    return page_name[:-11], page_name[-10:-6], page_name[-5:-3], page_name[-2:]
dayFinder()
import datetime

starttime = datetime.datetime.now()

dayFinder()

train.at['3C_zh.wikipedia.org_all-access_spider','2016-07-08']

endtime = (datetime.datetime.now() - starttime) * 8703780
(endtime * 8703780).seconds
import numpy as np
starttime = datetime.datetime.now()

for i, value in enumerate(submission.Id):

    page, year, month, day = dayFinder(value)

    previous_value = 0

    #previous = train.loc[train.Page == page]['2016-'+month+'-'+day].iloc[0]

    previous = train.at[page,'2016-'+month+'-'+day]

    if not np.isnan(previous): # if is nan... set 0

        previous_value = previous

    if i%100000 == 0:

        endtime = (datetime.datetime.now() - starttime)

        print(i, len(submission.Id), endtime)

    #print(page, year, month, day, previous_value)

    submission.set_value(i,'Visits',previous_value)
submission.to_csv('2016submission.csv',index=False)
submission