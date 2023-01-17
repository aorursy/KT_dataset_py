YEAR_SHIFT = 364 #number of days in a year, use multiple of 7 to be able to capture week behavior

PERIOD = 49 #number of days for median comparison

PREDICT_PERIOD = 75 #number of days which will be predicted



#evaluation function

def smape(x, y):

    if x == y:

        return 0

    else:

        return np.abs(x-y)/(x+y)

    

#median function ignoring nans

def safe_median(s):

    return np.median([x for x in s if ~np.isnan(x)])
import pandas as pd

import numpy as np



train = pd.read_csv("../input/train_2.csv")

train = pd.melt(train[list(train.columns[-(YEAR_SHIFT + 2*PERIOD):])+['Page']], id_vars='Page', var_name='date', value_name='Visits')

train['date'] = train['date'].astype('datetime64[ns]')



LAST_TRAIN_DAY = train['date'].max()



train = train.groupby(['Page'])["Visits"].apply(lambda x: list(x))
pred_dict = {}



count = 0

scount = 0



for page, row in zip(train.index, train):

    last_month = np.array(row[-PERIOD:])

    slast_month = np.array(row[-2*PERIOD:-PERIOD])

    prev_last_month = np.array(row[PERIOD:2*PERIOD])

    prev_slast_month = np.array(row[:PERIOD])

    

    use_last_year = False

    if ~np.isnan(row[0]):

        #calculate yearly prediction error

        year_increase = np.median(slast_month)/np.median(prev_slast_month)

        year_error = np.sum(list(map(lambda x: smape(x[0], x[1]), zip(last_month, prev_last_month * year_increase))))

        

        #calculate monthly prediction error

        smedian = np.median(slast_month)

        month_error = np.sum(list(map(lambda x: smape(x, smedian), last_month)))

        

        #check if yearly prediction is better than median prediction in the previous period

        error_diff = (month_error - year_error)/PERIOD

        if error_diff > 0.1:

            scount += 1

            use_last_year = True

    

    if use_last_year:

        last_year = np.array(row[2*PERIOD:2*PERIOD+PREDICT_PERIOD])

        preds = last_year * year_increase #consider yearly increase while using the last years visits

    else:

        preds = [0]*PREDICT_PERIOD

        windows = np.array([2, 3, 4, 7, 11, 18, 29, 47])*7 #kind of fibonacci

        medians = np.zeros((len(windows), 7))

        for i in range(7):

            for k in range(len(windows)):

                array = np.array(row[-windows[k]:]).reshape(-1, 7)

                # use 3-day window. for example, Friday: [Thursday, Friday, Saturday]

                s = np.hstack([array[:, (i-1)%7], array[:, i], array[:, (i+1)%7]]).reshape(-1)

                medians[k, i] = safe_median(s)

        for i in range(PREDICT_PERIOD):

            preds[i] = safe_median(medians[:, i%7])

                

    pred_dict[page] = preds

    

    count += 1        

    if count % 1000 == 0:

        print(count, scount)



del train

print("Yearly prediction is done on the percentage:", scount/count)
test = pd.read_csv("../input/key_2.csv")

test['date'] = test.Page.apply(lambda a: a[-10:])

test['Page'] = test.Page.apply(lambda a: a[:-11])

test['date'] = test['date'].astype('datetime64[ns]')



test["date"] = test["date"].apply(lambda x: int((x - LAST_TRAIN_DAY).days) - 1)



def func(row):

    return pred_dict[row["Page"]][row["date"]]



test["Visits"] = test.apply(func, axis=1)



test.loc[test.Visits.isnull(), 'Visits'] = 0

test['Visits'] = test['Visits'].values + test['Visits'].values*0.03 # overestimating is usually better for smape

test.Visits = test.Visits.round(4)

#test[['Id','Visits']].to_csv('submission.csv', index=False)