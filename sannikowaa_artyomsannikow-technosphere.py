# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

#train['Year'] = train['Date'].apply(lambda x: int(x[:4]))

#train['Month'] = train['Date'].apply(lambda x: int(x[5:7]))

#del train['Date'], train['Promo2SinceWeek'], train['PromoInterval']

del train['Customers']

train['Year'] = train['Date'].apply(lambda x: int(x[:4]))

train['Month'] = train['Date'].apply(lambda x: int(x[5:7]))

del train['Date']

train.head()
#binarizing data

bin_day_of_week = pd.get_dummies(train['DayOfWeek'])

bin_day_of_week = bin_day_of_week.rename(columns={1: 'Mo.', 2: 'Tu.', 3: 'We.', 

                                                  4: 'Th.',5: 'Fr.',6: 'Sa.', 7: 'Su.'})

bin_month = pd.get_dummies(train['Month'])

bin_month = bin_month.rename(columns={1:"Jan.", 2:"Feb.",3:"Mar.",4:"Apr.",

                                      5:"May" ,6:"June",7:"July",8:"Aug.",

                                      9:"Sept.",10:"Oct.",11:"Nov." ,12:"Dec."})



bin_holydays = pd.get_dummies(train['StateHoliday'])

bin_holydays = bin_holydays.rename(columns={'0':"Na_Holyday",'a':"Holyday_type_a",

                                         'b':"Holyday_type_b",

                                         'c':"Holyday_type_c"})

del bin_holydays[0]
bin_month.head()
train = train.join(bin_day_of_week, lsuffix='', rsuffix='')

train = train.join(bin_month, lsuffix='', rsuffix='')

train = train.join(bin_holydays, lsuffix='', rsuffix='')

del train['DayOfWeek'] 

del train['Month']

del train['StateHoliday']

train.head()
test = pd.read_csv('../input/test.csv')

test['Year'] = test['Date'].apply(lambda x: int(x[:4]))

test['Month'] = test['Date'].apply(lambda x: int(x[5:7]))

del test['Date']

test.head()
bin_day_of_week = pd.get_dummies(test['DayOfWeek'])

bin_day_of_week = bin_day_of_week.rename(columns={1: 'Mo.', 2: 'Tu.', 3: 'We.', 

                                                  4: 'Th.',5: 'Fr.',6: 'Sa.', 7: 'Su.'})

bin_month = pd.get_dummies(test['Month'])

bin_month = bin_month.rename(columns={1:"Jan.", 2:"Feb.",3:"Mar.",4:"Apr.",

                                      5:"May" ,6:"June",7:"July",8:"Aug.",

                                      9:"Sept.",10:"Oct.",11:"Nov." ,12:"Dec."})

bin_holydays = pd.get_dummies(test['StateHoliday'])



bin_holydays = bin_holydays.rename(columns={'0':"Na_Holyday",'a':"Holyday_type_a",

                                         'b':"Holyday_type_b",

                                        'c':"Holyday_type_c"})

(sLength, tmp)= bin_holydays.shape

bin_holydays["Holyday_type_b"] = pd.Series(np.zeros(sLength), index=bin_holydays.index)

bin_holydays["Holyday_type_c"] = pd.Series(np.zeros(sLength), index=bin_holydays.index)



bin_month.head()


bin_month["Jan."] = pd.Series(np.zeros(sLength), index=bin_holydays.index)

bin_month["Feb."] = pd.Series(np.zeros(sLength), index=bin_holydays.index)

bin_month["Mar."] = pd.Series(np.zeros(sLength), index=bin_holydays.index)

bin_month["Apr."] = pd.Series(np.zeros(sLength), index=bin_holydays.index)

bin_month["May"] = pd.Series(np.zeros(sLength), index=bin_holydays.index)

bin_month["June"] = pd.Series(np.zeros(sLength), index=bin_holydays.index)

bin_month["July"] = pd.Series(np.zeros(sLength), index=bin_holydays.index)

bin_month["Oct."] = pd.Series(np.zeros(sLength), index=bin_holydays.index)

bin_month["Nov." ] = pd.Series(np.zeros(sLength), index=bin_holydays.index)

bin_month["Dec."] = pd.Series(np.zeros(sLength), index=bin_holydays.index)
test = test.join(bin_day_of_week, lsuffix='', rsuffix='')

test = test.join(bin_month, lsuffix='', rsuffix='')

test = test.join(bin_holydays, lsuffix='', rsuffix='')



del test['DayOfWeek'] 

del test['Month']

del test['StateHoliday']

test.head()
store_inf = pd.read_csv('../input/store.csv')

store_inf.head()
s_types = pd.get_dummies(store_inf['StoreType'])

s_types = s_types.rename(columns={'a': 'StoreType_a', 'b': 'StoreType_b', 'c': 'StoreType_c', 'd': 'StoreType_d'})

assortment_types = pd.get_dummies(store_inf['Assortment'])

assortment_types = assortment_types.rename(columns={'a': 'Assortment_a', 'b': 'Assortment_b', 'c': 'Assortment_c'})
del store_inf['CompetitionOpenSinceMonth']

del store_inf['CompetitionOpenSinceYear']

del store_inf['PromoInterval']
store_inf['Promo2SinceWeek'].fillna(0, inplace=True)

store_inf['Promo2SinceYear'].fillna(0, inplace=True)

store_inf.head()
s_types = pd.get_dummies(store_inf['StoreType'])

s_types = s_types.rename(columns={'a': 'StoreType_a', 'b': 'StoreType_b', 'c': 'StoreType_c', 'd': 'StoreType_d'})

assortment_types = pd.get_dummies(store_inf['Assortment'])

assortment_types = assortment_types.rename(columns={'a': 'Assortment_a', 'b': 'Assortment_b', 'c': 'Assortment_c'})
del store_inf['StoreType']

del store_inf['Assortment']

store_inf.head()
store_inf = store_inf.join(s_types, lsuffix='', rsuffix='')

store_inf = store_inf.join(assortment_types, lsuffix='', rsuffix='')

store_inf.head()
train = pd.merge(train, store_inf, on='Store')

train.head()
test = pd.merge(test, store_inf, on='Store')

test.head()
test = test.set_index('Id')

test = test.sort_index()

test.head()
c_list = list(train)

print(c_list)

tc_list = list(test)
for column_name in c_list:

    train[column_name] = train[column_name].fillna(train[column_name].mean())

for column_name in tc_list:

    test[column_name] = test[column_name].fillna(test[column_name].mean())
X = train[c_list[3:]].values

Y = train['Sales'].values

X_test = test[c_list[3:]].values
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=False)

model.fit(X,Y)
y_test = model.predict(X_test)
y_test = np.array(y_test)
answer = pd.read_csv('../input/sample_submission.csv')
answer.head()
print(answer.shape, len(y_test))
answer['Sales'] = y_test

answer.head()
answer.to_csv('rossmann_submission.csv', index=False)