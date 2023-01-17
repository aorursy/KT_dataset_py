#All kinds of import

from pandas import read_csv, DataFrame

from os.path import isfile

from math import sqrt

from numpy import float64

from sklearn.utils.extmath import weighted_mode

import numpy as np

import random

import pandas
#Reading input data

train_input = read_csv('./train.csv')

sample_num = 300000 #

total_weeks = 157

total_days = total_weeks * 7

weeks = 50

days = weeks * 7

#train_input = train_input.sample(frac=1).reset_index(drop=True) #don't forget to delete before making mistakes
index  = [int((i-1)%7+1) for i in range(0, total_days * 7)]
#Splitting data and such

pd_visits = train_input['visits'].str.lstrip().str.split(' ')

visits = pd_visits.apply(pandas.to_numeric)
from tqdm import tqdm_notebook as tqdm

with tqdm(total=300000) as pbar:

    

    def to_numdays(row):

        global pbar

        inttmp = np.array(pandas.to_numeric(row))

        tmp = [index[i] for i in inttmp]

        pbar.update(1)

        return tmp



    num_days = visits.apply(to_numdays)
vec_weight = []

for i in num_days:

    array = []

    w = 1

    for j in range(len(i)):

        array.append(w)

        w = w + 0.1

    vec_weight.append(array)

#form results

vec_result = []

from tqdm import tqdm_notebook as tqdm

with tqdm(total=300000) as pbar:

    for i in range(len(num_days)):

        a, b = weighted_mode(num_days[i], vec_weight[i])

        days = np.array(num_days[i])

        totalvisits = (days != 0).sum()

        vec_result.append([(days==1).sum()/totalvisits,(days==2).sum()/totalvisits,(days==3).sum()/totalvisits,(days==4).sum()/totalvisits,(days==5).sum()/totalvisits,(days==6).sum()/totalvisits,(days==7).sum()/totalvisits,int(a[0])])

        pbar.update(1)
print(vec_result[0])
array_size = 15

array_amount = 4

num_days_train = num_days[:290000]

first = num_days.copy()

with tqdm(total=290000) as pbar:

    random.seed(0)

    for i in range(290000):

        row = num_days[i]

        ret = ()

        for j in range(array_amount):

            start = random.randrange(0, len(row) - array_size)

            a = vec_result[i]

            b = np.array(row)[start:start+array_size]

            ret = ret + ( np.concatenate([a,b]),)

        pbar.update(1)

        first[i] = ret

    #random.seed(0)

    #def first_fa(row):

    #    global array_size

    #    global array_amount

    #    global pbar



    #    ret = ()

    #    for i in range(array_amount):

    #        start = random.randrange(0, len(row) - array_size)

    #        ret = ret + (np.array(row)[start:start+array_size],)

    #    pbar.update(1)

    #    return ret

#

    #first = num_days.apply(first_fa)

#row = num_days[0]

#first = [(np.array(row)[i:i+features_amount]) for i in range(len(row) - features_amount + 1)]

#print(first)


k = 0

train = []

for i in tqdm(range(290000)):

    train.extend(first[k])

    k = k+1

    
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(categories='auto')

X = [np.zeros(array_size) + i+1for i in range(7)]

enc.fit(X)

df_train = DataFrame(train)

y_train = df_train.pop(array_size)

#print(np.array(df_train.iloc[:,:7]))

#x_train =np.concatenate([np.array(df_train.iloc[:,:7]), enc.transform(df_train.iloc[:,7:]).toarray()])

x_train = enc.transform(df_train.iloc[:,7:]).toarray()

print(np.array(df_train.iloc[:,:7]).shape)

print(x_train.shape)

x_train = np.hstack((np.array(df_train.iloc[:,:7]), x_train))

print(x_train[0].size)
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train, y_train)


num_days_cut = num_days[290000:300000]

tst = num_days_cut.copy()

with tqdm(total=10000) as pbar:

    for i in range(10000):

        row = num_days_cut[i+290000]

        ret = ()

        a = vec_result[i+290000]

        b = np.array(row)[len(row)-array_size:len(row)]

        ret = ret + ( np.concatenate([a,b]),)

        pbar.update(1)

        tst[i+290000] = ret



#with tqdm(total=10000) as pbar:

#    def test_fa(row):

#        global array_size

#        global pbar

#        pbar.update(1)

#        return [np.array(row)[len(row)-array_size:len(row)]]

#    tst = num_days_cut.apply(test_fa)
k = 0

test = []

for i in tqdm(range(10000)):

    test.extend(tst[k+290000])

    k = k+1

print(test[0])
df_test = DataFrame(test)

y_test = df_test.pop(array_size)



x_test = enc.transform(df_test.iloc[:,7:]).toarray()

x_test = np.hstack((np.array(df_test.iloc[:,:7]), x_test))

print(x_test[0].size)
y = clf.predict(x_test)

print((y == y_test).sum()/y.size)

y = clf.predict(x_train)

print((y == y_train).sum()/y.size)
ftst = num_days.copy()

with tqdm(total=300000) as pbar:

    for i in range(300000):

        row = num_days[i]

        ret = ()

        a = vec_result[i]

        b = np.array(row)[len(row)-array_size+1:len(row)]

        ret = ret + ( np.concatenate([a,b]),)

        pbar.update(1)

        ftst[i] = ret



#with tqdm(total=300000) as pbar:

#    def ftest_fa(row):

#        global array_size

#        global pbar

#        pbar.update(1)

#        return [np.array(row)[len(row)-array_size+1:len(row)]]

#    ftst = num_days.apply(ftest_fa)
k = 0

ftest = []

for i in tqdm(range(300000)):

    ftest.extend(ftst[k])

    k = k+1

print(ftest[0])
df_ftest = DataFrame(ftest)

x_ftest = enc.transform(df_ftest.iloc[:,7:]).toarray()

x_ftest = np.hstack((np.array(df_ftest.iloc[:,:7]), x_ftest))

print(x_ftest[0].size)

prediction = DataFrame(columns = ['id', 'nextvisit'])

prediction['id'] = train_input['id']

prediction['nextvisit'] = clf.predict(x_ftest)
prediction.to_csv('solutiontmp.csv', index=False, sep =',')

print(prediction)