# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pylab

from sklearn import svm

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/sf-crime/train.csv", parse_dates=['Dates'], index_col='Dates')

test = pd.read_csv("/kaggle/input/sf-crime/test.csv", parse_dates=['Dates'], index_col='Dates')

sample = pd.read_csv('/kaggle/input/sf-crime/sampleSubmission.csv')

data_count = len(train['Category'])
columns = train.columns

count = 1

figures_per_time = 1

for i in columns:

    if i == 'Dates' or i == 'Y' or i == 'X' or i == 'Id' or i == 'Descript' or i == 'Resolution':

        continue

    hist = train[i].value_counts()

    length = len(hist.keys())

    if length > 20:

        length = 20

    pos = np.arange(len(hist[0:length].keys()))

    plt.figure(figsize=(50, length*2))

    plt.subplot(1,figures_per_time,np.mod(count,1)+1)

    count+=1

    plt.barh(pos, hist[0:length].get_values(),  align='edge', alpha=0.8, color = 'black')

    plt.yticks(pos, map(lambda x:x.title(),hist[0:length].keys()), fontsize = 25)

    plt.xlabel('Number of occurences', fontsize = 50)

    plt.title(i, fontsize = 50)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
count = 0

target = train["Category"].unique()

data_dict = {}

data_dict_reverse = {}

for data in target:

    data_dict[data] = count

    data_dict_reverse[count] = data

    count+=1

train["Category"] = train["Category"].replace(data_dict)



data_week_dict = {

    "Monday": 1,

    "Tuesday":2,

    "Wednesday":3,

    "Thursday":4,

    "Friday":5,

    "Saturday":6,

    "Sunday":7

}

train["DayOfWeek"] = train["DayOfWeek"].replace(data_week_dict)

test["DayOfWeek"] = test["DayOfWeek"].replace(data_week_dict)



district = train["PdDistrict"].unique()

data_dict_district = {}

count = 1

for data in district:

    data_dict_district[data] = count

    count+=1 

train["PdDistrict"] = train["PdDistrict"].replace(data_dict_district)

test["PdDistrict"] = test["PdDistrict"].replace(data_dict_district)
train_columns = train.columns

train_columns = train_columns.drop('Category')

train_columns = train_columns.drop('Descript')

train_columns = train_columns.drop('Resolution')

train_columns = train_columns.drop('Address')

new_train = train[train_columns]

new_train
# d = list(zip(*np.array(train)))

# del d[0], d[0], d[2], d[2]



# d = list(zip(*d))

# t = list(train['Category'])

d = new_train

t = train['Category']
t_col = test.columns

t_col = t_col.drop('Id')

t_col = t_col.drop('Address')

test_new = np.array(test[t_col])
sample_col = sample.columns

sample_col = sample_col.drop('Id')
sample = sample[sample_col]
for i in sample:

    for j in range(len(sample[i][:data_count])):

        if sample[i][j] != 0:

            sample[i][j] = data_dict[i]

            

test_result = []

for row in np.array(sample[:data_count]):

    test_result.append(sum(row))



sample 
data_count = 10000

model = svm.SVC(gamma = 'auto')

#model.fit(d[:data_count],t[:data_count])
p = cross_val_predict(model,d[:data_count],t[:data_count], cv = 10) #model.predict(d[:data_count])

p
print(f1_score(t[:data_count], p[:data_count], average='macro'))

c = confusion_matrix(t[:data_count], p[:data_count])

reverse_c = list(zip(*np.array(c)))

for i in range(len(c[1])):

    print(data_dict_reverse[i])

    fn = sum(c[i])

    fp = sum(reverse_c[i])

    print("Приавильных результатовЖ: " + str(c[i][i]))

    print("Ошибки первого рода: "+ str(fn))

    print("Ошибки второго рода: " + str(fp))