from glob import glob

import pandas as pd

from sklearn.preprocessing import StandardScaler

import os



def parse_data(fname): #create a function for parsing train data

    data = pd.read_csv(fname) #read data using pandas' read_csv function

    event_data = fname.replace('_data', '_events') #you replace filename string '_data' with '_event' 

	#so that you can automate reading same filename parsed for event

    labels = pd.read_csv(event_data)

    clean = data.drop(['id'], axis =1)

    labels = labels.drop(['id'], axis = 1)

    return clean, labels



def parse_test_data(fname):

    data = pd.read_csv(fname)

    return data
subjects = range(1, 13)



idx_tot = []



import numpy as np

for subject in subjects:

    raw = [] #raw data features

    y_raw = [] #raw label data

    read_datas = glob('../input/train/subj%d_series*_data.csv' % (subject))

    for fname in read_datas:

        data,label = parse_data(fname)

        raw.append(data)

        y_raw.append(label)

        

    X = pd.concat(raw)

    y = pd.concat(y_raw)

    

    X_train = np.asarray(X.astype(float))

    y = np.asarray(y.astype(float))
for subject in subjects:

    import numpy as np

    read_test = glob('../input/test/subj%d_series*_data.csv' % (subject))

    test = [] #initiate a an empty list to put each row of the read test set data

    idx = []  #What is this one doing? need to find out but it looks a lot like this is the index

    for fname in read_test:

        label = parse_test_data(fname) #parse test data using initialized function for parsing test_data

        test.append(label)

        idx.append(np.array(label['id'])) # this adds the array dtype of index column

     

    X_test = pd.concat(test) #concatenate the list gotten for the Test data

    id = np.concatenate(idx) #concatenation done through numpy because idx is a numpy array object

	

	#We are appending id to 'idx_tot' empty list that was initiated in the begining, the WHY of this i am yet to get a

	#grasp on.

    idx_tot.append(id)

    X_test = X_test.drop(['id'], axis = 1)

	

	#Prepare the test data for the classifier

    X_test = np.asarray(X_test.astype(float))
pred = np.empty((X_test.shape[0], 6))

pred
y
for i in range(6):

    y_train = y[:, i]

    print(y_train)

sub1_events_file = '../input/train/subj1_series1_events.csv'

sub1_data_file = '../input/train/subj1_series1_data.csv'



sub1_events = pd.read_csv(sub1_events_file)

sub1_data = pd.read_csv(sub1_data_file)



sub1 = pd.concat([sub1_events, sub1_data], axis = 1)

sub1["time"] = range(0, len(sub1))



sample_sub1 = sub1[sub1["time"] < 5000]
new_subj = sub1_data_file.replace('_data', '_events')

new_subj
train_subj5 = pd.read_csv("../input/train/subj5_series7_data.csv")

train_subj5.head()
train_event = pd.read_csv("../input/train/subj7_series1_events.csv")

train_data = pd.read_csv("../input/train/subj7_series1_data.csv")

train_event.head()
import matplotlib.pyplot as plt

view_data1 = train_subj5[1500:3000]

show_data1 = view_data1.plot(subplots = True, figsize = (10, 50))

plt.show(show_data1)
view_data2 = train_subj5.loc[1500:3000, :]

view_data2.plot(subplots = True, figsize = (10, 50))

plt.show(view_data2)
sample_submit = pd.read_csv("../input/sample_submission.csv")

sample_submit.head()
os.listdir("../input/test")
"""

Acknowledgment to kagglers I learnt from, thanks



Abby Shockley

"""