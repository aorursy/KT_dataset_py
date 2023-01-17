
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

sub1 = pd.read_csv('../input/osic-temp-data/submission.csv')
sub2 = pd.read_csv('../input/osic-temp-data/submission(1).csv')
sub1[['Patient','Week']] = sub1.Patient_Week.str.split("_",expand=True)

sub2[['Patient','Week']] = sub1.Patient_Week.str.split("_",expand=True) 
train_data_list = []

for person in train.Patient.unique():
    person_data = train[train['Patient'] == person]
    week_list = person_data['Weeks']
    fvc_list = person_data['FVC']
    train_data_list.append([week_list,fvc_list])
submission_data_list_1 = []

for person in sub1.Patient.unique():
    person_data = sub1[sub1['Patient'] == person]
    week_list = person_data['Week']
    fvc_list = person_data['FVC']
    submission_data_list_1.append([week_list,fvc_list])
submission_data_list_2 = []

for person in sub2.Patient.unique():
    person_data = sub2[sub2['Patient'] == person]
    week_list = person_data['Week']
    fvc_list = person_data['FVC']
    submission_data_list_2.append([week_list,fvc_list])
for week_list,data_list in train_data_list:
    plt.plot(week_list.to_list(),data_list.to_list())
    
plt.show()
i = 1
lines_in_graph = 10
for week_list,data_list in train_data_list:
    i = i+1
    plt.plot(week_list.to_list(),data_list.to_list())
    if i%lines_in_graph==0:
        plt.show()
for week_list,data_list in submission_data_list_1:
    plt.plot(week_list.to_list(),data_list.to_list())
    plt.show()
for week_list,data_list in submission_data_list_1:
    plt.plot(week_list.to_list(),data_list.to_list())
plt.show()
for week_list,data_list in submission_data_list_2:
    plt.plot(week_list.to_list(),data_list.to_list())
    plt.show()
for week_list,data_list in submission_data_list_2:
    plt.plot(week_list.to_list(),data_list.to_list())
plt.show()
