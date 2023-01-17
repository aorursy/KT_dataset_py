import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import re

import matplotlib.pyplot as plt

import pydicom

import glob
traindt = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

traindt.head()
import seaborn as sns

def MinMaxFVC(feature):

    plt.figure(dpi=100)

    sns.distplot(traindt[feature],color='red')

    print("{}Max value of {} is: {} {:.2f} \n{}Min value of {} is: {} {:.2f}\n{}Mean of {} is: {}{:.2f}\n{}Standard Deviation of {} is:{}{:.2f}"\

      .format('',feature,'',traindt[feature].max(),'',feature,'',traindt[feature].min(),'',feature,'',traindt[feature].mean(),'',feature,'',traindt[feature].std()))
MinMaxFVC("FVC")

print("There are {} unique patients in Train Data.".format(len(traindt["Patient"].unique())), "\n")

dataMinentry = traindt.groupby(by="Patient")["Weeks"].count().reset_index(drop=False)

dataMinentry = dataMinentry.sort_values(['Weeks']).reset_index(drop=True)

print("The Min Week : {}".format(traindt['Weeks'].min()), "\n" +

      "The Max Week :{}".format(traindt['Weeks'].max()))

print("Minimum number of entries are: {}".format(dataMinentry["Weeks"].min()), "\n" +

      "Maximum number of entries are: {}".format(dataMinentry["Weeks"].max()))

print(f"The Max week of the patient {max(traindt['Weeks'])}")

print(f"The Min week of the patient {min(traindt['Weeks'])}")
traindt.corr()


def TrainDCM(dt, size=(4,4)):

    plt.figure(figsize=size)

    plt.imshow(dt.pixel_array, cmap=plt.cm.bone)

    plt.show()

def numberfrom(svalue, pvalue, ret=0):

    search = pvalue.search(svalue)

    if search:

        return int(search.groups()[0])

    else:

        return ret     

filepath = []

ID = "ID00007637202177411956430"



for file in glob.glob("../input/osic-pulmonary-fibrosis-progression/train/"+ ID +"/*.dcm"):

    filepath.append(file)

   

pvalue = re.compile(ID +"/"+"(\d+)")

filepath = sorted(filepath, key=lambda svalue: numberfrom(svalue, pvalue, float('inf'))) 

for i in range(5):

    plt.subplot(3, 6, i+1)

    file_path = filepath[i]

    dataset = pydicom.dcmread(file_path)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.title(file_path[77:])

    plt.tick_params(labelbottom=False,

                    labelleft=False,

                    labelright=False,

                    labeltop=False)

testdt.loc[testdt.Patient == ID]
# Create base director for Train .dcm files

director = "../input/osic-pulmonary-fibrosis-progression/train"



# Create path column with the path to each patient's CT

traindt["Path"] = director + "/" + traindt["Patient"]



# Create variable that shows how many CT scans each patient has

traindt["CT_number"] = 0



for k, path in enumerate(traindt["Path"]):

    traindt["CT_number"][k] = len(os.listdir(path))







traindt.head(50)
