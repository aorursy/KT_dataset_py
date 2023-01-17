# import necessary packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# set plot style

sns.set_style()
# import dataset from local file

dataset = pd.read_csv("../data/csv/South Asian Wireless Telecom Operator (SATO 2015).csv")

# check dataset shape

print("Number of observations in data ",dataset.shape)

print("\n\nDataset info ",dataset.info())
# check head of data

dataset.head()
# check distribution of each numeric features

dataset.describe()
# check target class is balanced or not

dataset.Class.value_counts()
# convert Clas s column to numeric form 

def convert_label_to_numeric_form(row):

    if row.Class == "Churned":

        return 0

    elif row.Class == "Active":

        return 1
# add new column as Class_Converted containing class labels in 0 or 1 form

dataset["Class_Converted"]=dataset.apply(convert_label_to_numeric_form,axis=1)
dataset[dataset["Class_Converted"]==0]["Class"].value_counts()
dataset[dataset["Class_Converted"]==1]["Class"].value_counts()
# lets check whether any relation exists between aggregate complaints counts by customer and customer churn



print("\nComplain count distribition for churned subscriber \n ",dataset[dataset["Class_Converted"]==0]["Aggregate_complaint_count"].describe())

print("\nComplain count distribition for active subscriber \n ",dataset[dataset["Class_Converted"]==1]["Aggregate_complaint_count"].describe())
# lets check Complain count distribition for each type of subscriber where complaint count is more than 2



print("\nComplain count distribition for churned subscriber \n ",dataset[(dataset["Class_Converted"]==0)&(dataset["Aggregate_complaint_count"]>2)]["Aggregate_complaint_count"].describe())

print("\nComplain count distribition for active subscriber \n ",dataset[(dataset["Class_Converted"]==1)&(dataset["Aggregate_complaint_count"]>2)]["Aggregate_complaint_count"].describe())
# lets check whether any relation exists between from how long time customer taking services and customer churn



print("\nNetwork age distribition for churned subscriber \n ",dataset[dataset["Class_Converted"]==0]["network_age"].describe())

print("\nNetwork age distribition for active subscriber \n ",dataset[dataset["Class_Converted"]==1]["network_age"].describe())

print("\n-ve Network age distribition for churned subscriber \n ",dataset[(dataset["Class_Converted"]==0)&(dataset["network_age"]<0)]["network_age"])
print("\nNetwork age distribition for churned subscriber \n ",dataset[(dataset["Class_Converted"]==0)&(dataset["network_age"]>=0)]["network_age"].describe())

print("\nNetwork age distribition for active subscriber \n ",dataset[dataset["Class_Converted"]==1]["network_age"].describe())

# create new object by removing entries for -ve network age for churned customers

dataset_copy = dataset[dataset.network_age>=0].copy()

dataset.shape,dataset_copy.shape
# check user type count for churned and active customers for the month of aug and sep

print("User type count for churned customer for the month of aug \n",

     dataset_copy[dataset_copy.Class_Converted==0].aug_user_type.value_counts())



print("\nUser type count for active customer for the month of aug \n",

     dataset_copy[dataset_copy.Class_Converted==1].aug_user_type.value_counts())





print("\nUser type count for churned customer for the month of sep \n",

     dataset_copy[dataset_copy.Class_Converted==0].sep_user_type.value_counts())



print("\nUser type count for active customer for the month of sep \n",

     dataset_copy[dataset_copy.Class_Converted==1].sep_user_type.value_counts())