import numpy as np

import pandas as pd

from colorama import Fore, Back, Style

import matplotlib.pyplot as plt

import seaborn as sns



# Set Style

sns.set_style("whitegrid")

sns.despine(left=True, bottom=True)



# Set Color Palettes for the notebook

colors_nude = ['#e0798c','#65365a','#da8886','#cfc4c4','#dfd7ca']

sns.palplot(sns.color_palette(colors_nude))
train = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

test  = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

sub = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

target = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
print(Fore.YELLOW+"Training dataset:", Style.RESET_ALL + "has {} rows and {} columns".format(train.shape[0],train.shape[1]))
train.head()
print(Fore.YELLOW+"Test dataset:", Style.RESET_ALL + "has {} rows and {} columns".format(test.shape[0],train.shape[1]))
test.head()
print(Fore.YELLOW+"Feature dataset:", Style.RESET_ALL + "has {} rows and {} columns".format(target.shape[0],target.shape[1]))
target.head()
sub = pd.read_csv("../input/lish-moa/sample_submission.csv")

sub.shape
# helper function to check null values

def check_nulls(data):

    isNull = "N"

    for col in data.columns:

        if data[col].isnull().sum() > 0:        

            print("{} has {} null values".format(col,data[col].isnull().sum()))

            isNull = "Y"



    if isNull == "N":

        print("No Null Values found in the dataset")   
check_nulls(train)
check_nulls(test)
check_nulls(target)
train.columns
#check for duplicate sig_ids in training dataset

print("Total no. of records in the training dataset: ",train.shape[0])

print("No. of unique sig_ids in the training dataset:",train.sig_id.nunique())
# Count of "c-" & "g-" features

c_count = 0

g_count = 0

others = []

for feat in train.columns:

        if (feat.find("c-")) != -1:

            c_count = c_count + 1

        elif (feat.find("g-")) != -1:

            g_count = g_count + 1

        else:

            others.append(feat)

            

print(Fore.YELLOW +"No. of g- features:", Style.RESET_ALL + "{}".format(g_count)) 

print(Fore.YELLOW +"No. of c- features:", Style.RESET_ALL + "{}".format(c_count)) 

print(Fore.YELLOW +"Other features:", Style.RESET_ALL + "{}".format(train.shape[1] 

                            - (c_count + g_count)),others)



# visualize the no. of g- & c- features in the dataset

others = train.shape[1] - (c_count + g_count)

plt.figure(figsize = (8,6))

plt.bar(["g-", "c-","others"], [g_count,c_count,others],color = colors_nude)

plt.title("Categorical Features Distribution")

plt.xlabel("Features")

plt.ylabel("Count")

plt.legend()
# helper function to plot categorical variables

def plot_cp(feats):    

    print("--------------------" + feats[0] + "------------------------")

    print(train[feats[0]].value_counts())

    print("---------------------------------------------------")

    print("\n")

    print("--------------------" + feats[1] + "------------------------")

    print(train[feats[1]].value_counts())

    print("---------------------------------------------------")

    print("\n")

    print("--------------------" + feats[2] + "------------------------")

    print(train[feats[2]].value_counts())

    print("---------------------------------------------------")

    

    plt.figure(figsize = (21,8))

    

    plt.subplot(1,3,1)

    sns.countplot(train[feats[0]],palette=colors_nude)

    plt.xlabel("Features Distribution - " + feats[0],fontsize=15)

    plt.ylabel("Count",fontsize=15)

     

    plt.subplot(1,3,2)

    sns.countplot(train[feats[1]],palette=colors_nude)

    plt.xlabel("Features Distribution - " + feats[1],fontsize=15)

    plt.ylabel("Count",fontsize=15)

        

    plt.subplot(1,3,3)

    sns.countplot(train[feats[2]],palette=colors_nude)

    plt.xlabel("Features Distribution - " + feats[2],fontsize=15)

    plt.ylabel("Count",fontsize=15)

    

    plt.suptitle("Feature Distribution for cp_ variable",fontsize=25)

    plt.show()
# lets check how the distribution of cp_ features looks like

plot_cp(['cp_type','cp_dose','cp_time'])
# helper function to plot distribution of g- & c- features

def plot_g_c(feats,type):    

        

    plt.figure(figsize = (15,30))

    

    for idx,feat in enumerate(feats):

        plt.subplot(5,2,idx+1)

        sns.distplot(train[feats[idx]],color = "red")

        plt.xlabel("Features Distribution - " + feats[idx],fontsize=15)

        plt.ylabel("Count",fontsize=15)

        plt.title(feats[idx],fontsize=15)

    

    plt.suptitle(type + " Features - Distribution",fontsize=20)

    plt.show()
# call the helper function to check c- feature distribution

plot_g_c(['c-1','c-20','c-30','c-40','c-50','c-60','c-65','c-70','c-75','c-99'],"Cell Viability")
# call the helper function to check g- feature distribution

plot_g_c(['g-1','g-6','g-11','g-16','g-21','g-26','g-31','g-36','g-41','g-46'],"Gene Expression")
target.head()
plt.figure(figsize = (35,350))

for idx,col in enumerate(target.columns[1:]):   

    plt.subplot(42,5,idx+1)       

    sns.countplot(target[col],palette=colors_nude)

    plt.xlabel("Target Distribution - " + col)

    plt.ylabel("Count")

plt.show()
new_df= target[target.sum(axis=1)>1]

new_df ['sum'] = new_df.sum(axis=1)

new_df.shape
plt.figure(figsize = (15,10))

    

plt.subplot(2,2,1)

sns.countplot(new_df['sum'],palette=colors_nude)

plt.xlabel("No. of Targets",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.title("Drugs classified into more than one categories",fontsize=20)
# we are using set.intersection to find out similarity between the sig_id values present in the two datasets

len(set(train.sig_id.values).intersection(set(target.sig_id.values)))
c_list = []

g_list = []

others = []

for feat in train.columns:

        if (feat.find("c-")) != -1:

            c_list.append(feat)

        elif (feat.find("g-")) != -1:

            g_list.append(feat)

        else:

            others.append(feat)
# create a master dataset by concatenating train and target dataframes

master_df = pd.concat([train,target],axis = 1)

master_df.shape
master_df.head()
for tar_col in target.columns[1:11]:

    plt.figure(figsize = (25,10))



    for idx in range(1,11):

        plt.subplot(2,5,idx)

        col = "g-" + str(idx)

        plt.scatter(x=master_df[col],y=master_df[tar_col])

        plt.xlabel(col,fontsize=10)

        plt.ylabel(tar_col,fontsize=10)

        plt.suptitle("Correlation between g-1 to 10 &  {}".format(tar_col),fontsize=15)

    plt.show()
for tar_col in target.columns[1:11]:

    plt.figure(figsize = (25,100))



    for idx in range(1,11):

        plt.subplot(20,5,idx)

        col = "c-" + str(idx)

        plt.scatter(x=master_df[col],y=master_df[tar_col])

        plt.xlabel(col,fontsize=10)

        plt.ylabel(tar_col,fontsize=10)

        #plt.suptitle("Correlation between c-1 to 100 &  {}".format(tar_col),fontsize=15)

    plt.show()
target.columns[25:26]
def bivariate(feat):

    plt.figure(figsize = (25,50))



    for idx,tar_col in enumerate(target.columns[25:50]):  

        plt.subplot(10,5,idx+1)

        sns.countplot(master_df[tar_col],hue = master_df[feat],palette=colors_nude)    

    plt.show()
bivariate("cp_type")
bivariate("cp_time")
bivariate("cp_dose")