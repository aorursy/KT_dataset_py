import pandas as pd

import numpy as np

import datetime

import scipy

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

sns.set_style("whitegrid",{'axes.grid':False})

from matplotlib import cm
df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')
df_train.info()
df_test.info()
categorical_features=[x for x in df_train.columns if "cat" in x]

print("The number of categorical features are {}".format(len(categorical_features)))
print("Checking distribution of categories across variables in Train Dataset")

train_categories=pd.DataFrame(columns=['Feature Name','Categories','Count'])

for i in range(14):

    train_temp=df_train.groupby(categorical_features[i]).size().to_frame().reset_index().sort_values(0,ascending=False).rename(columns={categorical_features[i]:"Categories",0:"Count"})

    train_temp['Feature Name']=categorical_features[i]

    train_categories=train_categories.append(train_temp,ignore_index=True)

train_categories['percentage']=train_categories['Count']*100.0/df_train.shape[0]

train_categories['percentage']=train_categories['percentage'].apply(lambda x:round(x,2))

ax=train_categories.pivot(index='Feature Name',columns='Categories',values='percentage').plot(kind='bar',stacked=True,label=None,figsize=(12,6))

ax.legend().set_visible(False)

plt.show()
print("Checking distribution of categories across variables in Test Dataset")

test_categories=pd.DataFrame(columns=['Feature Name','Categories','Count'])

for i in range(14):

    test_temp=df_test.groupby(categorical_features[i]).size().to_frame().reset_index().sort_values(0,ascending=False).rename(columns={categorical_features[i]:"Categories",0:"Count"})

    test_temp['Feature Name']=categorical_features[i]

    test_categories=test_categories.append(test_temp,ignore_index=True)

test_categories['percentage']=test_categories['Count']*100.0/df_test.shape[0]

test_categories['percentage']=test_categories['percentage'].apply(lambda x:round(x,2))

ax=test_categories.pivot(index='Feature Name',columns='Categories',values='percentage').plot(kind='bar',stacked=True,label=None,figsize=(12,6))

ax.legend().set_visible(False)

plt.show()
def plot_pie(column):

    train_dist=df_train.groupby(column).size().to_frame().reset_index().sort_values(0,ascending=False)

    print("{} contains {} categories in train data, top 2 categories contribute to {} percent entries ".format(column,train_dist.shape[0],round(train_dist.head(2)[0].sum()*100.0/df_train.shape[0],2)))

    test_dist=df_test.groupby(column).size().to_frame().reset_index().sort_values(0,ascending=False)

    print("{} contains {} categories in test data, top 2 categories contribute to {} percent entries ".format(column,test_dist.shape[0],round(test_dist.head(2)[0].sum()*100.0/df_test.shape[0],2)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,10),subplot_kw={'aspect':'equal'})

    ax1.pie(train_dist[0],labels=train_dist[column],autopct='%.2f')

    ax1.set_title("Distribution of %s categories in train data"%column)

    ax2.pie(test_dist[0],labels=test_dist[column],autopct='%.2f')

    ax2.set_title("Distribution of %s categories in test data"%column)

    plt.tight_layout()

    plt.show()

    

def plot_bar(column):

    mean=df_train.groupby(column)['target'].mean().to_frame().reset_index().rename(columns = {'target':'mean'})

    

    fig=plt.figure(figsize=(10,5))

    ax=sns.barplot(mean[column],mean['mean'])

    ax.set_title("Impact of categories of %s on target" %column)

    rects = ax.patches

    (y_bottom, y_top) = ax.get_ylim()

    y_height = y_top - y_bottom



    for rect in rects:

        height = rect.get_height()

        label_position = height - (y_height * 0.05)

        ax.text(rect.get_x() + rect.get_width()/2., label_position,round(height,4),ha='center', va='bottom', fontweight='bold')



    plt.ylabel("Avearge Target Value")

    plt.show()
print("Exploring {}".format(categorical_features[0])) 

plot_pie("ps_ind_02_cat")

plot_bar("ps_ind_02_cat")
df_train['target'].mean()
print("Exploring {}".format(categorical_features[1])) 

plot_pie(categorical_features[1])

plot_bar(categorical_features[1])
print("Exploring {}".format(categorical_features[2])) 

plot_pie(categorical_features[2])

plot_bar(categorical_features[2])
print("Exploring {}".format(categorical_features[3])) 

plot_pie(categorical_features[3])

plot_bar(categorical_features[3])
print("Exploring {}".format(categorical_features[4])) 

plot_pie(categorical_features[4])

plot_bar(categorical_features[4])
print("Exploring {}".format(categorical_features[5])) 

plot_pie(categorical_features[5])

plot_bar(categorical_features[5])
print("Exploring {}".format(categorical_features[6])) 

plot_pie(categorical_features[6])

plot_bar(categorical_features[6])
print("Exploring {}".format(categorical_features[7])) 

plot_pie(categorical_features[7])

plot_bar(categorical_features[7])
print("Exploring {}".format(categorical_features[8])) 

plot_pie(categorical_features[8])

plot_bar(categorical_features[8])
print("Exploring {}".format(categorical_features[9])) 

plot_pie(categorical_features[9])

plot_bar(categorical_features[9])
print("Exploring {}".format(categorical_features[10])) 

plot_pie(categorical_features[10])

plot_bar(categorical_features[10])
print("Exploring {}".format(categorical_features[11])) 

plot_pie(categorical_features[11])

plot_bar(categorical_features[11])
print("Exploring {}".format(categorical_features[12])) 

plot_pie(categorical_features[12])

plot_bar(categorical_features[12])
print("Exploring {}".format(categorical_features[13])) 

plot_pie(categorical_features[13])

plot_bar(categorical_features[13])
binary_features=[x for x in df_train.columns if "bin" in x]

len(binary_features)
#Checking composition of zero and one for each variable

label_count=df_train[binary_features].apply(pd.value_counts).transpose().sort_values(1,ascending=True)

fig=label_count.plot(kind='bar',label=['0','1'],color=['#457fbc','#f97325'],stacked=True,figsize=(12,6))

plt.legend(loc='best')

plt.show()
plt.figure(figsize=(15,10))

for i in range(17):

    plt.subplot(5,4,i+1)

    mean=df_train.groupby(binary_features[i])['target'].mean().to_frame().reset_index().rename(columns = {'target':'mean'})

    ax=sns.barplot(mean[binary_features[i]],mean['mean'],palette=['#457fbc','#f97325'])

    rects = ax.patches

    (y_bottom, y_top) = ax.get_ylim()

    y_height = y_top - y_bottom



    for rect in rects:

        height = rect.get_height()

        label_position = height - (y_height * 0.2)

        ax.text(rect.get_x() + rect.get_width()/2., label_position,round(height,4),ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

plt.show()
co_variables=[x for x in df_train.columns if (('bin' not in x) and ('cat' not in x))]

co_variables.remove('id')

co_variables.remove('target')

len(co_variables)
#Distribution of ordinal and continuous variables in train dataset

plt.figure(figsize=(15,10))

for i in range(26):

    plt.subplot(7,4,i+1)

    sns.distplot(df_train[co_variables[i]],kde=True,hist=False)

plt.tight_layout()

plt.show()
#Distribution of ordinal and continuous variables in test dataset

plt.figure(figsize=(15,10))

for i in range(26):

    plt.subplot(7,4,i+1)

    sns.distplot(df_test[co_variables[i]],kde=True,hist=False)

plt.tight_layout()

plt.show()