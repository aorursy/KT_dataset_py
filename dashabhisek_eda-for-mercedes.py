import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

sns.set_style("whitegrid",{'axes.grid':False})

from matplotlib import cm
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.info()
test.info()
plt.figure(figsize=(18,8))

plt.scatter(range(train.shape[0]), np.sort(train.y.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('y', fontsize=12)

plt.show()
train['y'].describe()
np.percentile(train['y'],[5,95])
plt.figure(figsize=(18,8))

train['y'].plot(kind='kde')

plt.xlim([50,200])

plt.xlabel('y')

plt.show()
categorical_features=[x for x in train.columns if train[x].dtype=='object']

print (categorical_features)
def plot_pie(column):

    train_dist=train.groupby(column).size().to_frame().reset_index().sort_values(0,ascending=False)

    print ("%s contains %s categories in train data and top 5 categories contribute to %s percent entries "%(column,train_dist.shape[0],round(train_dist.head()[0].sum()*100.0/train.shape[0],2)))

    test_dist=test.groupby(column).size().to_frame().reset_index().sort_values(0,ascending=False)

    print ("%s contains %s categories in test data and top 5 categories contribute to %s percent entries "%(column,test_dist.shape[0],round(test_dist.head()[0].sum()*100.0/test.shape[0],2)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,12),subplot_kw={'aspect':'equal'})

    ax1.pie(train_dist[0],labels=train_dist[column],autopct='%.2f')

    ax1.set_title("Distribution of %s categories in train data"%column)

    ax2.pie(test_dist[0],labels=test_dist[column],autopct='%.2f')

    ax2.set_title("Distribution of %s categories in test data"%column)

    plt.tight_layout()

    plt.show()
def plot_bar_with_error(column):

    mean=train.groupby(column)['y'].mean().to_frame().reset_index().rename(columns = {'y':'mean'})

    std=train.groupby(column)['y'].std().to_frame().reset_index().fillna(0).rename(columns = {'y':'std'})

    mean_std=pd.merge(mean,std,how='left',on=column).sort_values('mean',ascending=False)

    colors = cm.gist_rainbow((mean_std['mean']-mean_std['mean'].min()) / float(mean_std['mean'].min()))

    

    fig=plt.figure(figsize=(18,8))

    ax = fig.add_subplot(111)

    ax.bar(np.arange(mean_std.shape[0]),mean_std['mean'],yerr=mean_std['std'],color=colors)

    ax.set_xticks(np.arange(mean_std.shape[0]))

    ax.set_xticklabels(mean_std[column])

    ax.set_title("Impact of categories of %s on time" %column)

    plt.show()
# X0

plot_pie('X0')
train_X0=train.groupby('X0').size().to_frame().reset_index().rename(columns = {0:'train_x0'})

test_X0=test.groupby('X0').size().to_frame().reset_index().rename(columns = {0:'test_x0'})

train_test=pd.merge(test_X0,train_X0,how='left',on='X0')

train_test[train_test['train_x0'].isnull()]
plot_bar_with_error('X0')
#X1

plot_pie('X1')
plot_bar_with_error('X1')
plot_pie('X2')
plot_bar_with_error("X2")
train.groupby('X2').size().to_frame().reset_index().sort_values(0).head(8)
plot_pie('X3')
plot_bar_with_error('X3')
plot_pie('X4')
plot_pie('X5')
plot_bar_with_error('X5')
train.groupby('X5').size().to_frame().reset_index().sort_values(0).head(5)
plot_pie('X6')
plot_bar_with_error('X6')
plot_pie('X8')
plot_bar_with_error('X8')
integer=[x for x in train.columns if train[x].dtype==int]

print (integer)

print (len(integer))
integer.remove('ID')

train[integer].head(10)
label_count=train[integer].apply(pd.value_counts)  

label_count
#Columns containing only 1

one_label=label_count.columns[label_count.iloc[0,:].isnull()]

one_label
zero_label=label_count.columns[label_count.iloc[1,:].isnull()]

zero_label
#Drop columns containing only one values from the integer

integer=[x for x in integer if x not in zero_label]

len(integer)
label_count=train[integer].apply(pd.value_counts).transpose().sort_values(1,ascending=True)

fig=label_count.plot(kind='bar',label=['0','1'],color=['r','g'],stacked=True,figsize=(100,50),fontsize=50)

plt.legend(prop={'size':80})

plt.show()
fig=label_count[(label_count[1]>10) & (label_count[0]>10)].plot(kind='bar',label=['0','1'],color=['r','g'],stacked=True,figsize=(100,50),fontsize=50)

plt.legend(prop={'size':80},bbox_to_anchor=(1, 1))

plt.show()
label_count[(label_count[1]>10) & (label_count[0]>10)].shape[0]
integer=label_count[(label_count[1]>10) & (label_count[0]>10)].index

label_average=[]

for column in integer:

    grouped_df=train.groupby(column)['y'].mean().to_frame().reset_index()

    label_average.append([column,grouped_df.loc[0,'y'],grouped_df.loc[1,'y']])

    

label_average=pd.DataFrame(label_average,columns=['column','mean_0','mean_1'])

label_average['mean_difference']=label_average['mean_1']-label_average['mean_0']

fig=label_average.sort_values('mean_difference')['mean_difference'].plot(kind='bar',figsize=(100,50),fontsize=50)

plt.title("Difference in mean time taken according to presence or absence of a feature",fontsize=70)

plt.show()