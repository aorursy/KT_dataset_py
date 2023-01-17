import os

import pandas as pd 

import seaborn as sb

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
df = pd.read_csv('../input/StudentsPerformance.csv')
df.describe(include = 'all').transpose()
df.isnull().sum()
df.hist(bins=10)
def plot_piecharts(row,col,columns,df,title):

#     row,col = 3,2

    index = 0

    # columns = df.columns[:5]

    f, axes = plt.subplots(nrows=row,ncols=col, figsize=(20,12))

    f.suptitle(title, fontsize=16)

    for i in range(row):

        for j in range(col):

            

            if index < len(columns):

                

                axes[i][j].pie(df[columns[index]].value_counts(), labels=df[columns[index]].value_counts().index.tolist(), autopct='%1.1f%%',

                shadow=True, startangle=90)

                axes[i][j].set_title(columns[index])

                index+=1

#                 axes[i][j].axis('equal')

    axes[-1,-1].axis('off')

    plt.show()
plot_piecharts(2,3,df.columns[:5],df,'Raw Data')
#calculating the total score

if not 'total score' in df.columns:

    df['total score'] = df[['math score', 'reading score',

           'writing score']].sum(axis=1)

df.describe().transpose()

# BP

# [item.get_xdata()for item in BP['whiskers']]

BP = plt.boxplot(df['total score'],vert = False)
above_median = df[df['total score'] > df['total score'].median()]

plot_piecharts(2,3,df.columns[:5],above_median,'Total scorers above median')
above_q3 = df[df['total score'] > df['total score'].quantile(0.8)]

plot_piecharts(2,3,df.columns[:5],above_q3,'Total scorers above 80 percent')
fig1, ax1 = plt.subplots(1,3, figsize=(15,5))

fig2, ax2 = plt.subplots(1,3, figsize=(15,5))

col_name = 'race/ethnicity'

for i,column in enumerate(['math score','reading score', 'writing score']):

    temp_df = df[df[column] > df[column].median()]

    sb.countplot(x = col_name, data=temp_df, order = temp_df[col_name].value_counts().index, ax=ax1[i])

    ax1[i].set_title(column)

    fig1.suptitle('Above Median Scores',fontsize = 16)

    

    temp_df = df[df[column] > df[column].quantile(0.8)]

    sb.countplot(x = col_name, data=temp_df, order = temp_df[col_name].value_counts().index, ax=ax2[i])

    ax2[i].set_title(column)

    fig2.suptitle('More than 80 percentage',fontsize = 16)

#     plt.tight_layout()

plt.show()
below_median = df[df['total score'] < df['total score'].median()]

plot_piecharts(2,3,df.columns[:5],below_median,'Total scorers below median')
below_q1 = df[df['total score'] < df['total score'].quantile(0.25)]

plot_piecharts(2,3,df.columns[:5],below_median,'Total scorers below 25%')
figsize = (35,6)

fig1, ax1 = plt.subplots(1,3, figsize=figsize)

fig2, ax2 = plt.subplots(1,3, figsize=figsize)

col_name = 'parental level of education'

for i,column in enumerate(['math score','reading score', 'writing score']):

    temp_df = df[df[column] < df[column].median()]

    sb.countplot(x = col_name, data=temp_df, order = temp_df[col_name].value_counts().index, ax=ax1[i])

    ax1[i].set_title(column)

    fig1.suptitle('Below Median Scores',fontsize = 16)

    

    temp_df = df[df[column] < df[column].quantile(0.25)]

    sb.countplot(x = col_name, data=temp_df, order = temp_df[col_name].value_counts().index, ax=ax2[i])

    ax2[i].set_title(column)

    fig2.suptitle('Less than 25 percentage',fontsize = 16)

#     plt.tight_layout()

plt.show()
#Logistic regression

features = df[['math score', 'reading score','writing score']]

labels = pd.DataFrame(np.where(df['gender'] == 'male', 1, 0),columns=['gender'])



#train-test split

features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size = 0.2)



#train model

clf = LogisticRegression(random_state=0).fit(features_train,labels_train.to_numpy().ravel() )

# Predict model

labels_pred = clf.predict(features_test)



#accuracy

accuracy = clf.score(features_test,labels_test)



print ('The accuracy of the model is ',accuracy*100,'%')

print ('The confusion matrix for the model')

confusion_matrix(labels_test,labels_pred)