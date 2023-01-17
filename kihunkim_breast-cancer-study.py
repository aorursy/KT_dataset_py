# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

sns.set(style="whitegrid")

import warnings

warnings.filterwarnings("ignore") 



# Machine learning libraries

from sklearn.model_selection import train_test_split

# split test and train data

from sklearn.model_selection import KFold,cross_val_score,cross_val_predict

# K Fold cross validation 

from sklearn import metrics

# metrics will be use later to get a accuracy score 



# Prediction algoritms that I will use in this study 

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load csv file

df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
df.head()

# load 5 rows of df 

# There are a few unnecessary columns
df.drop(['id','Unnamed: 32'],axis=1,inplace=True)

#In this study 'id' and 'Unnamed: 32' are not needed 

#So drop both columns 
df.isna().sum()

# checking missing value 

# No missing values
df.info()

# get a information about each column
df.shape

# rows and columns
df.columns

# all columns in df 
#countplot

plt.subplots(figsize=(10,5))

sns.countplot(data=df,x='diagnosis');



plt.title('Diagnosis counting'.title(),

         fontsize = 14, weight="bold")



plt.xlabel('Type of diagonosis'.title(),

          fontsize=14,weight="bold")



plt.ylabel('Count'.title(),

           fontsize=14,weight="bold")



plt.legend(['malignant','benign'],loc='center right',bbox_to_anchor=(1.2, 0.93), 

           title="Diagonisis", title_fontsize = 14);
#pie chart  



plt.figure(figsize=(15,7))

sorted_counts = df['diagnosis'].value_counts()

# count the value of diagnosis 

ax=plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,

        counterclock = False,pctdistance=0.8 ,wedgeprops = {'width' : 0.4}, autopct='%1.0f%%');





plt.title('Proprotion of malignant and benign'.title(),

         fontsize = 14, weight="bold");



plt.legend(['Benign(B)','Malignant(M)'],bbox_to_anchor=(1,0.9));
#distplot = histogram + curveline 

# for example : radius mean



plt.subplots(figsize=(15,7))

x = df.radius_mean

bins = np.arange(0,30,1)

sns.distplot(x,bins=bins,color='black')



#ax.set_yticklabels([], minor = True);





plt.title('radius mean Histogram'.title(),

         fontsize = 14, weight="bold")



plt.xlabel('radius mean range'.title(),

          fontsize=14,weight="bold")



plt.ylabel('Count in percentage'.title(),

           fontsize=14,weight="bold");
# split table into different valriables 

y=df.diagnosis 

x = df.iloc[:,1:] 



# standardization

stand = (x - x.mean()) / (x.std())             
# Because we have 30 sub features we'll divide 3 groups to visualize



data = pd.concat([y,stand.iloc[:,0:10]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')



# In order to visualize different type of numeric value in one graph.We're going to melt df_new table into the new table called `data`.

# id_var : Column(s) to use as identifier variables.

# var_name : Name to use for the ‘variable’ column. If None it uses frame.columns.name or ‘variable’.

# value_name : Name to use for the ‘value’ column. 





plt.figure(figsize=(15,7))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")





plt.title('Sub features with standardization (first 10 features with violinplot)'.title(),

         fontsize = 14, weight="bold")



plt.xlabel('Sub features'.title(),

          fontsize=14,weight="bold")



plt.ylabel('z score'.title(),

           fontsize=14,weight="bold");



plt.xticks(rotation=45);

data = pd.concat([y,stand.iloc[:,10:20]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')



# In order to visualize different type of numeric value in one graph.We're going to melt df_new table into the new table called `data`.

# id_var : Column(s) to use as identifier variables.

# var_name : Name to use for the ‘variable’ column. If None it uses frame.columns.name or ‘variable’.

# value_name : Name to use for the ‘value’ column. 





plt.figure(figsize=(15,7))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")





plt.title('Sub features with standardization (Second 10 features with violinplot)'.title(),

         fontsize = 14, weight="bold")



plt.xlabel('Sub features'.title(),

          fontsize=14,weight="bold")



plt.ylabel('z score'.title(),

           fontsize=14,weight="bold");

plt.xticks(rotation=45);

data = pd.concat([y,stand.iloc[:,20:31]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')



# In order to visualize different type of numeric value in one graph.We're going to melt df_new table into the new table called `data`.

# id_var : Column(s) to use as identifier variables.

# var_name : Name to use for the ‘variable’ column. If None it uses frame.columns.name or ‘variable’.

# value_name : Name to use for the ‘value’ column. 





plt.figure(figsize=(15,7))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")





plt.title('Sub features with standardization (last 10 features with violinplot)'.title(),

         fontsize = 14, weight="bold")



plt.xlabel('Sub features'.title(),

          fontsize=14,weight="bold")



plt.ylabel('z score'.title(),

           fontsize=14,weight="bold");



plt.xticks(rotation=45);

data = pd.concat([y,stand.iloc[:,0:10]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')



plt.subplots(figsize=(15,7))

base_color = sns.color_palette()[6]

sns.pointplot(data=data,x='features',y='value',hue='diagnosis',dodge=True,ci=30,

              color=base_color)

# dodge: amount to separate the points for each level of the hue variable along the categorical axis.





plt.title('Sub features with standardization (first 10 features with pointplot)'.title(),

         fontsize = 14, weight="bold")



plt.xlabel('Sub features'.title(),

          fontsize=14,weight="bold")



plt.ylabel('z score'.title(),

           fontsize=14,weight="bold");



plt.xticks(rotation=45);

data = pd.concat([y,stand.iloc[:,10:20]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')



plt.subplots(figsize=(15,7))

base_color = sns.color_palette()[6]

sns.pointplot(data=data,x='features',y='value',hue='diagnosis',dodge=True,ci=30,

              color=base_color)

# dodge: amount to separate the points for each level of the hue variable along the categorical axis.





plt.title('Sub features with standardization (Second 10 features with pointplot)'.title(),

         fontsize = 14, weight="bold")



plt.xlabel('Sub features'.title(),

          fontsize=14,weight="bold")



plt.ylabel('z score'.title(),

           fontsize=14,weight="bold");



plt.xticks(rotation=45);

data = pd.concat([y,stand.iloc[:,20:31]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')



plt.subplots(figsize=(15,7))

base_color = sns.color_palette()[6]

sns.pointplot(data=data,x='features',y='value',hue='diagnosis',dodge=True,ci=30,

              color=base_color)

# dodge: amount to separate the points for each level of the hue variable along the categorical axis.





plt.title('Sub features with standardization (last 10 features with pointplot)'.title(),

         fontsize = 14, weight="bold")



plt.xlabel('Sub features'.title(),

          fontsize=14,weight="bold")



plt.ylabel('z score'.title(),

           fontsize=14,weight="bold");



plt.xticks(rotation=45);

data_new = pd.concat([y,stand],axis=1)

# build a new dataset with y and stand

# if you forgot what y and stand are. Just look at below 



# y=df.diagnosis 

# x = df.iloc[:,1:]  and stand = (x - x.mean()) / (x.std())          
# scatterplot with 2 high related sub features. Addtionally diagnosis will be used as a hue of different type of tumors

# Before we're going to plot scatter we'll find out which sub features are related together strongly 

# For that we will use pearson corrla



plt.figure(figsize=(28,13))

c= data_new.corr()

mask = np.triu(np.ones_like(c, dtype=np.bool))

cmap = sns.diverging_palette(220, 10, as_cmap=True) 

# color choose

sns.heatmap(c,cmap=cmap,mask=mask,center=0,annot=True);





plt.title('Sub title correlation'.title(),

         fontsize=20,weight='bold');
f,ax = plt.subplots(1,2,figsize=(20,7))



sns.scatterplot(data=data_new,x='radius_worst',y='perimeter_mean',hue='diagnosis',x_jitter=0.04,ax=ax[0])

ax[0].set_title('radius_worst vs perimeter_mean')

sns.scatterplot(data=data_new,x='area_mean',y='radius_mean',hue='diagnosis',x_jitter=0.04,ax=ax[1])

ax[1].set_title('area_mean vs radius_mean');
f,ax = plt.subplots(1,2,figsize=(20,7))



sns.scatterplot(data=data_new,x='fractal_dimension_mean',y='radius_mean',hue='diagnosis',x_jitter=0.04,ax=ax[0])

ax[0].set_title('radius_worst vs perimeter_mean')

sns.scatterplot(data=data_new,x='fractal_dimension_mean',y='area_mean',hue='diagnosis',x_jitter=0.04,ax=ax[1])

ax[1].set_title('area_mean vs radius_mean');
train,test = train_test_split(df,test_size=0.2,random_state=2019)



# test size =0.2 means I will use 20% for testing and 80% for training 

# Spliting test-set and training-set is very important.Because we have to use testdata to examine our prediction model and get a performance in numeric value.

# So never use testdata for training.Otherwise we can't get a exact result of prediction model.

# Reason why we use random_state : https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn



x_train = train.drop(['diagnosis'],axis=1)

y_train = train.diagnosis



# we should think about why we drop diagonosis column.Because we want to know the diagnosis in the end (That mean malignant or benign)

# We're going to use other columns as a x variable to get a diagonosis(y variable).That's the reason why we drop diagnosis in x_train and x_test



x_test = test.drop(['diagnosis'],axis=1)

y_test = test.diagnosis 



print(len(train),len(test))
model = svm.SVC(gamma='scale')

model.fit(x_train,y_train)

# learning train dataset



y_pred = model.predict(x_test)

# prediction test dataset



print('SVM: %.2f' % (metrics.accuracy_score(y_pred,y_test)*100))

# metrics.accuracy_score : measure the accurace_score

# so we compare prediction of y (prediction, y_pred) and test result of y (fact,y_test) how close our y_pred to y_test
model = DecisionTreeClassifier()

model.fit(x_train,y_train)



y_pred = model.predict(x_test)



print('DecisionTreeClassifier: %.2f' % (metrics.accuracy_score(y_pred,y_test)*100))

model = KNeighborsClassifier()

model.fit(x_train,y_train)



y_pred = model.predict(x_test)



print('KNeighborsClassifier: %.2f' % (metrics.accuracy_score(y_pred,y_test)*100))
model = LogisticRegression(solver='lbfgs',max_iter=2000)

# about parameters: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

model.fit(x_train,y_train)



y_pred = model.predict(x_test)



print('LogisticRegression: %.2f' % (metrics.accuracy_score(y_pred,y_test)*100))

model = RandomForestClassifier(n_estimators=100)

model.fit(x_train,y_train)



y_pred = model.predict(x_test)



print('RandomForestClassifier: %.2f' % (metrics.accuracy_score(y_pred,y_test)*100))

features = pd.Series(

     model.feature_importances_,

    index=x_train.columns).sort_values(ascending=False)



# model.feature_importances_ shows which paramet is important to predict the model 

# we are matching train dataset columns with model.feature_importances and saved in pandas series as a numeric values 

print(features)
### Extract Top 5 Features

top_5_features = features.keys()[:5]

# series.keys() : this function is an alias for index. It returns the index labels of the given series object.



print(top_5_features)
model = svm.SVC(gamma='scale')

model.fit(x_train[top_5_features],y_train)



y_pred = model.predict(x_test[top_5_features])

# prediction test dataset



print('SVM(Top5): %.2f' % (metrics.accuracy_score(y_pred,y_test)*100))
model = svm.SVC(gamma='scale')



cv = KFold(n_splits=5,random_state=2019)

# Interation : K=5



accs = []



for train_index,test_index in cv.split(df[top_5_features]):

    x_train = df.iloc[train_index][top_5_features]

    y_train = df.iloc[train_index].diagnosis

    

    x_test = df.iloc[test_index][top_5_features]

    y_test = df.iloc[test_index].diagnosis

    

    

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    accs.append(metrics.accuracy_score(y_pred,y_test))

    # position of y_pred and y_test are not important

    

print(accs)

    
model = svm.SVC(gamma='scale')

cv = KFold(n_splits=5,random_state=2019)



accs = cross_val_score(model,df[top_5_features],df.diagnosis,cv=cv)

# cross_vall_score : apply cross validation (in our case would be KFold) and learning.

# In the end will be print out the model score

# x variable : df[top_5_features] , y variable : di.diagnosis

print(accs)
model = {

    'SVM': svm.SVC(gamma='scale'),

    'DecisionTreeClassifier':DecisionTreeClassifier(),

    'KNeighborsClassifier': KNeighborsClassifier(),

    'LogisticRegression': LogisticRegression(solver='lbfgs',max_iter=2000),

    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)

    

}



cv = KFold(n_splits=5,random_state=2019)



for name, model in model.items():

    scores = cross_val_score(model,df[top_5_features],df.diagnosis,cv=cv)

    

    print('%s:%.2f%%' % (name,np.mean(scores)*100))



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=(0,1))

# scale the range between 0 and 1 

scaled_data = scaler.fit_transform(df[top_5_features])



model = {

    'SVM': svm.SVC(gamma='scale'),

    'DecisionTreeClassifier':DecisionTreeClassifier(),

    'KNeighborsClassifier': KNeighborsClassifier(),

    'LogisticRegression': LogisticRegression(solver='lbfgs',max_iter=2000),

    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)

    

}



cv = KFold(n_splits=5,random_state=2019)



for name, model in model.items():

    scores = cross_val_score(model,scaled_data,df.diagnosis,cv=cv)

    

    print('%s:%.2f%%' % (name,np.mean(scores)*100))
# First we will have a new table which contains only feature mean 

features_mean = list(df.columns[1:11])



# And then change diagnosis name

df['diagnosis']=df['diagnosis'].map({'M':0,'B':1})
from pandas.plotting import scatter_matrix



color_function = {0: "blue", 1: "red"} 

colors = df["diagnosis"].map(lambda x: color_function.get(x))

# mapping the color fuction with diagnosis column

pd.plotting.scatter_matrix(df[features_mean], c=colors, alpha = 0.5, figsize = (15, 15)); 

# plotting scatter plot matrix
df_new = pd.DataFrame(df,columns=['diagnosis','radius_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concavity_mean'] )

train,test = train_test_split(df_new,test_size=0.2,random_state=2019)



x_train = train.drop(['diagnosis'],axis=1)

y_train = train.diagnosis



x_test = test.drop(['diagnosis'],axis=1)

y_test = test.diagnosis 

model = {

    'SVM': svm.SVC(gamma='scale'),

    'DecisionTreeClassifier':DecisionTreeClassifier(),

    'KNeighborsClassifier': KNeighborsClassifier(),

    'LogisticRegression': LogisticRegression(solver='lbfgs',max_iter=2000),

    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)

    

}



cv = KFold(n_splits=5,random_state=2019)



prediction_var=['radius_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concavity_mean']

df_new[prediction_var]



for name, model in model.items():

    scores = cross_val_score(model,df_new[prediction_var],df_new.diagnosis,cv=cv)    

    print('%s:%.2f%%' % (name,np.mean(scores)*100))
