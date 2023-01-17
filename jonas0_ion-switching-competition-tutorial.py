# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')





print("loading successful!")
print(train.shape, "\n")

print(train.info(), "\n")

print(train.columns, "\n")

print(train.index, "\n")
print(test.shape, "\n")

print(test.info(), "\n")

print(test.columns, "\n")

print(test.index, "\n")
for i in train.columns:

    print(i, train[i].isnull().sum())
for i in test.columns:

    print(i, test[i].isnull().sum())
print(train.open_channels.value_counts())
plt.figure(figsize=(20,5))

plt.plot(train.time[::100], train.signal[::100])



plt.show()
plt.figure(figsize=(20,5))

plt.plot(train.time[::1000], train.open_channels[::1000], color = 'red')

plt.show()
corr_dataframe = train[["time", "signal"]]



corr_mat = corr_dataframe.corr()



print(corr_mat)
a = 500000

dist = 100



for i in range(0,10):

    

    print(i, "min: ", min(train.signal[0+i*a:(i+1)*a:dist].values), "max: ", max(train.signal[0+i*a:(i+1)*a:dist]))

    plt.figure(figsize=(20,5))

    plt.plot(train.time[0+i*a:(i+1)*a:dist], train.signal[0+i*a:(i+1)*a:dist])

    plt.plot(train.time[0+i*a:(i+1)*a:dist], train.open_channels[0+i*a:(i+1)*a:dist], color = 'red')

    plt.show()
plt.figure(figsize=(20,5))

plt.plot(test.time[::100], test.signal[::100])

plt.show()
'''

a = 100000

dist = 100





plt.figure(figsize=(20,5))

plt.plot(test.time[0:2*a:dist], test.signal[0:2*a:dist])

plt.show()



plt.figure(figsize=(20,5))

plt.plot(test.time[2*a:3*a:dist], test.signal[2*a:3*a:dist])

plt.show()



plt.figure(figsize=(20,5))

plt.plot(test.time[3*a:4*a:dist], test.signal[3*a:4*a:dist])

plt.show()



plt.figure(figsize=(20,5))

plt.plot(test.time[4*a:5*a:dist], test.signal[4*a:5*a:dist])

plt.show()



plt.figure(figsize=(20,5))

plt.plot(test.time[5*a:6*a:dist], test.signal[5*a:6*a:dist])

plt.show()



plt.figure(figsize=(20,5))

plt.plot(test.time[6*a:8*a:dist], test.signal[6*a:8*a:dist])

plt.show()



plt.figure(figsize=(20,5))

plt.plot(test.time[8*a:9*a:dist], test.signal[8*a:9*a:dist])

plt.show()



plt.figure(figsize=(20,5))

plt.plot(test.time[9*a:10*a:dist], test.signal[9*a:10*a:dist])

plt.show()



plt.figure(figsize=(20,5))

plt.plot(test.time[10*a:15*a:dist], test.signal[10*a:15*a:dist])

plt.show()





plt.figure(figsize=(20,5))

plt.plot(test.time[15*a:20*a:dist], test.signal[15*a:20*a:dist])

plt.show()

'''
a = 500000 

b = 600000 





plt.plot(train.time[0+1*a:(1+1)*a:dist], train.signal[0+1*a:(1+1)*a:dist])

plt.plot(train.time[0+1*a:(1+1)*a:dist], train.open_channels[0+1*a:(1+1)*a:dist], color = 'red')

plt.show()



#####################################

train2 = train.copy()



c = 0.3

d = 50



train2.signal[a:b] = train2.signal[a:b].values - c*(train2.time[a:b].values - d)

train.signal[a:b] = train2.signal[a:b]
a = 500000

dist = 100



plt.plot(train.time[0+1*a:(1+1)*a:dist], train.signal[0+1*a:(1+1)*a:dist])

plt.plot(train.time[0+1*a:(1+1)*a:dist], train.open_channels[0+1*a:(1+1)*a:dist], color = 'red')

plt.show()
a = 100000

dist = 100



print("part 1")

plt.figure(figsize=(20,5))

plt.plot(test.time[0:1*a:dist], test.signal[0:1*a:dist])

plt.show()



print("part 2")

plt.figure(figsize=(20,5))

plt.plot(test.time[1*a:2*a:dist], test.signal[1*a:2*a:dist])

plt.show()



print("part 3")

plt.figure(figsize=(20,5))

plt.plot(test.time[2*a:3*a:dist], test.signal[2*a:3*a:dist])

plt.show()



print("part 4")

plt.figure(figsize=(20,5))

plt.plot(test.time[3*a:4*a:dist], test.signal[3*a:4*a:dist])

plt.show()



print("part 5")

plt.figure(figsize=(20,5))

plt.plot(test.time[4*a:5*a:dist], test.signal[4*a:5*a:dist])

plt.show()



print("part 6")

plt.figure(figsize=(20,5))

plt.plot(test.time[5*a:6*a:dist], test.signal[5*a:6*a:dist])

plt.show()



print("part 7")

plt.figure(figsize=(20,5))

plt.plot(test.time[6*a:7*a:dist], test.signal[6*a:7*a:dist])

plt.show()



print("part 8")

plt.figure(figsize=(20,5))

plt.plot(test.time[7*a:8*a:dist], test.signal[7*a:8*a:dist])

plt.show()



print("part 9")

plt.figure(figsize=(20,5))

plt.plot(test.time[8*a:9*a:dist], test.signal[8*a:9*a:dist])

plt.show()



print("part 10")

plt.figure(figsize=(20,5))

plt.plot(test.time[9*a:10*a:dist], test.signal[9*a:10*a:dist])

plt.show()



print("part 11")

plt.figure(figsize=(20,5))

plt.plot(test.time[10*a:15*a:dist], test.signal[10*a:15*a:dist])

plt.show()



print("part 12")

plt.figure(figsize=(20,5))

plt.plot(test.time[15*a:20*a:dist], test.signal[15*a:20*a:dist])

plt.show()
################



test2 = test.copy()



################

# part 1:



a = 0

b = 100000



c = 0.3

d = 500



test2.signal[a:b] = test2.signal[a:b].values - c*(test2.time[a:b].values - d)

test.signal[a:b]  = test2.signal[a:b]

################

# part 2:



a = 100000

b = 200000



d =  510



test2.signal[a:b] = test2.signal[a:b].values - c*(test2.time[a:b].values - d)

test.signal[a:b]  = test2.signal[a:b]

################

# part 5:



a = 400000

b = 500000



d =  540



test2.signal[a:b] = test2.signal[a:b].values - c*(test2.time[a:b].values - d)

test.signal[a:b]  = test2.signal[a:b]

################

# part 7:



a = 600000

b = 700000



d =  560



test2.signal[a:b] = test2.signal[a:b].values - c*(test2.time[a:b].values - d)

test.signal[a:b]  = test2.signal[a:b]

################

# part 8:



# slope  =  3/10



a = 700000

b = 800000



d =  570



test2.signal[a:b] = test2.signal[a:b].values - c*(test2.time[a:b].values - d)

test.signal[a:b]  = test2.signal[a:b]

################

# part 9:



a = 800000

b = 900000



d =  580



test2.signal[a:b] = test2.signal[a:b].values - c*(test2.time[a:b].values - d)

test.signal[a:b]  = test2.signal[a:b]

################



print("correcting linear slopes in test successful!")
plt.figure(figsize=(20,5))

plt.plot(test.time[::100], test.signal[::100])

plt.show()
def remove_parabolic_shape(values, minimum, middle, maximum):

    

    a = maximum - minimum

    return -(a/625)*(values - middle)**2+a



################################################



# I really want to find out, how he found these perfectly working

# numbers, because I can't imagine, that he sat around for hours,

# tweaking these low and high values until it worked.



# idea1: get the min and max value by calculating the mean

# of a certain window at the beginning of the batch

# and at the middle of the batch.





################################################

# part 7 goes from 3000k to 3500k



#his values

#low  = -1.817

#high =  3.186



#my values

#min:   -2.9517 

#max:    4.366



a = 3000000

b = 3500000

minimum = -1.817

middle = 325

maximum = 3.186



train2.signal[a:b] = train2.signal[a:b].values - remove_parabolic_shape(train2.time[a:b].values, minimum, middle, maximum)

train.signal[a:b] = train2.signal[a:b]



################################################

# part 8 goes from 3500k to 4000k



#his values

#low  = -0.094

#high =  4.936



#my values

#min:   -3.0399 

#max:    9.9986



a = 3500000

b = 4000000

minimum = -0.094

middle = 375

maximum = 4.936



train2.signal[a:b] = train2.signal[a:b].values - remove_parabolic_shape(train2.time[a:b].values, minimum, middle, maximum)

train.signal[a:b] = train2.signal[a:b]



################################################

# part 9 goes from 4000k to 4500k



#his values

#low  =  1.715

#high =  6.689



#my values

#min:   -2.0985 

#max:    9.0889



a = 4000000

b = 4500000

minimum = 1.715

middle = 425

maximum = 6.689



train2.signal[a:b] = train2.signal[a:b].values - remove_parabolic_shape(train2.time[a:b].values, minimum, middle, maximum)

train.signal[a:b] = train2.signal[a:b]



################################################

# part10 goes from 4500k to 5000k



#his values

#low  =  3.361

#high =  8.45



#my values

#min:   -1.5457 

#max:   12.683



a = 4500000

b = 5000000

minimum = 3.361

middle = 475

maximum = 8.45



train2.signal[a:b] = train2.signal[a:b].values - remove_parabolic_shape(train2.time[a:b].values, minimum, middle, maximum)

train.signal[a:b] = train2.signal[a:b]



################################################
a = 500000

dist = 100



for i in range(6,10):    

    plt.figure(figsize=(20,5))

    plt.plot(train.time[0+i*a:(i+1)*a:dist], train.signal[0+i*a:(i+1)*a:dist])

    plt.plot(train.time[0+i*a:(i+1)*a:dist], train.open_channels[0+i*a:(i+1)*a:dist], color = 'red')

    plt.show()
#######################################################

# his magical function full of magical numbers



def f(x):

    return -(0.00788)*(x-625)**2+2.345 +2.58





#test2.loc[test2.index[a:b],'signal'] = test2.signal.values[a:b] - f(test2.time[a:b].values)

#######################################################



test2 = test.copy()





a = 1000000

b = 1500000



plt.figure(figsize=(20,5))

plt.plot(test.time[a:b], test.signal[a:b])

plt.show()



test2.signal[a:b] = test2.signal[a:b].values - f(test2.time[a:b].values)

#test2.signal[a:b] = test2.signal[a:b].values - remove_parabolic_shape(test2.time[a:b].values, minimum, middle, maximum)

test.signal[a:b] = test2.signal[a:b]



plt.figure(figsize=(20,5))

plt.plot(test.time[a:b], test.signal[a:b])

plt.show()
from sklearn.metrics import f1_score

import graphviz

from sklearn import tree
# 1 slow open channel



a =  0

b =  500000

c =  500000

d = 1000000



X_train = np.concatenate([train.signal.values[a:b],train.signal.values[c:d]]).reshape((-1,1))

y_train = np.concatenate([train.open_channels.values[a:b],train.open_channels.values[c:d]]).reshape((-1,1))



model_1_slow_channel = tree.DecisionTreeClassifier(max_depth=1)

model_1_slow_channel.fit(X_train,y_train)



print('Training model_1_slow_open_channel...')

preds = model_1_slow_channel.predict(X_train)





print('has f1 validation score =', f1_score(y_train,preds, average='macro'))





#tree_graph = tree.export_graphviz(model_1_slow_channel, out_file=None, max_depth = 10,

#    impurity = False, feature_names = ['signal'], class_names = ['0', '1'],

#    rounded = True, filled= True )

#graphviz.Source(tree_graph)  
a = 1000000

b = 1500000



c = 3000000 

d = 3500000



X_train = np.concatenate([train.signal.values[a:b],train.signal.values[c:d]]).reshape((-1,1))

y_train = np.concatenate([train.open_channels.values[a:b],train.open_channels.values[c:d]]).reshape((-1,1))



model_1_fast_channel = tree.DecisionTreeClassifier(max_depth=1)



model_1_fast_channel.fit(X_train, y_train)



print('Training model_1_fast_channel...')

preds = model_1_fast_channel.predict(X_train)



print('has f1 validation score =',f1_score(y_train,preds,average='macro'))



#tree_graph = tree.export_graphviz(clf1f, out_file=None, max_depth = 10,

#    impurity = False, feature_names = ['signal'], class_names = ['0', '1'],

#    rounded = True, filled= True )

#graphviz.Source(tree_graph) 
a = 1500000 

b = 2000000



c = 3500000 

d = 4000000



X_train = np.concatenate([train.signal.values[a:b],train.signal.values[c:d]]).reshape((-1,1))

y_train = np.concatenate([train.open_channels.values[a:b],train.open_channels.values[c:d]]).reshape((-1,1))



model_3_channels = tree.DecisionTreeClassifier(max_depth=4)

model_3_channels.fit(X_train,y_train)

print('Training model_3_open_channels')



preds = model_3_channels.predict(X_train)

print('has f1 validation score =',f1_score(y_train,preds,average='macro'))



#tree_graph = tree.export_graphviz(clf3, out_file=None, max_depth = 10,

#    impurity = False, feature_names = ['signal'], class_names = ['0', '1','2','3'],

#    rounded = True, filled= True )

#graphviz.Source(tree_graph) 
a = 2500000

b = 3000000



c = 4000000 

d = 4500000





X_train = np.concatenate([train.signal.values[a:b],train.signal.values[c:d]]).reshape((-1,1))

y_train = np.concatenate([train.open_channels.values[a:b],train.open_channels.values[c:d]]).reshape((-1,1))



model_5_channels = tree.DecisionTreeClassifier(max_depth=6)

model_5_channels.fit(X_train, y_train)

print('Training model_5_open_channels')

preds = model_5_channels.predict(X_train)

print('has f1 validation score =',f1_score(y_train,preds,average='macro'))



#tree_graph = tree.export_graphviz(clf5, out_file=None, max_depth = 10,

#    impurity = False, feature_names = ['signal'], class_names = ['0', '1','2','3','4','5'],

#    rounded = True, filled= True )

#graphviz.Source(tree_graph) 
a = 2000000

b = 2500000



c = 4500000 

d = 5000000



X_train = np.concatenate([train.signal.values[a:b],train.signal.values[c:d]]).reshape((-1,1))

y_train = np.concatenate([train.open_channels.values[a:b],train.open_channels.values[c:d]]).reshape((-1,1))



model_10_channels = tree.DecisionTreeClassifier(max_depth=9)  # max_depth = 9 may be overfitting, try 8 and see if priv/pub score gets better

model_10_channels.fit(X_train, y_train)



print('Training model_10_open_channels')

preds = model_10_channels.predict(X_train)

print('has f1 validation score =',f1_score(y_train,preds,average='macro'))



#tree_graph = tree.export_graphviz(clf10, out_file=None, max_depth = 10,

#    impurity = False, feature_names = ['signal'], class_names = [str(x) for x in range(11)],

#    rounded = True, filled= True )

#graphviz.Source(tree_graph) 
sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')



a = 100000



# part 1

sub.iloc[0*a:1*a,1] = model_1_slow_channel.predict(test.signal.values[0*a:1*a].reshape((-1,1)))



# part 2

sub.iloc[1*a:2*a,1] = model_3_channels.predict(test.signal.values[1*a:2*a].reshape((-1,1)))



# part 3

sub.iloc[2*a:3*a,1] = model_5_channels.predict(test.signal.values[2*a:3*a].reshape((-1,1)))



# part 4

sub.iloc[3*a:4*a,1] = model_1_slow_channel.predict(test.signal.values[3*a:4*a].reshape((-1,1)))



# part 5

sub.iloc[4*a:5*a,1] = model_1_fast_channel.predict(test.signal.values[4*a:5*a].reshape((-1,1)))



# part 6

sub.iloc[5*a:6*a,1] = model_10_channels.predict(test.signal.values[5*a:6*a].reshape((-1,1)))



# part 7

sub.iloc[6*a:7*a,1] = model_5_channels.predict(test.signal.values[6*a:7*a].reshape((-1,1)))



# part 8

sub.iloc[7*a:8*a,1] = model_10_channels.predict(test.signal.values[7*a:8*a].reshape((-1,1)))



# part 9

sub.iloc[8*a:9*a,1] = model_1_slow_channel.predict(test.signal.values[8*a:9*a].reshape((-1,1)))



# part 10

sub.iloc[9*a:10*a,1] = model_3_channels.predict(test.signal.values[9*a:10*a].reshape((-1,1)))



# part 11

sub.iloc[10*a:20*a,1] = model_1_slow_channel.predict(test.signal.values[10*a:20*a].reshape((-1,1)))



print("training successful!")
plt.figure(figsize=(20,5))

res = 1000

let = ['A','B','C','D','E','F','G','H','I','J']

plt.plot(range(0,test.shape[0],res),sub.open_channels[0::res])

for i in range(5): plt.plot([i*500000,i*500000],[-5,12.5],'r')

for i in range(21): plt.plot([i*100000,i*100000],[-5,12.5],'r:')

for k in range(4): plt.text(k*500000+250000,10,str(k+1),size=20)

for k in range(10): plt.text(k*100000+40000,7.5,let[k],size=16)

plt.title('Test Data Predictions',size=16)

plt.show()
sub.to_csv('submission.csv', index = False, float_format='%.4f')



print("submission.csv saved successfully!")







#################################################

# result so far:



# public uses 30% of the test data

# public score:   0.92874  

# public rank:  1888/2618





# private uses 70% of the test data

# private score:  0.91612

# private rank: 1711/2618



#################################################
# things to do:



# 1.) remove all these warnings, maybe try the df.loc[df.column, 'signal']  alternative

# 2.) understand how he got the parabola values to work so nicely





# things to improve performance:



# 1.) tweak max_depth  of the decisiontree models, then submit, and see if the private/public score improves

# 2.) use max_leaf_nodes instead of max_depth and see if the priv/pub score improves

# 3.) tweak other parameters of the models as well??