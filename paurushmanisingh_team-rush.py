import pandas as pd

import numpy as np

import scipy

import seaborn as sns

import sklearn.preprocessing
train_data = pd.read_csv("../input/equipfails/equip_failures_training_set.csv")    #read data into csv file

    

train_data.describe(include = 'all')     #display the raw data

    
#function to get rid of datapoints with more than 50 missing data values

def drop_with_na(thresh=50):

    a=len(train_data)

    print(train_data.iloc[[3]].isna().sum(axis=1)) 

    for i in range(a):

        c = train_data.iloc[[i]].isna().sum(axis=1)     #find the total number of "nan"



        print(i,c)

        if int(c) > thresh:

            print("!!!!")

            train_data = train_data.drop([i])

            i-=1

            a -=1

        if i==(a-1):

            break
def data_pre_process(file):

    file.replace('na',np.nan,inplace=True)     #replace 'na' with 'nan'

    for i in file.columns:

        file[i] = file[i].astype(float)        #make the data floats



    for i in file.columns:

        replacement = np.mean(file[i].dropna())

#         replacement = np.median(file[i].dropna())

#         replacement=-1

#         replacement=scipy.interp1d()





        

    #     print(file[i])

    #     print(median[0][0])

        file[i].replace(np.nan,replacement,inplace=True)     #replace the nan cells with the mean of the column

#         print(file[i])

    return file

train_data=data_pre_process(train_data)

# sns.heatmap(train_data)

train_data
#train_data=sklearn.preprocessing.normalize(train_data)
x = train_data.drop(['target','id'],axis=1)

y = train_data['target']

x=x.as_matrix()     #turn x and y into np array

y=y.as_matrix()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.267,random_state=42)#used a 26.7% test set to get 16000 values

print(x_train.shape,x_test.shape)                                                    #same as kaggle test set
from sklearn.ensemble import GradientBoostingClassifier 

import matplotlib.pyplot as plt

model=GradientBoostingClassifier().fit(x_train,y_train)     #train the model using the classifier



predictions=model.predict(x_test)                           #make the prediction



count=0

for i in range(len(predictions)):

    if predictions[i]==y_test[i]:

        count+=1

print(count/len(predictions))                               #calculate the precentage of correct predictions



ones_as_zeros = 0                               #number of times the model predicted zero where the correct value was one

zeros_as_ones = 0                               #number of times the model predicted one where the correct value was zero

for i in range(len(predictions)):               #find the number of times the model predicted wrong

    if y_test[i] != predictions[i]:

        if y_test[i] == 1:

            ones_as_zeros+=1

        else:

            zeros_as_ones +=1

print(ones_as_zeros,zeros_as_ones)

a = [ones_as_zeros,zeros_as_ones]

index = np.arange(len(a))

label = ['predicted ones as zeros','predicted zeros as ones']



#plot a bar graph showing the result

plt.bar(index,a)

plt.xticks(index,label)

plt.title('Prediction result for Gradient Boosting Classifier')

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x()+rect.get_width()/2.,1.05*height,

                '%f' %float(height),

        ha='center',va='bottom')

       

plt.show()
from sklearn.ensemble import RandomForestClassifier 

import matplotlib.pyplot as plt

model=RandomForestClassifier().fit(x_train,y_train)



predictions=model.predict(x_test)



count=0

for i in range(len(predictions)):

    if predictions[i]==y_test[i]:

        count+=1

print(count/len(predictions))



ones_as_zeros = 0

zeros_as_ones = 0

for i in range(len(predictions)):

    if y_test[i] != predictions[i]:

        if y_test[i] == 1:

            ones_as_zeros+=1

        else:

            zeros_as_ones +=1

print(ones_as_zeros,zeros_as_ones)

a = [ones_as_zeros,zeros_as_ones]

index = np.arange(len(a))

label = ['predicted ones as zeros','predicted zeros as ones']



plt.bar(index,a)

plt.xticks(index,label)

plt.title('Prediction result for Random Forest Classifier')

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x()+rect.get_width()/2.,1.05*height,

                '%f' %float(height),

        ha='center',va='bottom')

       

plt.show()
from sklearn.ensemble import AdaBoostClassifier 

import matplotlib.pyplot as plt

model=AdaBoostClassifier().fit(x_train,y_train)



predictions=model.predict(x_test)



count=0

for i in range(len(predictions)):

    if predictions[i]==y_test[i]:

        count+=1

print(count/len(predictions))



ones_as_zeros = 0

zeros_as_ones = 0

for i in range(len(predictions)):

    if y_test[i] != predictions[i]:

        if y_test[i] == 1:

            ones_as_zeros+=1

        else:

            zeros_as_ones +=1

print(ones_as_zeros,zeros_as_ones)

a = [ones_as_zeros,zeros_as_ones]

index = np.arange(len(a))

label = ['predicted ones as zeros','predicted zeros as ones']



plt.bar(index,a)

plt.xticks(index,label)

plt.title('Prediction result for Ada Boost Classifier')

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x()+rect.get_width()/2.,1.05*height,

                '%f' %float(height),

        ha='center',va='bottom')

       

plt.show()
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

model=LogisticRegression().fit(x_train,y_train)



predictions=model.predict(x_test)



count=0

for i in range(len(predictions)):

    if predictions[i]==y_test[i]:

        count+=1

print(count/len(predictions))



ones_as_zeros = 0

zeros_as_ones = 0

for i in range(len(predictions)):

    if y_test[i] != predictions[i]:

        if y_test[i] == 1:

            ones_as_zeros+=1

        else:

            zeros_as_ones +=1

print(ones_as_zeros,zeros_as_ones)

a = [ones_as_zeros,zeros_as_ones]

index = np.arange(len(a))

label = ['predicted ones as zeros','predicted zeros as ones']



plt.bar(index,a)

plt.xticks(index,label)

plt.title('Prediction result for Logistic Regression')

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x()+rect.get_width()/2.,1.05*height,

                '%f' %float(height),

        ha='center',va='bottom')

       

plt.show()
from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt

model=SGDClassifier().fit(x_train,y_train)



predictions=model.predict(x_test)



count=0

for i in range(len(predictions)):

    if predictions[i]==y_test[i]:

        count+=1

print(count/len(predictions))



ones_as_zeros = 0

zeros_as_ones = 0

for i in range(len(predictions)):

    if y_test[i] != predictions[i]:

        if y_test[i] == 1:

            ones_as_zeros+=1

        else:

            zeros_as_ones +=1

print(ones_as_zeros,zeros_as_ones)

a = [ones_as_zeros,zeros_as_ones]

index = np.arange(len(a))

label = ['predicted ones as zeros','predicted zeros as ones']



plt.bar(index,a)

plt.xticks(index,label)

plt.title('Prediction result for SGD Classifier')

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x()+rect.get_width()/2.,1.05*height,

                '%f' %float(height),

        ha='center',va='bottom')

       

plt.show()
from sklearn.linear_model import ElasticNet

import matplotlib.pyplot as plt

model=ElasticNet().fit(x_train,y_train)



predictions=model.predict(x_test)



count=0

for i in range(len(predictions)):

    if predictions[i]==y_test[i]:

        count+=1

print(count/len(predictions))



ones_as_zeros = 0

zeros_as_ones = 0

for i in range(len(predictions)):

    if y_test[i] != predictions[i]:

        if y_test[i] == 1:

            ones_as_zeros+=1

        else:

            zeros_as_ones +=1

print(ones_as_zeros,zeros_as_ones)

a = [ones_as_zeros,zeros_as_ones]

index = np.arange(len(a))

label = ['predicted ones as zeros','predicted zeros as ones']



plt.bar(index,a)

plt.xticks(index,label)

plt.title('Prediction result for ElasticNet')

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x()+rect.get_width()/2.,1.05*height,

                '%f' %float(height),

        ha='center',va='bottom')

       

plt.show()
import xgboost as xgb

from sklearn.ensemble import GradientBoostingClassifier 



model=xgb.XGBClassifier(verbosity=1,booster='gbtree',eta=0.3,max_depth=8,subsample=1,scale_pos_weight=1,gamma=1,max_delta_step=0,colsample_bytree=1,colsample_bylevel=0.5,colsample_bynode=1,min_child_weight=1, tree_method='exact',objective='binary:logistic').fit(x_train,y_train)

# model=GradientBoostingClassifier().fit(x_train,y_train)



predictions=model.predict(x_test)



count=0

for i in range(len(predictions)):

    if predictions[i]==y_test[i]:

        count+=1

print(count/len(predictions))





                           

              

ones_as_zeros = 0                           #number of times the model predicted zero where the correct value was one

zeros_as_ones = 0                           #number of times the model predicted one where the correct value was zero

for i in range(len(predictions)):           #find the number of times the model predicted wrong

    if y_test[i] != predictions[i]:

        if y_test[i] == 1:

            ones_as_zeros+=1

        else:

            zeros_as_ones +=1

print(zeros_as_ones,ones_as_zeros)

a = [zeros_as_ones,ones_as_zeros]

index = np.arange(len(a))

label = ['predicted ones as zeros','predicted zeros as ones']



#plot the bar graph showing the results

plt.bar(index,a)

plt.xticks(index,label)

plt.title('Prediction result for XGB')

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x()+rect.get_width()/2.,1.05*height,

                '%f' %float(height),

        ha='center',va='bottom')

       

plt.show()
testing=pd.read_csv("../input/equipfails/equip_failures_test_set.csv")

id_list=testing["id"]

testing=data_pre_process(testing)



test_final=testing.drop(["id"],axis=1)

test_final=test_final.as_matrix()





import pickle

# save model to file

pickle.dump(model, open("xgb.pickle.dat", "wb"))
import time

start_time = time.time()

2

# load model from file

loaded_model = pickle.load(open("xgb.pickle.dat", "rb"))

#make prediction

predictions=loaded_model.predict(test_final)



predictions=predictions.astype(int)

submission = pd.DataFrame({ 'id':id_list,

                            'target': predictions })

print(submission)

submission.to_csv("submission.csv", index=False)

print("--- %s seconds ---" % (time.time() - start_time))