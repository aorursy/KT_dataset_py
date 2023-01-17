import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

from datetime import tzinfo, timedelta, datetime



df_train = pd.read_csv("../input/sf-crime/train.csv", parse_dates=['Dates'])

df_test = pd.read_csv("../input/sf-crime/test.csv", parse_dates=['Dates'])

train_data = pd.read_csv("../input/sf-crime/train.csv")

test_data = pd.read_csv("../input/sf-crime/test.csv")

df_train.dtypes
df_train.head()
Crime_Categories = list(df_train.loc[:,"Category"].unique())

print("Number of crime categories: " + str(len(Crime_Categories)))

for crime in Crime_Categories:

    print(crime)
number_of_crimes = df_train.Category.value_counts()



_n_crime_plot = sns.barplot(x=number_of_crimes.index,y=number_of_crimes)

_n_crime_plot.set_xticklabels(number_of_crimes.index,rotation=90)
most_dangerous_districts = df_train.PdDistrict.value_counts()

_n_crime_plot = sns.barplot(x=most_dangerous_districts.index,y=most_dangerous_districts)

_n_crime_plot.set_xticklabels(most_dangerous_districts.index,rotation=90)
most_dangerous_days_of_week = df_train.DayOfWeek.value_counts()

_n_crime_plot = sns.barplot(x=most_dangerous_days_of_week.index,y=most_dangerous_days_of_week)

_n_crime_plot.set_xticklabels(most_dangerous_days_of_week.index,rotation=90)
pareto_crime = number_of_crimes / sum(number_of_crimes)

pareto_crime = pareto_crime.cumsum()

_pareto_crime_plot = sns.tsplot(data=pareto_crime)

_pareto_crime_plot.set_xticklabels(pareto_crime.index,rotation=90)

_pareto_crime_plot.set_xticks(np.arange(len(pareto_crime)))



Main_Crime_Categories = list(pareto_crime[0:8].index)

print("The following categories :")

print(Main_Crime_Categories)

print("make up to {:.2%} of the crimes".format(pareto_crime[8]))

#Cross-tabulate Category and Year

sf_df_crosstab_dt = pd.crosstab(train_data.Category,train_data.DayOfWeek,margins=True)

del sf_df_crosstab_dt['All']#delete All column

sf_df_crosstab_dt = sf_df_crosstab_dt.ix[:-1]#delete last row (All)



column_labels_dt = list(sf_df_crosstab_dt.columns.values)

row_labels_dt = sf_df_crosstab_dt.index.values.tolist()



fig,ax = plt.subplots()

heatmap = ax.pcolor(sf_df_crosstab_dt,cmap=plt.cm.Blues)

fig = plt.gcf()

fig.set_size_inches(5,10)

#turn off the frame

ax.set_frame_on(False)

# put the major ticks at the middle of each cell

ax.set_yticks(np.arange(sf_df_crosstab_dt.shape[0])+0.5, minor=False)

ax.set_xticks(np.arange(sf_df_crosstab_dt.shape[1])+0.5, minor=False)



# want a more natural, table-like display

ax.invert_yaxis()

ax.xaxis.tick_top()

ax.set_xticklabels(column_labels_dt, minor=False)

ax.set_yticklabels(row_labels_dt, minor=False)

#rotate

plt.xticks(rotation=90)

#remove gridlines

ax.grid(True)

# Turn off all the ticks

ax = plt.gca()

for t in ax.xaxis.get_major_ticks(): 

    t.tick1On = False 

    t.tick2On = False 

for t in ax.yaxis.get_major_ticks(): 

    t.tick1On = False 

    t.tick2On = False

plt.show()    
data_dict = {}

data_dict_reverse = {}

target = train_data["Category"].unique()

count = 1

for data in target:

    data_dict[data] = count

    data_dict_reverse[count] = data

    count+=1

train_data["Category"] = train_data["Category"].replace(data_dict)



#Replacing the day of weeks

data_week_dict = {

    "Monday": 1,

    "Tuesday":2,

    "Wednesday":3,

    "Thursday":4,

    "Friday":5,

    "Saturday":6,

    "Sunday":7

}



category_dist = {

    'LARCENY/THEFT': 1,

 'OTHER OFFENSES': 2,

 'NON-CRIMINAL': 3,

 'ASSAULT': 4,

 'DRUG/NARCOTIC': 5,

 'VEHICLE THEFT': 6,

 'VANDALISM': 7,

 'WARRANTS': 8,

 'BURGLARY': 9,

 'SUSPICIOUS OCC': 10,

 'MISSING PERSON': 11,

 'ROBBERY': 12,

 'FRAUD': 13,

 'FORGERY/COUNTERFEITING': 14,

 'SECONDARY CODES': 15,

 'WEAPON LAWS': 16,

 'PROSTITUTION': 17,

 'TRESPASS': 18,

 'STOLEN PROPERTY': 19,

 'SEX OFFENSES FORCIBLE': 20,

 'DISORDERLY CONDUCT': 21,

 'DRUNKENNESS': 22,

 'RECOVERED VEHICLE': 23,

 'KIDNAPPING': 24,

 'DRIVING UNDER THE INFLUENCE': 25,

 'RUNAWAY': 26,

 'LIQUOR LAWS': 27,

 'ARSON': 28,

 'LOITERING': 29,

 'EMBEZZLEMENT': 30,

 'SUICIDE': 31,

 'FAMILY OFFENSES': 32,

 'BAD CHECKS': 33,

 'BRIBERY': 34,

 'EXTORTION': 35,

 'SEX OFFENSES NON FORCIBLE': 36,

 'GAMBLING': 37,

 'PORNOGRAPHY/OBSCENE MAT': 38,

 'TREA': 39

}



train_data["DayOfWeek"] = train_data["DayOfWeek"].replace(data_week_dict)

test_data["DayOfWeek"] = test_data["DayOfWeek"].replace(data_week_dict)



train_data.head()



##train_data["Category"] = train_data["Category"].replace(category_dist)



#District

district = train_data["PdDistrict"].unique()

data_dict_district = {}

count = 1

for data in district:

    data_dict_district[data] = count

    count+=1 

train_data["PdDistrict"] = train_data["PdDistrict"].replace(data_dict_district)

test_data["PdDistrict"] = test_data["PdDistrict"].replace(data_dict_district)
from matplotlib.colors import ListedColormap

features = ["DayOfWeek", "PdDistrict",  "X", "Y"]

X_train = train_data[features]

y_train = train_data["Category"]

X_test = test_data[features]

n_neighbors = 5

h= .2

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF','#FF0000', '#00FF00','#66AAFF'])



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors)

knn.fit(X_train, y_train)







predictionsTest = knn.predict(X_test)

predictionTrain = knn.predict(X_train)



plt.scatter(predictionTrain, y_train, alpha=.75, color='g')

plt.xlabel('Predicted ')

plt.ylabel('Actual ')

plt.show()
from collections import OrderedDict

data_dict_new = OrderedDict(sorted(data_dict.items()))

result_dataframe = pd.DataFrame({

    "Id": test_data["Id"]

})

for key,value in data_dict_new.items():

    result_dataframe[key] = 0

count = 0

for item in predictionTrain:

    for key,value in data_dict.items():

        if(value == item):

            result_dataframe[key][count] = 1

    count+=1


from sklearn import metrics

from sklearn.metrics import f1_score,confusion_matrix



print("F1 score",f1_score(y_train,predictionTrain,average='macro'))

print(metrics.accuracy_score(y_train, predictionTrain))





result_dataframe.head()

testTable = np.vstack((y_train, predictionTrain))

resultTestDataFrame = pd.DataFrame(testTable)

resultTestDataFrame.head()
c = confusion_matrix(y_train, predictionTrain)

reverse_c = list(zip(*np.array(c)))

for i in range(len(c[1])):

    #print(data_dict_reverse[i])

    fn = sum(c[i])

    fp = sum(reverse_c[i])

    print("Правильных результатов: " + str(c[i][i]))

    print("Ошибки первого рода: "+ str(fn))

    print("Ошибки второго рода: " + str(fp))
##from sklearn.metrics import f1_score,confusion_matrix,roc_auc_score

#f1_list=[]

#k_list=[]

#for k in range(1,10):

    #clf=KNeighborsClassifier(n_neighbors=k,n_jobs=-1)

    #clf.fit(X_train,y_train)

   # pred=clf.predict(X_train)

    #f=f1_score(y_train,pred,average='macro')

    #f1_list.append(f)

    #k_list.append(k)

    

    

#best_f1_score=max(f1_list)

#best_k=k_list[f1_list.index(best_f1_score)]       

#print(f1_list)

#print("Optimum K value=",best_k," with F1-Score=",best_f1_score)