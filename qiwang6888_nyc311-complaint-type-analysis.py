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





%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

data_1 = pd.read_csv("../input/311-nyc-open-data-hpd/fhrw-4uyv.csv")

data_1.head()


# Combine values because heat/hot water is same as heating, and so on. 

data_1['complaint_type'] = data_1['complaint_type'].replace("HEAT/HOT WATER","HEATING").replace("Plumbing","PLUMBING").replace(("General","GENERAL","CONSTRUCTION"),"GENERAL CONSTRUCTION").replace("Unsanitary Condition","UNSANITARY CONDITION").replace("Safety","SAFETY").replace("Outside Building","OUTSIDE BUILDING").replace("Appliance","APPLIANCE").replace("Electric","ELECTRIC").replace("PAINT - PLASTER","PAINT/PLASTER").replace(("STRUCTURAL","AGENCY","VACANT APARTMENT","Mold"),"OTHER")

data_1['complaint_type'].value_counts()
#pie chart

df_complaint = data_1['complaint_type'].value_counts()

df_complaint = pd.DataFrame(df_complaint)

df_complaint = df_complaint.reset_index()

df_complaint.columns = ['Complaint_type',"Count"]

df_complaint = df_complaint.set_index("Complaint_type")



explode_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0.2,0.2]

df_complaint['Count'].plot(kind="pie",

                 figsize=(14,8),

                 autopct='%1.2f%%',

                 shadow=True,

                 labels=None,

                 pctdistance=1.12,

                 explode=explode_list)





plt.title("Pie Chart")

plt.axis('equal')

plt.legend(labels=df_complaint.index, loc='best') 
df_complaint.plot(kind='barh',

                 figsize=(14,8))

plt.title("Bar Chart for Complaint Types")

plt.xlabel("Total Number")

plt.ylabel("Complaint Types")
data_heat_time = data_1[['created_date','complaint_type']]

data_heat_time = data_heat_time.loc[data_heat_time['complaint_type']=='HEATING']

data_heat_time.created_date = pd.to_datetime(data_heat_time.created_date)

data_heat_time = data_heat_time.groupby(data_heat_time['created_date'].dt.year).count()

data_heat_time = data_heat_time.assign(complaint_type='HEATING')

data_heat_time.drop('complaint_type',inplace=True,axis=1)

data_heat_time

plt.bar(data_heat_time.index, data_heat_time.created_date,

       color='red')

plt.xlabel("Year")

plt.ylabel("Number")

plt.title("The Total Number of Complaint Type HEATING Yearly Occurance")
# I created a new dataframe with slected objects.

top_complaint = data_1[['complaint_type','incident_zip','street_name','borough']]

top_complaint = top_complaint[top_complaint.complaint_type=='HEATING'].reset_index(drop=True)

top_complaint.head()
# Bar chart based zip code.

colors=['r','b','yellow','grey','green','pink','orange','black','brown','purple']

series_zip = top_complaint['incident_zip'].value_counts().head(10)

total_value = series_zip.sum()

top_complaint['incident_zip'].value_counts().head(10).plot(kind="barh",

                                                          figsize=(10,6),

                                                          color=colors)

plt.xlabel('Zip Code') # add to x-label to the plot

plt.ylabel('Number of occurance') # add y-label to the plot

plt.title('HEATING Reports Amount by Borough') # add title to the plot

#

for index, value in enumerate(series_zip):

    label =  '{}%'.format(round((value/total_value)*100, 2)) 

    plt.annotate(label, xy=(value + 2000, index- 0.05), color='black')



plt.show()
# Bar chart based borough.

colors=['r','b','yellow','grey','green','pink','orange','black','brown','purple']

series_zip = top_complaint['borough'].value_counts().head(10)

total_value = series_zip.sum()

top_complaint['borough'].value_counts().head(10).plot(kind="barh",

                                                          figsize=(10,6),

                                                          color=colors)

plt.xlabel('Borough') # add to x-label to the plot

plt.ylabel('Number of occurance') # add y-label to the plot

plt.title('HEATING Reports Amount by Borough') # add title to the plot

#

for index, value in enumerate(series_zip):

    label =  '{}%'.format(round((value/total_value)*100, 2)) 

    plt.annotate(label, xy=(value + 2000, index- 0.05), color='black')



plt.show()
# Bar chart based on street.

colors=['r','b','yellow','grey','green','pink','orange','black','brown','purple']

series_zip = top_complaint['street_name'].value_counts().head(10)

total_value = series_zip.sum()

top_complaint['street_name'].value_counts().head(10).plot(kind="barh",

                                                          figsize=(10,6),

                                                          color=colors)

plt.xlabel('Number of occurance') # add to x-label to the plot

plt.ylabel('Street Name') # add y-label to the plot

plt.title('HEATING Reports Amount by street_name') # add title to the plot

#

for index, value in enumerate(series_zip):

    label =  '{}%'.format(round((value/total_value)*100, 2)) 

    plt.annotate(label, xy=(value + 2000, index- 0.05), color='black')



plt.show()
top_complaint[top_complaint.street_name == 'GRAND CONCOURSE'].head(10)

top_complaint[top_complaint.incident_zip == 11226.0].head(10)
#Create a new data fram that only contain HEATING complaint type, incident address and borough that is "BRONX"

df_bronx_complaint = data_1[["complaint_type", 'incident_address',"borough"]]

print(df_bronx_complaint.shape)

df_bronx_complaint = df_bronx_complaint[(df_bronx_complaint.complaint_type=="HEATING") &(df_bronx_complaint.borough=="BRONX")].reset_index(drop=True)

print(df_bronx_complaint.shape)

print(df_bronx_complaint.head())

#Create a data frame that has total number of complaint reports regarding to heating for each address

df_bronx_total = df_bronx_complaint.groupby("incident_address").count()

df_bronx_total.drop("borough",inplace=True,axis=1)

df_bronx_total.columns=['Total']

df_bronx_total
#Import dataset for the house characterics of borough of BRONX

df_bronx = pd.read_csv(("../input/311-nyc-open-data-hpd/BX_18v1.csv"))



#Select some important features.

df_bronx = df_bronx[['BldgArea','BldgDepth','Address',"BuiltFAR", "CommFAR", 

                    "FacilFAR", "Lot", "LotArea", "LotDepth", "NumBldgs", "NumFloors", 

                    "OfficeArea", "ResArea", "ResidFAR", "RetailArea", "YearBuilt", "YearAlter1", "ZipCode", "YCoord","XCoord"]]

print("original shape:", df_bronx.shape)



#Check missing value from address columns and then drop rows

print("number of missing address: ",df_bronx['Address'].isnull().sum())

df_bronx = df_bronx.dropna(subset=['Address'])

print("after removing missing address shape: ",df_bronx.shape)

print("check to see if there are duplicate address:" ,all(df_bronx['Address'].value_counts()==1))

#check and remove duplicated address

df_bronx.drop_duplicates(subset="Address", keep="first", inplace=True)

print("After removing duplicated rows: ",df_bronx.shape)



print("check to see if there are duplicate address: ",any(df_bronx['Address'].value_counts()==1))

#Merge two data frame

df_merged = pd.merge(df_bronx_total, df_bronx, right_on = "Address", left_index = True, how ="right")

df_merged.head()



#correlation

corr1 = df_merged.corr(method="pearson")

corr1
fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(111)

cax = ax.matshow(corr1,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df_merged.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df_merged.columns)

ax.set_yticklabels(df_merged.columns)

plt.show()
# In this case, the better correlation is Spearman.In a monotonic relationship, the variables tend to move in the same relative direction, but not necessarily at a constant rate.

corr2 = df_merged.corr(method="spearman")

corr2
fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(111)

cax = ax.matshow(corr2,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df_merged.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df_merged.columns)

ax.set_yticklabels(df_merged.columns)

plt.show()
new_merged = df_merged.dropna(subset=['Total'])

new_merged.corr(method="spearman")

# Na values do not affect the result of correlation
from scipy import stats



pval_df = pd.DataFrame(columns = new_merged.columns[1:], index=['Spearman','P-value'])



for i in pval_df.columns:

    spearman_coef, p_value = stats.spearmanr(new_merged['Total'],new_merged[i])

    pval_df[i]['Spearman'] = spearman_coef

    pval_df[i]['P-value'] = p_value

pval_df
df_merged.head()

df_merged.plot(kind="scatter",x='BldgArea', y= 'Total')
# First approch is classification. we create a new data frame contain response variable complaint type that can either be heating or not heating. 

# I use the predictive variables I achieved from question 3, they are "BldgArea","BldgDepth","BuiltFAR","LotArea","NumFloors","ResArea". 



#Create a datafram with binary value for response variable

df_cf = df_merged[['Total',"BldgArea","BldgDepth","BuiltFAR","LotArea","NumFloors","ResArea","LotArea"]]

df_cf = df_cf.fillna(0)

df_cf['Total'] = np.where(df_cf['Total']>0, 1, 0)

df_cf.head()



# split dataset into train and test set.

from sklearn.model_selection import train_test_split

x = df_cf.drop(['Total'], axis = 1)

y = df_cf[['Total']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=3)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics



for i in range(1,15):

    Tree = DecisionTreeClassifier(criterion="entropy", max_depth = i)

    Tree # it shows the default parameters

    Tree.fit(x_train, y_train)

    predTree = Tree.predict(x_test)

    print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree),"When max depth is: ",i)



    

# The best max depth is 5 because it has highest accuracy
# SVM

from sklearn import svm

clf = svm.SVC(kernel='rbf')

clf.fit(x_train, y_train) 

yhat = clf.predict(x_test)

print("SVM's Accuracy: ", metrics.accuracy_score(y_test, predTree))

# Logistic Regression



#standarlize all predictive variables because their values are 

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

x_train = preprocessing.StandardScaler().fit(x_train).transform(x_train)

x_test = preprocessing.StandardScaler().fit(x_test).transform(x_test)



logit = LogisticRegression()

logit.fit(x_train,y_train)

yhat = logit.predict(x_test)

print("The accuracy of logistic is: ",round(metrics.accuracy_score(y_test, yhat),2))





# K-Nearest Neighbors

import warnings

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from sklearn.neighbors import KNeighborsClassifier





for k in range(4,10): 

    neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)

    yhat = neigh.predict(x_test)

    print("The accuracy of KNN is :", round(metrics.accuracy_score(yhat, y_test),2), "when k is :" ,k)

    

    
