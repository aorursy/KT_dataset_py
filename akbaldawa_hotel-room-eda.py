# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
hotel = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
print(hotel.shape)
print(hotel.info())
print(hotel.head().T)
# Checking outliers at 25%,50%,75%,90%,95% and 99%
hotel.describe(percentiles=[.25,.5,.75,.90,.95,.99]).T
## Lets drop these rows as there are no actual bookings made and it may not be of much use for Analysis

hotel.drop(hotel[((hotel['children'] == 0) & 
                  (hotel['babies']   == 0) & 
                  (hotel['adults']   == 0))].index,inplace = True)

from collections import Counter 
# Outlier detection

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        print(col)
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        #print(outlier_step)
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
        # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    #return outlier_indices
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n )
    #multiple_outliers = list(outlier_indices)
    return multiple_outliers
hotel_num = hotel.dtypes[hotel.dtypes != 'object']
hotel_num = hotel_num.index.to_list()

Date_Drop = {'is_canceled','company'}
hotel_num = [ele for ele in hotel_num if ele not in Date_Drop]
hotel_num

#hot_num = hotel[hotel_num].copy()
Outliers_to_drop = detect_outliers(hotel,2,hotel_num)
# for i in hot_num.columns:
#     hot_num.boxplot(column=i)
#     plt.show()
hotel_num.remove('arrival_date_year') 
hotel_num.remove('arrival_date_week_number')
hotel_num.remove('arrival_date_day_of_month') 
# Outliers_to_drop = detect_outliers(hotel,["lead_time",
#  "stays_in_weekend_nights"])
# #  'stays_in_week_nights',
#  'adults',
#  'children',
#  'babies',
#  'is_repeated_guest',
#  'previous_cancellations',
#  'previous_bookings_not_canceled',
#  'booking_changes',
#  'agent',
#  'days_in_waiting_list',
#  'adr',
#  'required_car_parking_spaces',
#  'total_of_special_requests'])
len(Outliers_to_drop)
# Drop outliers
hotel = hotel.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
hotel.isna().sum()
## The country columns missing values can be replaced with unknown
## Agent missing can be replaced by 0 as these booking are not doe via an agent
## missing value for company could be replaced 
## Looking at the number of adults, may be there are no children accompanying them,we will replace the missing values with 0

## Looking at the unique values of the company and agents columns gives, they do not seem to be numerical data,these seem to be different codes for the gent or company
## masked while the data set was released for maintaining data provancy  
## There are 4 columns with missing values

# country                               488
# agent                               16192
# company                            109588
# Children                                4
hotel.company = hotel.company.fillna(0)
hotel.agent   = hotel.agent.fillna(0)
hotel.children = hotel.children.fillna(0)
hotel.country = hotel.country.fillna('unknown')
hotel_clean = hotel.copy()
hotel.describe(percentiles=[.25,.5,.75,.90,.95,.99]).T
hotel['hotel'] = hotel['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})
# Creating a dummy variable for the variable 'meal' and dropping the first one.
cont = pd.get_dummies(hotel['meal'],prefix='meal',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'market_segment' and dropping the first one.
cont = pd.get_dummies(hotel['market_segment'],prefix='market_segment',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'distribution_channel' and dropping the first one.
cont = pd.get_dummies(hotel['distribution_channel'],prefix='distribution_channel',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'reserved_room_type' and dropping the first one.
cont = pd.get_dummies(hotel['reserved_room_type'],prefix='reserved_room_type',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'assigned_room_type' and dropping the first one.
cont = pd.get_dummies(hotel['assigned_room_type'],prefix='assigned_room_type',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'deposit_type' and dropping the first one.
cont = pd.get_dummies(hotel['deposit_type'],prefix='deposit_type',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'customer_type' and dropping the first one.
cont = pd.get_dummies(hotel['customer_type'],prefix='customer_type',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'reservation_status' and dropping the first one.
cont = pd.get_dummies(hotel['reservation_status'],prefix='reservation_status',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)
hotel = hotel.drop(['meal',
 'country',
 'market_segment',
 'distribution_channel',
 'reserved_room_type',
 'assigned_room_type',
 'deposit_type',
 'customer_type',
 'reservation_status'],1)
# Normalising continuous features
df = hotel[hotel_num]
# normalized_df=(df-df.mean())/df.std()
# hotel = hotel.drop(hotel_num, 1)
# hotel = pd.concat([hotel,normalized_df],axis=1)
hotel.head()
import datetime

hotel['reserve_year'] = pd.DatetimeIndex(hotel['reservation_status_date']).year
hotel['reserve_month'] = pd.DatetimeIndex(hotel['reservation_status_date']).month
hotel['reserve_day'] = pd.DatetimeIndex(hotel['reservation_status_date']).day


hotel['arrival_date_month'] = pd.to_datetime(hotel['arrival_date_month'], format='%B').dt.month
hotel = hotel.drop('reservation_status_date',axis = 1)
hotel.info()
#Split the data into Test and Train

x = hotel.drop('is_canceled',axis = 1)
y = hotel['is_canceled']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
# Machine learning tools.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
models = []
models.append(SVC())
models.append(LinearSVC())
models.append(Perceptron())
models.append(GaussianNB())
models.append(SGDClassifier())
models.append(LogisticRegression())
models.append(KNeighborsClassifier())
models.append(RandomForestClassifier())
models.append(DecisionTreeClassifier())
models.append(GradientBoostingClassifier())

accuracy_list = []
for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = (accuracy_score(y_pred, y_test, normalize=True)*100)
    accuracy_list.append(accuracy)


model_name_list = ["SVM","Linear SVC","Perceptron","Gaussian NB","SGD Classifier","Logistic Regression",
                   "K-Neighbors Classifier","Random Forest Classifier","Decision Tree","Gradient Boosting"]

best_model = pd.DataFrame({"Model": model_name_list, "Score": accuracy_list})
best_model.sort_values(by="Score", ascending=False)
DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)
hotel_clean = hotel.copy()
hotel_clean[hotel_clean['lead_time'] == 0].count()
# pred = GB.predict(test_data)
# predictions = pd.DataFrame({ "PassengerId" : passenger_id, "Survived": pred })
Cat_Var = hotel_clean.dtypes[hotel.dtypes == 'object']
Cat_Var = Cat_Var.index.to_list()

Date_Drop = {'reservation_status_date'}
Cat_Var = [ele for ele in Cat_Var if ele not in Date_Drop]
Cat_Var
def cnt_plot(a):
    col = hotel[a]
    plt.figure(figsize = (16,10))
    sns.countplot(col,order = col.value_counts().index)
    title = 'Category wise count of' + ' ' + a
    plt.title(title)
    #sns.countplot(col.value_counts())
    #plt.xticks(rotation= 90)
    plt.show()
for col in Cat_Var:
    cnt_plot(col)
hotel.columns
def num_plot(x):
    fea = hotel[x]
    plt.figure(figsize = (16,10))
    sns.distplot(a = fea,kde = False)
    #title = 'Category wise count of' + ' ' + a
    #plt.title(title)
    #sns.countplot(col.value_counts())
    #plt.xticks(rotation= 90)
    plt.show()
s = hotel_clean['country'].value_counts()

hotel_clean['country'] = np.where(hotel_clean['country'].isin(s.index[s == 1]),'EQ1' , hotel_clean['country'])
hotel_clean['country'] = np.where(hotel_clean['country'].isin(s.index[s > 1] & s.index[s <= 10]),'LT10' , hotel_clean['country'])
hotel_clean['country'] = np.where(hotel_clean['country'].isin(s.index[s > 10] & s.index[s <= 49]),'LT50' , hotel_clean['country'])
hotel_clean['country'] = np.where(hotel_clean['country'].isin(s.index[s > 49] & s.index[s <= 99]),'LT100' , hotel_clean['country'])
hotel_clean['country'] = np.where(hotel_clean['country'].isin(s.index[s > 99] & s.index[s <= 499]),'LT500' , hotel_clean['country'])
bins = [0,1,8,29,85,169,366,737]
#labels = [0-8","20-29","30-39","40-49","50+"]

labels = ["0","1-7","8-28","29-84","85-168","169-365","366-737"]
hotel_clean["LeadtGroup"] = pd.cut(hotel_clean["lead_time"],bins, labels = labels, include_lowest = True)

leadt_mapping = {"0": 0,"1-7": 1,"8-28":2,"29-84":3,"85-168":4,"169-365":5,"366-737":6}
hotel_clean["LeadtGroup"] = hotel_clean["LeadtGroup"].map(leadt_mapping)
#data.drop("age", axis=1, inplace=True)
hotel_clean["LeadtGroup"].value_counts()
hotel_clean.head().T
hotel_clean['Members'] = hotel_clean['adults'] + hotel_clean['children'] + hotel_clean['babies']
hotel_clean.head().T
## Offline TA/TO May refer to Through Agent/Through Operator
## There are 2 extra type of room in the assigned Room Type as compared to the reserved Room Type/This will need further Analysis
## It will be interesting to compare  distribution_channel and market_segment, there may be some correlation
## There are very few bookingd for room C,H,P,I ,we can combine them to a separate category
## There are many countries from where less than 10 bookings are being done,These can be grouped together under a single category
hotel_clean['arrival_month'] = pd.to_datetime(hotel_clean['arrival_date_month'], format='%B').dt.month
arrival_time_df = hotel_clean[['arrival_date_year','arrival_month','arrival_date_day_of_month']].copy()
arrival_time_df.columns = ["year", "month", "day"]
arrival_time_df["month"] = arrival_time_df.month.map("{:02}".format)
arrival_time_df["day"] = arrival_time_df.day.map("{:02}".format)
hotel_clean['arrival_date'] = pd.to_datetime(arrival_time_df[['year','month','day']])
hotel_clean['arrival_date'] = pd.to_datetime(hotel_clean['arrival_date']).dt.date
hotel_clean['cost_per_member_night'] = hotel_clean['adr']/(hotel_clean['adults'] + hotel_clean['children'])
hotel_clean.groupby('hotel')['cost_per_member_night'].mean()
hotel_clean.groupby('arrival_date_month')['cost_per_member_night'].mean()
# order the hotel dataset by month:
ordered_months = ["January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]
hotel_clean["arrival_date_month"] = pd.Categorical(hotel_clean["arrival_date_month"], categories=ordered_months, ordered=True)
def lin_plot(a):
    plt.figure(figsize = (16,10))
    sns.lineplot(x = a, y ='cost_per_member_night' ,hue = 'hotel',data = hotel_clean)
    plt.title('Cost per night Vs ' + a)
    plt.show()
lin_cols = ['arrival_date_month','arrival_date_year','meal']
for i in lin_cols:
    lin_plot(i)
def lin_plot(a):
    plt.figure(figsize = (16,10))
    sns.lineplot(x = a, y ='cost_per_member_night' ,hue = 'hotel',data = hotel_clean)
    plt.title('Cost per night Vs ' + a)
    plt.show()
df_cntry_adr = pd.DataFrame(hotel_clean.groupby('country')['adr'].sum())
df_cntry_adr.reset_index(inplace = True)
df_cntry_cnt = pd.DataFrame(hotel_clean['country'].value_counts())
df_cntry_cnt.reset_index(inplace = True)
df_cntry_cnt = df_cntry_cnt.rename(columns = {'country': 'Tot Bookings','index' : 'country'})
df_cntry_data = pd.merge(df_cntry_adr,df_cntry_cnt, on='country')
df_cntry_data.columns
#Create combo chart

fig, ax1 = plt.subplots(figsize=(10,6))
ax1 = sns.lineplot(x = 'country', y = 'adr',data = df_cntry_data,color = 'Green' )
plt.xticks(rotation = 90)
plt.title('Country wise Total Bookings/Adr')
ax2 = ax1.twinx()
ax2 = sns.lineplot(x = 'country', y = 'Tot Bookings', data = df_cntry_data, color = 'Yellow')
plt.show()
hotel_clean.columns
import matplotlib.pyplot as plt

plt.figure(figsize = (15,15))
values = hotel_clean.groupby('market_segment')['adr'].sum().values
#colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w', 'f']
labels = hotel_clean.groupby('market_segment')['adr'].sum().index
#explode = (0.2, 0, 0, 0, 0, 0,0,0)
plt.pie(values, labels= values,autopct='%1.1f%%',counterclock=False, shadow=True)
plt.title('Market Segment Wise Revenue')
plt.legend(labels,loc=2)
plt.show()
hotel_clean.groupby('market_segment')['adr'].sum().values
hotel_clean.groupby('distribution_channel')['adr'].sum()
hotel_clean.groupby('country')['adr'].sum()
hotel_clean['country'].value_counts()
hotel_clean.head().T
def fact_plot(row):
    plt.figure(figsize = (16,10))
    #g = sns.catplot(x=row,y="is_canceled",data=hotel,kind="bar")
    sns.countplot(x=row,data=hotel,hue='is_canceled',palette='pastel')
    #g = g.set_ylabels("Canceled Status")
    #g = plt.xticks(rotation= 90)
    title = 'Plot of ' + row + ' Vs' + " is_canceled"
    plt.title(title)
    plt.legend()
    plt.show()
for col in Cat_Var:
    fact_plot(col)
## Lets plot Correlation plot for the DataFrame:
corrmap = hotel.corr()
plt.subplots(figsize = (16,10))
sns.heatmap(corrmap,annot= True)
plt.show()
Cat_df = hotel[['hotel',
 'meal',
 'country',
 'market_segment',
 'distribution_channel',
 'reserved_room_type',
 'assigned_room_type',
 'deposit_type',
 'customer_type',
 'reservation_status']].copy()
for col in Cat_df:
    print(Cat_df[col].unique())
hotel['hotel'] = hotel['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})
# Creating a dummy variable for the variable 'meal' and dropping the first one.
cont = pd.get_dummies(hotel['meal'],prefix='meal',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'market_segment' and dropping the first one.
cont = pd.get_dummies(hotel['market_segment'],prefix='market_segment',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'distribution_channel' and dropping the first one.
cont = pd.get_dummies(hotel['distribution_channel'],prefix='distribution_channel',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'reserved_room_type' and dropping the first one.
cont = pd.get_dummies(hotel['reserved_room_type'],prefix='reserved_room_type',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'assigned_room_type' and dropping the first one.
cont = pd.get_dummies(hotel['assigned_room_type'],prefix='assigned_room_type',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'deposit_type' and dropping the first one.
cont = pd.get_dummies(hotel['deposit_type'],prefix='deposit_type',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'customer_type' and dropping the first one.
cont = pd.get_dummies(hotel['customer_type'],prefix='customer_type',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)

# Creating a dummy variable for the variable 'reservation_status' and dropping the first one.
cont = pd.get_dummies(hotel['reservation_status'],prefix='reservation_status',drop_first=True)
#Adding the results to the master dataframe
hotel = pd.concat([hotel,cont],axis=1)
hotel = hotel.drop(['meal',
 'country',
 'market_segment',
 'distribution_channel',
 'reserved_room_type',
 'assigned_room_type',
 'deposit_type',
 'customer_type',
 'reservation_status'],1)
hotel.head()
# for cat in Cat_df:
#     Cat_df[cat]=Cat_df[cat].astype('category')

# from sklearn.preprocessing import OneHotEncoder

# one_hot = OneHotEncoder()
# one_hot.fit(Cat_df)
# cat_enc = pd.DataFrame((one_hot.transform(Cat_df)).toarray())
# #df[cat] = le.fit_transform(df[cat].astype(str))

# print('the number of rows in train is {} and columns is {}'.format(cat_enc.shape[0],cat_enc.shape[1]))
## Lets plot Correlation plot again for the modified DataFrame:
corr = hotel.corr()
corrmap = np.triu(corr)

fig, ax = plt.subplots(figsize = (60,60))

sns.heatmap(hotel.corr(),annot= True,square = True,cmap= 'coolwarm', linewidths=3, linecolor='black')
ax.set_xticklabels(corr.columns, fontsize=30)
ax.set_yticklabels(corr.columns, fontsize=30)
plt.show()
##Assigned vs Reserved have correlation for E,F,G,H categories
#is_cancelled has correlation only with Reservation status check out and deposit type non-refund
#customer type Transient and Transient type have correaltion
#distrubtion channel direct vs market segment direct,market segment direct vs market segment TA/TD
#distrubtion channel direct vs distrubtion channel TA/TD, distrubtion channel TA/TD
#hotel vs Agent
#distribution channel undefined vs market segment undefined
hotel['reservation_status_date'].head()
hotel.info()
import datetime

hotel['reserve_year'] = pd.DatetimeIndex(hotel['reservation_status_date']).year
hotel['reserve_month'] = pd.DatetimeIndex(hotel['reservation_status_date']).month
hotel['reserve_day'] = pd.DatetimeIndex(hotel['reservation_status_date']).day
hotel['arrival_date_month'].unique()
hotel['arrival_date_month'] = pd.to_datetime(hotel['arrival_date_month'], format='%B').dt.month
hotel = hotel.drop('reservation_status_date',axis = 1)
#Split the data into Test and Train

x = hotel.drop('is_canceled',axis = 1)
y = hotel['is_canceled']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
y_train.value_counts()
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, x_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING

# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(x_train,y_train)

ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_
#ExtraTrees 
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(x_train,y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_
# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(x_train,y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_
# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(x_train,y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_,"RF learning curves",x_train,y_train,cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",x_train,y_train,cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",x_train,y_train,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",x_train,y_train,cv=kfold)
nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=x_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1
