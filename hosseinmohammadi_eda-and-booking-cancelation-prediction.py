# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import pyplot

import seaborn as sns



from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score, cross_val_predict

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV



import xgboost as xgb

from xgboost import XGBClassifier



from sklearn.inspection import permutation_importance



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier



from sklearn.metrics import accuracy_score 

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Import Data



hotel_data = pd.read_csv("../input/hotelbookings/hotel_bookings.csv")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Show first 10 rows



hotel_data.head(10)
# Data summary



hotel_data.info()
## Display sum of null data



hotel_data.isnull().sum()


# Replace missing values:

# agent: If no agency is given, booking was most likely made without one.

# company: If none given, it was most likely private.

# rest schould be self-explanatory.

nan_replacements = {"children:": 0.0,"country": "Unknown", "agent": 0, "company": 0}

cleaned_hotel_data = hotel_data.fillna(nan_replacements)



# "meal" contains values "Undefined", which is equal to SC.

cleaned_hotel_data["meal"].replace("Undefined", "SC", inplace=True)



# Some rows contain entreis with 0 adults, 0 children and 0 babies. 

# I'm dropping these entries with no guests.

zero_guests = list(cleaned_hotel_data.loc[cleaned_hotel_data["adults"]

                   + cleaned_hotel_data["children"]

                   + cleaned_hotel_data["babies"]==0].index)

cleaned_hotel_data.drop(cleaned_hotel_data.index[zero_guests], inplace=True)
## Display sum of null data



cleaned_hotel_data.isnull().sum()
# After cleaning, separate Resort and City hotel

# To know the acutal visitor numbers, only bookings that were not canceled are included. 

rh = cleaned_hotel_data.loc[(cleaned_hotel_data["hotel"] == "Resort Hotel") & (cleaned_hotel_data["is_canceled"] == 0)]

ch = cleaned_hotel_data.loc[(cleaned_hotel_data["hotel"] == "City Hotel") & (cleaned_hotel_data["is_canceled"] == 0)]
# Hotel types details



plt.figure(figsize=(5,7))

sns.countplot(x='hotel', data = cleaned_hotel_data, palette='gist_earth')

plt.title('Hotel Types', weight='bold')

plt.xlabel('Hotel', fontsize=12)

plt.ylabel('Count', fontsize=12)
# `is_canceled` graph



plt.figure(figsize=(7,5))

sns.countplot(y='is_canceled', data= cleaned_hotel_data, palette='gist_stern', orient = 'v',hue="hotel", hue_order = ["City Hotel", "Resort Hotel"])

plt.title('Canceled Situation', weight='bold')

plt.xlabel('Count', fontsize=12)

plt.ylabel('Canceled or Not Canceled', fontsize=12)
# `arrival_date_year` vs `lead_time` vs `is_canceled` exploration with violin plot



plt.figure(figsize=(8,8))

sns.violinplot(x='arrival_date_year', y ='lead_time', hue="is_canceled", data=cleaned_hotel_data, palette="Set3", bw=.2,

               cut=2, linewidth=2, iner= 'box', split = True, )

sns.despine(left=True)

plt.title('Arrival Year vs Lead Time vs Canceled Situation', weight='bold')

plt.xlabel('Year', fontsize=12)

plt.ylabel('Lead Time', fontsize=12)
# `arrival_date_year` vs `lead_time` vs `is_canceled` exploration with violin plot



plt.figure(figsize=(18,10))

sns.countplot(x='is_canceled', data= cleaned_hotel_data, palette='gist_stern', orient = 'v',hue="arrival_date_month",)

plt.title('Canceled Situation', weight='bold')

plt.xlabel('Count', fontsize=12)

plt.ylabel('Canceled or Not Canceled', fontsize=12)




cross_table = pd.crosstab(cleaned_hotel_data['arrival_date_month'],cleaned_hotel_data['is_canceled'])

cross_table.div(cross_table.sum(1), axis = 0).plot(kind = 'bar', stacked = True)



#`arrival_date_month` names converted to the numbers



# cleaned_hotel_data['arrival_date_month'].replace({'January' : '01',

#         'February' : '02',

#         'March' : '03',

#         'April' : '04',

#         'May' : '05',

#         'June' : '06',

#         'July' : '07',

#         'August' : '08',

#         'September' : '09', 

#         'October' : '10',

#         'November' : '11',

#         'December' : '12'}, inplace=True)



#`arrival_date_month` exploration 



plt.figure(figsize=(10,7))

sns.countplot(x='arrival_date_month', data = cleaned_hotel_data,

              order=pd.value_counts(cleaned_hotel_data['arrival_date_month']).index, palette='YlOrBr_r',hue="hotel", hue_order = ["City Hotel", "Resort Hotel"])

plt.title('Arrival Month', weight='bold')

plt.xlabel('Month', fontsize=12)

plt.ylabel('Count', fontsize=12)
cleaned_hotel_data["adr_pp"] = cleaned_hotel_data["adr"] / (cleaned_hotel_data["adults"] + cleaned_hotel_data["children"])



hotel_data_guests = cleaned_hotel_data.loc[cleaned_hotel_data["is_canceled"] == 0] # only actual gusts

room_prices = hotel_data_guests[["hotel", "reserved_room_type", "adr_pp"]].sort_values("reserved_room_type")



# grab data:

room_prices_mothly = hotel_data_guests[["hotel", "arrival_date_month", "adr_pp"]].sort_values("arrival_date_month")



# order by month:

ordered_months = ["January", "February", "March", "April", "May", "June", 

          "July", "August", "September", "October", "November", "December"]

room_prices_mothly["arrival_date_month"] = pd.Categorical(room_prices_mothly["arrival_date_month"], categories=ordered_months, ordered=True)



# barplot with standard deviation:

plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="adr_pp", hue="hotel", data=room_prices_mothly, 

            hue_order = ["City Hotel", "Resort Hotel"], ci="sd", size="hotel", sizes=(2.5, 2.5))

plt.title("Room price per night and person over the year", fontsize=16)

plt.xlabel("Month", fontsize=16)

plt.xticks(rotation=45)

plt.ylabel("Price [EUR]", fontsize=16)

plt.show()
# Create a DateFrame with the relevant data:

rh["total_nights"] = rh["stays_in_weekend_nights"] + rh["stays_in_week_nights"]

ch["total_nights"] = ch["stays_in_weekend_nights"] + ch["stays_in_week_nights"]



num_nights_res = list(rh["total_nights"].value_counts().index)

num_bookings_res = list(rh["total_nights"].value_counts())

rel_bookings_res = rh["total_nights"].value_counts() / sum(num_bookings_res) * 100 # convert to percent



num_nights_cty = list(ch["total_nights"].value_counts().index)

num_bookings_cty = list(ch["total_nights"].value_counts())

rel_bookings_cty = ch["total_nights"].value_counts() / sum(num_bookings_cty) * 100 # convert to percent



res_nights = pd.DataFrame({"hotel": "Resort hotel",

                           "num_nights": num_nights_res,

                           "rel_num_bookings": rel_bookings_res})



cty_nights = pd.DataFrame({"hotel": "City hotel",

                           "num_nights": num_nights_cty,

                           "rel_num_bookings": rel_bookings_cty})



nights_data = pd.concat([res_nights, cty_nights], ignore_index=True)
plt.figure(figsize=(16, 8))

sns.barplot(x = "num_nights", y = "rel_num_bookings", hue="hotel", data=nights_data,

            hue_order = ["City hotel", "Resort hotel"])

plt.title("Length of stay", fontsize=16)

plt.xlabel("Number of nights", fontsize=16)

plt.ylabel("Guests [%]", fontsize=16)

plt.legend(loc="upper right")

plt.xlim(0,22)

plt.show()
## Creating new feature: `Weekday vs Weekend` 



pd.options.mode.chained_assignment = None

def week_function(feature1, feature2, data_source):

    data_source['weekend_or_weekday'] = 0

    for i in range(0, len(data_source)):

        if feature2.iloc[i] == 0 and feature1.iloc[i] > 0:

            cleaned_hotel_data['weekend_or_weekday'].iloc[i] = 'stay_just_weekend'

        if feature2.iloc[i] > 0 and feature1.iloc[i] == 0:

            cleaned_hotel_data['weekend_or_weekday'].iloc[i] = 'stay_just_weekday'

        if feature2.iloc[i] > 0 and feature1.iloc[i] > 0:

            cleaned_hotel_data['weekend_or_weekday'].iloc[i] = 'stay_both_weekday_and_weekend'

        if feature2.iloc[i] == 0 and feature1.iloc[i] == 0:

            cleaned_hotel_data['weekend_or_weekday'].iloc[i] = 'undefined_data'



            

week_function(cleaned_hotel_data['stays_in_weekend_nights'],cleaned_hotel_data['stays_in_week_nights'], cleaned_hotel_data)
#`arrival_date_month` vs `weekend_or_weekday` graph 



# cleaned_hotel_data['arrival_date_month']= cleaned_hotel_data['arrival_date_month'].astype('int64')

group_data = cleaned_hotel_data.groupby([ 'arrival_date_month','weekend_or_weekday']).size().unstack(fill_value=0)

group_data.sort_values('arrival_date_month', ascending = True).plot(kind='bar',stacked=True, cmap='Set2',figsize=(15,10))

plt.title('Arrival Month vs Staying Weekend or Weekday', weight='bold')

plt.xlabel('Arrival Month', fontsize=12)

plt.xticks(rotation=360)

plt.ylabel('Count', fontsize=12)
import plotly.express as px

segments=cleaned_hotel_data["market_segment"].value_counts()



# pie plot

fig = px.pie(segments,

             values=segments.values,

             names=segments.index,

             title="Bookings per market segment",

             )

fig.update_traces(rotation=-90, textinfo="percent+label")

fig.show()
# `Market_segment` feature exploration



plt.figure(figsize=(10,10))

sns.countplot(cleaned_hotel_data['market_segment'], palette='spring_r', 

              order=pd.value_counts(cleaned_hotel_data['market_segment']).index)

plt.title('Market Segment Types', weight='bold')

plt.xlabel('Market Segment', fontsize=12)

plt.ylabel('Count', fontsize=12)
# price per night (ADR) and person based on booking and room.

# show figure:

plt.figure(figsize=(12, 8))

sns.barplot(x="market_segment",

            y="adr_pp",

            hue="reserved_room_type",

            data=cleaned_hotel_data,

            ci="sd",

            errwidth=1,

            capsize=0.1)

plt.title("ADR by market segment and room type", fontsize=16)

plt.xlabel("Market segment", fontsize=16)

plt.xticks(rotation=45)

plt.ylabel("ADR per person [EUR]", fontsize=16)

plt.legend(loc="upper left")

plt.show()
meal_labels= ['BB','HB', 'SC', 'FB']

size = cleaned_hotel_data['meal'].value_counts()

plt.figure(figsize=(10,10))

cmap =plt.get_cmap("Pastel2")

colors = cmap(np.arange(3)*4)

my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(size, labels=meal_labels, colors=colors, wedgeprops = { 'linewidth' : 5, 'edgecolor' : 'white' })

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title('Meal Types', weight='bold')

plt.show()
# Create Top 10 Country of Origin graph



plt.figure(figsize=(10,10))

sns.countplot(x='country', data=cleaned_hotel_data, 

              order=pd.value_counts(cleaned_hotel_data['country']).iloc[:10].index, palette="brg")

plt.title('Top 10 Country of Origin', weight='bold')

plt.xlabel('Country', fontsize=12)

plt.ylabel('Count', fontsize=12)
# `total_of_special_requests` graph



plt.figure(figsize=(10,10))

sns.countplot(x='total_of_special_requests', data=cleaned_hotel_data, palette = 'ocean_r')

plt.title('Total Special Request', weight='bold')

plt.xlabel('Number of Special Request', fontsize=12)

plt.ylabel('Count', fontsize=12)
# Group by `total_of_special_requests` and `is_canceled` features



group_adr_request = cleaned_hotel_data.groupby([ 'total_of_special_requests', 'is_canceled']).size().unstack(fill_value=0)

group_adr_request.plot(kind='bar', stacked=True, cmap='vlag', figsize=(10,10))

plt.title('Total Special Request vs Booking Cancellation Status', weight='bold')

plt.xlabel('Number of Special Request', fontsize=12)

plt.xticks(rotation=360)

plt.ylabel('Count', fontsize=12)
#Create new dataframe for categorical data



labelencoder = LabelEncoder()

cat_features = ['hotel','arrival_date_month','meal',

                                     'country','market_segment','distribution_channel', 

                                     'is_repeated_guest', 'reserved_room_type',

                                     'assigned_room_type','deposit_type','agent',

                                     'customer_type','reservation_status', 

                                     'weekend_or_weekday']



for feature in cat_features:

    cleaned_hotel_data[feature] = labelencoder.fit_transform(cleaned_hotel_data[feature])





hotel_data_categorical = cleaned_hotel_data[cat_features + ['is_canceled']]

# hotel_data_categorical.info()







#Create new dataframe for numerical data



hotel_data_numerical= cleaned_hotel_data.drop(cat_features + ['is_canceled'], axis = 1)

# hotel_data_numerical.info()





# Correlation Matrix with Spearman method



plt.figure(figsize=(15,15))

corr_categorical=hotel_data_categorical.corr(method='spearman')

mask_categorical = np.triu(np.ones_like(corr_categorical, dtype=np.bool))

sns.heatmap(corr_categorical, annot=True, fmt=".2f", cmap='BrBG', vmin=-1, vmax=1, center= 0,

            square=True, linewidths=2, cbar_kws={"shrink": .5}).set(ylim=(15, 0))

plt.title("Correlation Matrix Spearman Method- Categorical Data ",size=15, weight='bold')




# Correlation Matrix with pearson method



plt.figure(figsize=(15,15))

corr_numerical=hotel_data_numerical.corr(method='pearson')

mask_numerical = np.triu(np.ones_like(corr_numerical, dtype=np.bool))

sns.heatmap(corr_numerical, annot=True, fmt=".2f", cmap='RdBu', mask= mask_numerical, vmin=-1, vmax=1, center= 0,

            square=True, linewidths=2, cbar_kws={"shrink": .5}).set(ylim=(17, 0))

plt.title("Correlation Matrix Pearson Method- Numerical Data ",size=15, weight='bold')



# `reservation_status` vs `is_canceled` table

pd.crosstab(columns = cleaned_hotel_data['reservation_status'], index = cleaned_hotel_data['is_canceled'],

           margins=True, margins_name = 'Total')
#Dropping some features from data



hotel_data_model = cleaned_hotel_data.drop(['reservation_status',

                                            'children',

                                            'reservation_status_date',

                                            'adr_pp',

                                           ], axis=1)

# Seperate target variable

hotel_data_tunning = hotel_data_model

y = hotel_data_tunning.iloc[:,1]

X = pd.concat([hotel_data_tunning.iloc[:,0],hotel_data_tunning.iloc[:,2:30]], axis=1)



# X = cleaned_hotel_data.drop(["is_canceled"], axis=1)

# y = cleaned_hotel_data["is_canceled"]
# X.info()



params = {

#     'criterion': 'giny', 

    'learning_rate': 0.01, 

    'max_depth': 5,

    'n_estimators': 100, 

    'objective': 'binary:logistic', 

}

model = XGBClassifier(**params)

# fit the model

model.fit(X, y)

# perform permutation importance

result = permutation_importance(model, X, y, scoring='accuracy', n_repeats = 5, n_jobs=-1)

sorted_idx = result.importances_mean.argsort()

# Feature scores table



for i,v in enumerate(sorted_idx):

    print('Feature: %0d, Score: %.5f' % (i,v))




#Permutation Importance graph 



fig, ax = plt.subplots(figsize=(20,15))



ax.boxplot(result.importances[sorted_idx].T,

           vert=False, labels=X.columns[sorted_idx])

ax.set_title("Permutation Importance")

fig.tight_layout()

plt.show()



# group data for deposit_type:

country_cancel_data = cleaned_hotel_data.groupby("country")["is_canceled"].describe()



#show figure:

plt.figure(figsize=(12, 8))

sns.barplot(x=country_cancel_data.index, y=country_cancel_data["mean"] * 100)

plt.title("Effect of country on cancelation", fontsize=16)

plt.xlabel("Country", fontsize=16)

plt.ylabel("Cancelations [%]", fontsize=16)

plt.show()
# Seperate target variable

hotel_data_model = hotel_data_model.drop(['babies', 'days_in_waiting_list', 'assigned_room_type', 'country'], axis=1)

hotel_data_tunning = hotel_data_model

y = hotel_data_tunning.iloc[:,1]

X = pd.concat([hotel_data_tunning.iloc[:,0],hotel_data_tunning.iloc[:,2:30]], axis=1)
# split data into 'kfolds' parts for cross validation,

# use shuffle to ensure random distribution of data:

kfolds = 4 # 4 = 75% train, 25% validation

split = KFold(n_splits=kfolds, shuffle=True, random_state=42)
# print(y)

# print(X)

# print(split)



dtc_model = DecisionTreeClassifier(criterion= 'gini', min_samples_split=8,

                                  min_samples_leaf = 4, max_features = 'auto')



cv_results = cross_val_score(dtc_model, 

                                 X, y, 

                                 cv=split,

                                 scoring="accuracy",

                                 n_jobs=-1)



# output:

min_score = round(min(cv_results), 4)

max_score = round(max(cv_results), 4)

mean_score = round(np.mean(cv_results), 4)

std_dev = round(np.std(cv_results), 4)

print("Decision Tree Classifier")

print(f"cross validation accuarcy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")



cv_predict = cross_val_predict(dtc_model, 

                                 X, y, 

                                 cv=split,

                                 n_jobs=-1)



print(classification_report(y, cv_predict))



conf_matrix = confusion_matrix(y, cv_predict)



plt.figure(figsize=(7,7))

sns.heatmap(conf_matrix,annot=True, fmt="d" ,cbar=False, cmap="tab20").set_ylim([0,2])

plt.title("Decision Tree Classifier", weight='bold')

plt.xlabel('Predicted Labels')

plt.ylabel('Actual Labels')




# Random Forest Model Building



rf_model = RandomForestClassifier(min_samples_leaf = 6, min_samples_split=6,

                                  n_estimators = 100)





cv_results = cross_val_score(rf_model, 

                                 X, y, 

                                 cv=split,

                                 scoring="accuracy",

                                 n_jobs=-1)



# output:

min_score = round(min(cv_results), 4)

max_score = round(max(cv_results), 4)

mean_score = round(np.mean(cv_results), 4)

std_dev = round(np.std(cv_results), 4)

print("Random Forest Classifier")

print(f"cross validation accuarcy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")



cv_predict = cross_val_predict(rf_model, 

                                 X, y, 

                                 cv=split,

                                 n_jobs=-1)



print(classification_report(y, cv_predict))



conf_matrix = confusion_matrix(y, cv_predict)



plt.figure(figsize=(7,7))

sns.heatmap(conf_matrix,annot=True, fmt="d" ,cbar=False, cmap="tab20").set_ylim([0,2])

plt.title("Random Forest Classifier", weight='bold')

plt.xlabel('Predicted Labels')

plt.ylabel('Actual Labels')
etc_model = ExtraTreesClassifier(min_samples_leaf = 7, min_samples_split=2,

                                  n_estimators = 100)



cv_results = cross_val_score(etc_model, 

                                 X, y, 

                                 cv=split,

                                 scoring="accuracy",

                                 n_jobs=-1)



# output:

min_score = round(min(cv_results), 4)

max_score = round(max(cv_results), 4)

mean_score = round(np.mean(cv_results), 4)

std_dev = round(np.std(cv_results), 4)

print("Extra Trees Classifier")

print(f"cross validation accuarcy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")



cv_predict = cross_val_predict(etc_model, 

                                 X, y, 

                                 cv=split,

                                 n_jobs=-1)



print(classification_report(y, cv_predict))



conf_matrix = confusion_matrix(y, cv_predict)



plt.figure(figsize=(7,7))

sns.heatmap(conf_matrix,annot=True, fmt="d" ,cbar=False, cmap="tab20").set_ylim([0,2])

plt.title("Extra Trees Classifier", weight='bold')

plt.xlabel('Predicted Labels')

plt.ylabel('Actual Labels')
xgb_model = XGBClassifier(criterion = 'giny', learning_rate = 0.01, max_depth = 5, n_estimators = 100,

                          objective ='binary:logistic', subsample = 1.0)



cv_results = cross_val_score(xgb_model, 

                                 X, y, 

                                 cv=split,

                                 scoring="accuracy",

                                 n_jobs=-1)



# output:

min_score = round(min(cv_results), 4)

max_score = round(max(cv_results), 4)

mean_score = round(np.mean(cv_results), 4)

std_dev = round(np.std(cv_results), 4)

print("XGB Classifier")

print(f"cross validation accuarcy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")



cv_predict = cross_val_predict(xgb_model, 

                                 X, y, 

                                 cv=split,

                                 n_jobs=-1)



print(classification_report(y, cv_predict))



conf_matrix = confusion_matrix(y, cv_predict)



plt.figure(figsize=(7,7))

sns.heatmap(conf_matrix,annot=True, fmt="d" ,cbar=False, cmap="tab20").set_ylim([0,2])

plt.title("XGB Classifier", weight='bold')

plt.xlabel('Predicted Labels')

plt.ylabel('Actual Labels')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)







# Enhanced RF model

enhanced_rf_model = RandomForestClassifier(n_estimators=160,

                               max_features=0.4,

                               min_samples_split=2,

                               n_jobs=-1,

                               random_state=0)



enhanced_rf_model.fit(X_train, y_train) 

score = enhanced_rf_model.score(X_test, y_test)

print("Enhanced Random Forest Score is : " , score)



feature_importances = pd.DataFrame(enhanced_rf_model.feature_importances_,

                                   index = X.columns,

                                    columns=['importance']).sort_values('importance',

                                                                        ascending=False)

feature_importances.head(10)




# group data for deposit_type:

deposit_cancel_data = cleaned_hotel_data.groupby("deposit_type")["is_canceled"].describe()



#show figure:

plt.figure(figsize=(12, 8))

sns.barplot(x=deposit_cancel_data.index, y=deposit_cancel_data["mean"] * 100)

plt.title("Effect of deposit_type on cancelation", fontsize=16)

plt.xlabel("Deposit type", fontsize=16)

plt.ylabel("Cancelations [%]", fontsize=16)

plt.show()



lead_cancel_data = cleaned_hotel_data.groupby("lead_time")["is_canceled"].describe()

# use only lead_times wih more than 10 bookings for graph:

lead_cancel_data_10 = lead_cancel_data.loc[lead_cancel_data["count"] >= 10]



#show figure:

plt.figure(figsize=(12, 8))

sns.regplot(x=lead_cancel_data_10.index, y=lead_cancel_data_10["mean"].values * 100)

plt.title("Effect of lead time on cancelation", fontsize=16)

plt.xlabel("Lead time", fontsize=16)

plt.ylabel("Cancelations [%]", fontsize=16)

# plt.xlim(0,365)

plt.show()





# group data for adr:

adr_cancel_data = cleaned_hotel_data.groupby("adr")["is_canceled"].describe()

#show figure:

plt.figure(figsize=(12, 8))

sns.regplot(x=adr_cancel_data.index, y=adr_cancel_data["mean"].values * 100)

plt.title("Effect of ADR on cancelation", fontsize=16)

plt.xlabel("ADR", fontsize=16)

plt.ylabel("Cancelations [%]", fontsize=16)

plt.xlim(0,400)

plt.ylim(0,100)

plt.show()


