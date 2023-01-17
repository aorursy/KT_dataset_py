import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix, auc, roc_curve, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier
# Import Data

hotel_df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
# Show first 5 rows

hotel_df.head(5)
# print some information about data

hotel_df.info()
# print the size of the data
hotel_df.shape
plt.figure(figsize=(8,6))
sns.countplot(x='hotel', data = hotel_df, palette='gist_earth')
plt.title('Hotel Types', weight='bold')
plt.xlabel('Hotel', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.figure(figsize=(8,6))
sns.countplot(x='is_canceled', data= hotel_df, palette='gist_stern')
plt.title('Canceled Situation', weight='bold')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Canceled or Not Canceled', fontsize=12)
plt.figure(figsize=(8,6))
sns.violinplot(x='arrival_date_year', y ='lead_time', hue="is_canceled", data=hotel_df, palette="Set2", bw=.2,
               cut=2, linewidth=2, iner= 'box', split = True)
sns.despine(left=True)
plt.title('Arrival Year vs Lead Time vs Canceled Situation', weight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Lead Time', fontsize=12)
#`arrival_date_month` names converted to the numbers

hotel_df['arrival_date_month'].replace({'January' : '1',
        'February' : '2',
        'March' : '3',
        'April' : '4',
        'May' : '5',
        'June' : '6',
        'July' : '7',
        'August' : '8',
        'September' : '9', 
        'October' : '10',
        'November' : '11',
        'December' : '12'}, inplace=True)
#`arrival_date_month` exploration 

plt.figure(figsize=(10,10))
sns.countplot(x='arrival_date_month', data = hotel_df,
              order=pd.value_counts(hotel_df['arrival_date_month']).index, palette='YlOrBr_r')
plt.title('Arrival Month', weight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
# Table of `stay_in_weekend` and `stay_in_week_nights` features

pd.crosstab(index = hotel_df['stays_in_week_nights'],columns=hotel_df['stays_in_weekend_nights'], margins=True, margins_name = 'Total').iloc[:10]
## Creating new feature: `Weekday vs Weekend` 

pd.options.mode.chained_assignment = None
def week_function(feature1, feature2, data_source):
    data_source['weekend_or_weekday'] = 0
    for i in range(0, len(data_source)):
        if feature2.iloc[i] == 0 and feature1.iloc[i] > 0:
            hotel_df['weekend_or_weekday'].iloc[i] = 'stay_just_weekend'
        if feature2.iloc[i] > 0 and feature1.iloc[i] == 0:
            hotel_df['weekend_or_weekday'].iloc[i] = 'stay_just_weekday'
        if feature2.iloc[i] > 0 and feature1.iloc[i] > 0:
            hotel_df['weekend_or_weekday'].iloc[i] = 'stay_both_weekday_and_weekend'
        if feature2.iloc[i] == 0 and feature1.iloc[i] == 0:
            hotel_df['weekend_or_weekday'].iloc[i] = 'undefined_data'

            
week_function(hotel_df['stays_in_weekend_nights'],hotel_df['stays_in_week_nights'], hotel_df)
#`arrival_date_month` vs `weekend_or_weekday` graph 

hotel_df['arrival_date_month']= hotel_df['arrival_date_month'].astype('int64')
group_data = hotel_df.groupby([ 'arrival_date_month','weekend_or_weekday']).size().unstack(fill_value=0)

group_data.sort_values('arrival_date_month', ascending = True).plot(kind='bar',stacked=True, cmap='Set3',figsize=(12,8))
plt.title('Arrival Month vs Staying Weekend or Weekday', weight='bold')
plt.xlabel('Arrival Month', fontsize=12)
plt.xticks(rotation=360)
plt.ylabel('Count', fontsize=12)
# Create new feature:`all_children` with merge children and baby features

hotel_df['all_children'] = hotel_df['children'] + hotel_df['babies']
pd.crosstab(hotel_df['adults'], hotel_df['all_children'], margins=True, margins_name = 'Total')
# Groupby `Meal` and `Hotel` features

group_meal_data = hotel_df.groupby(['hotel','meal']).size().unstack(fill_value=0).transform(lambda x: x/x.sum())
group_meal_data.applymap('{:.2f}'.format)
# Create Top 10 Country of Origin graph

plt.figure(figsize=(10,10))
sns.countplot(x='country', data=hotel_df, 
              order=pd.value_counts(hotel_df['country']).iloc[:10].index, palette="brg")
plt.title('Top 10 Country of Origin', weight='bold')
plt.xlabel('Country', fontsize=12)
plt.ylabel('Count', fontsize=12)
# `Arrival Month` vs `ADR` vs `Booking Cancellation Status`

hotel_df['adr'] = hotel_df['adr'].astype(float)
plt.figure(figsize=(15,10))
sns.barplot(x='arrival_date_month', y='adr', hue='is_canceled', dodge=True, palette= 'PuBu_r', data=hotel_df)
plt.title('Arrival Month vs ADR vs Booking Cancellation Status', weight='bold')
plt.xlabel('Arrival Month', fontsize=12)
plt.ylabel('ADR', fontsize=12)
## Display sum of null data

hotel_df.isnull().sum()
# Fill missing data

hotel_df['children'] =  hotel_df['children'].fillna(0)
hotel_df['all_children'] =  hotel_df['all_children'].fillna(0)
hotel_df['country'] = hotel_df['country'].fillna(hotel_df['country'].mode().index[0])
hotel_df['agent']= hotel_df['agent'].fillna('0')
hotel_df=hotel_df.drop(['company'], axis =1)
# Change data type

hotel_df['agent']= hotel_df['agent'].astype(int)
#hotel_df['country']= hotel_df['country'].astype(O)
#Using Label Encoder method for categorical features

cols =  [cols for cols in hotel_df.columns if hotel_df[cols].dtype == 'O']

hotel_df.loc[:, cols] = hotel_df.loc[:, cols].astype(str).apply(LabelEncoder().fit_transform)
hotel_df.head()
#Create new dataframe for categorical data

hotel_data_categorical = hotel_df[['hotel','is_canceled','arrival_date_month','meal',
                                     'country','market_segment','distribution_channel', 
                                     'is_repeated_guest', 'reserved_room_type',
                                     'assigned_room_type','deposit_type','agent',
                                     'customer_type','reservation_status', 
                                     'weekend_or_weekday']]
hotel_data_categorical.info()
#Create new dataframe for numerical data

hotel_data_numerical= hotel_df.drop(['hotel','is_canceled', 'arrival_date_month','meal',
                                       'country','market_segment','distribution_channel', 
                                       'is_repeated_guest', 'reserved_room_type', 
                                       'assigned_room_type','deposit_type','agent', 
                                       'customer_type','reservation_status',
                                       'weekend_or_weekday'], axis = 1)
hotel_data_numerical.info()
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
# Finding high correlated features

corr_mask_categorical = corr_categorical.mask(mask_categorical)
corr_values_categorical = [c for c in corr_mask_categorical.columns if any (corr_mask_categorical[c] > 0.90)]
corr_mask_numerical = corr_numerical.mask(mask_numerical)
corr_values_numerical = [c for c in corr_mask_numerical.columns if any (corr_mask_numerical[c] > 0.90)]
print(corr_values_categorical, corr_values_numerical)
# drop the highly correlated features

hotel_df = hotel_df.drop(['reservation_status', 'children', 'reservation_status_date'], axis=1)
# Seperate target variable

hotel_data_tunning = hotel_df
y = hotel_data_tunning.iloc[:,1]
x = pd.concat([hotel_data_tunning.iloc[:,0],hotel_data_tunning.iloc[:,2:30]], axis=1)
# train and test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
print('X train size: ', x_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', x_test.shape)
print('y test size: ', y_test.shape)
# Create adaboost classifer object
lr = LogisticRegression()

# Train Adaboost Classifer
lr.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = lr.predict(x_test)
precision_score_lr =  precision_score(y_test, y_pred)
accuracy_score_lr = accuracy_score(y_test, y_pred)
print('The precision score is : ',round(precision_score_lr * 100,2), '%')
print('The accuracy score is : ',round(accuracy_score_lr * 100,2), '%')
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred))
# base estimator (optional)
dt = DecisionTreeClassifier() 

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=250, base_estimator=dt,learning_rate=1.0, random_state=0)

# Train Adaboost Classifer
abc.fit(x_train, y_train)

#Predict the response for test dataset
y_pred_lg = abc.predict(x_test)
precision_score_ab =  precision_score(y_test, y_pred_lg)
accuracy_score_ab = accuracy_score(y_test, y_pred_lg)
print('The precision score is : ',round(precision_score_ab * 100,2), '%')
print('The accuracy score is : ',round(accuracy_score_ab * 100,2), '%')
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred_lg))
# create object
gbc= GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=10, min_samples_split=200, max_features='sqrt',random_state=10)

# Train Adaboost Classifer
gbc.fit(x_train, y_train)

#Predict the response for test dataset
y_pred_gbc = gbc.predict(x_test)
precision_score_gbc =  precision_score(y_test, y_pred_gbc)
accuracy_score_gbc = accuracy_score(y_test, y_pred_gbc)
print('The precision score  is : ',round(precision_score_gbc * 100,2), '%')
print('The accuracy score  is : ',round(accuracy_score_gbc * 100,2), '%')
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred_gbc))
xgbc = XGBClassifier(max_depth=13,n_estimators=300,learning_rate=0.5)
    
# Train Adaboost Classifer
xgbc.fit(x_train, y_train)

#Predict the response for test dataset
y_pred_xgbc = xgbc.predict(x_test)
precision_score_xgbc =  precision_score(y_test, y_pred_xgbc)
accuracy_score_xgbc = accuracy_score(y_test, y_pred_xgbc)
print('The precision score  is : ',round(precision_score_xgbc * 100,2), '%')
print('The accuracy score is : ',round(accuracy_score_xgbc * 100,2), '%')
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred_xgbc))
print('Logistic Regression accuracy score is : ',round(accuracy_score_lr * 100,2), '%')
print('AdaBoost accuracy score is : ',round(accuracy_score_ab * 100,2), '%')
print('Gradient boosting  accuracy score  is : ',round(accuracy_score_gbc * 100,2), '%')
print('XGBoost accuracy score is : ',round(accuracy_score_xgbc * 100,2), '%')
