import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import pyplot

import seaborn as sns



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
# Import Data



hotel_data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
# Show first 10 rows



hotel_data.head(10)
# Data summary



hotel_data.info()
# Hotel types details



plt.figure(figsize=(10,10))

sns.countplot(x='hotel', data = hotel_data, palette='gist_earth')

plt.title('Hotel Types', weight='bold')

plt.xlabel('Hotel', fontsize=12)

plt.ylabel('Count', fontsize=12)
# `is_canceled` graph



plt.figure(figsize=(10,10))

sns.countplot(y='is_canceled', data= hotel_data, palette='gist_stern', orient = 'v')

plt.title('Canceled Situation', weight='bold')

plt.xlabel('Count', fontsize=12)

plt.ylabel('Canceled or Not Canceled', fontsize=12)
# `arrival_date_year` vs `lead_time` vs `is_canceled` exploration with violin plot



plt.figure(figsize=(10,10))

sns.violinplot(x='arrival_date_year', y ='lead_time', hue="is_canceled", data=hotel_data, palette="Set3", bw=.2,

               cut=2, linewidth=2, iner= 'box', split = True)

sns.despine(left=True)

plt.title('Arrival Year vs Lead Time vs Canceled Situation', weight='bold')

plt.xlabel('Year', fontsize=12)

plt.ylabel('Lead Time', fontsize=12)
#`arrival_date_month` names converted to the numbers



hotel_data['arrival_date_month'].replace({'January' : '1',

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



plt.figure(figsize=(15,10))

sns.countplot(x='arrival_date_month', data = hotel_data,

              order=pd.value_counts(hotel_data['arrival_date_month']).index, palette='YlOrBr_r')

plt.title('Arrival Month', weight='bold')

plt.xlabel('Month', fontsize=12)

plt.ylabel('Count', fontsize=12)
# Table of `stay_in_weekend` and `stay_in_week_nights` features



pd.crosstab(index = hotel_data['stays_in_week_nights'],columns=hotel_data['stays_in_weekend_nights'], margins=True, margins_name = 'Total').iloc[:10]
## Creating new feature: `Weekday vs Weekend` 



pd.options.mode.chained_assignment = None

def week_function(feature1, feature2, data_source):

    data_source['weekend_or_weekday'] = 0

    for i in range(0, len(data_source)):

        if feature2.iloc[i] == 0 and feature1.iloc[i] > 0:

            hotel_data['weekend_or_weekday'].iloc[i] = 'stay_just_weekend'

        if feature2.iloc[i] > 0 and feature1.iloc[i] == 0:

            hotel_data['weekend_or_weekday'].iloc[i] = 'stay_just_weekday'

        if feature2.iloc[i] > 0 and feature1.iloc[i] > 0:

            hotel_data['weekend_or_weekday'].iloc[i] = 'stay_both_weekday_and_weekend'

        if feature2.iloc[i] == 0 and feature1.iloc[i] == 0:

            hotel_data['weekend_or_weekday'].iloc[i] = 'undefined_data'



            

week_function(hotel_data['stays_in_weekend_nights'],hotel_data['stays_in_week_nights'], hotel_data)
#`arrival_date_month` vs `weekend_or_weekday` graph 



hotel_data['arrival_date_month']= hotel_data['arrival_date_month'].astype('int64')

group_data = hotel_data.groupby([ 'arrival_date_month','weekend_or_weekday']).size().unstack(fill_value=0)

group_data.sort_values('arrival_date_month', ascending = True).plot(kind='bar',stacked=True, cmap='Set2',figsize=(15,10))

plt.title('Arrival Month vs Staying Weekend or Weekday', weight='bold')

plt.xlabel('Arrival Month', fontsize=12)

plt.xticks(rotation=360)

plt.ylabel('Count', fontsize=12)
# Create new feature:`all_children` with merge children and baby features



hotel_data['all_children'] = hotel_data['children'] + hotel_data['babies']

pd.crosstab(hotel_data['adults'], hotel_data['all_children'], margins=True, margins_name = 'Total').iloc[:10]
# `Meal` feature donut chart



meal_labels= ['BB','HB', 'SC', 'Undefined', 'FB']

size = hotel_data['meal'].value_counts()

plt.figure(figsize=(10,10))

cmap =plt.get_cmap("Pastel2")

colors = cmap(np.arange(3)*4)

my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(size, labels=meal_labels, colors=colors, wedgeprops = { 'linewidth' : 5, 'edgecolor' : 'white' })

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title('Meal Types', weight='bold')

plt.show()
# Groupby `Meal` and `Hotel` features



group_meal_data = hotel_data.groupby(['hotel','meal']).size().unstack(fill_value=0).transform(lambda x: x/x.sum())

group_meal_data.applymap('{:.2f}'.format)
# Create Top 10 Country of Origin graph



plt.figure(figsize=(10,10))

sns.countplot(x='country', data=hotel_data, 

              order=pd.value_counts(hotel_data['country']).iloc[:10].index, palette="brg")

plt.title('Top 10 Country of Origin', weight='bold')

plt.xlabel('Country', fontsize=12)

plt.ylabel('Count', fontsize=12)
# `Market_segment` feature exploration



plt.figure(figsize=(10,10))

sns.countplot(hotel_data['market_segment'], palette='spring_r', 

              order=pd.value_counts(hotel_data['market_segment']).index)

plt.title('Market Segment Types', weight='bold')

plt.xlabel('Market Segment', fontsize=12)

plt.ylabel('Count', fontsize=12)
# Reserved vs Assigned room types table



pd.crosstab(index = hotel_data['reserved_room_type'], 

            columns = hotel_data['assigned_room_type'],normalize='index').round(2)*100
# `Arrival Month` vs `ADR` vs `Booking Cancellation Status`



hotel_data['adr'] = hotel_data['adr'].astype(float)

plt.figure(figsize=(15,10))

sns.barplot(x='arrival_date_month', y='adr', hue='is_canceled', dodge=True, palette= 'PuBu_r', data=hotel_data)

plt.title('Arrival Month vs ADR vs Booking Cancellation Status', weight='bold')

plt.xlabel('Arrival Month', fontsize=12)

plt.ylabel('ADR', fontsize=12)
# `total_of_special_requests` graph



plt.figure(figsize=(10,10))

sns.countplot(x='total_of_special_requests', data=hotel_data, palette = 'ocean_r')

plt.title('Total Special Request', weight='bold')

plt.xlabel('Number of Special Request', fontsize=12)

plt.ylabel('Count', fontsize=12)
# Group by `total_of_special_requests` and `is_canceled` features



group_adr_request = hotel_data.groupby([ 'total_of_special_requests', 'is_canceled']).size().unstack(fill_value=0)

group_adr_request.plot(kind='bar', stacked=True, cmap='vlag', figsize=(10,10))

plt.title('Total Special Request vs Booking Cancellation Status', weight='bold')

plt.xlabel('Number of Special Request', fontsize=12)

plt.xticks(rotation=360)

plt.ylabel('Count', fontsize=12)
## Display sum of null data



hotel_data.isnull().sum()
# Fill missing data



hotel_data['children'] =  hotel_data['children'].fillna(0)

hotel_data['all_children'] =  hotel_data['all_children'].fillna(0)

hotel_data['country'] = hotel_data['country'].fillna(hotel_data['country'].mode().index[0])

hotel_data['agent']= hotel_data['agent'].fillna('0')

hotel_data=hotel_data.drop(['company'], axis =1)
# Change data structure



hotel_data['agent']= hotel_data['agent'].astype(int)

hotel_data['country']= hotel_data['country'].astype(str)
#Using Label Encoder method for categorical features



labelencoder = LabelEncoder()

hotel_data['hotel'] = labelencoder.fit_transform(hotel_data['hotel'])

hotel_data['arrival_date_month'] = labelencoder.fit_transform(hotel_data['arrival_date_month'])

hotel_data['meal'] = labelencoder.fit_transform(hotel_data['meal'])

hotel_data['country'] = labelencoder.fit_transform(hotel_data['country'])

hotel_data['market_segment']= labelencoder.fit_transform(hotel_data['market_segment'])

hotel_data['distribution_channel']=labelencoder.fit_transform(hotel_data['distribution_channel'])

hotel_data['is_repeated_guest'] = labelencoder.fit_transform(hotel_data['is_repeated_guest'])

hotel_data['reserved_room_type'] = labelencoder.fit_transform(hotel_data['reserved_room_type'])

hotel_data['assigned_room_type'] = labelencoder.fit_transform(hotel_data['assigned_room_type'])

hotel_data['deposit_type'] = labelencoder.fit_transform(hotel_data['deposit_type'])

hotel_data['agent'] = labelencoder.fit_transform(hotel_data['agent'])

hotel_data['customer_type'] = labelencoder.fit_transform(hotel_data['customer_type'])

hotel_data['reservation_status'] = labelencoder.fit_transform(hotel_data['reservation_status'])

hotel_data['weekend_or_weekday'] = labelencoder.fit_transform(hotel_data['weekend_or_weekday'])
#Create new dataframe for categorical data



hotel_data_categorical = hotel_data[['hotel','is_canceled','arrival_date_month','meal',

                                     'country','market_segment','distribution_channel', 

                                     'is_repeated_guest', 'reserved_room_type',

                                     'assigned_room_type','deposit_type','agent',

                                     'customer_type','reservation_status', 

                                     'weekend_or_weekday']]

hotel_data_categorical.info()
#Create new dataframe for numerical data



hotel_data_numerical= hotel_data.drop(['hotel','is_canceled', 'arrival_date_month','meal',

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
# `reservation_status` vs `is_canceled` table



pd.crosstab(columns = hotel_data['reservation_status'], index = hotel_data['is_canceled'],

           margins=True, margins_name = 'Total')
#Dropping some features from data



hotel_data = hotel_data.drop(['reservation_status', 'children', 'reservation_status_date'], axis=1)
#Copying data to used next parts



hotel_data_model = hotel_data
# Seperate target variable



hotel_data_tunning = hotel_data

y = hotel_data_tunning.iloc[:,1]

X = pd.concat([hotel_data_tunning.iloc[:,0],hotel_data_tunning.iloc[:,2:30]], axis=1)
## Finding parameters for XGBoost model



# model = XGBClassifier()

# parameters = {

# 'n_estimators' : [100,250,500],

# 'learning_rate' : [0.01, 0.1],

# 'subsample' :[0.5, 1.0],

# 'max_depth' : [3,5,7],

# 'criterion' : ['giny','entropy'],

# 'objective':['binary:logistic'],

# }



# grid_search = GridSearchCV(estimator=model, param_grid=parameters,

#                           cv=5, scoring='f1', verbose=True, n_jobs=-1)

# grid_search.fit(X, y)

# print(grid_search.best_score_)

# print(grid_search.best_params_)
## Finding parameters for RF model



# model_rfc_gs = RandomForestClassifier()

# parameters_rfc = {

# 'n_estimators' : [100,200,500],

# 'min_samples_split' : [2,4,6,8],

# 'min_samples_leaf': [1,2,4,6]

# }



# grid_search_rfc = GridSearchCV(estimator=model_rfc_gs, param_grid=parameters_rfc,

#                           cv=5, scoring='f1', verbose=True, n_jobs=-1)

# grid_search_rfc.fit(X, y)

# grid_search_rfc.best_params_
## Finding parameters for Extra Tree Classifier



# model_etc_gs = ExtraTreesClassifier()

# parameters_etc = {

# 'n_estimators' : [100,250,500],

# 'min_samples_split' : [2,4,6,8],

# 'min_samples_leaf': [1,3,5,7]

# }



# grid_search_etc = GridSearchCV(estimator=model_etc_gs, param_grid=parameters_etc,

#                           cv=5, scoring='f1', verbose=True, n_jobs=-1)

# grid_search_etc.fit(X, y)

# grid_search_etc.best_params_
## Finding parameters for Decision Tree



# model_dtc_gs = DecisionTreeClassifier()

# parameters_dtc = {

# 'criterion' : ['gini', 'entropy'],

# 'min_samples_split' : [2,4,6,8],

# 'min_samples_leaf': [1,2,3,4,5],

# 'max_features' : ['auto', 'sqrt']

# }



# grid_search_dtc = GridSearchCV(estimator=model_dtc_gs, param_grid=parameters_dtc,

#                           cv=5, scoring='f1', verbose=True, n_jobs =-1)

# grid_search_dtc.fit(X, y)

# grid_search_dtc.best_params_
# Permutation Importance graph with XGB Classifier algorithm.



params = {

    'criterion': 'giny', 

    'learning_rate': 0.01, 

    'max_depth': 5,

    'n_estimators': 100, 

    'objective': 'binary:logistic', 

}

model = XGBClassifier(parameters=params)

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
# Drop `baby` feature from data



hotel_data_model = hotel_data_model.drop(['babies'], axis=1)
# Seperate target variable for model building 



y_model = hotel_data_model.iloc[:,1]

X_model = pd.concat([hotel_data_tunning.iloc[:,0],hotel_data_tunning.iloc[:,2:30]], axis=1)

y_model.describe()
# Split to train and test with 70-30 ratio



X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.3, random_state=42, stratify = y)
# Implement standart scaler method



standardScalerX = StandardScaler()

X_train = standardScalerX.fit_transform(X_train)

X_test = standardScalerX.fit_transform(X_test)
# Stratified K-Fold Cross Validation Method



kfold_cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True)



for train_index, test_index in kfold_cv.split(X_model,y_model):

    X_train, X_test = X_model.iloc[train_index], X_model.iloc[test_index]

    y_train, y_test = y_model.iloc[train_index], y_model.iloc[test_index]
# Decision Tree Model Building



dtc_model = DecisionTreeClassifier(criterion= 'gini', min_samples_split=8,

                                  min_samples_leaf = 4, max_features = 'auto')

# fit the model

dtc_model.fit(X_train, y_train)



#Predict Model

predict_dtc = dtc_model.predict(X_test)
# Random Forest Model Building



rf_model = RandomForestClassifier(min_samples_leaf = 6, min_samples_split=6,

                                  n_estimators = 100)



# fit the model

estimator= rf_model.fit(X_train, y_train)

#Predict Model

predict_rf = rf_model.predict(X_test)
# Extra Treees Classsifier Model Building



etc_model = ExtraTreesClassifier(min_samples_leaf = 7, min_samples_split=2,

                                  n_estimators = 100)

# fit the model

etc_model.fit(X_train, y_train)



#Predict Model

predict_etc = etc_model.predict(X_test)
# Extreme Gradient Boosting Model Building



xgb_model = XGBClassifier(criterion = 'giny', learning_rate = 0.01, max_depth = 5, n_estimators = 100,

                          objective ='binary:logistic', subsample = 1.0)

# fit the model

xgb_model.fit(X_train, y_train)

#Predict Model

predict_xgb = xgb_model.predict(X_test)
# Classification Reports 



print("RF", classification_report(y_test, predict_rf))

print("DTC",classification_report(y_test, predict_dtc))

print("ETC", classification_report(y_test, predict_etc))

print("XGB", classification_report(y_test, predict_xgb))
# Confusion Matrix 



DTC_matrix = confusion_matrix(y_test, predict_dtc)

RF_matrix = confusion_matrix(y_test, predict_rf)

ETC_matrix = confusion_matrix(y_test, predict_etc)

XGB_matrix = confusion_matrix(y_test, predict_xgb) 



fig, ax = plt.subplots(1, 2, figsize=(15, 8))

sns.heatmap(RF_matrix,annot=True, fmt="d", cbar=False, cmap="Pastel2",  ax = ax[0]).set_ylim([0,2])

ax[0].set_title("Random Forest", weight='bold')

ax[0].set_xlabel('Predicted Labels')

ax[0].set_ylabel('Actual Labels')

sns.heatmap(DTC_matrix,annot=True, fmt="d" ,cbar=False, cmap="tab20", ax = ax[1]).set_ylim([0,2])

ax[1].set_title("Decision Tree", weight='bold')

ax[1].set_xlabel('Predicted Labels')

ax[1].set_ylabel('Actual Labels')



fig, axe = plt.subplots(1, 2, figsize=(15, 8))

sns.heatmap(ETC_matrix,annot=True, fmt="d", cbar=False, cmap="Paired", ax = axe[0]).set_ylim([0,2])

axe[0].set_title("Extra Tree Classifier", weight='bold')

axe[0].set_xlabel('Predicted Labels')

axe[0].set_ylabel('Actual Labels')

sns.heatmap(XGB_matrix,annot=True, fmt="d", cbar=False, cmap="Pastel1", ax = axe[1]).set_ylim([0,2])

axe[1].set_title("Gradient Boosting", weight='bold')

axe[1].set_xlabel('Predicted Labels')

axe[1].set_ylabel('Actual Labels')