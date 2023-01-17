import pandas as pd

import numpy as np

import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow
def plot_ann_auc(model,X_test,y_test,model_name='model_name'):

    n_probs = [0 for _ in range(len(y_test))]

    x_probs = model.predict_proba(X_test)

    x_probs = x_probs[:,0]

    n_auc = roc_auc_score(y_test, n_probs)

    x_auc = roc_auc_score(y_test, x_probs)



    print('No Skill Rate AUC: ',n_auc)

    print('Learned AUC: ',x_auc)



    n_fpr, n_tpr, _ = roc_curve(y_test,n_probs)

    l_fpr, l_tpr, _ = roc_curve(y_test,x_probs)



    plt.figure(figsize=(16,8))

    plt.plot(n_fpr,n_tpr,linestyle='--',label='No skill rate')

    plt.plot(l_fpr,l_tpr,marker='.',label=model_name)

    plt.xlabel('FPR')

    plt.ylabel('TPR')

    plt.legend()

    plt.show()
def plot_prob_auc(model,X_test,y_test,model_name='model_name'):

    n_probs = [0 for _ in range(len(y_test))]

    x_probs = model.predict_proba(X_test)

    x_probs = x_probs[:, 1]

    n_auc = roc_auc_score(y_test, n_probs)

    x_auc = roc_auc_score(y_test, x_probs)



    print('No Skill Rate AUC: ',n_auc)

    print('Learned AUC: ',x_auc)



    n_fpr, n_tpr, _ = roc_curve(y_test,n_probs)

    l_fpr, l_tpr, _ = roc_curve(y_test,x_probs)



    plt.figure(figsize=(16,8))

    plt.plot(n_fpr,n_tpr,linestyle='--',label='No skill rate')

    plt.plot(l_fpr,l_tpr,marker='.',label=model_name)

    plt.xlabel('FPR')

    plt.ylabel('TPR')

    plt.legend()

    plt.show()
data = pd.read_csv('../input/deep-solar-dataset/deepsolar_tract.csv', delimiter=',',encoding='latin-1')
use_cols = ['tile_count','average_household_income','county','education_bachelor','education_college',

            'education_doctoral','education_high_school_graduate','education_less_than_high_school',

            'education_master','education_population','education_professional_school','land_area',

            'per_capita_income','population','population_density','poverty_family_below_poverty_level',

            'poverty_family_count','state','total_area','unemployed','water_area','employ_rate',

            'poverty_family_below_poverty_level_rate','median_household_income','electricity_price_residential',

            'electricity_price_commercial','electricity_price_industrial','electricity_price_transportation',

            'electricity_price_overall','electricity_consume_residential','electricity_consume_commercial',

            'electricity_consume_industrial','electricity_consume_total','household_count','average_household_size',

            'housing_unit_count','housing_unit_occupied_count','housing_unit_median_value',

            'housing_unit_median_gross_rent','heating_design_temperature',

            'cooling_design_temperature','earth_temperature_amplitude','frost_days','air_temperature',

            'relative_humidity','daily_solar_radiation','atmospheric_pressure','wind_speed','earth_temperature',

            'heating_degree_days','cooling_degree_days','age_18_24_rate','age_25_34_rate','age_more_than_85_rate',

            'age_75_84_rate','age_35_44_rate','age_45_54_rate','age_65_74_rate','age_55_64_rate','age_10_14_rate',

            'age_15_17_rate','age_5_9_rate','household_type_family_rate','dropout_16_19_inschool_rate',

            'occupation_construction_rate','occupation_public_rate','occupation_information_rate',

            'occupation_finance_rate','occupation_education_rate','occupation_administrative_rate',

            'occupation_manufacturing_rate','occupation_wholesale_rate','occupation_retail_rate',

            'occupation_transportation_rate','occupation_arts_rate','occupation_agriculture_rate',

            'occupancy_vacant_rate','occupancy_owner_rate','mortgage_with_rate','transportation_home_rate',

            'transportation_car_alone_rate','transportation_walk_rate','transportation_carpool_rate',

            'transportation_motorcycle_rate','transportation_bicycle_rate','transportation_public_rate',

            'travel_time_less_than_10_rate','travel_time_10_19_rate','travel_time_20_29_rate',

            'travel_time_30_39_rate','travel_time_40_59_rate','travel_time_60_89_rate','health_insurance_public_rate',

            'health_insurance_none_rate','age_median','travel_time_average','voting_2016_dem_percentage',

            'voting_2016_gop_percentage','voting_2012_dem_percentage','voting_2012_gop_percentage',

            'number_of_years_of_education','diversity','incentive_count_residential',

            'incentive_count_nonresidential','incentive_residential_state_level','incentive_nonresidential_state_level',

            'net_metering','feedin_tariff','cooperate_tax','property_tax','sales_tax','rebate','avg_electricity_retail_rate']
# features not used per the aforementioned reasoning

dropped = [col for col in data.columns if col not in use_cols]

np.array(dropped).flatten()
df = data[use_cols]

# A look at the predicted variable of which we'll create a proxy, could be used for regression

df.tile_count.describe()
# Set new column where fips sub-county areas return 1 if there are solar panels, else 0

df.loc[df.tile_count == 0,'adoption'] = 0.0

df.loc[(df.tile_count>0),'adoption'] = 1
# Quick look at class sizes

df.adoption.value_counts()
# most regulations for utilities span intra-state, so we'll get a glimps of what to expect

# by looking at the states adoption levels, required for location is the state shortname

df.state = df.state.str.upper()

state_df = df.groupby('state').sum()
fig = px.choropleth(locations=state_df.index, locationmode="USA-states", color=state_df.tile_count, scope="usa")

fig.show()
corr = df.drop(['tile_count','county','state'],axis=1).corr()

plt.figure(figsize=(30,30))

fig = sns.heatmap(data=corr,cmap='binary')
from sklearn.preprocessing import MinMaxScaler,LabelEncoder

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, roc_curve

from sklearn.model_selection import GridSearchCV, train_test_split, ShuffleSplit
# encode a numeric state and county name value

encoder = LabelEncoder()

df['county_code'] = encoder.fit_transform(df.county)

df['state_code'] = encoder.fit_transform(df.state)

df.loc[df.electricity_price_transportation==df.electricity_price_transportation.value_counts().index[0],'electricity_price_transportation'] = 0

df.electricity_price_transportation = df.electricity_price_transportation.astype(float)
state_meds = df.groupby('state').median()

for col in state_meds.columns:

    print(col,' : ',round((state_meds[col].isna().sum()/72537)*100,2),'%')
county_meds = df.groupby('county').median()

for col in county_meds.columns:

    print(col,' : ',round((county_meds[col].isna().sum()/72537)*100,2),'%')
county_impute = ['average_household_income','land_area','per_capita_income','population_density',

                 'total_area','water_area','employ_rate','poverty_family_below_poverty_level_rate',

                 'median_household_income','average_household_size','housing_unit_median_value',

                 'housing_unit_median_gross_rent','age_18_24_rate','age_25_34_rate','age_more_than_85_rate',

                 'age_75_84_rate','age_35_44_rate','age_45_54_rate','age_65_74_rate','age_55_64_rate','age_10_14_rate',

                 'age_15_17_rate','age_5_9_rate','household_type_family_rate','dropout_16_19_inschool_rate',

                 'occupation_construction_rate','occupation_public_rate','occupation_information_rate','occupation_finance_rate',

                 'occupation_education_rate','occupation_administrative_rate','occupation_manufacturing_rate',

                 'occupation_wholesale_rate','occupation_retail_rate','occupation_transportation_rate','occupation_arts_rate',

                 'occupation_agriculture_rate','occupancy_vacant_rate','occupancy_owner_rate','mortgage_with_rate',

                 'transportation_home_rate','transportation_car_alone_rate','transportation_walk_rate','transportation_carpool_rate',

                 'transportation_motorcycle_rate','transportation_bicycle_rate','transportation_public_rate',

                 'travel_time_less_than_10_rate','travel_time_10_19_rate','travel_time_20_29_rate','travel_time_30_39_rate',

                 'travel_time_40_59_rate','travel_time_60_89_rate','health_insurance_public_rate','health_insurance_none_rate',

                 'age_median','travel_time_average','voting_2012_dem_percentage','voting_2012_gop_percentage',

                 'number_of_years_of_education','diversity']



state_impute = ['heating_design_temperature','cooling_design_temperature',

                'earth_temperature_amplitude','frost_days','air_temperature','relative_humidity',

                'daily_solar_radiation','atmospheric_pressure','wind_speed','earth_temperature',

                'heating_degree_days','cooling_degree_days','voting_2012_dem_percentage','voting_2012_gop_percentage']
# impute null values with the median in their respective county or state if not available

# reason: nearest neighbor approach



for col in county_impute:

    df[col] = df.groupby('county')[col].transform(lambda x: x.fillna(x.median()))

    

for col in state_impute:

    df[col] = df.groupby('state')[col].transform(lambda x: x.fillna(x.median()))
# impute null voting results with median if no median was available

df.loc[df.voting_2012_dem_percentage.isnull(),'voting_2012_dem_percentage'] = df.voting_2012_dem_percentage.median()

df.loc[df.voting_2012_gop_percentage.isnull(),'voting_2012_gop_percentage'] = df.voting_2012_gop_percentage.median()
# Check for nulls

for col in df.columns:

    print(col,' : ',round((df[col].isna().sum()/72537)*100,2),'%')
# although no null values are explicitly found, there are some values I'm assuming were meant to be null

# within the dataset, as some functions which handle only numeric were throwing Nan errors

# decided simply to drop the rest of the nulls

df = df.dropna()
# set X and y variables and train test for decision tree classifiers or potentially bayesian classifiers

X = df.drop(['tile_count','adoption','county','state'],axis=1)

y = df['adoption']

X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.3)
# scale data for kmeans or others

scaler = MinMaxScaler()

scaled_data = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)

scaled_data['adoption'] = df['adoption']

scaled_data = scaled_data.dropna()

# split scaled data, prepare for models

X_scaled = scaled_data.drop('adoption',axis=1)

y_scaled = scaled_data['adoption']

X_trains, X_tests, y_trains, y_tests = train_test_split(X_scaled, y_scaled, random_state=42, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)

print(classification_report(y_test,rf_pred))
# ran on local environment, see results below

"""

rf_auc = GridSearchCV(rf,param_grid={'n_estimators':np.arange(80,240,20),

                                     'criterion':['gini','entropy'],

                                     'max_depth':[5,10,25,50,100],

                                     'min_samples_split':[10,50,250,500]},

      NOTICE!!!  -->  scoring='roc_auc',

                      cv=5,

                      n_jobs=-1)

rf_auc.fit(X_train,y_train)

"""
rf_clf = RandomForestClassifier(n_estimators=500,

                                max_depth=25,

                                criterion='entropy',

                                min_samples_split=10,

                                random_state=42,

                                n_jobs=-1)

rf_clf.fit(X_train,y_train)
rfclf_pred = rf_clf.predict(X_test)

print(classification_report(y_test,rfclf_pred))
plot_prob_auc(rf_clf,X_test,y_test,model_name='Random Forest')
from tensorflow.keras import models

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras import initializers, optimizers

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.metrics import AUC
perc = Sequential()

perc.add(Dense(112,

               input_dim=112,

               activation='relu',

               kernel_initializer=initializers.glorot_uniform(seed=42),

               bias_initializer='zeros'))

perc.add(Dense(1,

               input_dim=112,

               activation='sigmoid',

               kernel_initializer=initializers.glorot_uniform(seed=42),

               bias_initializer='zeros'))
earlystop_callback = EarlyStopping(monitor='accuracy',

                                   min_delta=0.0001,

                                   patience=7)

auc = AUC()
perc.compile(optimizer='nadam',loss='binary_crossentropy',metrics=['accuracy',auc])

perc.fit(X_trains.values,y_trains.values,epochs=100,callbacks=[earlystop_callback])
plot_ann_auc(perc,X_tests.values,y_tests.values,model_name='Perceptron')
perc_pred = perc.predict_classes(X_tests.values).flatten()

print(classification_report(y_tests,perc_pred))
ann = Sequential()

ann.add(Dense(112,

              input_dim=112,

              activation='relu',

              kernel_initializer=initializers.glorot_uniform(seed=42),

              bias_initializer='zeros'))

ann.add(Dense(224,

              input_dim=112,

              activation='relu',

              kernel_initializer=initializers.glorot_uniform(seed=42),

              bias_initializer='zeros'))

ann.add(Dropout(0.5,seed=42))

ann.add(Dense(112,

              input_dim=112,

              activation='relu',

              kernel_initializer=initializers.glorot_uniform(seed=42),

              bias_initializer='zeros'))

ann.add(Dense(224,

              input_dim=112,

              activation='relu',

              kernel_initializer=initializers.glorot_uniform(seed=42),

              bias_initializer='zeros'))

ann.add(Dropout(0.5,seed=42))

ann.add(Dense(112,

              input_dim=112,

              activation='relu',

              kernel_initializer=initializers.glorot_uniform(seed=42),

              bias_initializer='zeros'))

ann.add(Dense(1,

              input_dim=112,

              activation='sigmoid',

              kernel_initializer=initializers.glorot_uniform(seed=42),

              bias_initializer='zeros'))
ann.compile(optimizer='nadam',loss='binary_crossentropy',metrics=['accuracy',auc])

ann.fit(X_trains.values,y_trains.values,epochs=100,callbacks=[earlystop_callback])
plot_ann_auc(ann,X_tests.values,y_tests.values,model_name='ANN W/ Dropout')
ann_pred = ann.predict_classes(X_tests.values).flatten()

print(classification_report(y_tests,ann_pred))
print("Perceptron:\n",classification_report(y_tests,perc_pred),"ANN W/ Dropout Regularization:\n",classification_report(y_tests,ann_pred))
importances = rf_clf.feature_importances_

indices = np.argsort(importances)

features = X_trains.columns
plt.figure(figsize=(20,20))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='g', align='center')

plt.yticks(range(len(indices)), features[indices],fontsize=12)

plt.xlabel('Relative Importance')