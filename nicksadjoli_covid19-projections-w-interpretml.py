!pip install -U interpret
from interpret import show

from interpret.data import Marginal

from interpret.glassbox import ExplainableBoostingRegressor, LinearRegression, RegressionTree

from interpret.perf import RegressionPerf



import datetime

import math

import numpy as np

from sklearn.metrics import mean_squared_error

import copy
import pandas as pd 

import numpy as np 



train_default_path = "../input/covid19-global-forecasting-week-4/train.csv"

test_default_path = "../input/covid19-global-forecasting-week-4/test.csv"



train_default_data = pd.read_csv(train_default_path)

#train_default_data
test_default_data = pd.read_csv(test_default_path)

#test_default_data
train_appended_df = pd.read_csv("../input/training-data-appended/training_data_with_weather_info_week_4.csv")

print("Current columns:", train_appended_df.columns)

#train_appended_df
training_data_unique_regions = train_appended_df['Province_State'].unique()

#training_data_unique_regions, len(training_data_unique_regions)
population_df = pd.read_csv("../input/covid19populationtestingdataset/Worldometer_Population_Regional_Latest.csv")

#population_df
#list the unique regions in the population_df DataFrame, while also removing the 'All_Regions' tag (which indicate it's the population of the whole country, and not just a region)

popdf_unique_regions = population_df['Region'].unique()

popdf_unique_regions = np.sort(popdf_unique_regions[popdf_unique_regions != 'All_Regions'])

print("All {} unique regions recorded:".format(str(len(popdf_unique_regions))))

print(popdf_unique_regions, "True Victoria" in popdf_unique_regions)
population_density_area_df = pd.read_csv("../input/covid19populationtestingdataset/OECD_PopulationDensity_and_Area-T2_T3_Regions-2018_2019.csv")

print("Columns available:", population_density_area_df.columns)

population_density_area_df.head()
#Limit to only population density data, and in Year 2019 only

population_density_only = population_density_area_df[population_density_area_df["VAR"] == "POP_DEN"]

population_density_only.drop(['SEX', 'Gender', 'POS', 'Position', 'PowerCode Code', 'Reference Period Code', 'Reference Period'], axis=1)

population_density_2019 = population_density_only[population_density_only["Year"] == 2019]

population_density_unique_regions = population_density_2019['Region'].unique()

print("All unique {} regions recorded for OECD's population density data: ".format(str(len(population_density_unique_regions))), 

                                                                                    population_density_unique_regions)

population_density_2019.head()
#train_appended_df = train_appended_df.copy()

#Initiate new feature columns

added_features = ['Population (2020)', 'Population Density']

for feature in added_features:

    train_appended_df[feature] = 0



for country in train_appended_df['Country_Region'].unique():

    #print(train_appended_df['Population (2020)'].unique())

    country_segment = train_appended_df[train_appended_df['Country_Region'] == country]



    #Sanity check for several countries, as they're apparently named quite differently in Worldometers vs the training data

    if country == "Burma":

        country = "Myanmar" #Burma in Training data is actually Myanmar. History stuff I guess?

    elif country == "Korea, South":

        country = "South Korea" #this one is honestly just trolling at this point...

    elif country == "US":

        country = "United States"



    population_df_country = population_df[population_df['Country (or dependency)'] == country]#['Region'] == 'All_Regions'



    #check whehter the current country has any listed states/regions in the original training data.

    #If yes: Add regional population and regional population density data

    #If not: Only add country population and population density data.

    country_regions_training = list(country_segment['Province_State'].unique())



    #Apparently there are 2 'Congo'-s: Republic of Congo/Brazzaville, vs DEMOCRATIC Republic of Congo/Zaire (as how it's differentiated in Worldometers)

    if country == "Congo (Brazzaville)": 

        country = "Congo"

        country_regions_training = ["Brazzaville"]

    elif country == "Congo (Kinshasa)":

        country = "Congo"

        country_regions_training = ["Kinshasa"]



    if country_regions_training == [np.NaN]:

        #print(country, country_regions_training)

        

        #sanity check: in case country isn't listed in the worldometers population data, 

        #then query to the population_df would return DataFrame of 0

        if len(population_df_country) != 0:

            try:

                country_population = int(population_df_country[population_df_country['Region'] == "All_Regions"]["Population (2020)"].values[0].replace(",", "")) 

            except:

                print("Problematic country for pop. df", country, country_regions_training)

            try:

                country_population_density = population_df_country[population_df_country['Region'] == "All_Regions"]["Density (P/Km²)"].values[0]

            except:

                print("Problematic country for pop_density df", country, country_regions_training)

                break

        else:

            continue

            

        country_ids = country_segment.index.tolist()

        train_appended_df.loc[country_ids, ['Population (2020)']] = country_population

        train_appended_df.loc[country_ids, ['Population Density']] = country_population_density

    else:

        for region in country_regions_training:

            region_segment = country_segment[country_segment['Province_State'] == region]



            #sanity check, as apparently the region names are not truly unique to a country in Worldometer's data

            #(in particular, the region 'Victoria' which is unique to Australia in training data, is not present in Worldometer's Australia,

            # and instead available for other countries.)

            if region == "Australian Capital Territory":

                region_popdf_segment = population_df_country[population_df_country['Region'] == "Canberra"]

            else:

                region_popdf_segment = population_df_country[population_df_country['Region'] == region]

            region_popdensity_segment = population_density_2019[population_density_2019['Region'] == region]



            if len(region_popdf_segment) == 1: #Means that there is a valid row available in Worldometer's population_data

                region_population = int(region_popdf_segment["Population (2020)"].values[0].replace(",", "") )

                #region_population = int(population_df_country[population_df_country['Region'] == region]["Population (2020)"].values[0].replace(",", ""))

            else:

                region_population = np.NaN

            



            if len(region_popdensity_segment) == 1:

                region_population_density = region_popdensity_segment['Value'].values[0]#population_density_2019[population_density_2019['Region'] == region]['Value'].values[0]

            elif len(region_popdensity_segment) > 1:

                picked_segment = region_popdensity_segment[region_popdensity_segment["Territory Level and Typology"] == "Large regions (TL2)"]

                region_population_density = picked_segment['Value'].values[0]

            else:

                region_population_density = np.NaN



            region_ids = region_segment.index.tolist()

            train_appended_df.loc[region_ids, ['Population (2020)']] = region_population

            train_appended_df.loc[region_ids,['Population Density']] = region_population_density

train_appended_df

#print("Done")
ourworldindata_testing_df = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv")

ourworldindata_testing_df.head()
#ourworldindata_testing_df['Entity'].unique()[0].split("-")[0].rstrip().lstrip()

columns_to_be_added = ['Data Type', 'Country']

for column in columns_to_be_added:

    if column not in ourworldindata_testing_df.columns:

        ourworldindata_testing_df.insert(1, column, '')

for i in ourworldindata_testing_df.index:

    entity_data = ourworldindata_testing_df.loc[i]['Entity'].split("-")

    ourworldindata_testing_df.loc[i, ['Country']] = entity_data[0].rstrip().lstrip()

    ourworldindata_testing_df.loc[i, ['Data Type']] = entity_data[1].rstrip().lstrip()

    

ourworldindata_testing_df = ourworldindata_testing_df.drop(['Entity'], axis=1)

print(ourworldindata_testing_df.columns)

ourworldindata_testing_df.head()
ourworldindata_features_to_append = ['Cumulative Total Tests', 'Daily Change in Cumulative Total Tests', 'Cumulative Tests per Thousand', 'Daily change in Cumulative Tests per Thousand', '3-day rolling mean daily change', '3-day rolling mean daily change per thousand']



for feature in ourworldindata_features_to_append:

    train_appended_df[feature] = 0.0



for country in train_appended_df['Country_Region'].unique():

    #print(train_appended_df['Population (2020)'].unique())

    country_segment = train_appended_df[train_appended_df['Country_Region'] == country]



    #Sanity check for South Korea, as they're apparently named quite differently in Worldometers vs the training data

    if country == "Korea, South":

        country = "South Korea" 

    elif country == "US":

        country = "United States"

    

    #sanity check for case of United Kingdom

    if country == "United Kingdom":

        country_segment = country_segment[country_segment['Province_State'].isnull()]

        #print(country_segment)



    #check whehter the current country has any listed states/regions in the original training data. For ourworldindata tests, only countries 

    #*without* regions in the training data will be used and appended.

    country_regions_training = list(country_segment['Province_State'].unique())



    #OWID dataset only covers country level (and not regional level). Hence, only countries without regions will be appended.

    if country_regions_training == [np.NaN]:

        ourworldindata_country_segment = ourworldindata_testing_df[ourworldindata_testing_df['Country'] == country]

        country_ids = country_segment.index



        #sanity check: in case country isn't listed in the worldometers population data, 

        #then query to the population_df would return DataFrame of 0

        if len(ourworldindata_country_segment) != 0:



            #print(country_ids) #ourworldindata_country_segment)

            for cur_id in country_ids:

                cur_date = train_appended_df.loc[cur_id]['Date']

                owid_test_data = ourworldindata_country_segment[pd.to_datetime(ourworldindata_country_segment['Date']) == cur_date]



                if country == "Japan":

                    owid_test_data = owid_test_data[owid_test_data['Data Type'] == 'tests performed']

                elif country == "Singapore":

                    owid_test_data = owid_test_data[owid_test_data['Data Type'] == 'swabs tested']

                elif country == "United States":

                    owid_test_data = owid_test_data[owid_test_data['Data Type'] == 'inconsistent units (COVID Tracking Project)']



                

                #sanity check for dates that were not recorded in ourworldindata's dataset

                if len(owid_test_data) != 0:

                    train_appended_df.loc[cur_id, ['Cumulative Total Tests']] = owid_test_data['Cumulative total'].values[0]

                    train_appended_df.loc[cur_id, ['Daily Change in Cumulative Total Tests']] = owid_test_data['Daily change in cumulative total'].values[0]

                    train_appended_df.loc[cur_id, ['Cumulative Tests per Thousand']] = owid_test_data['Cumulative total per thousand'].values[0]

                    train_appended_df.loc[cur_id, ['Daily change in Cumulative Tests per Thousand']] = owid_test_data['Daily change in cumulative total per thousand'].values[0]

                    train_appended_df.loc[cur_id, ['3-day rolling mean daily change']] = owid_test_data['3-day rolling mean daily change'].values[0]

                    train_appended_df.loc[cur_id, ['3-day rolling mean daily change per thousand']] = owid_test_data['3-day rolling mean daily change per thousand'].values[0]

                else:

                    train_appended_df.loc[cur_id, ourworldindata_features_to_append] = 0.0



        else:

            continue

            

    else:

        pass



train_appended_df[train_appended_df['Country_Region'] == "Korea, South"]
feature_dummy = "max"

missing_index = np.where(train_appended_df[feature_dummy].isnull())[0]

train_appended_df.at[missing_index,feature_dummy] = train_appended_df["temp"][missing_index]



feature_dummy = "min"

missing_index = np.where(train_appended_df[feature_dummy].isnull())[0]

train_appended_df.at[missing_index,feature_dummy] = train_appended_df["temp"][missing_index]



"""

feature_dummy = "Population Density"

missing_index = np.where(train_appended_df[feature_dummy].isnull())[0]

missing_country_province = train_appended_df["country+province"][missing_index].unique()

for reg in missing_country_province:

    mask_ = train_appended_df["country+province"] == reg

    mask_ix_ = np.where(mask_)[0]

    mask_country_ = train_appended_df["country+province"] == reg.split('')

    if(len(np.where())):

        replacement = 

    else:

        print("No replacement found for {}".format(reg))

    train_appended_df[feature_dummy].at[mask_ix_,feature_dummy] = 

"""

#np.where(train_appended_df["country+province"] =="Australia")



#feature_dummy = "Population (2020)"
def preprocess(df,

               features = ['Days','Region',"prev_ConfirmedCases","prev_Fatalities"],

               targets = ["ConfirmedCases", "Fatalities"]):



    # Create category called Region: country_province

    region_list = ["{}_{}".format(df["Country_Region"][i], df["Province_State"][i]) for i in range(df.shape[0])]

    df["Region"]=region_list



    # Get first day of corona virus for each region

    unique_region_list = list(set(region_list))

    unique_region_list.sort()

    first_date_dict = {}

    for region in unique_region_list:

        mask = df["Region"]==region

        first_ix = np.where(df[mask]["ConfirmedCases"]>0)[0][0] -1    

        first_date = df[mask]["Date"].iloc[first_ix]

        first_date_dict[region] = first_date



    # add column "Days": number of days since the first day of case per each region

    def get_days(dt):

        return dt.days

    dummy = [first_date_dict[region] for region in df["Region"]]

    df["Days"]=(pd.to_datetime(df['Date'])-pd.to_datetime(dummy)).apply(get_days)



    # Add previous confirmed cases and previous fatalities to df

    loc_group=["Region"]

    for target in targets:

        df["prev_{}".format(target)] = df.groupby(loc_group)[target].shift()

        df["prev_{}".format(target)].fillna(0, inplace=True)

    

    for target in targets:

        df[target] = np.log1p(df[target])

        df["prev_{}".format(target)] = np.log1p(df["prev_{}".format(target)])

    

    X = df[features]

    Y = df[targets]

    

    return X,Y
features = ['Region',"prev_ConfirmedCases","prev_Fatalities",

            'Days',"day_from_jan_first",

            "temp","max","min","prcp","stp","prcp","fog","wdsp", # weather

            "Lat","Long"]

X,Y = preprocess(train_appended_df,features=features)
marginal = Marginal().explain_data(X, Y["ConfirmedCases"],"ConfirmedCases")

show(marginal)
important_features = ['Region',"prev_ConfirmedCases","prev_Fatalities",

                        'Days',

                        "Lat","Long"]

X = X[important_features]
def split_train_val(X,Y, unique_region_list,num_of_val_days):

    

    train_ix = []

    val_ix = []

    for region in unique_region_list:

        

        mask = X["Region"]==region

        ix = np.where(mask)[0]

        

        train_ix += list(ix[:-num_of_val_days].flatten())

        val_ix += list(ix[-num_of_val_days:].flatten())

        

    return X.iloc[train_ix],X.iloc[val_ix],Y.iloc[train_ix],Y.iloc[val_ix]    



# IMPORTANT NOTE: We can only use prev_ConfirmedCases for the first day to predict
# IMPORTANT NOTE: assuming that X_features is sorted by number of days "Days"

ENFORCE_CONSTRAINT = True

seed = 1

def evaluate_rmse(Y_predicted,Y_true):

    """

    Y_predicted: n-by-d n is the number of data points, d is the number of criteria

    Y_true: n-by-d

    OUTPUT

    d elements

    """

    return np.sqrt(mean_squared_error(Y_predicted,Y_true,multioutput='raw_values'))



def predict(X_features,Y,num_validation_days,num_days_to_predict):

    unique_region_list = list(set(X_features["Region"]))

    unique_region_list.sort()

    print("No of unique region list: {}".format(len(unique_region_list)))

    

    ##################################################################

    # Train and Validation

    ##################################################################

    # Split to train and validation

    X_train,X_val,Y_train,Y_val = split_train_val(X,Y, unique_region_list,num_validation_days)

    

    # Train

    model_ConfirmedCases = ExplainableBoostingRegressor(random_state=seed)

    model_ConfirmedCases.fit(X_train,Y_train["ConfirmedCases"])

    model_Fatalities = ExplainableBoostingRegressor(random_state=seed)

    model_Fatalities.fit(X_train,Y_train["Fatalities"])

    

    # Predict for val

    Y_val_predicted = np.zeros((X_val.shape[0],2))

    

    for i in range(X_val.shape[0]):

        

        if(i==0 or X_val.iloc[i-1]["Region"] != X_val.iloc[i]["Region"]):

            pred_ConfirmedCases = model_ConfirmedCases.predict(X_val.iloc[[i]])[0]

            pred_Fatalities = model_Fatalities.predict(X_val.iloc[[i]])[0]

            

            if(ENFORCE_CONSTRAINT):

                if(pred_ConfirmedCases<X_val.iloc[[i]]["prev_ConfirmedCases"].item()):

                    pred_ConfirmedCases = 1.*X_val.iloc[[i]]["prev_ConfirmedCases"].item()

                if(pred_Fatalities<X_val.iloc[[i]]["prev_Fatalities"].item()):

                    pred_Fatalities = X_val.iloc[[i]]["prev_Fatalities"].item()

                    

        else:

            X_dummy  = X_val.iloc[[i]].copy(deep=True)

            X_dummy["prev_ConfirmedCases"] = pred_ConfirmedCases

            X_dummy["prev_Fatalities"] = pred_Fatalities

            pred_ConfirmedCases = model_ConfirmedCases.predict(X_dummy)

            pred_Fatalities =model_Fatalities.predict(X_dummy)

        

            if(ENFORCE_CONSTRAINT):

                if(pred_ConfirmedCases<X_dummy["prev_ConfirmedCases"].item()):

                    pred_ConfirmedCases = 1.* X_dummy["prev_ConfirmedCases"].item()

                if(pred_Fatalities<X_dummy["prev_Fatalities"].item()):

                    pred_Fatalities = X_dummy["prev_Fatalities"].item()

                    

        Y_val_predicted[i,0] = pred_ConfirmedCases

        Y_val_predicted[i,1] = pred_Fatalities

        

    # Report validation accuracy

    val_rmse = evaluate_rmse(Y_val,Y_val_predicted)

    

    ##################################################################

    # Train w Full Model and Predict for Test

    ##################################################################

    # Train with full data

    model_full_ConfirmedCases = ExplainableBoostingRegressor(random_state=seed)

    model_full_ConfirmedCases.fit(X_features,Y["ConfirmedCases"])

    model_full_Fatalities = ExplainableBoostingRegressor(random_state=seed)

    model_full_Fatalities.fit(X_features,Y["Fatalities"])

    

    # Predict for test

    Y_test_predicted = np.zeros((len(unique_region_list)*num_days_to_predict,2))

    count=0

    for region in unique_region_list:

        mask = X_features["Region"]==region

        

        prev_ConfirmedCase_ = Y[mask]["ConfirmedCases"].iloc[-1]

        prev_Fatality_ = Y[mask]["Fatalities"].iloc[-1]

        

        #print(prev_ConfirmedCase_,np.exp(prev_ConfirmedCase_)-1, prev_Fatality_, np.exp(prev_Fatality_)-1)

        

        X_dummy = X[mask].iloc[[-1]].copy(deep=True)

        X_dummy["prev_ConfirmedCases"] = prev_ConfirmedCase_

        X_dummy["prev_Fatalities"] = prev_Fatality_

        X_dummy["Days"] = X_dummy["Days"]+1

        

        pred_ConfirmedCases = model_full_ConfirmedCases.predict(X_dummy)

        pred_Fatalities = model_full_Fatalities.predict(X_dummy)

        

        if(ENFORCE_CONSTRAINT):

            if(pred_ConfirmedCases<X_dummy["prev_ConfirmedCases"].item()):

                pred_ConfirmedCases = X_dummy["prev_ConfirmedCases"].item()

            if(pred_Fatalities<X_dummy["prev_Fatalities"].item()):

                pred_Fatalities = X_dummy["prev_Fatalities"].item()

                

        Y_test_predicted[count,0] = pred_ConfirmedCases

        Y_test_predicted[count,1] = pred_Fatalities

        count = count+1

        

        for days_ahead in range(2,num_days_to_predict+1):

            

            X_dummy["prev_ConfirmedCases"] = pred_ConfirmedCases

            X_dummy["prev_Fatalities"] = pred_Fatalities

            X_dummy["Days"] = X_dummy["Days"]+1

            pred_ConfirmedCases = model_full_ConfirmedCases.predict(X_dummy)

            pred_Fatalities = model_full_Fatalities.predict(X_dummy)

            

            if(ENFORCE_CONSTRAINT):

                if(pred_ConfirmedCases<X_dummy["prev_ConfirmedCases"].item()):

                    pred_ConfirmedCases = X_dummy["prev_ConfirmedCases"].item()

                if(pred_Fatalities<X_dummy["prev_Fatalities"].item()):

                    pred_Fatalities = X_dummy["prev_Fatalities"].item()

                

            Y_test_predicted[count,0] = pred_ConfirmedCases

            Y_test_predicted[count,1] = pred_Fatalities

            

            count = count+1

      

    assert count==len(Y_test_predicted), "Something wrong"

    



    return unique_region_list,X_val,Y_val,Y_val_predicted,val_rmse,Y_test_predicted
num_days_to_predict = 43

num_validation_days = 10

unique_region_list,X_val,Y_val,Y_val_predicted,val_rmse,Y_test_predicted=predict(X,Y,num_validation_days,num_days_to_predict)
# Validation error

print("RMSE for ConfirmedCases and Fatalities: {}".format(val_rmse))
# This is the final value ConfirmedCases and Fatalities

# Convert back to linear scale

Y_test_predicted_final = np.exp(Y_test_predicted)-1

Y_val_predicted_final = np.exp(Y_val_predicted)-1
import matplotlib.pyplot as plt

from matplotlib import gridspec
# Choose using region_ix

#region_ix = 3

#region = unique_region_list[region_ix]



# Choose using region

region = "Indonesia_nan"

region_ix = unique_region_list.index(region)



USE_LOG_SCALE=False

PLOT_LINE = False

##############################################



mask = X["Region"]==region

N = Y[mask].shape[0]

x_ = np.arange(N+num_days_to_predict)



validation_confirmed_cases = Y_val_predicted_final[region_ix*num_validation_days:(region_ix+1)*num_validation_days,0]

validation_fatalities = Y_val_predicted_final[region_ix*num_validation_days:(region_ix+1)*num_validation_days,1]

predicted_confirmed_cases = Y_test_predicted_final[region_ix*num_days_to_predict:(region_ix+1)*num_days_to_predict,0]

predicted_fatalities = Y_test_predicted_final[region_ix*num_days_to_predict:(region_ix+1)*num_days_to_predict,1]
sz = 8

gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1],wspace=0.25)

fig = plt.figure(figsize=(12,5))

ax1 = fig.add_subplot(gs[0,0])

ax2 = fig.add_subplot(gs[0,1])

if(USE_LOG_SCALE):

    ax1.set_yscale('log')

    ax2.set_yscale('log')

ax1.scatter(x_[:N],np.exp(Y[mask]["ConfirmedCases"])-1, label="Real Data",s=sz)

ax1.scatter(x_[N:],predicted_confirmed_cases, label="Predicted",s=sz)

ax1.scatter(x_[N-num_validation_days:N],validation_confirmed_cases,label="Validation",s=sz)

if(PLOT_LINE):

    ax1.plot(x_[:N],np.exp(Y[mask]["ConfirmedCases"])-1, label="Real Data")

    ax1.plot(x_[N:],predicted_confirmed_cases, label="Predicted")

    ax1.plot(x_[N-num_validation_days:N],validation_confirmed_cases,label="Validation")

ax1.set_title(region+"Confirmed Cases")

ax1.legend()



ax2.scatter(x_[:N],np.exp(Y[mask]["Fatalities"])-1, label="Real Data",s=sz)

ax2.scatter(x_[N:],predicted_fatalities, label="Predicted",s=sz)

ax2.scatter(x_[N-num_validation_days:N],validation_fatalities,label="Validation",s=sz)

if(PLOT_LINE):

    ax2.plot(x_[:N],np.exp(Y[mask]["Fatalities"])-1, label="Real Data")

    ax2.plot(x_[N:],predicted_fatalities, label="Predicted")

    ax2.plot(x_[N-num_validation_days:N],validation_fatalities,label="Validation")



ax2.set_title(region+"Fatalities")

ax2.legend()
#submission

submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

submission['ConfirmedCases'] = Y_test_predicted_final[:, 0]

submission['Fatalities'] = Y_test_predicted_final[:,1]

train_appended_df.to_csv("appended_training_week4.csv")

submission.to_csv("submission.csv", index=False)