import pandas as pd
import requests
from pandas.io.json import json_normalize
r = requests.get("https://im6qye3mc3.execute-api.eu-central-1.amazonaws.com/prod", headers={'Accept': 'application/json'})
json_body = r.json()["body"]
import math
import dateutil.parser
from collections import defaultdict

social_distancing_dict = defaultdict(list)
sources = ["hystreet_score", "zug_score", "nationalExpress_score", "regional_score", "suburban_score", "national_score", "bus_score", "tomtom_score", "webcam_score", "bike_score", "gmap_score", "lemgoDigital", "date", "airquality_score"]
for date, districtJson in json_body.items():
    for ags, district in districtJson.items():
        social_distancing_dict["key"].append(ags+"_"+date)
        social_distancing_dict["date"].append(dateutil.parser.parse(date))
        social_distancing_dict["AGS"].append(ags)
        copySources = sources.copy()
        for key, value in district.items():
            try:
                copySources.remove(key)
                if(not (key == "date")):
                    social_distancing_dict[key].append(value)
            except:
                print("Problem with: "+key)
        for valuesLeft in copySources:
            social_distancing_dict[valuesLeft].append(math.nan)

# for key, list in social_distancing_dict.items():
#    print(key+" "+str(len(list)))

social_distancing = pd.DataFrame.from_dict(social_distancing_dict)

# copy Berlin values to all suburbs
for idx, row in social_distancing[social_distancing['AGS'] == "11000"].iterrows():
    for berlinAgs in ["11002", "11001", "11008", "11010", "11004", "11011", "11007", "11012", "11005", "11006", "11003", "11009"]:
        x1=social_distancing.loc[[idx],:]
        x1.key = berlinAgs+"_"+row.date.date().isoformat()
        x1.AGS=berlinAgs
#        print(x1)
        social_distancing = social_distancing.append(x1, ignore_index=True)

# social_distancing[social_distancing['AGS'] == '11002']
# social_distancing.key = social_distancing["AGS"]+"_"+social_distancing["date"].astype(str)
social_distancing.head()
social_distancing.describe()
hamburg_sd_rows = social_distancing["AGS"] == "02000"
hamburg_sd = social_distancing[hamburg_sd_rows]
muenchen_sd_rows = social_distancing["AGS"] == "09162"
muenchen_sd = social_distancing[muenchen_sd_rows]
hamburg_sd.head()
import seaborn as sn
import matplotlib.pyplot as plt
df = hamburg_sd.pivot(index='date', columns='AGS', values='hystreet_score')
df.plot()
plt.show()
landkreise = pd.read_csv("../input/landkreise/Landkreise.csv",dtype={"AGS":"str", "RS": "str"})
import folium
from folium import Choropleth, Circle, Marker
foliumMap = folium.Map(location=[51.0,9.0], tiles='openstreetmap', zoom_start=5)

# Add points to the map
for idx, row in landkreise.iterrows():
    Marker([row['Y'], row['X']], popup=row['GEN']).add_to(foliumMap)
foliumMap
import pylab as pl
hist = landkreise.hist(column="EWZ", bins=100)
pl.suptitle("Count of districts with amount of inhabitans")
r = requests.get("https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/Covid19_RKI_Sums/FeatureServer/0/query?where=%28Meldedatum%3Etimestamp+%272020-01-01+22%3A59%3A59%27+AND+%28Meldedatum%3Ctimestamp+%272020-12-31+22%3A00%3A00%27+OR+Meldedatum+%3E+timestamp+%272020-04-05+21%3A59%3A59%27%29%29&objectIds=&time=&resultType=none&outFields=ObjectId%2CSummeFall%2CSummeTodesfall%2CMeldedatum%2CIdLandkreis%2CAnzahlFall%2CAnzahlTodesfall&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=true&returnDistinctValues=false&cacheHint=true&orderByFields=Meldedatum+asc&groupByFieldsForStatistics=&outStatistics=&having=&sqlFormat=none&f=json&token=", headers={'Accept': 'application/json'})
count = r.json()["count"]
json_features = []
for i in range(0, count, 2000):
    r = requests.get("https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/Covid19_RKI_Sums/FeatureServer/0/query?where=%28Meldedatum%3Etimestamp+%272020-01-01+22%3A59%3A59%27+AND+%28Meldedatum%3Ctimestamp+%272020-12-31+22%3A00%3A00%27+OR+Meldedatum+%3E+timestamp+%272020-04-05+21%3A59%3A59%27%29%29&objectIds=&time=&resultType=none&outFields=ObjectId%2CSummeFall%2CSummeTodesfall%2CMeldedatum%2CIdLandkreis%2CAnzahlFall%2CAnzahlTodesfall&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=true&orderByFields=IdLandkreis,Meldedatum+asc&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset="+str(i)+"&resultRecordCount=2000&sqlFormat=none&f=json&token=", headers={'Accept': 'application/json'})
    r.encoding = "utf-8"
    json_features.extend(r.json()["features"])
from datetime import datetime, timedelta
Covid19_RKI_Sums_Dict = defaultdict(list)
SummeFall = 0
AnzahlFall = 0
SummeTodesfall = 0
relativeGrowthCases = 0
relativeGrowthDeath = 0
ags2firstCaseDate = {}
ags2casesDate = {}
agsDayAfterRow = {}
json_features.sort(key=lambda o: o.attributes.Meldedatum if 'attribute' in o.keys() else 0 )
for feature in json_features:
    keyDate = datetime.fromtimestamp(feature["attributes"]["Meldedatum"]/1000-60*60*24*7)
    date = datetime.fromtimestamp(feature["attributes"]["Meldedatum"]/1000)
    ags = feature["attributes"]["IdLandkreis"]
    Covid19_RKI_Sums_Dict["key"].append(ags+"_"+keyDate.date().isoformat())
    Covid19_RKI_Sums_Dict["Weekday"].append(keyDate.date().weekday())
    # next key
    Covid19_RKI_Sums_Dict["next_key"].append(ags+"_"+(keyDate.date()+timedelta(days=1)).isoformat())
    for key, value in feature["attributes"].items():
        Covid19_RKI_Sums_Dict[key].append(date if key == "Meldedatum" else value)
        if(key == "AnzahlFall"):
            AnzahlFall = value
        if(key == "SummeFall"):
            relativeGrowthCases = value/SummeFall if SummeFall != 0 else 0
            SummeFall = value
            # If there is a case and we don't have a first date case yet or the date is smaller than the date we already have
            if(SummeFall != 0 and ((not ags in ags2firstCaseDate.keys()) or date < ags2firstCaseDate[ags])):
                ags2firstCaseDate[ags] = date
        if(key == "SummeTodesfall"):
            relativeGrowthDeath = value/SummeTodesfall if SummeTodesfall != 0 else 0
            SummeTodesfall = value
    Covid19_RKI_Sums_Dict["relativeGrowthCases"].append(math.nan if relativeGrowthCases == 0 else relativeGrowthCases-1)
    Covid19_RKI_Sums_Dict["relativeGrowthDeath"].append(math.nan if relativeGrowthDeath == 0 else relativeGrowthDeath-1)
    Covid19_RKI_Sums_Dict["daysSinceFirstCase"].append(math.nan if not (ags in ags2firstCaseDate) else (date-ags2firstCaseDate[ags]).days)
    
    for d in range(0,7):
        Covid19_RKI_Sums_Dict["cases"+str(d+1)+"DaysBefore"].append(0 if (not ags in ags2casesDate.keys() or len(ags2casesDate[ags]) < d) else ags2casesDate[ags][len(ags2casesDate[ags])-d-1])
    
    # create features with cases the last 7 days before
    if(not ags in ags2casesDate):
        ags2casesDate[ags] = []
    ags2casesDate[ags].append(AnzahlFall)
        

Covid19_RKI_Sums = pd.DataFrame.from_dict(Covid19_RKI_Sums_Dict)

# Covid19_RKI_Sums[Covid19_RKI_Sums['IdLandkreis'] == '11001']
df = Covid19_RKI_Sums.pivot(index='Meldedatum', columns='IdLandkreis', values='SummeFall')
df.plot(figsize=(20,10))
plt.show()
biggestCitiesOver250000 = {
    "02000": "Hamburg",
    "09162": "München",
    "05315": "Köln",
    "06412": "Frankfurt am Main",
    "05111": "Düsseldorf",
    "14713": "Leipzig",
    "05913": "Dortmund",
    "05113": "Essen",
    "04011": "Bremen",
    "14612": "Dresden",
    "09564": "Nürnberg",
    "05112": "Duisburg",
    "05911": "Bochum",
    "05124": "Wuppertal",
    "05711": "Bielefeld",
    "05314": "Bonn",
    "05515": "Münster",
    "09761": "Augsburg",
    "06414": "Wiesbaden",
    "05116": "Mönchengladbach",
    "05513": "Gelsenkirchen"
}

cityRows = Covid19_RKI_Sums["IdLandkreis"].isin(biggestCitiesOver250000.keys())

df = Covid19_RKI_Sums[cityRows].pivot(index='Meldedatum', columns='IdLandkreis', values='SummeFall')
df.plot(figsize=(20,10))
plt.show()
hamburg_rows = Covid19_RKI_Sums["IdLandkreis"] == "02000"
hamburg = Covid19_RKI_Sums[hamburg_rows]
hamburg
df = hamburg.pivot(index='Meldedatum', columns='IdLandkreis', values='SummeFall')
df.plot(figsize=(20,10))
plt.show()
Covid19_RKI_Sums["AGS"] = Covid19_RKI_Sums["IdLandkreis"]
social_distancing = social_distancing.drop(columns=['AGS'])
Covid19 = Covid19_RKI_Sums.merge(social_distancing, on="key", how="left").merge(landkreise, on="AGS")
# social_distancing.AGS.unique()
# Covid19_RKI_Sums.AGS.unique()
# landkreise.AGS.unique()
from datetime import date, datetime, timedelta
import numpy as np
from scipy import stats

CityVsCountry = Covid19.copy()
CityVsCountry["CasesPer100000"] = Covid19.AnzahlFall/Covid19.EWZ/100000

yesterday = np.datetime64(date.today() - timedelta(days=1))
cities = CityVsCountry[(CityVsCountry['BEZ'] == "Kreisfreie Stadt") & (CityVsCountry['Meldedatum'] == yesterday)]
countryside = CityVsCountry[(CityVsCountry['BEZ'] == "Landkreis") & (CityVsCountry['Meldedatum'] == yesterday)]
cities.hist(column="CasesPer100000", bins=100)
countryside.hist(column="CasesPer100000", bins=100)
stats.ttest_ind(cities['CasesPer100000'],countryside['CasesPer100000'], equal_var = False)
Covid19
# Covid19_RKI_Sums[Covid19_RKI_Sums['IdLandkreis'] == '11001']
# social_distancing[social_distancing["key"] == ]
# Covid19[Covid19['AGS'] == '11001']
for ags, name in biggestCitiesOver250000.items():
    city_covid19_rows = Covid19["AGS"] == ags
    cityFrame = Covid19[city_covid19_rows]
    sn.lmplot(x='hystreet_score',y='relativeGrowthCases',data=cityFrame) 
    ax = plt.gca()
    ax.set_title(name); #+" ("+np.corrcoef(np.array(cityFrame["hystreet_score"]), np.array(cityFrame["relativeGrowthCases"]))+")")
    plt.show()


Covid19.to_csv('COVID19.csv',index=False)
import numpy as np
import lightgbm as lgb
import hyperopt
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Define the official root mean square logaritmic error function, that is officially used in the forecast compeition 0.2 - excellent, 1 - no so good, >1 terrible 
def RMSLE(predict, target):
    total = 0 
    for k in range(len(predict)):
        LPred= np.log1p(predict[k]+1)
        LTarg = np.log1p(target[k] + 1)
        if not (math.isnan(LPred)) and  not (math.isnan(LTarg)): 
            total = total + ((LPred-LTarg) **2)
        
    total = total / len(predict)        
    return np.sqrt(total)

# Copy the dataframe because we are going to do some heavy modifications
LightGBMCovid19 = Covid19.copy()
# Sort the data frame by data, this is necessary to have a split between old data and newer data
LightGBMCovid19 = LightGBMCovid19.sort_values(by=['date'])
# Save the next key, this is later needed to build the data for prediction
LightGBMCovid19NextKeys = LightGBMCovid19['next_key']
# Drop all columns that we are not going to need
LightGBMCovid19 = LightGBMCovid19.drop(columns=['key', 'next_key', 'Meldedatum', 'IdLandkreis', 'date', 'GEN', 'BEM', 'NBD', 'FK_S3', 'NUTS', 'WSK', 'DEBKG_ID', 'relativeGrowthCases',
                                                'X', 'Y', 'SummeTodesfall', 'SummeFall', 'AnzahlTodesfall', 'SN_K', 'OBJECTID',
                                                'KFL', 'ObjectId', 'RS', 'SDV_RS', 'RS_0', 'AGS_0', 'SN_L', 'SN_R',
                                                'relativeGrowthDeath', 'IBZ', 'ADE', 'GF', 'BSG', 'SN_V1', 'SN_V2', 'SN_G', 'Shape_Length'])

#LightGBMCovid19 = LightGBMCovid19.drop(columns=['cases1DaysBefore', 'cases2DaysBefore', 'cases3DaysBefore', 'cases4DaysBefore', 'cases5DaysBefore',
#                                                'cases6DaysBefore', 'cases7DaysBefore', "zug_score", "nationalExpress_score",
#                                                "regional_score", "suburban_score", "national_score", "bus_score", "tomtom_score",
#                                                "webcam_score", "bike_score", "gmap_score", "lemgoDigitalLightGBMCovid19", "EWZ", "Shape_Area", "daysSinceFirstCase", 'hystreet_score'])

# Make a category feature out of the AGS
LightGBMCovid19.AGS = LightGBMCovid19.AGS.astype('category')
LightGBMCovid19.BEZ = LightGBMCovid19.BEZ.astype('category')
LightGBMCovid19.Weekday = LightGBMCovid19.Weekday.astype('category')
LightGBMCovid19['PopulationDensity'] = LightGBMCovid19.EWZ/LightGBMCovid19.Shape_Area

#y = np.log1p(LightGBMCovid19.AnzahlFall)
# We are going to predict the cases on a daily basis
y = LightGBMCovid19.AnzahlFall
AnzahlFall = LightGBMCovid19.AnzahlFall
# Drop the column
LightGBMCovid19 = LightGBMCovid19.drop(columns=['AnzahlFall'])

# We are splitting the set for 90% training set and 10% test set, we are using date for finding the spliting point
X_train, X_test, y_train, y_test = train_test_split(LightGBMCovid19, y, test_size=0.10, shuffle=False)
for n in range(7, 10):
    # check the RMSE for taking a constant as the new cases
    rmse = mean_squared_error(y_test, np.repeat(n, len(y_test))) ** 0.5
    print('The rmse of constant '+str(n)+' benchmark is:', rmse)

rmse = mean_squared_error(y_test, X_test["cases1DaysBefore"]) ** 0.5
print('The rmse of using the value from the day before benchmark is:', rmse)
# Create LightGBM datasets 
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


# specify your configurations as a dict
#params = {
#    'boosting_type': 'gbdt',
#    'objective': 'regression',
#    'metric': {'l2', 'l1'},
#    'num_leaves': 30,
#    'learning_rate': 0.05,
#    'feature_fraction': 0.9,
#    'bagging_fraction': 0.8,
#    'bagging_freq': 5,
#    'verbose': 0
#}
# optimized with hyperopt
params = {
    'bagging_fraction': 0.9009356793140582,
    'boosting_type': 'dart',
    'metric': {'l2', 'l1'},
    'colsample_bytree': None,
    'feature_fraction': 0.9406373736120387,
    'lambda_l1': 4.318973164557415,
    'lambda_l2': 0.3976487343083751,
    'learning_rate': 0.030871120513205383,
    'min_child_samples': None,
    'min_child_weight': 0.000420873079335671,
    'min_data_in_leaf': 7,
    'min_sum_hessian_in_leaf': None,
    'num_leaves': 149,
    'objective': 'regression',
    'reg_alpha': None,
    'reg_lambda': None,
    'subsample': None,
    'subsample_for_bin': 40000,
    'verbosity': 0
}
evals_result = {}
gbm = lgb.train(params,
            lgb_train,
            #num_boost_round=35,
            valid_sets=lgb_eval,
            evals_result=evals_result,
            #early_stopping_rounds=5
)

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print('The rmse of prediction is:', rmse)
print('The RMSLE of prediction is:', RMSLE(y_pred.tolist(), y_test.tolist()))


print('Saving model...')
# save model to file
gbm.save_model('model.txt')



print('Plotting metrics recorded during training...')
ax = lgb.plot_metric(evals_result, metric='l1')
plt.show()

print('Plotting feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=10)
plt.show()

print('Plotting split value histogram...')
ax = lgb.plot_split_value_histogram(gbm, feature='cases1DaysBefore', bins='auto')
plt.show()

all_params = []

space = {
    #this is just piling on most of the possible parameter values for LGBM
    #some of them apparently don't make sense together, but works for now.. :)
    'objective':'regression',
    'boosting_type': hp.choice('boosting_type',
                               [{'boosting_type': 'gbdt',
#                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)
                                 },
                                {'boosting_type': 'dart',
#                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)
                                 },
                                {'boosting_type': 'goss'}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1), #alias "subsample"
    'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
    'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
    'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
    'verbosity': 0,
    #the LGBM parameters docs list various aliases, and the LGBM implementation seems to complain about
    #the following not being used due to other params, so trying to silence the complaints by setting to None
    'subsample': None, #overridden by bagging_fraction
    'reg_alpha': None, #overridden by lambda_l1
    'reg_lambda': None, #overridden by lambda_l2
    'min_sum_hessian_in_leaf': None, #overrides min_child_weight
    'min_child_samples': None, #overridden by min_data_in_leaf
    'colsample_bytree': None, #overridden by feature_fraction
#        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'min_child_weight': hp.loguniform('min_child_weight', -16, 5), #also aliases to min_sum_hessian
#        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
#        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
#        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
}
#check if given parameter can be interpreted as a numerical value
def is_number(s):
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

#convert given set of paramaters to integer values
#this at least cuts the excess float decimals if they are there
def convert_int_params(names, params):
    for int_type in names:
        #sometimes the parameters can be choices between options or numerical values. like "log2" vs "1-10"
        raw_val = params[int_type]
        if is_number(raw_val):
            params[int_type] = int(raw_val)
    return params

print('Starting training...')
# i call it objective_sklearn because the lgbm functions called use sklearn API
def objective_sklearn(params):
    global X_train, X_test, y_train, y_test, lgb_eval
    evals_result= {}
    int_types = ["num_leaves", "min_child_samples", "subsample_for_bin", "min_data_in_leaf"]
    params = convert_int_params(int_types, params)
    all_params.append(params)

    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    #    print("running with params:"+str(params))

    gbm = lgb.train(params,
                lgb_train,
                #num_boost_round=35,
                valid_sets=lgb_eval,
                evals_result=evals_result
                #early_stopping_rounds=5
    )
    
    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print('The rmse of prediction is:', rmse)
    print('The RMSLE of prediction is:', RMSLE(y_pred.tolist(), y_test.tolist()))
    result = {"loss": rmse, "params": params, 'status': hyperopt.STATUS_OK}
    return result

n_trials=100
trials = Trials()
# Train parameters with hyperopt
#best = fmin(fn=objective_sklearn,
#            space=space,
#            algo=tpe.suggest,
#            max_evals=n_trials,
#            trials=trials)

# find the trial with lowest loss value. this is what we consider the best one
#idx = np.argmin(trials.losses())

#print(all_params[idx])
tomorrow_test = defaultdict(list)

for ags in LightGBMCovid19.AGS.unique():
    agsRows = LightGBMCovid19[LightGBMCovid19.AGS == ags]
    maxDaysRowIndex = agsRows.daysSinceFirstCase.idxmax()
    nextKey = LightGBMCovid19NextKeys[maxDaysRowIndex]

    maxDaysRow = LightGBMCovid19.loc[maxDaysRowIndex, : ]
    tomorrow_test["AGS"].append(maxDaysRow.AGS)
    tomorrow_test["BEZ"].append(maxDaysRow.BEZ)
    tomorrow_test["PopulationDensity"].append(maxDaysRow.PopulationDensity)
    tomorrow_test["daysSinceFirstCase"].append(maxDaysRow.daysSinceFirstCase+1)
    tomorrow_test["Weekday"].append(7 % (maxDaysRow.Weekday+1))
    tomorrow_test["cases1DaysBefore"].append(AnzahlFall[maxDaysRowIndex])
    tomorrow_test["cases2DaysBefore"].append(maxDaysRow.cases1DaysBefore)
    tomorrow_test["cases3DaysBefore"].append(maxDaysRow.cases2DaysBefore)
    tomorrow_test["cases4DaysBefore"].append(maxDaysRow.cases3DaysBefore)
    tomorrow_test["cases5DaysBefore"].append(maxDaysRow.cases4DaysBefore)
    tomorrow_test["cases6DaysBefore"].append(maxDaysRow.cases5DaysBefore)
    tomorrow_test["cases7DaysBefore"].append(maxDaysRow.cases6DaysBefore)
    tomorrow_test['EWZ'].append(maxDaysRow.EWZ)
    tomorrow_test['Shape_Area'].append(maxDaysRow.Shape_Area)

    aWeekAgowRow = social_distancing[social_distancing.key == nextKey]
    if len(aWeekAgowRow) > 0:
        tomorrow_test['hystreet_score'].append(aWeekAgowRow.hystreet_score.iloc[0])
        tomorrow_test['zug_score'].append(aWeekAgowRow.zug_score.iloc[0])
        tomorrow_test['nationalExpress_score'].append(aWeekAgowRow.nationalExpress_score.iloc[0])
        tomorrow_test['regional_score'].append(aWeekAgowRow.regional_score.iloc[0])
        tomorrow_test['suburban_score'].append(aWeekAgowRow.suburban_score.iloc[0])
        tomorrow_test['national_score'].append(aWeekAgowRow.national_score.iloc[0])
        tomorrow_test['bus_score'].append(aWeekAgowRow.bus_score.iloc[0])
        tomorrow_test['tomtom_score'].append(aWeekAgowRow.tomtom_score.iloc[0])
        tomorrow_test['webcam_score'].append(aWeekAgowRow.webcam_score.iloc[0])
        tomorrow_test['bike_score'].append(aWeekAgowRow.bike_score.iloc[0])
        tomorrow_test['gmap_score'].append(aWeekAgowRow.gmap_score.iloc[0])
        tomorrow_test['lemgoDigital'].append(aWeekAgowRow.lemgoDigital.iloc[0])
        tomorrow_test['airquality_score'].append(aWeekAgowRow.airquality_score.iloc[0])
    else:
        tomorrow_test['hystreet_score'].append(math.nan)
        tomorrow_test['zug_score'].append(math.nan)
        tomorrow_test['nationalExpress_score'].append(math.nan)
        tomorrow_test['regional_score'].append(math.nan)
        tomorrow_test['suburban_score'].append(math.nan)
        tomorrow_test['national_score'].append(math.nan)
        tomorrow_test['bus_score'].append(math.nan)
        tomorrow_test['tomtom_score'].append(math.nan)
        tomorrow_test['webcam_score'].append(math.nan)
        tomorrow_test['bike_score'].append(math.nan)
        tomorrow_test['gmap_score'].append(math.nan)
        tomorrow_test['lemgoDigital'].append(math.nan)
        tomorrow_test['airquality_score'].append(math.nan)

        
#for key, value in tomorrow_test.items():
#    print(key+" "+str(len(value)))

tomorrow_test_df = pd.DataFrame.from_dict(tomorrow_test)
tomorrow_test_df.AGS = tomorrow_test_df.AGS.astype('category')
tomorrow_test_df.BEZ = tomorrow_test_df.BEZ.astype('category')
tomorrow_test_df.Weekday = tomorrow_test_df.Weekday.astype('category')

tomorrow_pred = gbm.predict(tomorrow_test_df, num_iteration=gbm.best_iteration)
#tomorrow_test_df['SummeFall'] = pd.Series(np.expm1(tomorrow_pred))
tomorrow_test_df['AnzahlFall'] = pd.Series(tomorrow_pred)
tomorrow_test_df
tomorrow_test_df = tomorrow_test_df.merge(landkreise, on="AGS")
from folium.plugins import HeatMap

foliumMap = folium.Map(location=[51.0,9.0], tiles='openstreetmap', zoom_start=5)
HeatMap(data=tomorrow_test_df[['Y', 'X', 'AnzahlFall']].groupby(['Y', 'X']).sum().reset_index().values.tolist(), radius=25, max_zoom=13).add_to(foliumMap)

foliumMap
foliumMap = folium.Map(location=[51.0,9.0], tiles='openstreetmap', zoom_start=5)

# Add points to the map
for idx, row in tomorrow_test_df.iterrows():
    Marker([row['Y'], row['X']], popup=row['GEN']+'\n <strong>Predicted Cases: '+str(round(row['AnzahlFall']))+'</strong> Cases in the last seven days: <ol><li>'+str(row.cases1DaysBefore)+'</li><li>'+str(row.cases2DaysBefore)+'</li><li>'+str(row.cases3DaysBefore)+'</li><li>'+str(row.cases4DaysBefore)+'</li><li>'+str(row.cases5DaysBefore)+'</li><li>'+str(row.cases6DaysBefore)+'</li><li>'+str(row.cases7DaysBefore)+'</li></ol>').add_to(foliumMap)
foliumMap
from IPython.core.display import HTML
# does not work
HTML("<style type='text/css'>@import 'https://cdn.jsdelivr.net/npm/leaflet-timedimension@1.1.1/dist/leaflet.timedimension.control.min.css'; </style>")

from folium.plugins import HeatMapWithTime

heatMap = folium.Map(location=[51.0,9.0], tiles='openstreetmap', zoom_start=5)

listOfHeatmaps = []
for attribute in ["cases1DaysBefore", "cases2DaysBefore", "cases3DaysBefore", "cases4DaysBefore", "cases5DaysBefore", "cases6DaysBefore", "cases7DaysBefore", "AnzahlFall"]:
    listOfHeatmaps.append(tomorrow_test_df[['Y', 'X', attribute]].groupby(['Y', 'X']).sum().reset_index().values.tolist())

    
flatten = lambda l: [item for sublist in l for item in sublist]

maxCases = max(map(lambda item : item[2], flatten(listOfHeatmaps)))

listOfHeapMapsNormalized = []
for listOfHeatmap in listOfHeatmaps:
    listOfHeapMapsNormalized.append(list(map(lambda item : [item[0], item[1], item[2]/maxCases], listOfHeatmap)))

HeatMapWithTime(data=listOfHeapMapsNormalized, radius=25).add_to(heatMap)

heatMap
foliumMap = folium.Map(location=[51.0,9.0], tiles='openstreetmap', zoom_start=5)
tomorrow_test_df["Cases-7-days-by-EWZ"] = (tomorrow_test_df["cases1DaysBefore"]+tomorrow_test_df["cases2DaysBefore"]+tomorrow_test_df["cases3DaysBefore"]+tomorrow_test_df["cases4DaysBefore"]+tomorrow_test_df["cases5DaysBefore"]+tomorrow_test_df["cases6DaysBefore"]+tomorrow_test_df["AnzahlFall"])/(tomorrow_test_df["EWZ_y"]/100000)

morethan45 = tomorrow_test_df[tomorrow_test_df["Cases-7-days-by-EWZ"] >= 40]

# Add points to the map
for idx, row in morethan45.iterrows():
    Marker([row['Y'], row['X']], popup=row['GEN']+'\n <strong>Total Cases with prediction for 7 days by 100.000 capita: '+str(round(row['Cases-7-days-by-EWZ']))+'</strong>\n Predicted cases '+str(round(row['AnzahlFall']))+' Cases in the last seven days: <ol><li>'+str(row.cases1DaysBefore)+'</li><li>'+str(row.cases2DaysBefore)+'</li><li>'+str(row.cases3DaysBefore)+'</li><li>'+str(row.cases4DaysBefore)+'</li><li>'+str(row.cases5DaysBefore)+'</li><li>'+str(row.cases6DaysBefore)+'</li><li>'+str(row.cases7DaysBefore)+'</li></ol>').add_to(foliumMap)
foliumMap