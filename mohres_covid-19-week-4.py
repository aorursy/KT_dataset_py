# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



import geopandas as gpd

from shapely.geometry import Point

import tensorflow as tf

from tqdm import tqdm

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_log_error



from datetime import datetime

from datetime import timedelta



from tensorflow.keras import layers

from tensorflow.keras import Input

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.
train_data = gpd.read_file('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

train_data = train_data.rename(columns={'Province_State': 'State', 'Country_Region': 'Country', 'ConfirmedCases': 'Confirmed'})

train_data["Confirmed"] = train_data["Confirmed"].astype("float")

train_data["Fatalities"] = train_data["Fatalities"].astype("float")

#The country_region got modified in the enriched dataset by @optimo, 

# so we have to apply the same change to this Dataframe to facilitate the merge.

train_data["Country"] = [row.Country.replace("'","").strip(" ") if row.State=="" else str(row.Country+"_"+row.State).replace("'","").strip(" ") for idx,row in train_data.iterrows()]
#Still using the enriched data from week 2 as there is everything required for the model's training

extra_data_df = gpd.read_file("/kaggle/input/covid-19-enriched-dataset-week-2/enriched_covid_19_week_2.csv")

extra_data_df = extra_data_df.rename(columns={'Province_State': 'State', 

                                              'Country_Region': 'Country', 

                                              'ConfirmedCases': 'Confirmed'})

extra_data_df["Country"] = [country_name.replace("'","") for country_name in extra_data_df["Country"]]

extra_data_df["restrictions"] = extra_data_df["restrictions"].astype("int")

extra_data_df["quarantine"] = extra_data_df["quarantine"].astype("int")

extra_data_df["schools"] = extra_data_df["schools"].astype("int")

extra_data_df["total_pop"] = extra_data_df["total_pop"].astype("float")

extra_data_df["density"] = extra_data_df["density"].astype("float")

extra_data_df["hospibed"] = extra_data_df["hospibed"].astype("float")

extra_data_df["lung"] = extra_data_df["lung"].astype("float")

extra_data_df["total_pop"] = extra_data_df["total_pop"]/max(extra_data_df["total_pop"])

extra_data_df["density"] = extra_data_df["density"]/max(extra_data_df["density"])

extra_data_df["hospibed"] = extra_data_df["hospibed"]/max(extra_data_df["hospibed"])

extra_data_df["lung"] = extra_data_df["lung"]/max(extra_data_df["lung"])

extra_data_df["age_100+"] = extra_data_df["age_100+"].astype("float")

extra_data_df["age_100+"] = extra_data_df["age_100+"]/max(extra_data_df["age_100+"])



extra_data_df = extra_data_df[["Country","Date","restrictions","quarantine","schools","hospibed","lung","total_pop","density","age_100+"]]

extra_data_df.head()
train_data = train_data.merge(extra_data_df, how="left", on=['Country','Date']).drop_duplicates()

train_data.head()
for country_region in train_data.Country.unique():

    query_df = train_data.query("Country=='"+country_region+"' and Date=='2020-03-25'")

    train_data.loc[(train_data["Country"]==country_region) & (train_data["Date"]>"2020-03-25"),"total_pop"] = query_df.total_pop.values[0]

    train_data.loc[(train_data["Country"]==country_region) & (train_data["Date"]>"2020-03-25"),"hospibed"] = query_df.hospibed.values[0]

    train_data.loc[(train_data["Country"]==country_region) & (train_data["Date"]>"2020-03-25"),"density"] = query_df.density.values[0]

    train_data.loc[(train_data["Country"]==country_region) & (train_data["Date"]>"2020-03-25"),"lung"] = query_df.lung.values[0]

    train_data.loc[(train_data["Country"]==country_region) & (train_data["Date"]>"2020-03-25"),"age_100+"] = query_df["age_100+"].values[0]

    train_data.loc[(train_data["Country"]==country_region) & (train_data["Date"]>"2020-03-25"),"restrictions"] = query_df.restrictions.values[0]

    train_data.loc[(train_data["Country"]==country_region) & (train_data["Date"]>"2020-03-25"),"quarantine"] = query_df.quarantine.values[0]

    train_data.loc[(train_data["Country"]==country_region) & (train_data["Date"]>"2020-03-25"),"schools"] = query_df.schools.values[0]
median_pop = np.median(extra_data_df.total_pop)

median_hospibed = np.median(extra_data_df.hospibed)

median_density = np.median(extra_data_df.density)

median_lung = np.median(extra_data_df.lung)

median_centenarian_pop = np.median(extra_data_df["age_100+"])

#need to replace that with a joint using Pandas

print("The missing countries/region are:")

for country_region in train_data.Country.unique():

    if extra_data_df.query("Country=='"+country_region+"'").empty:

        print(country_region)

        

        train_data.loc[train_data["Country"]==country_region,"total_pop"] = median_pop

        train_data.loc[train_data["Country"]==country_region,"hospibed"] = median_hospibed

        train_data.loc[train_data["Country"]==country_region,"density"] = median_density

        train_data.loc[train_data["Country"]==country_region,"lung"] = median_lung

        train_data.loc[train_data["Country"]==country_region,"age_100+"] = median_centenarian_pop

        train_data.loc[train_data["Country"]==country_region,"restrictions"] = 0

        train_data.loc[train_data["Country"]==country_region,"quarantine"] = 0

        train_data.loc[train_data["Country"]==country_region,"schools"] = 0
# #https://ourworldindata.org/smoking#prevalence-of-smoking-across-the-world

# smokers = pd.read_csv('/kaggle/input/smokingstats/share-of-adults-who-smoke.csv')

# smokers = smokers[smokers.Year == 2016].reset_index(drop=True)



# smokers_country_dict = {'North America' : "US",

#  'Gambia' : "The Gambia",

#  'Bahamas': "The Bahamas",

#  "'South Korea'" : "Korea, South",

# 'Papua New Guinea' : "Guinea",

#  "'Czech Republic'" : "Czechia",

#  'Congo' : "Congo (Brazzaville)"}



# smokers['Entity'] = smokers.Entity.apply(lambda x : rename_countries(x, smokers_country_dict))



# no_datas_smoker = []

# for country in df['Country'].unique():

#     if country not in smokers.Entity.unique():

#         mean_score = smokers[['Smoking prevalence, total (ages 15+) (% of adults)']].mean().to_dict()

#         mean_score['Entity'] = country

#         no_datas_smoker.append(mean_score)

# no_data_smoker_df = pd.DataFrame(no_datas_smoker)   

# clean_smoke_data = pd.concat([smokers, no_data_smoker_df], axis=0)[['Entity','Smoking prevalence, total (ages 15+) (% of adults)']]

# clean_smoke_data.rename(columns={"Entity": "Country",

#                                   "Smoking prevalence, total (ages 15+) (% of adults)" : "smokers_perc"}, inplace=True)



# df = df.merge(clean_smoke_data, on="Country", how='left')
# df.describe()
# icu_df = pd.read_csv("../input/hospital-beds-by-country/API_SH.MED.BEDS.ZS_DS2_en_csv_v2_887506.csv")

# icu_df['Country Name'] = icu_df['Country Name'].replace('United States', 'US')

# icu_df['Country Name'] = icu_df['Country Name'].replace('Russian Federation', 'Russia')

# icu_df['Country Name'] = icu_df['Country Name'].replace('Iran, Islamic Rep.', 'Iran')

# icu_df['Country Name'] = icu_df['Country Name'].replace('Egypt, Arab Rep.', 'Egypt')

# icu_df['Country Name'] = icu_df['Country Name'].replace('Venezuela, RB', 'Venezuela')

# data['Country'] = data['Country'].replace('Czechia', 'Czech Republic')
# icu_cleaned = pd.DataFrame()

# icu_cleaned["Country"] = icu_df["Country Name"]

# icu_cleaned["ICU"] = np.nan



# for year in range(1960, 2020):

#     year_df = icu_df[str(year)].dropna()

#     icu_cleaned["ICU"].loc[year_df.index] = year_df.values
# data = pd.merge(data, icu_cleaned, on='Country')
# data['State'] = data['State'].fillna('')

# temp = data[[col for col in data.columns if col != 'State']]



# latest = temp[temp['Date'] == max(temp['Date'])].reset_index()

# latest_grouped = latest.groupby('Country')['ICU'].mean().reset_index()





# fig = px.bar(latest_grouped.sort_values('ICU', ascending=False)[:12][::-1], 

#              x='ICU', y='Country',

#              title='Ratio of ICU Beds per 1000 People', text='ICU', orientation='h',color_discrete_sequence=['green'] )

# fig.show()
trend_df = pd.DataFrame(columns={"infection_trend",

                                 "fatality_trend",

                                 "quarantine_trend",

                                 "school_trend",

                                 "total_population",

                                 "expected_cases",

                                 "expected_fatalities"})
#Just getting rid of the first days to have a multiple of 7

#Makes it easier to generate the sequences

train_data = train_data.query("Date>'2020-01-22'and Date<'2020-04-01'")

days_in_sequence = 21



trend_list = []



with tqdm(total=len(list(train_data.Country.unique()))) as pbar:

    for country in train_data.Country.unique():

        for province in train_data.query(f"Country=='{country}'").State.unique():

            province_df = train_data.query(f"Country=='{country}' and State=='{province}'")

            

            #I added a quick hack to double the number of sequences

            #Warning: This will later create a minor leakage from the 

            # training set into the validation set.

            for i in range(0,len(province_df),int(days_in_sequence/3)):

                if i+days_in_sequence<=len(province_df):

                    #prepare all the temporal inputs

                    infection_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].Confirmed.values]

                    fatality_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].Fatalities.values]

                    restriction_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].restrictions.values]

                    quarantine_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].quarantine.values]

                    school_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].schools.values]



                    #preparing all the demographic inputs

                    total_population = float(province_df.iloc[i].total_pop)

                    density = float(province_df.iloc[i].density)

                    hospibed = float(province_df.iloc[i].hospibed)

                    lung = float(province_df.iloc[i].lung)

                    centenarian_pop = float(province_df.iloc[i]["age_100+"])



                    expected_cases = float(province_df.iloc[i+days_in_sequence-1].Confirmed)

                    expected_fatalities = float(province_df.iloc[i+days_in_sequence-1].Fatalities)



                    trend_list.append({"infection_trend":infection_trend,

                                     "fatality_trend":fatality_trend,

                                     "restriction_trend":restriction_trend,

                                     "quarantine_trend":quarantine_trend,

                                     "school_trend":school_trend,

                                     "demographic_inputs":[total_population,density,hospibed,lung,centenarian_pop],

                                     "expected_cases":expected_cases,

                                     "expected_fatalities":expected_fatalities})

        pbar.update(1)

trend_df = pd.DataFrame(trend_list)
trend_df["temporal_inputs"] = [np.asarray([trends["infection_trend"],trends["fatality_trend"],trends["restriction_trend"],trends["quarantine_trend"],trends["school_trend"]]) for idx,trends in trend_df.iterrows()]



trend_df = shuffle(trend_df)
trend_df.head()
i=0

temp_df = pd.DataFrame()

for idx,row in trend_df.iterrows():

    if sum(row.infection_trend)>0:

        temp_df = temp_df.append(row)

    else:

        if i<25:

            temp_df = temp_df.append(row)

            i+=1

trend_df = temp_df
trend_df.head()
sequence_length = 20

training_percentage = 0.9
training_item_count = int(len(trend_df)*training_percentage)

validation_item_count = len(trend_df)-int(len(trend_df)*training_percentage)

training_df = trend_df[:training_item_count]

validation_df = trend_df[training_item_count:]
X_temporal_train = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in training_df["temporal_inputs"].values]),(training_item_count,5,sequence_length)),(0,2,1) )).astype(np.float32)

X_demographic_train = np.asarray([np.asarray(x) for x in training_df["demographic_inputs"]]).astype(np.float32)

Y_cases_train = np.asarray([np.asarray(x) for x in training_df["expected_cases"]]).astype(np.float32)

Y_fatalities_train = np.asarray([np.asarray(x) for x in training_df["expected_fatalities"]]).astype(np.float32)
X_temporal_test = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in validation_df["temporal_inputs"]]),(validation_item_count,5,sequence_length)),(0,2,1)) ).astype(np.float32)

X_demographic_test = np.asarray([np.asarray(x) for x in validation_df["demographic_inputs"]]).astype(np.float32)

Y_cases_test = np.asarray([np.asarray(x) for x in validation_df["expected_cases"]]).astype(np.float32)

Y_fatalities_test = np.asarray([np.asarray(x) for x in validation_df["expected_fatalities"]]).astype(np.float32)
#temporal input branch

temporal_input_layer = Input(shape=(sequence_length, 5))

main_rnn_layer = layers.LSTM(64, return_sequences=True, recurrent_dropout=0.2)(temporal_input_layer)



#demographic input branch

demographic_input_layer = Input(shape=(5))

demographic_dense = layers.Dense(32)(demographic_input_layer)

demographic_dropout = layers.Dropout(0.2)(demographic_dense)



#cases output branch

rnn_c = layers.LSTM(64)(main_rnn_layer)

merge_c = layers.Concatenate(axis=-1)([rnn_c,demographic_dropout])

dense_c = layers.Dense(256)(merge_c)

dropout_c = layers.Dropout(0.2)(dense_c)

cases = layers.Dense(1, activation=layers.LeakyReLU(alpha=0.1),name="cases")(dropout_c)



#fatality output branch

rnn_f = layers.LSTM(64)(main_rnn_layer)

merge_f = layers.Concatenate(axis=-1)([rnn_f,demographic_dropout])

dense_f = layers.Dense(256)(merge_f)

dropout_f = layers.Dropout(0.2)(dense_f)

fatalities = layers.Dense(1, activation=layers.LeakyReLU(alpha=0.1), name="fatalities")(dropout_f)





model = Model([temporal_input_layer,demographic_input_layer], [cases,fatalities])



model.summary()
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, factor=0.6),

             EarlyStopping(monitor='val_loss', patience=20),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

model.compile(loss=[tf.keras.losses.MeanSquaredLogarithmicError(),tf.keras.losses.MeanSquaredLogarithmicError()], optimizer="adam")

history = model.fit([X_temporal_train,X_demographic_train], [Y_cases_train, Y_fatalities_train], 

          epochs = 250, 

          batch_size = 16, 

          validation_data=([X_temporal_test,X_demographic_test],  [Y_cases_test, Y_fatalities_test]),

                   callbacks=callbacks)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss over epochs')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()
plt.plot(history.history['cases_loss'])

plt.plot(history.history['val_cases_loss'])

plt.title('Loss over epochs for the number of cases')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()
plt.plot(history.history['fatalities_loss'])

plt.plot(history.history['val_fatalities_loss'])

plt.title('Loss over epochs for the number of fatalities')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()
model.load_weights("best_model.h5")
predictions = model.predict([X_temporal_test,X_demographic_test])
display_limit = 30

for inputs, pred_cases, exp_cases, pred_fatalities, exp_fatalities in zip(X_temporal_test,predictions[0][:display_limit], Y_cases_test[:display_limit], predictions[1][:display_limit], Y_fatalities_test[:display_limit]):

    print("================================================")

    print(inputs)

    print("Expected cases:", exp_cases, " Prediction:", pred_cases[0], "Expected fatalities:", exp_fatalities, " Prediction:", pred_fatalities[0] )
#Will retrieve the number of cases and fatalities for the past 6 days from the given date

def build_inputs_for_date(country, province, date, df):

    start_date = date - timedelta(days=20)

    end_date = date - timedelta(days=1)

    

    str_start_date = start_date.strftime("%Y-%m-%d")

    str_end_date = end_date.strftime("%Y-%m-%d")

    df = df.query("Country=='"+country+"' and State=='"+province+"' and Date>='"+str_start_date+"' and Date<='"+str_end_date+"'")

    

    #preparing the temporal inputs

    temporal_input_data = np.transpose(np.reshape(np.asarray([df["Confirmed"],

                                                 df["Fatalities"],

                                                 df["restrictions"],

                                                 df["quarantine"],

                                                 df["schools"]]),

                                     (5,sequence_length)), (1,0) ).astype(np.float32)

    

    #preparing all the demographic inputs

    total_population = float(province_df.iloc[i].total_pop)

    density = float(province_df.iloc[i].density)

    hospibed = float(province_df.iloc[i].hospibed)

    lung = float(province_df.iloc[i].lung)

    centenarian_pop = float(province_df.iloc[i]["age_100+"])

    demographic_input_data = [total_population,density,hospibed,lung,centenarian_pop]

    

    return [np.array([temporal_input_data]), np.array([demographic_input_data])]
#Take a dataframe in input, will do the predictions and return the dataframe with extra rows

#containing the predictions

def predict_for_region(country, province, df):

    begin_prediction = "2020-04-01"

    start_date = datetime.strptime(begin_prediction,"%Y-%m-%d")

    end_prediction = "2020-05-14"

    end_date = datetime.strptime(end_prediction,"%Y-%m-%d")

    

    date_list = [start_date + timedelta(days=x) for x in range((end_date-start_date).days+1)]

    for date in date_list:

        input_data = build_inputs_for_date(country, province, date, df)

        result = model.predict(input_data)

        

        #just ensuring that the outputs is

        #higher than the previous counts

        result[0] = np.round(result[0])

        if result[0]<input_data[0][0][-1][0]:

            result[0]=np.array([[input_data[0][0][-1][0]]])

        

        result[1] = np.round(result[1])

        if result[1]<input_data[0][0][-1][1]:

            result[1]=np.array([[input_data[0][0][-1][1]]])

        

        #We assign the quarantine and school status

        #depending on previous values

        #e.g Once a country is locked, it will stay locked until the end

        df = df.append({"Country":country, 

                        "State":province, 

                        "Date":date.strftime("%Y-%m-%d"), 

                        "restrictions": 1 if any(input_data[0][0][2]) else 0,

                        "quarantine": 1 if any(input_data[0][0][3]) else 0,

                        "schools": 1 if any(input_data[0][0][4]) else 0,

                        "total_pop": input_data[1][0],

                        "density": input_data[1][0][1],

                        "hospibed": input_data[1][0][2],

                        "lung": input_data[1][0][3],

                        "age_100+": input_data[1][0][4],

                        "Confirmed":round(result[0][0][0]),	

                        "Fatalities":round(result[1][0][0])},

                       ignore_index=True)

    return df
#The functions that are called here need to optimise, sorry about that!

copy_df = train_data

with tqdm(total=len(list(copy_df.Country.unique()))) as pbar:

    for country in copy_df.Country.unique():

        for province in copy_df.query("Country=='"+country+"'").State.unique():

            copy_df = predict_for_region(country, province, copy_df)

        pbar.update(1)
groundtruth_df = gpd.read_file("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

groundtruth_df = groundtruth_df.rename(columns={'Province_State': 'State', 

                                              'Country_Region': 'Country', 

                                              'ConfirmedCases': 'Confirmed'})

groundtruth_df["Confirmed"] = groundtruth_df["Confirmed"].astype("float")

groundtruth_df["Fatalities"] = groundtruth_df["Fatalities"].astype("float")

#The country_region got modifying in the enriched dataset by @optimo, 

# so we have to apply the same change to this Dataframe.

groundtruth_df["Country"] = [ row.Country.replace("'","").strip(" ") if row.State=="" else str(row.Country+"_"+row.State).replace("'","").strip(" ") for idx,row in groundtruth_df.iterrows()]



last_date = groundtruth_df.Date.unique()[-1]
#to remove annoying warnings from pandas

pd.options.mode.chained_assignment = None



def get_RMSLE_per_region(region, groundtruth_df, display_only=False):

    groundtruth_df["Confirmed"] = groundtruth_df["Confirmed"].astype("float")

    groundtruth_df["Fatalities"] = groundtruth_df["Fatalities"].astype("float")

    

    #we only take data until the 30th of March 2020 as the groundtruth was not available for later dates.

    groundtruth = groundtruth_df.query("Country=='"+region+"' and Date>='2020-04-01' and Date<='"+last_date+"'")

    predictions = copy_df.query("Country=='"+region+"' and Date>='2020-04-01' and Date<='"+last_date+"'")

    

    RMSLE_cases = np.sqrt(mean_squared_log_error( groundtruth.Confirmed.values, predictions.Confirmed.values ))

    RMSLE_fatalities = np.sqrt(mean_squared_log_error( groundtruth.Fatalities.values, predictions.Fatalities.values ))

    

    if display_only:

        print(region)

        print("RMSLE on cases:",np.mean(RMSLE_cases))

        print("RMSLE on fatalities:",np.mean(RMSLE_fatalities))

    else:

        return RMSLE_cases, RMSLE_fatalities
def get_RMSLE_for_all_regions(groundtruth_df):

    RMSLE_cases_list = []

    RMSLE_fatalities_list = []

    for region in groundtruth_df.Country.unique():

        RMSLE_cases, RMSLE_fatalities = get_RMSLE_per_region(region, groundtruth_df, False)

        RMSLE_cases_list.append(RMSLE_cases)

        RMSLE_fatalities_list.append(RMSLE_fatalities)

    print("RMSLE on cases:",np.mean(RMSLE_cases_list))

    print("RMSLE on fatalities:",np.mean(RMSLE_fatalities_list))
get_RMSLE_for_all_regions(groundtruth_df)
badly_affected_countries = ["France","Italy","United Kingdom","Spain","Iran","Germany", "Turkey"]

for country in badly_affected_countries:

    get_RMSLE_per_region(country, groundtruth_df, display_only=True)
healthy_countries = ["Taiwan*","Singapore","Kenya","Slovenia","Portugal", "Israel"]

for country in healthy_countries:

    get_RMSLE_per_region(country, groundtruth_df, display_only=True)
test_data = gpd.read_file("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

#The country_region got modifying in the enriched dataset by @optimo, 

# so we have to apply the same change to the test Dataframe.

test_data["Country_Region"] = [ row.Country_Region if row.Province_State=="" else row.Country_Region+"_"+row.Province_State for idx,row in test_data.iterrows() ]

test_data.head()
submission_df = pd.DataFrame(columns=["ForecastId","ConfirmedCases","Fatalities"])

with tqdm(total=len(test_data)) as pbar:

    for idx, row in test_data.iterrows():

        #Had to remove single quotes because of countries like Cote D'Ivoire for example

        country_region = row.Country_Region.replace("'","").strip(" ")

        province_state = row.Province_State.replace("'","").strip(" ")

        item = copy_df.query("Country=='"+country_region+"' and State=='"+province_state+"' and Date=='"+row.Date+"'")

        submission_df = submission_df.append({"ForecastId":row.ForecastId,

                                              "ConfirmedCases":int(item.Confirmed.values[0]),

                                              "Fatalities":int(item.Fatalities.values[0])},

                                             ignore_index=True)

        pbar.update(1)
submission_df.sample(20)
copy_df.to_csv('covid_19_week_4_data.csv', index=None)
submission_df.to_csv("submission.csv",index=False)