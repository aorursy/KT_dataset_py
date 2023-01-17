import numpy as np

import pandas as pd

import geopandas as gpd

from shapely.geometry import Point

import os

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

dataset = gpd.read_file("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

dataset["ConfirmedCases"] = dataset["ConfirmedCases"].astype("float")

dataset["Fatalities"] = dataset["Fatalities"].astype("float")

dataset["Country_Region"] = [ row.Country_Region.replace("'","").strip(" ") if row.Province_State=="" else str(row.Country_Region+"_"+row.Province_State).replace("'","").strip(" ") for idx,row in dataset.iterrows()]

dataset.head()
enriched_dataset = gpd.read_file("/kaggle/input/enriched-covid-19-week-2/enriched_covid_19_week_2.csv")

enriched_dataset["Country_Region"] = [country_name.replace("'","") for country_name in enriched_dataset["Country_Region"]]

enriched_dataset["restrictions"] = enriched_dataset["restrictions"].astype("int")

enriched_dataset["quarantine"] = enriched_dataset["quarantine"].astype("int")

enriched_dataset["schools"] = enriched_dataset["schools"].astype("int")

enriched_dataset["total_pop"] = enriched_dataset["total_pop"].astype("float")

enriched_dataset["density"] = enriched_dataset["density"].astype("float")

enriched_dataset["hospibed"] = enriched_dataset["hospibed"].astype("float")

enriched_dataset["lung"] = enriched_dataset["lung"].astype("float")

enriched_dataset["total_pop"] = enriched_dataset["total_pop"]/max(enriched_dataset["total_pop"])

enriched_dataset["density"] = enriched_dataset["density"]/max(enriched_dataset["density"])

enriched_dataset["hospibed"] = enriched_dataset["hospibed"]/max(enriched_dataset["hospibed"])

enriched_dataset["lung"] = enriched_dataset["lung"]/max(enriched_dataset["lung"])

enriched_dataset["age_100+"] = enriched_dataset["age_100+"].astype("float")

enriched_dataset["age_100+"] = enriched_dataset["age_100+"]/max(enriched_dataset["age_100+"])

enriched_dataset = enriched_dataset[["Country_Region","Date","restrictions","quarantine","schools","hospibed","lung","total_pop","density","age_100+"]]

enriched_dataset.head()
merge_dataset = dataset.merge(enriched_dataset,how="left", on=["Country_Region","Date"]).drop_duplicates()

merge_dataset.head()
for country_region in merge_dataset.Country_Region.unique():

    query_df = merge_dataset.query("Country_Region=='"+country_region+"' and Date=='2020-03-25'")

    merge_dataset.loc[(merge_dataset["Country_Region"]==country_region) & (merge_dataset["Date"]>"2020-03-25"),"total_pop"] = query_df.total_pop.values[0]

    merge_dataset.loc[(merge_dataset["Country_Region"]==country_region) & (merge_dataset["Date"]>"2020-03-25"),"hospibed"] = query_df.hospibed.values[0]

    merge_dataset.loc[(merge_dataset["Country_Region"]==country_region) & (merge_dataset["Date"]>"2020-03-25"),"density"] = query_df.density.values[0]

    merge_dataset.loc[(merge_dataset["Country_Region"]==country_region) & (merge_dataset["Date"]>"2020-03-25"),"lung"] = query_df.lung.values[0]

    merge_dataset.loc[(merge_dataset["Country_Region"]==country_region) & (merge_dataset["Date"]>"2020-03-25"),"age_100+"] = query_df["age_100+"].values[0]

    merge_dataset.loc[(merge_dataset["Country_Region"]==country_region) & (merge_dataset["Date"]>"2020-03-25"),"restrictions"] = query_df.restrictions.values[0]

    merge_dataset.loc[(merge_dataset["Country_Region"]==country_region) & (merge_dataset["Date"]>"2020-03-25"),"quarantine"] = query_df.quarantine.values[0]

    merge_dataset.loc[(merge_dataset["Country_Region"]==country_region) & (merge_dataset["Date"]>"2020-03-25"),"schools"] = query_df.schools.values[0]


median_pop = np.median(enriched_dataset.total_pop)

median_hospibed = np.median(enriched_dataset.hospibed)

median_density = np.median(enriched_dataset.density)

median_lung = np.median(enriched_dataset.lung)

median_centenarian_pop = np.median(enriched_dataset["age_100+"])

#need to replace that with a joint using Pandas

print("The missing countries/region are:")

for country_region in merge_dataset.Country_Region.unique():

    if enriched_dataset.query("Country_Region=='"+country_region+"'").empty:

        print(country_region)

        

        merge_dataset.loc[merge_dataset["Country_Region"]==country_region,"total_pop"] = median_pop

        merge_dataset.loc[merge_dataset["Country_Region"]==country_region,"hospibed"] = median_hospibed

        merge_dataset.loc[merge_dataset["Country_Region"]==country_region,"density"] = median_density

        merge_dataset.loc[merge_dataset["Country_Region"]==country_region,"lung"] = median_lung

        merge_dataset.loc[merge_dataset["Country_Region"]==country_region,"age_100+"] = median_centenarian_pop

        merge_dataset.loc[merge_dataset["Country_Region"]==country_region,"restrictions"] = 0

        merge_dataset.loc[merge_dataset["Country_Region"]==country_region,"quarantine"] = 0

        merge_dataset.loc[merge_dataset["Country_Region"]==country_region,"schools"] = 0
feed_dataset = pd.DataFrame(columns={"infection_past","fatality_past","quarantine_past","school_past","demographic_inputs","expected_cases","expected_fatalities"})
#Makes it easier to generate the sequences

# train_df = train_df.query("Date>'2020-01-22'and Date<'2020-04-01'")

days_in_sequence = 10



trend_list = []



with tqdm(total=len(list(merge_dataset.Country_Region.unique()))) as pbar:

    for country in merge_dataset.Country_Region.unique():

        for province in merge_dataset.query(f"Country_Region=='{country}'").Province_State.unique():

            province_df = merge_dataset.query(f"Country_Region=='{country}' and Province_State=='{province}'")

            

            #I added a quick hack to double the number of sequences

            #Warning: This will later create a minor leakage from the 

            # training set into the validation set.

            for i in range(0,len(province_df),int(days_in_sequence/2)):

                if i+days_in_sequence<=len(province_df):

                    #prepare all the temporal inputs

                    infection_past = [float(x) for x in province_df[i:i+days_in_sequence-1].ConfirmedCases.values]

                    fatality_past = [float(x) for x in province_df[i:i+days_in_sequence-1].Fatalities.values]

                    restriction_past = [float(x) for x in province_df[i:i+days_in_sequence-1].restrictions.values]

                    quarantine_past = [float(x) for x in province_df[i:i+days_in_sequence-1].quarantine.values]

                    school_past = [float(x) for x in province_df[i:i+days_in_sequence-1].schools.values]



                    #preparing all the demographic inputs

                    total_population = float(province_df.iloc[i].total_pop)

                    density = float(province_df.iloc[i].density)

                    hospibed = float(province_df.iloc[i].hospibed)

                    lung = float(province_df.iloc[i].lung)

                    centenarian_pop = float(province_df.iloc[i]["age_100+"])



                    expected_cases = float(province_df.iloc[i+days_in_sequence-1].ConfirmedCases)

                    expected_fatalities = float(province_df.iloc[i+days_in_sequence-1].Fatalities)



                    trend_list.append({"infection_past":infection_past,

                                     "fatality_past":fatality_past,

                                     "restriction_past":restriction_past,

                                     "quarantine_past":quarantine_past,

                                     "school_past":school_past,

                                     "demographic_inputs":[total_population,density,hospibed,lung,centenarian_pop],

                                     "expected_cases":expected_cases,

                                     "expected_fatalities":expected_fatalities})

        pbar.update(1)

feed_dataset = pd.DataFrame(trend_list)
feed_dataset["temporal_inputs"] = [np.asarray([trends["infection_past"],trends["fatality_past"],trends["restriction_past"],trends["quarantine_past"],trends["school_past"]]) for idx,trends in feed_dataset.iterrows()]



feed_dataset = shuffle(feed_dataset)
feed_dataset.head()
i=0

temp_df = pd.DataFrame()

for idx,row in feed_dataset.iterrows():

    if sum(row.infection_past)>0:

        temp_df = temp_df.append(row)

    else:

        if i<25:

            temp_df = temp_df.append(row)

            i+=1

feed_dataset = temp_df
feed_dataset.head()
sequence_length=9

from sklearn.model_selection import train_test_split

train_dataset, validation_dataset = train_test_split(feed_dataset, test_size=0.1)
X_temporal_train = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in train_dataset["temporal_inputs"].values]),(train_dataset.shape[0],5,sequence_length)),(0,2,1) )).astype(np.float32)

X_demographic_train = np.asarray([np.asarray(x) for x in train_dataset["demographic_inputs"]]).astype(np.float32)

Y_cases_train = np.asarray([np.asarray(x) for x in train_dataset["expected_cases"]]).astype(np.float32)

Y_fatalities_train = np.asarray([np.asarray(x) for x in train_dataset["expected_fatalities"]]).astype(np.float32)
X_temporal_test = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in validation_dataset["temporal_inputs"]]),(validation_dataset.shape[0],5,sequence_length)),(0,2,1)) ).astype(np.float32)

X_demographic_test = np.asarray([np.asarray(x) for x in validation_dataset["demographic_inputs"]]).astype(np.float32)

Y_cases_test = np.asarray([np.asarray(x) for x in validation_dataset["expected_cases"]]).astype(np.float32)

Y_fatalities_test = np.asarray([np.asarray(x) for x in validation_dataset["expected_fatalities"]]).astype(np.float32)
#temporal input branch

temporal_input_layer = Input(shape=(sequence_length,5))

main_rnn_layer = layers.LSTM(64, return_sequences=True, recurrent_dropout=0.25)(temporal_input_layer)

main_rnn_layer = layers.LSTM(64, return_sequences=True, recurrent_dropout=0.25)(main_rnn_layer)



#demographic input branch

demographic_input_layer = Input(shape=(5))

demographic_dense = layers.Dense(16)(demographic_input_layer)

demographic_dropout = layers.Dropout(0.2)(demographic_dense)



#cases output branch

rnn_c = layers.LSTM(32)(main_rnn_layer)

merge_c = layers.Concatenate(axis=-1)([rnn_c,demographic_dropout])

dense_c = layers.Dense(128)(merge_c)

dropout_c = layers.Dropout(0.3)(dense_c)

cases = layers.Dense(1, activation=layers.LeakyReLU(alpha=0.1),name="cases")(dropout_c)



#fatality output branch

rnn_f = layers.LSTM(32)(main_rnn_layer)

merge_f = layers.Concatenate(axis=-1)([rnn_f,demographic_dropout])

dense_f = layers.Dense(128)(merge_f)

dropout_f = layers.Dropout(0.3)(dense_f)

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
model.load_weights("best_model.h5")
predictions = model.predict([X_temporal_test,X_demographic_test])
display_limit = 5

for inputs, pred_cases, exp_cases, pred_fatalities, exp_fatalities in zip(X_temporal_test,predictions[0][:display_limit], Y_cases_test[:display_limit], predictions[1][:display_limit], Y_fatalities_test[:display_limit]):

    print("================================================")

    print(inputs)

    print("Expected cases:", exp_cases, " Prediction:", pred_cases[0], "Expected fatalities:", exp_fatalities, " Prediction:", pred_fatalities[0] )
merge_dataset.head()
sequence_length=9

def build_inputs_for_date(country, province, date, df):

    start_date = date - timedelta(days=sequence_length)

    end_date = date - timedelta(days=1)

    

    str_start_date = start_date.strftime("%Y-%m-%d")

    str_end_date = end_date.strftime("%Y-%m-%d")

    # print(date)

    # print(str_start_date)

    # print(str_end_date)

    df = df.query("Country_Region=='"+country+"' and Province_State=='"+province+"' and Date>='"+str_start_date+"' and Date<='"+str_end_date+"'")

    # print(df.Date)

    #preparing the temporal inputs

    temporal_input_data = np.transpose(np.reshape(np.asarray([df["ConfirmedCases"],

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

        df = df.append({"Country_Region":country, 

                        "Province_State":province, 

                        "Date":date.strftime("%Y-%m-%d"), 

                        "restrictions": 1 if any(input_data[0][0][2]) else 0,

                        "quarantine": 1 if any(input_data[0][0][3]) else 0,

                        "schools": 1 if any(input_data[0][0][4]) else 0,

                        "total_pop": input_data[1][0],

                        "density": input_data[1][0][1],

                        "hospibed": input_data[1][0][2],

                        "lung": input_data[1][0][3],

                        "age_100+": input_data[1][0][4],

                        "ConfirmedCases":round(result[0][0][0]),	

                        "Fatalities":round(result[1][0][0])},

                       ignore_index=True)

    return df
# train_df = train_df.query("Date>'2020-01-22'and Date<'2020-04-01'")

copy_df = merge_dataset.query("Date>'2020-01-22'and Date<'2020-04-01'")

with tqdm(total=len(list(copy_df.Country_Region.unique()))) as pbar:

    for country in copy_df.Country_Region.unique():

        for province in copy_df.query("Country_Region=='"+country+"'").Province_State.unique():

            copy_df = predict_for_region(country, province, copy_df)

        pbar.update(1)
test_dataset = gpd.read_file("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

#The country_region got modifying in the enriched dataset by @optimo, 

# so we have to apply the same change to the test Dataframe.

test_dataset["Country_Region"] = [ row.Country_Region if row.Province_State=="" else row.Country_Region+"_"+row.Province_State for idx,row in test_dataset.iterrows() ]

test_dataset.head()
submission_df = pd.DataFrame(columns=["ForecastId","ConfirmedCases","Fatalities"])

with tqdm(total=len(test_dataset)) as pbar:

    for idx, row in test_dataset.iterrows():

        #Had to remove single quotes because of countries like Cote D'Ivoire for example

        country_region = row.Country_Region.replace("'","").strip(" ")

        province_state = row.Province_State.replace("'","").strip(" ")

        item = copy_df.query("Country_Region=='"+country_region+"' and Province_State=='"+province_state+"' and Date=='"+row.Date+"'")

        submission_df = submission_df.append({"ForecastId":row.ForecastId,

                                              "ConfirmedCases":int(item.ConfirmedCases.values[0]),

                                              "Fatalities":int(item.Fatalities.values[0])},

                                             ignore_index=True)

        pbar.update(1)
submission_df.sample(20)
submission_df.to_csv("submission.csv",index=False)