import numpy as np

import pandas as pd

import geopandas as gpd

from shapely.geometry import Point

import os

import tensorflow as tf

from tqdm import tqdm

from sklearn.utils import shuffle





from datetime import datetime

from datetime import timedelta



from tensorflow.keras import layers

from tensorflow.keras import Input

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = gpd.read_file("/kaggle/input/enrichedcovid19week2/enriched_covid_19_week_2.csv")

train_df["Country_Region"] = [country_name.replace("'","") for country_name in train_df["Country_Region"]]

train_df["restrictions"] = train_df["restrictions"].astype("int")

train_df["quarantine"] = train_df["quarantine"].astype("int")

train_df["schools"] = train_df["schools"].astype("int")

train_df["total_pop"] = train_df["total_pop"].astype("float")

train_df["density"] = train_df["density"].astype("float")

train_df["hospibed"] = train_df["hospibed"].astype("float")

train_df["lung"] = train_df["lung"].astype("float")

train_df["total_pop"] = train_df["total_pop"]/max(train_df["total_pop"])

train_df["density"] = train_df["density"]/max(train_df["density"])

train_df["hospibed"] = train_df["hospibed"]/max(train_df["hospibed"])

train_df["lung"] = train_df["lung"]/max(train_df["lung"])

train_df.head()
train_df.columns
trend_df = pd.DataFrame(columns={"infection_trend","fatality_trend","quarantine_trend","school_trend","total_population","expected_cases","expected_fatalities"})
#Just getting rid of the first days to have a multiple of 14

#Makes it easier to generate the sequences

train_df = train_df.query("Date>'2020-01-22'and Date<='2020-03-18'")

days_in_sequence = 14



trend_list = []



with tqdm(total=len(list(train_df.Country_Region.unique()))) as pbar:

    for country in train_df.Country_Region.unique():

        for province in train_df.query(f"Country_Region=='{country}'").Province_State.unique():

            province_df = train_df.query(f"Country_Region=='{country}' and Province_State=='{province}'")

            

            #I added a quick hack to double the number of sequences

            #Warning: This will later create a minor leakage from the 

            # training set into the validation set. TO FIX.

            for i in range(0,len(province_df),int(days_in_sequence/2)):

                if i+days_in_sequence<=len(province_df):

                    #prepare all the temporal inputs

                    infection_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].ConfirmedCases.values]

                    fatality_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].Fatalities.values]

                    restriction_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].restrictions.values]

                    quarantine_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].quarantine.values]

                    school_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].schools.values]



                    #preparing all the demographic inputs

                    total_population = float(province_df.iloc[i].total_pop)

                    density = float(province_df.iloc[i].density)

                    hospibed = float(province_df.iloc[i].hospibed)

                    lung = float(province_df.iloc[i].lung)



                    expected_cases = float(province_df.iloc[i+days_in_sequence-1].ConfirmedCases)

                    expected_fatalities = float(province_df.iloc[i+days_in_sequence-1].Fatalities)



                    trend_list.append({"infection_trend":infection_trend,

                                     "fatality_trend":fatality_trend,

                                     "restriction_trend":restriction_trend,

                                     "quarantine_trend":quarantine_trend,

                                     "school_trend":school_trend,

                                     "demographic_inputs":[total_population,density,hospibed,lung],

                                     "expected_cases":expected_cases,

                                     "expected_fatalities":expected_fatalities})

        pbar.update(1)

trend_df = pd.DataFrame(trend_list)
trend_df["temporal_inputs"] = [np.asarray([trends["infection_trend"],trends["fatality_trend"],trends["restriction_trend"],trends["quarantine_trend"],trends["school_trend"]]) for idx,trends in trend_df.iterrows()]



trend_df = shuffle(trend_df)
i=0

y=0

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
sequence_length = 13

training_percentage = 0.8
training_item_count = int(len(trend_df)*training_percentage)

validation_item_count = len(trend_df)-int(len(trend_df)*training_percentage)

training_df = trend_df[:training_item_count]

validation_df = trend_df[training_item_count:]
X_temporal_train = np.asarray(np.reshape(np.asarray([np.asarray(x) for x in training_df["temporal_inputs"].values]),(training_item_count,5,sequence_length))).astype(np.float32)

X_demographic_train = np.asarray([np.asarray(x) for x in training_df["demographic_inputs"]]).astype(np.float32)

Y_cases_train = np.asarray([np.asarray(x) for x in training_df["expected_cases"]]).astype(np.float32)

Y_fatalities_train = np.asarray([np.asarray(x) for x in training_df["expected_fatalities"]]).astype(np.float32)
X_temporal_test = np.asarray(np.reshape(np.asarray([np.asarray(x) for x in validation_df["temporal_inputs"]]),(validation_item_count,5,sequence_length))).astype(np.float32)

X_demographic_test = np.asarray([np.asarray(x) for x in validation_df["demographic_inputs"]]).astype(np.float32)

Y_cases_test = np.asarray([np.asarray(x) for x in validation_df["expected_cases"]]).astype(np.float32)

Y_fatalities_test = np.asarray([np.asarray(x) for x in validation_df["expected_fatalities"]]).astype(np.float32)
#temporal input branch

temporal_input_layer = Input(shape=(5,sequence_length))

main_rnn_layer = layers.LSTM(128, return_sequences=True, recurrent_dropout=0.2)(temporal_input_layer)



#demographic input branch

demographic_input_layer = Input(shape=(4))

demographic_dense = layers.Dense(16)(demographic_input_layer)

demographic_dropout = layers.Dropout(0.2)(demographic_dense)



#cases output branch

rnn_c = layers.LSTM(64)(main_rnn_layer)

merge_c = layers.Concatenate(axis=-1)([rnn_c,demographic_dropout])

dense_c = layers.Dense(256)(merge_c)

dropout_c = layers.Dropout(0.3)(dense_c)

cases = layers.Dense(1, activation="relu",name="cases")(dropout_c)



#fatality output branch

rnn_f = layers.LSTM(64)(main_rnn_layer)

merge_f = layers.Concatenate(axis=-1)([rnn_f,demographic_dropout])

dense_f = layers.Dense(256)(merge_f)

dropout_f = layers.Dropout(0.3)(dense_f)

fatalities = layers.Dense(1, activation="relu", name="fatalities")(dropout_f)





model = Model([temporal_input_layer,demographic_input_layer], [cases,fatalities])



model.summary()
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, factor=0.6),

             EarlyStopping(monitor='val_loss', patience=20),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

model.compile(loss=[tf.keras.losses.MeanSquaredLogarithmicError(),tf.keras.losses.MeanSquaredLogarithmicError()], optimizer="adam")
history = model.fit([X_temporal_train,X_demographic_train], [Y_cases_train, Y_fatalities_train], 

          epochs = 200, 

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

    start_date = date - timedelta(days=13)

    end_date = date - timedelta(days=1)

    

    str_start_date = start_date.strftime("%Y-%m-%d")

    str_end_date = end_date.strftime("%Y-%m-%d")

    df = df.query("Country_Region=='"+country+"' and Province_State=='"+province+"' and Date>='"+str_start_date+"' and Date<='"+str_end_date+"'")

    

    #preparing the temporal inputs

    temporal_input_data = np.reshape(np.asarray([df["ConfirmedCases"],

                                                 df["Fatalities"],

                                                 df["restrictions"],

                                                 df["quarantine"],

                                                 df["schools"]]),

                                     (5,sequence_length)).astype(np.float32)

    

    #preparing all the demographic inputs

    total_population = float(province_df.iloc[i].total_pop)

    density = float(province_df.iloc[i].density)

    hospibed = float(province_df.iloc[i].hospibed)

    lung = float(province_df.iloc[i].lung)

    demographic_input_data = [total_population,density,hospibed,lung]

    

    return [np.array([temporal_input_data]), np.array([demographic_input_data])]
#Take a dataframe in input, will do the predictions and return the dataframe with extra rows

#containing the predictions

def predict_for_region(country, province, df):

    begin_prediction = "2020-03-19"

    start_date = datetime.strptime(begin_prediction,"%Y-%m-%d")

    end_prediction = "2020-04-30"

    end_date = datetime.strptime(end_prediction,"%Y-%m-%d")

    

    date_list = [start_date + timedelta(days=x) for x in range((end_date-start_date).days+1)]

    for date in date_list:

        input_data = build_inputs_for_date(country, province, date, df)

        result = model.predict(input_data)

        

        #just ensuring that the outputs is

        #higher than the previous counts

        

        result[0] = np.round(result[0])

        if result[0]<input_data[0][0][0][-1]:

            result[0]=np.array([[input_data[0][0][0][-1]]])

        

        result[1] = np.round(result[1])

        if result[1]<input_data[0][0][1][-1]:

            result[1]=np.array([[input_data[0][0][1][-1]]])

        

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

                        "ConfirmedCases":round(result[0][0][0]),	

                        "Fatalities":round(result[1][0][0])},

                       ignore_index=True)

    return df
#The functions that are called here need to optimise, sorry about that!

copy_df = train_df

with tqdm(total=len(list(copy_df.Country_Region.unique()))) as pbar:

    for country in copy_df.Country_Region.unique():

        for province in copy_df.query("Country_Region=='"+country+"'").Province_State.unique():

            copy_df = predict_for_region(country, province, copy_df)

        pbar.update(1)
def display_comparison(region):

    groundtruth_df = gpd.read_file("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

    groundtruth_df["ConfirmedCases"] = groundtruth_df["ConfirmedCases"].astype("float")

    groundtruth_df["Fatalities"] = groundtruth_df["Fatalities"].astype("float")

    groundtruth = groundtruth_df.query("Country_Region=='"+region+"' and Date>='2020-03-19' and Date<='2020-04-01'")

    prediction = copy_df.query("Country_Region=='"+region+"' and Date>='2020-03-19' and Date<='2020-04-01'")

    

    plt.plot(groundtruth.ConfirmedCases.values)

    plt.plot(prediction.ConfirmedCases.values)

    plt.title("Comparison between the actual data and our predictions for the number of cases")

    plt.ylabel('Number of cases')

    plt.xlabel('Date')

    plt.xticks(range(len(prediction.Date.values)),prediction.Date.values,rotation='vertical')

    plt.legend(['Groundtruth', 'Prediction'], loc='best')

    plt.show()

    

    plt.plot(groundtruth.Fatalities.values)

    plt.plot(prediction.Fatalities.values)

    plt.title("Comparison between the actual data and our predictions for the number of fatalities")

    plt.ylabel('Number of fatalities')

    plt.xlabel('Date')

    plt.xticks(range(len(prediction.Date.values)),prediction.Date.values,rotation='vertical')

    plt.legend(['Groundtruth', 'Prediction'], loc='best')

    plt.show()
display_comparison("Kenya")
display_comparison("Germany")
test_df = gpd.read_file("../input/covid19-global-forecasting-week-2/test.csv")

#The country_region got modifying in the enriched dataset by @optimo, 

# so we have to apply the same change to the test Dataframe.

test_df["Country_Region"] = [ row.Country_Region if row.Province_State=="" else row.Country_Region+"_"+row.Province_State for idx,row in test_df.iterrows() ]

test_df.head()
submission_df = pd.DataFrame(columns=["ForecastId","ConfirmedCases","Fatalities"])

with tqdm(total=len(test_df)) as pbar:

    for idx, row in test_df.iterrows():

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
submission_df.describe(include='all')