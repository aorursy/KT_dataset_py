# Exploratory notebook for the UK 2016 Road Safety Dataset to investigate prediction 

# capability for cycling accidents



# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

sns.set(style="white")



from datetime import datetime



from keras.layers import Activation

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

from keras.models import load_model



from sklearn.model_selection import train_test_split
# Read in the accident data. Each unique accident in the accident data can have multiple 

# casualties and vehicles

accident_data = pd.read_csv('../input/dftRoadSafety_Accidents_2016.csv', 

                            dtype={"Did_Police_Officer_Attend_Scene_of_Accident": int})



# Extract month from date

def month_from_date(date):

    dmyyyy = datetime.strptime(date, '%d/%m/%Y')

    return int(datetime.strftime(dmyyyy, '%m'))



accident_data['Month_of_Year'] = accident_data['Date'].apply(month_from_date)



# Extract hour from time

def hour_from_time(time):

    try:

        hhmm = datetime.strptime(time, '%H:%M')

        return int(datetime.strftime(hhmm, '%H'))

    except Exception:

        # Some times are nan

        return 0



accident_data['Hour_of_Day'] = accident_data['Time'].apply(hour_from_time)



# Drop columns which cannot be used as predictors of cycling accidents

accident_data = accident_data.drop(['Number_of_Vehicles', 'Local_Authority_(District)', 

                                    'Police_Force', 'Location_Easting_OSGR', 

                                    'Location_Northing_OSGR', 'Speed_limit',

                                    '1st_Road_Class', 'Road_Type', 'Date', 'Time', 

                                    'Junction_Detail', 'Junction_Control', '2nd_Road_Class', 

                                    '2nd_Road_Number', 'Special_Conditions_at_Site', 

                                    'Carriageway_Hazards', 'Number_of_Casualties',

                                    'Pedestrian_Crossing-Physical_Facilities', 

                                    'Pedestrian_Crossing-Human_Control',

                                    'Did_Police_Officer_Attend_Scene_of_Accident', 

                                    'Local_Authority_(Highway)', 

                                    'LSOA_of_Accident_Location', 'Accident_Severity', 

                                    '1st_Road_Number'], axis=1)



# Remove rows where latitude or longitude is NaN

accident_data = accident_data[np.isfinite(accident_data['Longitude'])]

accident_data = accident_data[np.isfinite(accident_data['Latitude'])]



accident_data.head()
# Plot the accident coordinates

plt.plot(accident_data.Longitude, accident_data.Latitude, 'b.', ms=0.5)
print ("There are", len(accident_data.index), "unique accidents")
# Read in the casualty data

casualty_data = pd.read_csv('../input/Cas.csv')



# Remove non-cycling casualties

casualty_data = casualty_data[casualty_data.Casualty_Type==1]



# Drop columns which cannot be used as predictors of cycling accidents

casualty_data = casualty_data.drop(['Bus_or_Coach_Passenger', 'Car_Passenger', 

                                    'Pedestrian_Movement', 'Pedestrian_Location', 

                                    'Casualty_Reference', 'Vehicle_Reference', 

                                    'Casualty_Class', 'Age_of_Casualty',  

                                    'Casualty_Severity', 'Casualty_IMD_Decile', 

                                    'Pedestrian_Road_Maintenance_Worker'], axis=1)

casualty_data.head()
print ("There are", len(casualty_data.index), "cycling casualties")
# Merge accident and casualty data for cycling accidents

casualty_and_accident_data = pd.merge(accident_data, casualty_data, on='Accident_Index', 

                                      how='left')

casualty_and_accident_data = casualty_and_accident_data.drop(['Accident_Index'], axis=1)



# Replace NaN with zero

casualty_and_accident_data.fillna(0, inplace=True)



# Casualty data after merge is cast to float. Cast back to integer

casualty_and_accident_data['Casualty_Type'] = casualty_and_accident_data['Casualty_Type'].astype(int)



# We will use Casualty_Type as an indicator of a cycling accident. If 1 it is a cycling accident and if 0 a non-cycling accident

casualty_and_accident_data = casualty_and_accident_data.rename(columns={'Casualty_Type': 'Cycling_Accident'})



# Display our data

casualty_and_accident_data



# Age_Band_of_Casualty

#

# 1: 0-5

# 2: 6-10

# 3: 11-15

# 4: 16-20

# 5: 21-25

# 6: 26-35

# 7: 36-45

# 8: 46-55

# 9: 56-65

# 10: 66-75

# 11: >75

# -1: Unknown



# Sex_of_Casualty

#

# 1: Male

# 2: Femals

# -1: Unknown



# Casualty_Severity

#

# 1: Fatal

# 2: Serious

# 3: Slight



# Day_of_Week

#

# 1: Sunday

# 2: Monday

# 3: Tuesday

# 4: Wednesday

# 5: Thursday

# 6: Friday

# 7: Saturday



# Road_Surface_Conditions

#

# 1: Dry

# 2: Wet / damp

# 3: Snow

# 4: Frost / ice

# 5: Flood

# 6: Oil

# 7: Mud

# -1: No data



# Cycling_Accident

#

# 1: Yes

# 0: No
# Convert the latitude and longitude coordinates to area squares

latitude_min = min(casualty_and_accident_data['Latitude'])

latitude_max = max(casualty_and_accident_data['Latitude'])

longitude_min = min(casualty_and_accident_data['Longitude'])

longitude_max = max(casualty_and_accident_data['Longitude'])



# Determine the required step for a given number of areas

latitude_step = (latitude_max - latitude_min) / 1000

longitude_step = (longitude_max - longitude_min) / 1000



# Determine the latitude area

def calc_lat_area(current_lat):

    return int((current_lat - latitude_min) / latitude_step)



# Determine the longitude area

def calc_lon_area(current_lon):

    return int((current_lon - longitude_min) / longitude_step)



# Create new columns with the latitude and longitude areas

casualty_and_accident_data["Latitude_Area"] = list(map(calc_lat_area, casualty_and_accident_data["Latitude"]))

casualty_and_accident_data["Longitude_Area"] = list(map(calc_lon_area, casualty_and_accident_data["Longitude"]))



# Drop the old latitude and logitude columns

casualty_and_accident_data = casualty_and_accident_data.drop(['Longitude', 'Latitude'], axis=1)



# Display our data

casualty_and_accident_data.head()
# Compute the correlation matrix

corr = casualty_and_accident_data.corr()



# Draw the correlation matrix

sns.heatmap(corr, square=True, linewidths=.4, cbar_kws={"shrink": .4})



plt.title("Accident Correlation")

plt.show()
# Split the data, 75% for training and 25% for validation

x_data = casualty_and_accident_data.drop(['Cycling_Accident'], axis=1)

y_data = casualty_and_accident_data['Cycling_Accident']



(trainData, testData, trainLabels, testLabels) = train_test_split(x_data.values, 

                                                                  y_data.values, 

                                                                  test_size=0.25, 

                                                                  random_state=42)
# Setup our Keras model

model = Sequential()

model.add(Dense(2000, input_dim=12, kernel_initializer="uniform", activation="relu"))

model.add(Dense(500))

model.add(Dense(80))

model.add(Dense(1))

model.add(Activation("sigmoid"))



# Train the model using SGD

sgd = SGD(lr=0.01)

model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.fit(trainData, trainLabels, epochs=4, batch_size=30, verbose=1)
# Evaluate accuracy on testing data set

(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=3, verbose=1)

print("\nloss = {:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
# Accuracy is reported high but in raslity we dont have enough cycling accidents for 

# meaningful training

ANN_pred = np.round(model.predict(testData))

ANN_cycling_accident = ANN_pred.astype(int)

ANN_cycling_accident.min()
ANN_cycling_accident.max()