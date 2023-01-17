# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing libraries

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential , load_model
from keras.layers import Dense , Dropout



# Importing dataset
dataset = pd.read_csv('../input/homicide-reports/database.csv' , low_memory = False)
X = dataset.iloc[: , [2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,9,20,21,22,23]].values
y = dataset.iloc[: , 10:11].values


# Encoding data 
encoder = LabelEncoder()

y = encoder.fit_transform(y)

agency_code_encoded = encoder.fit_transform(X[: , 0]).reshape((-1,1))
agency_type_encoded = encoder.fit_transform(X[: , 1]).reshape((-1,1))
city_encoded = encoder.fit_transform(X[: , 2]).reshape((-1,1))
state_encoded = encoder.fit_transform(X[: , 3]).reshape((-1,1))
year_encoded = encoder.fit_transform(X[: , 4]).reshape((-1,1))
month_encoded = encoder.fit_transform(X[: , 5]).reshape((-1,1))
incident_encoded = encoder.fit_transform(X[: , 6]).reshape((-1,1))
crime_type_encoded = encoder.fit_transform(X[: , 7]).reshape((-1,1))
victim_sex_encoded = encoder.fit_transform(X[: , 8]).reshape((-1,1))
victim_age_encoded = encoder.fit_transform(X[: , 9]).reshape((-1,1))
victim_race_encoded = encoder.fit_transform(X[: , 10]).reshape((-1,1))
victim_ethnicity_encoded = encoder.fit_transform(X[: , 11]).reshape((-1,1))
perpetrator_sex_encoded = encoder.fit_transform(X[: , 12]).reshape((-1,1))
perpetrator_age_encoded = encoder.fit_transform(X[: , 13]).reshape((-1,1))
perpetrator_race_encoded = encoder.fit_transform(X[: , 14]).reshape((-1,1))
perpetrator_ethnicity_encoded = encoder.fit_transform(X[: , 15]).reshape((-1,1))
relationship_encoded = encoder.fit_transform(X[: , 16]).reshape((-1,1))
weapon_encoded = encoder.fit_transform(X[: , 17]).reshape((-1,1))
victim_count_encoded = encoder.fit_transform(X[: , 18]).reshape((-1,1))
perpetrator_count_encoded = encoder.fit_transform(X[: , 19]).reshape((-1,1))
record_source_encoded = encoder.fit_transform(X[: , 20]).reshape((-1,1))



# Convert various data into binary
def get_binary(data , length):    
    bin_data = bin(data)[2:].zfill(length)
    node = []
    
    for i in bin_data:
        node.append(int(i))        
    return node



# List for various features
i_agency_code = []
i_agency_type = []
i_city = []
i_state = []
i_year = []
i_month = []
i_incident = []
i_crime_type = []
i_victim_sex = []
i_victim_age = []
i_victim_race = []
i_victim_ethnicity = []
i_perpetrator_sex = []
i_perpetrator_age = []
i_perpetrator_race = []
i_perpetrator_ethnicity = []
i_relationship = []
i_weapon = []
i_victim_count = []
i_perpetrator_count = []
i_record_source = []

# Populating list with binary value of ffeatures

for i in range (len(X)):    
    i_agency_code.append([])
    i_agency_code[i].append(get_binary(agency_code_encoded[i][0] , 15))
    i_agency_type.append([])
    i_agency_type[i].append(get_binary(agency_type_encoded[i][0] , 4))
    i_city.append([])
    i_city[i].append(get_binary(city_encoded[i][0] , 12))
    i_state.append([])
    i_state[i].append(get_binary(state_encoded[i][0] , 6))
    i_year.append([])
    i_year[i].append(get_binary(year_encoded[i][0] , 6))
    i_month.append([])
    i_month[i].append(get_binary(month_encoded[i][0] , 4))
    i_incident.append([])
    i_incident[i].append(get_binary(incident_encoded[i][0] , 10))
    i_crime_type.append([])
    i_crime_type[i].append(crime_type_encoded[i][0])
    i_victim_sex.append([])
    i_victim_sex[i].append(victim_sex_encoded[i][0])
    i_victim_age.append([])
    i_victim_age[i].append(get_binary(victim_age_encoded[i][0] , 7))
    i_victim_race.append([])
    i_victim_race[i].append(get_binary(victim_race_encoded[i][0] , 3))
    i_victim_ethnicity.append([])
    i_victim_ethnicity[i].append(victim_ethnicity_encoded[i][0])
    i_perpetrator_sex.append([])
    i_perpetrator_sex[i].append(perpetrator_sex_encoded[i][0])
    i_perpetrator_age.append([])
    i_perpetrator_age[i].append(get_binary(perpetrator_age_encoded[i][0] , 7))
    i_perpetrator_race.append([])
    i_perpetrator_race[i].append(get_binary(perpetrator_race_encoded[i][0] , 3))
    i_perpetrator_ethnicity.append([])
    i_perpetrator_ethnicity[i].append(perpetrator_ethnicity_encoded[i][0])
    i_relationship.append([])
    i_relationship[i].append(relationship_encoded[i][0])
    i_weapon.append([])
    i_weapon[i].append(get_binary(weapon_encoded[i][0] , 4))
    i_victim_count.append([])
    i_victim_count[i].append(get_binary(victim_count_encoded[i][0] , 4))
    i_perpetrator_count.append([])
    i_perpetrator_count[i].append(get_binary(perpetrator_count_encoded[i][0] , 4))
    i_record_source.append([])
    i_record_source[i].append(record_source_encoded[i][0])



# Creating final input data by concatinating all the features

input_array = np.concatenate((np.array(i_agency_code).reshape(-1 , 15) , 
                             np.array(i_agency_type).reshape(-1 , 4) ,
                             np.array(i_city).reshape(-1 , 12) ,
                             np.array(i_state).reshape(-1 , 6) ,
                             np.array(i_year).reshape(-1 , 6) ,
                             np.array(i_month).reshape(-1 , 4) ,
                             np.array(i_incident).reshape(-1 , 10) ,
                             np.array(i_crime_type).reshape(-1 , 1) ,
                             np.array(i_victim_sex).reshape(-1 , 1) ,
                             #np.array(i_victim_age).reshape(-1 , 7) ,
                             np.array(i_victim_race).reshape(-1 , 3) ,
                             np.array(i_victim_ethnicity).reshape(-1 , 1) ,
                             np.array(i_perpetrator_sex).reshape(-1 , 1) ,
                             np.array(i_perpetrator_age).reshape(-1 , 7) ,
                             np.array(i_perpetrator_race).reshape(-1 , 3) ,
                             np.array(i_perpetrator_ethnicity).reshape(-1 , 1) ,
                             np.array(i_relationship).reshape(-1 , 1) ,
                             np.array(i_weapon).reshape(-1 , 4) ,
                             np.array(i_victim_count).reshape(-1 , 4) ,
                             np.array(i_perpetrator_count).reshape(-1 , 4) ,
                             np.array(i_record_source).reshape(-1 , 1)) ,
                             axis = 1)



# Creating the Neural Network Model
    
model = Sequential()
model.add(Dense(units = 96 , kernel_initializer = 'glorot_uniform' , activation = 'relu' , input_dim = input_array.shape[1]))
model.add(Dense(units = 81 , kernel_initializer = 'glorot_uniform' , activation = 'relu'))
model.add(Dense(units = 65 , kernel_initializer = 'glorot_uniform' , activation = 'relu'))
model.add(Dense(units = 49 , kernel_initializer = 'glorot_uniform' , activation = 'relu'))
model.add(Dense(units = 33 , kernel_initializer = 'glorot_uniform' , activation = 'relu'))
model.add(Dense(units = 17 , kernel_initializer = 'glorot_uniform' , activation = 'relu'))
model.add(Dense(units = 1 , kernel_initializer = 'glorot_uniform' , activation = 'sigmoid'))
model.compile('adam' , 'binary_crossentropy' , ['accuracy'])

# Fitting data into our model

crime_solver.fit(input_array , y , batch_size = 1000 , epochs = 10)

