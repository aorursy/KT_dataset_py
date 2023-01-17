# do frequent garbage collection via gc.collect() and/or do "del variable"

import gc



# general things used for EDA steps

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import os

from time import time

from time import strftime
from IPython.display import HTML



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

$( document ).ready(code_toggle);

</script>



<font size=3><b>

Note: the code cells for this IPython notebook may be hidden -- <br>

<a href="javascript:code_toggle()"> - Click HERE - Click HERE - Click HERE </a> <br>

to toggle them between visible-hidden.</b></font><br><br>

This is useful because of all the code: on the plus side there are comments,

on the minus side the coding style is verbose-unimaginative ;-)''')
# Running on Kaggle (True), or on my local machine (False)

LOCATION_KAGGLE = True



# Key parameters to set upfront?



# For testing, 

# can fill the Test data with the Training data

TEST_IS_TRAIN = False   # False when submitting!

# and/or reduce the train size

REDUCED_SIZE = False



# Select if an "intersection", the InterCode value, is

# a unique geographic location:

#   City-LatOff-LongOff - False

# or a unique "entry-section":

#   City-LatOff-LongOff-InHeading - True

ENTRYSECTIONS = True



# Comparisons to use for the training NoWait, LoWait, HiWait determination.

# 0: use Total_p80 for all three thresholds

# 1: use Total, TTS, and DTFS, for No, Lo, Hi respectively.

WAIT_CHOICE = 0



# Use ML to assign iWait values

# (if False, use the Known values when Test is train)

ML_WAITS = True

# Override the ML iWait for the Test-data 'northern unknows'

# (-1=use the ML, 0,1,2=override with this value)

northern_iwait = -1  # Do ML



# Can turn on/off EDA output and plots (not needed for ML)

SHOW_EDA = True

# Create and show results from

# an (entry-)intersection dataframe.

# This adds features so should be True.

INTER_DF = True



# version string to include in filenames of plots, etc. (except submission file)

version_str = "v54"

# Create an output directory for this version, if running locally,

# otherwise use the current dir when on Kaggle.

if LOCATION_KAGGLE:

    out_dir = "."

else:

    out_dir = "Out_"+version_str

    try:

        os.mkdir(out_dir)

    except FileExistsError:

        pass

    

# The seed is set once here at beginning of notebook.

RANDOM_SEED = 360

# Uncomment this to get a time-based random value, 0 to 1023

##RANDOM_SEED = int(time()) % 2**10

# in either case initialize the seed

np.random.seed(RANDOM_SEED)
# Collecting some of the 'global' variables here.

# Unless noted, these are set here and used elsewhere below.



# The names of the cities:

cities=['Atlanta','Boston','Chicago','Philadelphia']



# The direction heading names in order

headings=['N','NE','E','SE','S','SW','W','NW']



# Locations of the 'center' of the cities

# The integer LatOff and LongOff (in 10^-4 degree units)

# are created so that (5000,5000) = this center.

# 

# Atlanta: Georgia State Capitol  33.7489, -84.3881

# Boston:  Park St subway         42.3564, -71.0625

# Chicago: Architecture Center    41.8878, -87.6233

# Philly:  City Hall              39.9525, -75.1633

lat_centers =  [ 33.7489,  42.3564,  41.8878,  39.9525]

long_centers = [-84.3881, -71.0625, -87.6233, -75.1633]



# Locations of major airport in each city

#

# Atlanta: H-J AIA          33.6404, -84.4198

# Boston:  Logan            42.3670, -71.0224

# Chicago: O'Hare           41.9786, -87.9047  (there's also Midway airport)

# Phil.:  Phil.Int.         39.8719, -75.2411

lat_airport =  [ 33.6404,  42.3670,  41.9786,  39.8719]

long_airport = [-84.4198, -71.0224, -87.9047, -75.2411]



# The six output column names

out_cols = ['TotalTimeStopped_p20','TotalTimeStopped_p50','TotalTimeStopped_p80',

           'DistanceToFirstStop_p20','DistanceToFirstStop_p50','DistanceToFirstStop_p80']

# Where are the data files

# Data dir

if LOCATION_KAGGLE:

    dat_dir ='../input/bigquery-geotab-intersection-congestion/'

else:

    dat_dir ="../input/"

# CSV files

train_csv = "train.csv"

test_csv = "test.csv"



# show the files and dirs

##print(os.listdir(dat_dir))
# Read the files and do All the basic feature processing

# time it

t_preproc = time()



# Read in the train and test data



# = = = = =

# Train

df_train = pd.read_csv(dat_dir+train_csv)

if REDUCED_SIZE:

    divide_by = 3

    rand_indices = np.random.choice(df_train.index,

                    size=int(len(df_train)/divide_by), replace=False)

    # reduce the size and re-index

    df_train = df_train.loc[rand_indices].reset_index().drop('index',axis=1)



# = = = = =

# Test

# For testing, can fill the Test data with the Training data

if TEST_IS_TRAIN:

    df_test = df_train.copy()

else:

    # Read in the TEST data

    df_test = pd.read_csv(dat_dir+test_csv)

    if REDUCED_SIZE:

        rand_indices = np.random.choice(df_test.index,

                    size=int(len(df_test)/divide_by), replace=False)

        df_test = df_test.loc[rand_indices].reset_index().drop('index',axis=1)

        

if REDUCED_SIZE:

    del rand_indices

    gc.collect()



print("{:.2f} seconds -- read in data files".format(time() - t_preproc))

 



# Create some other columns, etc. (in both test and train):



# - - - - -

# Add the output columns to test, to be filled in later

# Set the columns to 0

for new_col in out_cols:

    # Set all to 0:

    df_test[new_col] = 0



# - - - - -

# The months:

if True:

    # Because there are so few Jan(1) and May(5) month values, and none in Feb(2)-April(4):

    # Set the Jan ones to Dec and the May ones to June to have only 7 month values: 6 through 12.

    select = df_train['Month'] == 1

    df_train.loc[select,'Month'] = 12

    select = df_train['Month'] == 5

    df_train.loc[select,'Month'] = 6

    #

    select = df_test['Month'] == 1

    df_test.loc[select,'Month'] = 12

    select = df_test['Month'] == 5

    df_test.loc[select,'Month'] = 6

    

    # Further combine the months into just 3 groups:

    #  1, 2, 3  =  (5+)6-8, 9-10, 11-12(+1)

    select = df_train['Month'] >= 11

    df_train.loc[select,'Month'] = 3

    select = df_train['Month'] >= 9

    df_train.loc[select,'Month'] = 2

    select = df_train['Month'] >= 6

    df_train.loc[select,'Month'] = 1

    #

    select = df_test['Month'] >= 11

    df_test.loc[select,'Month'] = 3

    select = df_test['Month'] >= 9

    df_test.loc[select,'Month'] = 2

    select = df_test['Month'] >= 6

    df_test.loc[select,'Month'] = 1



# - - - - -

# Add an iCity column = 0,1,2,3

df_train['iCity'] = -1

df_test['iCity'] = -1

for icity, this_city in enumerate(cities):

    df_train.loc[df_train['City'] == this_city, 'iCity'] = icity

    df_test.loc[df_test['City'] == this_city, 'iCity'] = icity

#

# One-hot encoding of City

# This may be better than iCity for some ML methods ?

# from https://www.kaggle.com/dcaichara/feature-engineering-and-lightgbm

if False:

    df_train = pd.concat([df_train,pd.get_dummies(df_train["City"],

                dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)

    df_test = pd.concat([df_test,pd.get_dummies(df_test["City"],

                dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)



# - - - - -

# Add an HrWk column made by combining the Hour and Weekend: 

#   HrWk = 0 - 23 weekdays, 24 - 47 weekends

df_train['HrWk'] = df_train['Hour'] + 24*df_train['Weekend']

df_test['HrWk'] = df_test['Hour'] + 24*df_test['Weekend']



# - - - - -

# Add InHeading and ExHeading numeric columns = 0--7 (N to NW)

df_train['InHeading'] = 0

df_test['InHeading'] = 0

df_train['ExHeading'] = 0

df_test['ExHeading'] = 0



# Assign the heading numbers

for ihead, this_head in enumerate(headings):

    df_train.loc[df_train['EntryHeading'] == this_head, 'InHeading'] = ihead

    df_train.loc[df_train['ExitHeading'] == this_head, 'ExHeading'] = ihead

    df_test.loc[df_test['EntryHeading'] == this_head, 'InHeading'] = ihead

    df_test.loc[df_test['ExitHeading'] == this_head, 'ExHeading'] = ihead



# - - - - -

# Calculate the Turn amount

# 0 = straight; 1,2,3 = right; -1,-2,-3,-4 = left to U-turn

df_train['Turn'] = df_train['ExHeading'] - df_train['InHeading']

df_test['Turn'] = df_test['ExHeading'] - df_test['InHeading']

# Keep it between -4 to 3:

# If Turn > 3 then Turn = Turn - 8

select = df_train['Turn'] > 3

df_train.loc[select,'Turn'] = df_train.loc[select,'Turn'] - 8

select = df_test['Turn'] > 3

df_test.loc[select,'Turn'] = df_test.loc[select,'Turn'] - 8

# If Turn < -4 then Turn = Turn + 8

select = df_train['Turn'] < -4

df_train.loc[select,'Turn'] = df_train.loc[select,'Turn'] + 8

select = df_test['Turn'] < -4

df_test.loc[select,'Turn'] = df_test.loc[select,'Turn'] + 8



# - - - - -

# Make a coarser turn value: Left, Straight, or Right

df_train['TurnLSR'] = 0

df_test['TurnLSR'] = 0

select = df_train['Turn'] > 0

df_train.loc[select,'TurnLSR'] = 1

select = df_test['Turn'] > 0

df_test.loc[select,'TurnLSR'] = 1

select = df_train['Turn'] < 0

df_train.loc[select,'TurnLSR'] = -1

select = df_test['Turn'] < 0

df_test.loc[select,'TurnLSR'] = -1



# - - - - -

# Create LatOff and LongOff values for the intersections,

# these are 4 digit numbers in units of 10^-4 degrees.

# They are set to give (5000, 5000) at each city center.

df_train['LatOff'] = 0.0

df_train['LongOff'] = 0.0

df_test['LatOff'] = 0.0

df_test['LongOff'] = 0.0

#

for icity in range(4):

    select = df_train['iCity'] == icity

    df_train.loc[select,'LatOff'] = df_train.loc[select,'Latitude'] - lat_centers[icity]

    df_train.loc[select,'LongOff'] = df_train.loc[select,'Longitude'] - long_centers[icity]

df_train['LatOff'] = (5000 + 10000*df_train['LatOff']).astype(int)

df_train['LongOff'] = (5000 + 10000*df_train['LongOff']).astype(int)

#

for icity in range(4):

    select = df_test['iCity'] == icity

    df_test.loc[select,'LatOff'] = df_test.loc[select,'Latitude'] - lat_centers[icity]

    df_test.loc[select,'LongOff'] = df_test.loc[select,'Longitude'] - long_centers[icity]

df_test['LatOff'] = (5000 + 10000*df_test['LatOff']).astype(int)

df_test['LongOff'] = (5000 + 10000*df_test['LongOff']).astype(int)



# - - - - -

# Convert the Airport Lat,Long into offset values in the center=(5000,5000) system:

# (this is in features in case "distance to airpor" would be useful...

longoff_air = np.zeros(4)

latoff_air = np.zeros(4)

for icity in range(4):

    longoff_air[icity] = np.int(5000 + 10000*(long_airport[icity] - long_centers[icity]))

    latoff_air[icity] = np.int(5000 + 10000*(lat_airport[icity] - lat_centers[icity]))



# - - - - -

# Create unique integer intersection codes

if ENTRYSECTIONS:

    # Create unique integer intersection codes --> "Entry-section" codes:

    # Combine iCity, LatOff and LongOff *** and InHeading *** into one integer

    df_train['InterCode'] = (1000000000*(df_train['iCity'] + 1) + 100000*df_train['LatOff'] + 

                         10*df_train['LongOff'] + df_train['InHeading'])

    df_test['InterCode'] = (1000000000*(df_test['iCity'] + 1) + 100000*df_test['LatOff'] + 

                        10*df_test['LongOff'] + df_test['InHeading'])

else:

    # just intersections:

    # Combine iCity, LatOff and LongOff into one integer

    df_train['InterCode'] = (100000000*(df_train['iCity'] + 1) + 10000*df_train['LatOff'] + 

                         df_train['LongOff'])

    df_test['InterCode'] = (100000000*(df_test['iCity'] + 1) + 10000*df_test['LatOff'] + 

                        df_test['LongOff'])



# - - - - -

# Create distance from city center values,

# in 10^-4 degree units from LatOff LongOff:

df_train['DistToCenter'] = df_train.apply(lambda row: np.sqrt((row.LatOff - 5000) ** 2 +

                                                    (row.LongOff - 5000) ** 2) , axis=1)

df_test['DistToCenter'] = df_test.apply(lambda row: np.sqrt((row.LatOff - 5000) ** 2 +

                                                    (row.LongOff - 5000) ** 2) , axis=1)



# - - - - -

# Create the dot product between intersection-to-Center and InHeading unit vectors.

# So that: 100 means InHeading is toward Center, -100 means InHeading opposite Center.

# Add Center*InHead, -100 to 100 (will become an integer):

df_train['Center*InHead'] = 100.0*( (5000 - df_train['LatOff'])*np.cos(np.pi*df_train['InHeading']/4.0) + 

                         (5000 - df_train['LongOff'])*np.sin(np.pi*df_train['InHeading']/4.0)

                        ) / df_train['DistToCenter']

df_test['Center*InHead'] = 100.0*( (5000 - df_test['LatOff'])*np.cos(np.pi*df_test['InHeading']/4.0) + 

                         (5000 - df_test['LongOff'])*np.sin(np.pi*df_test['InHeading']/4.0)

                        ) / df_test['DistToCenter']



# - - - - -

# Create Total_p80 as a single measure of delay, Total_p80 =  TTS_p80 + DTFS_p80

# Covert to integers for cleaner histograming.

df_train['Total_p80'] = df_train['DistanceToFirstStop_p80'] + df_train['TotalTimeStopped_p80']

df_train['Total_p80'] = df_train['Total_p80'].astype(int)

#

# Add these shorter names to Train:

df_train['DTFS_p80'] = df_train['DistanceToFirstStop_p80'].astype(int)

df_train['TTS_p80'] = df_train['TotalTimeStopped_p80'].astype(int)

#

# Make the columns in Test too:

df_test['Total_p80'] = -1

df_test['DTFS_p80'] = -1

df_test['TTS_p80'] = -1



print("\n{:.2f} seconds -- added basic new feature columns".format(time() - t_preproc))

# Look for NAs - only have them in some street names



##df_train.isnull().sum()

# Non-zero ones:

# EntryStreetName            8189

# ExitStreetName             5534



##df_test.isnull().sum()

# Non-zero ones:

# EntryStreetName            19157

# ExitStreetName             16340

if SHOW_EDA:

    # Alert if Test is Train:

    if TEST_IS_TRAIN:

        print("\n"+20*" *"+"\n   TEST is Train !!!\n"+20*" *"+"\n")



    # Number of locations, i.e., Lattitude and Longitude values:

    print('Number of Latitudes in Train Set: ', len(df_train.Latitude.unique()))

    print('Number of Longitudes in Train Set: ', len(df_train.Longitude.unique()))

    print('Number of Latitudes in TEST Set: ', len(df_test.Latitude.unique()))

    print('Number of Longitudes in TEST Set: ', len(df_test.Longitude.unique()))

    print("")

    # Numbers of entry, exit streets in the data

    # *** There are more/different intersecions in TEST than in Train ***

    #From https://www.kaggle.com/harisyammnv/initial-eda-with-maps)

    print('Number of Entry Street Names in Train Set: ', len(df_train.EntryStreetName.unique()))

    print('Number of Exit Street Names in Train Set: ', len(df_train.ExitStreetName.unique()))

    print('Number of Entry Street Names in TEST Set: ', len(df_test.EntryStreetName.unique()))

    print('Number of Exit Street Names in TEST Set: ', len(df_test.ExitStreetName.unique()))

    print("")
# Form this to count how many unique InterCode - Month - HrWk combinations show up

# (InterCode used here was for Intersections, not entry-sections.)

#  Train:  524,711 from 4793 InterCodes,

# This is about 1/3 of full coverage (full = 4793 * 7 * 48 = 1,610,448.)

#

##df_train['InterMonHrWk'] = (df_train['InterCode'].astype(str) + "-" + df_train['Month'].astype(str) + 

##                            "-" + df_train['HrWk'].astype(str))



# Include EntryHeading ExitHeading too - expect these to be unique

# Mostly, but about 1.5% of entries are duplicates with different street names (same headings).

##df_train['InterMonHrWk'] = (df_train['InterCode'].astype(str) + "-" + df_train['Month'].astype(str) + 

##                            "-" + df_train['HrWk'].astype(str) + "-" +

##                            df_train['EntryHeading'] + "-" + df_train['ExitHeading'])
# Look at some strange ones

#

# Inter-Mon-HrWk:

#  Cambridge Street - Monsignor O'Brien Highway - East Street

##df_train[df_train['InterMonHrWk']=='237063240-7-7']

#  Main Street - Vassar Street - Galileo Galilei Way

##df_train[df_train['InterMonHrWk']=='236263100-8-9']



# Inter-Mon-HrWk-Entry-Exit with 6 times:

#   Albany Street - Frontage Road Southbound - NaN

##df_train[df_train['InterMonHrWk']=='234033361-12-14-SW-SW']
# Get all the unique street names

street_names = []



# In training:

entry_names = df_train['EntryStreetName'].unique()

exit_names = df_train['ExitStreetName'].unique()

for name in entry_names:

    street_names.append(name)

for name in exit_names:

    street_names.append(name)

# and the test ones too:

entry_names = df_test['EntryStreetName'].unique()

exit_names = df_test['ExitStreetName'].unique()    

for name in entry_names:

    street_names.append(name)

for name in exit_names:

    street_names.append(name)

    

unique_streets = pd.Series(street_names).unique()



# Alert if Test is Train:

if TEST_IS_TRAIN:

    print("\n"+20*" *"+"\n       TEST is Train !!!\n"+20*" *"+"\n")

    

print("Number of street names in Train & Test: {}".format(len(unique_streets)))
# Dictionary of recognized types of "streets".

# Use to encode ExitType and EntryType

# "nan" streets will be code 16 (was 0)

# ones not in this list will be code 25  (about 27 of them)

street_type_dict = {'Street':1, 'St':1, 'Boulevard':2, 'Bld':2, 'Avenue':3, 'Ave':3,

                     'Road':4, 'Rd':4, 'Lane':5, 'Drive':6,

                     'Parkway':7, 'Pkwy':7, 'Place':8, 'Way':9, 'Highway':10, 'Circle':11, 'Terrace':12,

                     'Square':13, 'Court':14, 'Connector':15, 'Bridge':16, 'Overpass':17, 'Tunnel':18,

                     'Mall':19, 'Wharf':20, 'Expressway':21,

                     # ones added for Test but not in Train:

                     'Pike':22}

# Indices for EntryType sorted on Total_p80 averages (using all training data):

# [11 12 15 13  2  1 25 17 18 21  7  5  4 19 10  9  0  3 16 14  6  8 20]

# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22

# Relabel the values in the dictionary using this, 

# e.g., 'Circle':11 becomes 'Circle':0;  'Terrace':12 becomes 'Terrace':1;  etc.

street_type_dict = {'Street':5, 'St':5, 'Boulevard':4, 'Bld':4, 'Avenue':17, 'Ave':17,

                     'Road':12, 'Rd':12, 'Lane':11, 'Drive':20,

                     'Parkway':10, 'Pkwy':10, 'Place':21, 'Way':15, 'Highway':14, 'Circle':0, 'Terrace':1,

                     'Square':3, 'Court':19, 'Connector':2, 'Bridge':18, 'Overpass':7, 'Tunnel':8,

                     'Mall':13, 'Wharf':22, 'Expressway':9,

                     # ones added for Test but not in Train:

                     'Pike':14}  # put same as Highway



if SHOW_EDA:

    print("   These words in street names are used to form groups:")

    for stkey in street_type_dict.keys():

        print(stkey)



# Show the streets that are being left out, i.e. will be code 6 (was 25):

if SHOW_EDA:

    print("\n   These street names are 'unique' and put in one category:")

    for street in unique_streets:

        if pd.isna(street):

            pass

        else:

            # is it in the dictionary?

            in_dict = False

            for street_type in street_type_dict:

                if street_type in street:

                    in_dict = True

            if in_dict == False:

                print(street)
# Set an EntryType feature using the dictionary,

# adapted the code in https://www.kaggle.com/dcaichara/feature-engineering-and-lightgbm

def encode(x):

    global street_type_dict

    if pd.isna(x):

        return 16   # was 0

    for street in street_type_dict.keys():

        if street in x:

            return street_type_dict[street]

    # otherwise

    return 6  # was 25



df_train['EntryType'] = df_train['EntryStreetName'].apply(encode)

df_train['ExitType'] = df_train['ExitStreetName'].apply(encode)

df_test['EntryType'] = df_test['EntryStreetName'].apply(encode)

df_test['ExitType'] = df_test['ExitStreetName'].apply(encode)
t_lohi = time()

print("             ...now adding LoWait, HiWait, iWait...")



# Categorize each (Entry-)section by some average p80 value(s),

# setting flags for: NoWait, LoWait, and HiWait.



# - - - - -

# Set Thresholds and new Columns

#

# Threshold Choices:

#   0: use Total_p80 value for all three of No, Lo, Hi

#   1: use Total, TTS, and DTFS for No, Lo, Hi, respectively.

if WAIT_CHOICE == 0:

    if ENTRYSECTIONS:

        # These are estimates

        NoWait_Threshold = 20.0

        LoWait_Thresh = [120.0, 120.0, 120.0, 120.0]

        HiWait_Thresh = [500.0,300.0,500.0,500.0]

    else:  # Intersections

        # * These have been optimized * (v20)

        NoWait_Threshold = 20.0

        LoWait_Thresh = [120.0, 120.0, 120.0, 120.0]

        HiWait_Thresh = [280.0, 280.0, 280.0, 280.0]

    print("\n    NoWait_Threshold = {}  on Total_p80".format(NoWait_Threshold))

    print("    LoWait_Thresh.s = {}, {}, {}, {}  on Total_p80".format(

        LoWait_Thresh[0], LoWait_Thresh[1], LoWait_Thresh[2], LoWait_Thresh[3] ))

    print("    HiWait_Thresh.s = {}, {}, {}, {}  on Total_p80 \n".format(

        HiWait_Thresh[0], HiWait_Thresh[1], HiWait_Thresh[2], HiWait_Thresh[3] ))  

else:  # WAIT_CHOICE = 1

    if ENTRYSECTIONS:

        # * These have been optimized * (v32)

        NoWait_Threshold = 20.0

        LoWait_Thresh = [18.0,18.0,18.0,18.0]

        HiWait_Thresh = [400.0,210.0,400.0,400.0]

    else:  # Intersections

        # * These have been optimized * (v21)

        NoWait_Threshold = 20.0

        LoWait_Thresh = [18.0,18.0,18.0,18.0]

        HiWait_Thresh = [210.0,210.0,210.0,210.0]

    print("\n    NoWait_Threshold = {}  on Total_p80".format(NoWait_Threshold))

    print("    LoWait_Thresh.s = {}, {}, {}, {}  on TTS_p80".format(

        LoWait_Thresh[0], LoWait_Thresh[1], LoWait_Thresh[2], LoWait_Thresh[3] ))

    print("    HiWait_Thresh.s = {}, {}, {}, {}  on DTFS_p80 \n".format(

        HiWait_Thresh[0], HiWait_Thresh[1], HiWait_Thresh[2], HiWait_Thresh[3] ))   



# Flag (entry-)intersections that have No, Low or High average observed wait times;

# medium is if neither low nor high. The Low includes the No:

# the No values are just for information, they contribute very little to the RMSE.

df_train['NoWait'] = 0

df_train['LoWait'] = 0

df_train['HiWait'] = 0

# signal these are not determined with -1

df_test['NoWait'] = -1

df_test['LoWait'] = -1

df_test['HiWait'] = -1



# - - - - -

# For each (entry-)intersection assign the No, Lo, Hi status

# using its calculated average p80 value(s) in the training dataset

feat_name = 'InterCode'

val_counts = df_train[feat_name].value_counts()

feat_values = np.sort(val_counts.index)

for this_val in feat_values:

    # find this feat_value in the df

    select = df_train[feat_name] == this_val

    # and get a dataframe of just those

    df_feat = df_train[select].copy()

    total_mean = df_feat['Total_p80'].mean()

    iCity = (df_feat['iCity'].values)[0]

    if WAIT_CHOICE == 0:

        # NoWait

        if total_mean < NoWait_Threshold:

            df_train.loc[select,'NoWait'] = 1

        # LoWait (includes NoWait)

        if total_mean < LoWait_Thresh[iCity]:

            df_train.loc[select,'LoWait'] = 1

        # HiWait

        if total_mean > HiWait_Thresh[iCity]:

            df_train.loc[select,'HiWait'] = 1  

    else:

        # Here wait is assigned depending on

        # Total_p80 and TTS_p80 and DTFS_p80

        # get means of these other two:

        tts_mean = df_feat['TTS_p80'].mean()

        dtfs_mean = df_feat['DTFS_p80'].mean()

        # NoWait

        if total_mean < NoWait_Threshold:

            # Very low value for the Total_p80:

            df_train.loc[select,'NoWait'] = 1

        # LoWait (includes NoWait)

        if tts_mean < LoWait_Thresh[iCity]:

            # Low TTS

            df_train.loc[select,'LoWait'] = 1

        # HiWait

        if dtfs_mean > HiWait_Thresh[iCity]:

            # High DTFS

            df_train.loc[select,'HiWait'] = 1



# - - - - -

# Include an iWait with values: 0-low, 1-medium, 2-high:

# Define the wait selections

select_no = df_train['NoWait'] == 1

select_lo = df_train['LoWait'] == 1

select_hi = df_train['HiWait'] == 1

# and a medium selection

select_me = (df_train['LoWait'] == 0) & (df_train['HiWait'] == 0)



df_train['iWait'] = 1

df_train.loc[select_lo, 'iWait'] = 0

df_train.loc[select_hi, 'iWait'] = 2

# and set up iWait for Test too, set to -1 for undetermined

df_test['iWait'] = -1





print("\n{:.2f} seconds to add No,Lo,Hi and iWait columns".format(time() - t_lohi))
# Print out information about the Waits

print("\nTrain NoWait fraction = {:.2f}%".format(100*df_train['NoWait'].mean()))

print("Train LoWait(w/NoWait) fraction = {:.2f}%".format(100*df_train['LoWait'].mean()))

print("Train HiWait fraction = {:.2f}%\n".format(100*df_train['HiWait'].mean()))



# Get the averages for the NoWait, LoWait, MeWait and HiWait rows:

nowait_aves = [0,0,0,0,0,0]

lowait_aves = [0,0,0,0,0,0]

mewait_aves = [0,0,0,0,0,0]

hiwait_aves = [0,0,0,0,0,0]

# The NoWait averages

for icol, this_col in enumerate(out_cols):

    nowait_aves[icol] = df_train.loc[select_no,this_col].mean()

# The LoWait averages (including NoWait)

for icol, this_col in enumerate(out_cols):

    lowait_aves[icol] = df_train.loc[select_lo,this_col].mean()

# The medium-wait averages

for icol, this_col in enumerate(out_cols):

    mewait_aves[icol] = df_train.loc[select_me,this_col].mean()

# The HiWait averages:

for icol, this_col in enumerate(out_cols):

    hiwait_aves[icol] = df_train.loc[select_hi,this_col].mean()



print("Averages of the 6 target values for the NoWait training data: \n" +

      "    {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(

    nowait_aves[0],nowait_aves[1],nowait_aves[2],nowait_aves[3],nowait_aves[4],nowait_aves[5]))

print("Averages of the 6 target values for the LoWait training data: \n" +

      "    {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(

    lowait_aves[0],lowait_aves[1],lowait_aves[2],lowait_aves[3],lowait_aves[4],lowait_aves[5]))

print("Averages of the 6 target values for the medium-wait training data: \n" +

      "    {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(

    mewait_aves[0],mewait_aves[1],mewait_aves[2],mewait_aves[3],mewait_aves[4],mewait_aves[5]))

print("Averages of the 6 target values for the HiWait training data: \n" +

      "    {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(

    hiwait_aves[0],hiwait_aves[1],hiwait_aves[2],hiwait_aves[3],hiwait_aves[4],hiwait_aves[5]))

# Try to free up memory I don't need...

del df_feat, val_counts, feat_values, select, select_no, select_lo, select_me, select_hi

# collect any garbage?

gc_dummy = gc.collect()
if SHOW_EDA:

    # Alert if Test is Train:

    if TEST_IS_TRAIN:

        print("\n"+20*" *"+"\n   TEST is Train !!!\n"+20*" *"+"\n")



    print("\n   Number of Unique (Entry-)Intersection Codes\n")

    for icity, this_city in enumerate(cities):

        # Numer of unique intersections in each city

        print(this_city+': Train Set: ',

              len(df_train[df_train['iCity'] == icity].InterCode.unique()),

              "  No waits: ", len(df_train[(df_train['iCity'] == icity) &

                           (df_train['NoWait'] == 1)].InterCode.unique()),

              ",  Low(&No) waits: ", len(df_train[(df_train['iCity'] == icity) &

                           (df_train['LoWait'] == 1)].InterCode.unique()),

              ",  med waits: ", len(df_train[(df_train['iCity'] == icity) &

                           (df_train['iWait'] == 1)].InterCode.unique()),

              ",  High waits: ", len(df_train[(df_train['iCity'] == icity) &

                           (df_train['HiWait'] == 1)].InterCode.unique()))

        print(this_city+':  TEST Set: ',

              len(df_test[df_test['iCity'] == icity].InterCode.unique()), '\n')

    # show the totals too

    print("      All Cities Train :  {}".format(len(df_train.InterCode.unique())))

    print("      All Cities  TEST :  {}".format(len(df_test.InterCode.unique())))
if SHOW_EDA:

    # Show all columns

    print("All columns:\n")

    print(df_train.columns)
# Down-select to just columns I may use,

# add more if/as desired.



using_cols = (['RowId', 'Latitude', 'Longitude', 'Month', 'iCity', 'HrWk', 'TurnLSR', 'Turn',

             'LatOff', 'LongOff', 'InterCode',

             'ExHeading', 'InHeading',

             'DistToCenter', 'Center*InHead',

             'Total_p80', 'DTFS_p80', 'TTS_p80',

             ##'Atlanta', 'Boston', 'Chicago', 'Philadelphia',

             'NoWait', 'LoWait', 'HiWait', 'iWait', 

             'EntryType', 'ExitType'] + 

              out_cols )   # the targets are included



# Make sure that some of the columns are integers

integer_cols = ['RowId', 'Month', 'iCity', 'HrWk', 'TurnLSR', 'Turn',

             'LatOff', 'LongOff', 'InterCode',

             'ExHeading', 'InHeading',

             ##'DistToCenter', 'Center*InHead',   # leave these as floats

             ##'Atlanta', 'Boston', 'Chicago', 'Philadelphia',

             'NoWait', 'LoWait', 'HiWait', 'iWait',

             'EntryType', 'ExitType']



df_train = df_train[using_cols].copy()

df_test = df_test[using_cols].copy()

for col in integer_cols:

    df_train[col] = df_train[col].astype(int)

    df_test[col] = df_test[col].astype(int)

    
if INTER_DF:

    # Assemble a dataframe of the intersections and their properties.

    # Some new features are created as well.

    t_df_inter = time()

    

    # Train

    val_counts = df_train['InterCode'].value_counts()



    train_inter = pd.DataFrame(val_counts)



    train_inter = train_inter.reset_index()

    train_inter.columns = ['InterCode','num_train']

    train_inter = train_inter.sort_values(by='InterCode').reset_index().drop('index',axis=1)



    # TEST

    val_counts = df_test['InterCode'].value_counts()



    test_inter = pd.DataFrame(val_counts)



    test_inter = test_inter.reset_index()

    test_inter.columns = ['InterCode','num_test']

    test_inter = test_inter.sort_values(by='InterCode').reset_index().drop('index',axis=1)

if INTER_DF:

    # Combine the two

    inter_merge = pd.merge(train_inter, test_inter, on='InterCode', how='outer')



    # Fill NaNs and make all entries integers

    inter_merge = inter_merge.fillna(value={'num_train':-1.0, 'num_test':-1.0})

    inter_merge = inter_merge.astype(int)



    if SHOW_EDA:

        print(inter_merge.head(5))
if INTER_DF:

    # Add other columns to the merged df

    # - ones directly from Train/Test

    # - means of p80s: Total, DTFS, TTS

    # - number of unique ExHeadings for the entry-section

    t_df_inter = time()



    # Flag if it is in Train and/or Test (can be both)

    inter_merge['Train'] = (inter_merge['num_train'] > 0).astype(int)

    inter_merge['Test'] = (inter_merge['num_test'] > 0).astype(int)



    # columns to add that depend only on InterCode

    add_cols = ['iCity','LatOff','LongOff','DistToCenter',

            'iWait','EntryType','InHeading','Center*InHead']

    # averages of targets over entries with the same InterCode

    p80_cols = ['Total_p80', 'DTFS_p80', 'TTS_p80']

    

    

    # setup the new columns

    for icol in add_cols:

        inter_merge[icol] = -1

    for icol in p80_cols:

        inter_merge["Ave"+icol] = -1

    inter_merge['UniqueExits'] = -1

    

    # go through the InterCodes and fill columns

    for indx in (inter_merge.index):

        intercode = inter_merge.loc[indx,'InterCode']

        # In training?

        if inter_merge.loc[indx,'Train'] > 0:

            df_select = df_train[df_train['InterCode'] == intercode].copy()

            # target values only available in train:

            for icol in p80_cols:

                inter_merge.loc[indx,"Ave"+icol] = df_select[icol].mean() 

        else:

            # get info from test

            df_select = df_test[df_test['InterCode'] == intercode].copy()

        # these are the same for train/test once df_select is set

        for icol in add_cols:

            inter_merge.loc[indx,icol] = df_select[icol].mean()

        # number of unique exit headings

        inter_merge.loc[indx,'UniqueExits'] = len(df_select['ExHeading'].unique())



    # Make values integers (doing this for histogramsing?)

    inter_merge = inter_merge.fillna(value=-1)

    inter_merge = inter_merge.astype(int)



    print("\n {:.2f} seconds to fill basic (entry-)intersection dataframe.\n".format(time() - t_df_inter))

if INTER_DF:

    # Add column(s) calculated just from the inter_merge data,

    # For each entry-section calculate:

    # - the spatial density of that City around the entry-section (number in small, local region)

    # - the distance to the closest other entry-section that is in the -InHead direction

    #   (When ENTRYSECTIONS=False the InHeading is an average so this may not mean much?)

    t_intercols = time()

    

    # Local density, in square region (2*radius x 2*radius)

    radius = 100

    # Calculate the values into a list and then load results into the df

    local_dens = []

    inter_merge['LocalDensity'] = -1

    

    # Length of the entry-section

    entry_lens = []

    inter_merge['EntryLength'] = -1  

    

    # Go through the entry-sections

    for iinter in inter_merge.index:

        this_lat = inter_merge.loc[iinter,'LatOff']

        this_long = inter_merge.loc[iinter,'LongOff']

        this_city = inter_merge.loc[iinter,'iCity']

        # select the nearby ones from the same city

        select = ( (inter_merge.loc[iinter,'iCity'] == this_city) &

                                (inter_merge['LatOff'] < (this_lat + radius)) &

                                (inter_merge['LatOff'] > (this_lat - radius)) &

                                (inter_merge['LongOff'] < (this_long + radius)) &

                                (inter_merge['LongOff'] > (this_long - radius)) )

        # LocalDensity is the number in the selection

        local_dens.append(sum(select))

        #

        # Find closest entry-section in the -InHead direction

        theta_inhead = (np.pi*inter_merge.loc[iinter,'InHeading'])/4.0

        delta_lat = inter_merge.loc[select,'LatOff'] - this_lat

        delta_long = inter_merge.loc[select,'LongOff'] - this_long

        delta_dot_inhead = (np.sin(theta_inhead) * delta_long +

                    np.cos(theta_inhead) * delta_lat)/np.sqrt(delta_lat**2 + delta_long**2)

        # want delta_dot_inhead to be negative, say < -0.5

        # down-select to just those:

        select = (select & (delta_dot_inhead < -0.85))

        # Now, find the closest, non-zero, entry-section among these:

        d_sqrs = np.sqrt(np.sort((inter_merge.loc[select,'LatOff'] - this_lat)**2 +

                (inter_merge.loc[select,'LongOff'] - this_long)**2))

        this_len = 1.4*radius

        for this_d in d_sqrs:

            if this_d > 0:

                this_len = this_d

                break

        entry_lens.append(this_len)

        

    # Put the values in the dataframe           leave them as floats

    inter_merge['LocalDensity'] = local_dens

    inter_merge['EntryLength'] = entry_lens

    

    print("\n {:.2f} seconds to add LocalDensity and EntryLength to (entry-)intersection dataframe.\n".format(time() - t_intercols))

if INTER_DF:

    del val_counts, train_inter, test_inter, df_select

    gc_dummy = gc.collect()
if INTER_DF and SHOW_EDA:

    # show some of the dataframe:

    print(inter_merge.head(5))

    print(inter_merge.tail(5))

    

    # Note if ENTRYSECTIONS=False:

    if ENTRYSECTIONS == False:

        print("\nNOTE: These rows are  *** Intersections ***\n" +

              "      so the following are averages over all the Entries of each intersection:\n" +

             "          EntryType, InHeading, Center*InHead")
if INTER_DF and SHOW_EDA:

    # Scatter plot between these two new entry-section features:

    inter_merge.plot.scatter('LocalDensity','EntryLength',figsize=(9,6),alpha=0.5)



    plt.savefig(out_dir+"/"+"LocalDensity_EntryLength"+"_scatter_"+version_str+".png")

    plt.show()
if INTER_DF and SHOW_EDA:

    # Look at the histograms of some value



    feat_to_hist = 'EntryLength'

    ##feat_to_hist = 'LocalDensity'

    ##feat_to_hist = 'AveDTFS_p80'

    ##feat_to_hist = 'AveTTS_p80'

    ##feat_to_hist = 'DistToCenter'

    ##feat_to_hist = 'InHeading'

    ##feat_to_hist = 'iWait'

    ##feat_to_hist = 'num_test'

    ##feat_to_hist = 'EntryType'

    

    ##by_feat = 'iCity'

    by_feat = 'iWait'

    ##by_feat = 'InHeading'





    print("\n Histograms of  log10( "+feat_to_hist+" )  for different  "+by_feat+"  values:")

    if by_feat == 'iWait':

        print(40*" "+"(iWait = -1 for Test-only entry-sections.)")

    # Calculate the value to histogram/plot - usually log scale

    inter_merge['plot_this'] = np.log10(2.0+inter_merge[feat_to_hist])

    inter_merge.hist('plot_this',bins=50,by=by_feat,sharex=True,figsize=(12,8))



    plt.savefig(out_dir+"/"+feat_to_hist+"_hists_"+version_str+".png")

    plt.show()



    # Drop the plotting column

    inter_merge = inter_merge.drop('plot_this', axis=1)
# Plot Intersections

if INTER_DF and SHOW_EDA:

    # zoom in on city center

    zoom_inter = False

    for iCity in range(4):

   

        # Discrete colors for iWait

        if True:

            print("\n"+8*" "+"Showing iWait values color-coded:  Yellow(0) - Blue(1) - Red(2) \n" +

                 "\n"+13*" "+"Test-only intersections indicated with a '+'")

        

            ax = inter_merge[(inter_merge['iCity'] == iCity) & (inter_merge['iWait'] == 0) &

               (inter_merge['Train'] == 1)].plot.scatter("LongOff","LatOff",

                        figsize=(10,8),c='yellow',alpha=0.7,s=10)

            inter_merge[(inter_merge['iCity'] == iCity) & (inter_merge['iWait'] == 1) &

               (inter_merge['Train'] == 1)].plot.scatter("LongOff","LatOff",

                        figsize=(10,8),c='blue',alpha=0.7,s=5, ax=ax)

            inter_merge[(inter_merge['iCity'] == iCity) & (inter_merge['iWait'] == 2) &

               (inter_merge['Train'] == 1)].plot.scatter("LongOff","LatOff",

                        figsize=(10,8),c='red',alpha=0.7,s=16, ax=ax)



        # Points colored by DTFS_p80

        # colormaps: viridis_r   gnuplot_r    plasma_r

        if False:

            inter_merge[(inter_merge['iCity'] == iCity) &

                   (inter_merge['Train'] == 1)].plot.scatter("LongOff","LatOff",

                        figsize=(10,8),c='DTFS_p80',alpha=0.7,

                        s=4+4*(inter_merge['iWait'])**2,colormap="viridis_r",colorbar=True)

 



        # Overplot the locations of the Test-only intersections (unknown iWait)

        test_only = ((inter_merge['iCity'] == iCity) & (inter_merge['Train']== 0) &

                            (inter_merge['Test']== 1))

        test_lats = inter_merge[test_only].LatOff

        test_longs = inter_merge[test_only].LongOff

        plt.plot(test_longs,test_lats,color='black',marker='+',linestyle='',markersize=5)

    

        # and the locations of the city center and airport

        plt.plot([5000],[5000],color='lime',marker='*',linestyle='',markersize=12)

        plt.plot([longoff_air[iCity]],[latoff_air[iCity]],color='lime',marker='*',linestyle='',markersize=12)

    

        if zoom_inter:

            plt.xlim(4500,5500)

            plt.ylim(4500,5500)

        

        plt.title(cities[iCity]+" ")

        if zoom_inter:

            plt.savefig(out_dir+"/"+cities[iCity]+"_congestion_zoom_"+version_str+".png")

        else:

            plt.savefig(out_dir+"/"+cities[iCity]+"_congestion_map_"+version_str+".png")

        plt.show()
if INTER_DF:

    # Finally, transfer new values in the inter_merge df to the Train and Test dfs:

    t_density = time()

    

    df_train['UniqueExits'] = -1

    df_test['UniqueExits'] = -1

    df_train['LocalDensity'] = -1

    df_test['LocalDensity'] = -1

    df_train['EntryLength'] = -1

    df_test['EntryLength'] = -1

    for iinter in inter_merge.index:

        inter_code = inter_merge.loc[iinter,'InterCode']

        #

        select = (df_train['InterCode'] == inter_code)

        df_train.loc[select,'UniqueExits'] = inter_merge.loc[iinter,'UniqueExits']

        df_train.loc[select,'LocalDensity'] = inter_merge.loc[iinter,'LocalDensity']

        df_train.loc[select,'EntryLength'] = inter_merge.loc[iinter,'EntryLength']

        #

        select = (df_test['InterCode'] == inter_code)

        df_test.loc[select,'UniqueExits'] = inter_merge.loc[iinter,'UniqueExits']

        df_test.loc[select,'LocalDensity'] = inter_merge.loc[iinter,'LocalDensity']

        df_test.loc[select,'EntryLength'] = inter_merge.loc[iinter,'EntryLength']



    print("\n {:.2f} seconds, added UniqueExits, LocalDensity, EntryLength to Train,Test.\n".format(time() - t_density))
##df_train.tail(10)
if SHOW_EDA:

    # List the column names

    print("Selected columns:\n")

    print(df_train.columns)
print("\nSize of Train and Test: {}, {}".format(len(df_train), len(df_test)))
if SHOW_EDA:

    # Show all the stats of the numeric columns

    desc_train = df_train.describe()

    # Transpose for better printing out

    print("Stats for the Training data:\n")

    print((desc_train.T)[['count','mean','min','max']])
if SHOW_EDA:

    # Alert if Test is Train:

    if TEST_IS_TRAIN:

        print("\n"+20*" *"+"\n   TEST is Train !!!\n"+20*" *")

    # Compare Train and Test averages of the feature values

    # Estimate the z-score of the difference, significant if outside +/-5.

    # Do this by City

    for icity, this_city in enumerate(cities):

        # Using a z-score with standard error based on the number of samples

        descr_train_c = df_train[df_train['iCity'] == icity].describe()

        descr_test_c = df_test[df_test['iCity'] == icity].describe()

        print("\n\n"+5*" "+this_city+

              "  (Lat.: {} -- {}".format(descr_test_c.loc["min","LatOff"],descr_test_c.loc["max","LatOff"]) +

                ",  Long.: {} -- {}".format(descr_test_c.loc["min","LongOff"],descr_test_c.loc["max","LongOff"]) +

              ") \n")

        # Number of samples in the test set

        n_test = descr_test_c.loc["count","HrWk"]

        n_train = descr_train_c.loc["count","HrWk"]

        print("     --column--    z-score      TEST Mean     Train Mean")

        # Select the columns to show here, most of them:

        ##for col in descr_test.columns.drop('iCity').drop('RowId'):

        # or just some chosen (non-target) ones:

        ##for col in ['Month','Hour','Weekend','HrWk','Latitude','Longitude',

        ##            'InHeading','ExHeading','Turn','TurnLSR','Total_p80']:

        for col in descr_test_c.columns.drop('iCity').drop('RowId'):

            ave_test = descr_test_c.loc["mean",col]

            ave_train = descr_train_c.loc["mean",col]

            std_train = descr_train_c.loc["std",col]

            if np.isnan(std_train):

                std_train = 1.0

            print(col.rjust(15), 

                    '{:.4f}'.format((ave_test - ave_train)/

                               (std_train*np.sqrt(1.0/n_test+1.0/n_train))).rjust(10),

                    '{:.4f}'.format(ave_test).rjust(14),

                    '{:.4f}'.format(ave_train).rjust(14))

if SHOW_EDA:

    # Look at a feature

    #

    # Common to Train and Test:

    #  RowId, IntersectionId, Latitude, Longitude,

    #  EntryStreetName, ExitStreetName, EntryHeading, ExitHeading,

    #  Hour, Weekend, Month, Path, City

    #

    # The "y"s that are unique to Train:

    #  TotalTimeStopped_p    20,40,50,60,80

    #  TimeFromFirstStop_p   20,40,50,60,80

    #  DistanceToFirstStop_p 20,40,50,60,80

    

    ##feat_name = 'EntryLength'

    ##feat_name = 'LocalDensity'

    ##feat_name = 'UniqueExits'

    ##feat_name = 'Center*InHead'

    ##feat_name = 'DistToCenter'

    ##feat_name = 'EntryType'

    ##feat_name = 'InterCode'

    feat_name = 'HrWk'

    ##feat_name = 'Turn'

    

    ##feat_name = 'TotalTimeStopped_p80'

    ##feat_name = 'DistanceToFirstStop_p80'

    

    val_counts = df_train[feat_name].value_counts()

    feat_values = np.sort(val_counts.index)

    print(feat_name+" has {} distinct values".format(len(val_counts)) +

            " from {} to {}".format(feat_values[0],feat_values[-1]))

    # Find the mean of the non-zero values:

    if '_p' in feat_name:

        df_temp2 = df_train[df_train[feat_name] > 0.0].copy()

        print(15*" "+"Mean of All: {:.2f}, Mean of non-zeros: {:.2f}".format(df_train[feat_name].mean(),

                                    df_temp2[feat_name].mean()))

    # Show them if not too many

    if (len(val_counts) < 30.0):

        print(val_counts)

    else:

        print(val_counts[val_counts.index[0:10]])

        print(val_counts[val_counts.index[-5:-1]])

# Look at the histogram of this feature's TRAINING values

# provided that it is numeric:

if SHOW_EDA and (feat_name in desc_train.columns):

    

    # Histogram of Training values

    df_train[feat_name].hist(bins=2*(2*48-1), figsize=(10,4), grid=False, color='orange')

    plt.xlabel(feat_name + " values")

    plt.ylabel("Number of samples")

    plt.title(feat_name + " -- Training")

    plt.savefig(out_dir+"/"+feat_name+"_Train_"+version_str+".png", bbox_inches='tight')

    plt.show()



    # Calculate the average Total_p80 for each feature value

    total_p80_aves = []

    # Number of feature values with NoWait

    n_now = 0

    for this_val in feat_values:

        # df of ones with this feature value

        df_feat = df_train[(df_train[feat_name] == this_val)]

        feat_mean = df_feat['Total_p80'].mean()

        total_p80_aves.append(feat_mean)

        if feat_mean < NoWait_Threshold:

            n_now += 1

    

    if False:

        # List the features in total_p80 sorted order

        # Did this to change the street-type dictionary so that 

        # EntryType numerical values are in the same order as their total_p80_aves.

        sort_indx = np.argsort(total_p80_aves)

        print(feat_values[sort_indx])

        print((np.array(total_p80_aves))[sort_indx])

        # Low to Hi indices for EntryType:

        # [11 12 15 21 13  2 25  1 17  4  9 19 14  7 10 18  5  3  0  6 16 20  8]

        #   . . . for ExitType it's different:

        # 

    

    

    # Plot the Ave'p80 vs feat_values

    plt.figure(figsize=(10,4))

    plt.plot(feat_values, total_p80_aves, marker="o", linestyle='')

    plt.xlabel(feat_name + " values")

    plt.ylabel("Average Total_p80")

    plt.title(feat_name + " -- Training"+

          "  (The number below {} is: {} / {})".format(NoWait_Threshold,n_now,len(feat_values)))

    # start at 0:

    plt.ylim(0.0,)

    plt.savefig(out_dir+"/"+feat_name+"_Train_p80_"+version_str+".png", bbox_inches='tight')

    plt.show()



    # Histogram of this feature's TEST values

    if feat_name in df_test.columns:

        df_test[feat_name].hist(bins=2*(2*48-1), figsize=(10,4), grid=False, color='orange')

        plt.xlabel(feat_name + " values")

        plt.ylabel("Number of samples")

        plt.title(feat_name + " -- TEST")

        plt.savefig(out_dir+"/"+feat_name+"_TEST_"+version_str+".png", bbox_inches='tight')

        plt.show()
# Make an array to store the average values of the 6 target values

# based on the City, Month, Hour+Weekend, and iWait values.

# This divides the samples into :

#    4x12(really 7*)x48x3 = 4032 categories.

#              (* only 7 months have appreciable data)

# Here the months have also been collapsed to 3, so the total is

#     4 x 3 x 48 x 3  =  1728 different lookup-value sets of the 6 targets.

t_lookup = time()



# Lookup array of mean values of the 6 output values

# (the T in the variable is a hold-over from including Turn)

lookup_CMHWTV = np.zeros([4,3,48,3,6])



# Fill the array with average values from the training set



for iCity in range(4):

    for iMonth in [0,1,2]:      # instead of 0-11, range(12):

        if SHOW_EDA:

            print("... doing iCity = {},  Month = {}".format(iCity,iMonth+1))

        # get just that part of the df_train

        df_lookup = df_train[(df_train['iCity'] == iCity) &

                          (df_train['Month'] == iMonth+1)].copy()

        for iHrWk in range(48):

            for iWait in range(3):

                if True:

                    select = ( (df_lookup['HrWk'] == iHrWk) &

                          (df_lookup['iWait'] == iWait)  )

                    df_temp2 = df_lookup.loc[select].copy()

                    if len(df_temp2) > 0:

                        # Get all six of the output values

                        for icol in range(6):

                            lookup_CMHWTV[iCity,iMonth,iHrWk,iWait,icol] = int(

                                0.5+df_temp2[out_cols[icol]].mean())



print("Filling the lookup array took {:.3f} seconds.".format(time() -  t_lookup))
# There are not as many of the iWait=2 values,

# so smooth the values vs HrWk for them.

# do "1 2 1" smothing, twice: --> equivalent to 1,4,6,4,1



for iCity in range(4):

    for iMonth in [0,1,2]:

        iWait=2

        if True:

            for icol in range(6):

                new_vals = np.zeros(48)

                for iHrWk in range(1,47):

                    new_vals[iHrWk] = int(0.5 +

                                    0.25*lookup_CMHWTV[iCity,iMonth,iHrWk-1,iWait,icol] +

                                    0.50*lookup_CMHWTV[iCity,iMonth,iHrWk,iWait,icol] +

                                    0.25*lookup_CMHWTV[iCity,iMonth,iHrWk+1,iWait,icol] )

                new_vals[0] = int(0.5 +

                                    0.25*lookup_CMHWTV[iCity,iMonth,47,iWait,icol] +

                                    0.50*lookup_CMHWTV[iCity,iMonth,0,iWait,icol] +

                                    0.25*lookup_CMHWTV[iCity,iMonth,1,iWait,icol] )

                new_vals[47] = int(0.5 +

                                    0.25*lookup_CMHWTV[iCity,iMonth,46,iWait,icol] +

                                    0.50*lookup_CMHWTV[iCity,iMonth,47,iWait,icol] +

                                    0.25*lookup_CMHWTV[iCity,iMonth,0,iWait,icol] )

                lookup_CMHWTV[iCity,iMonth, : ,iWait,icol] = new_vals

# and again

for iCity in range(4):

    for iMonth in [0,1,2]:

        iWait=2

        if True:

            for icol in range(6):

                new_vals = np.zeros(48)

                for iHrWk in range(1,47):

                    new_vals[iHrWk] = int(0.5 +

                                    0.25*lookup_CMHWTV[iCity,iMonth,iHrWk-1,iWait,icol] +

                                    0.50*lookup_CMHWTV[iCity,iMonth,iHrWk,iWait,icol] +

                                    0.25*lookup_CMHWTV[iCity,iMonth,iHrWk+1,iWait,icol] )

                new_vals[0] = int(0.5 +

                                    0.25*lookup_CMHWTV[iCity,iMonth,47,iWait,icol] +

                                    0.50*lookup_CMHWTV[iCity,iMonth,0,iWait,icol] +

                                    0.25*lookup_CMHWTV[iCity,iMonth,1,iWait,icol] )

                new_vals[47] = int(0.5 +

                                    0.25*lookup_CMHWTV[iCity,iMonth,46,iWait,icol] +

                                    0.50*lookup_CMHWTV[iCity,iMonth,47,iWait,icol] +

                                    0.25*lookup_CMHWTV[iCity,iMonth,0,iWait,icol] )

                lookup_CMHWTV[iCity,iMonth, : ,iWait,icol] = new_vals

# Fill arrays with correction lookup factors for DTFS and TTS p80 vs the Turn value

# Just by City and Wait and Turn (-4,-3,-2,-1,0,1,2,3).

if True:

    

    tts_turn_lookup = np.zeros([4,3,8])

    # For TTS   out_cols[2]

    for iCity in range(4):

        for iWait in range(3):

            ##print("iCity,iWait = {},{}:".format(iCity,iWait))

            ave_all_turns = df_train.loc[((df_train['iCity'] == iCity) &

                       (df_train['iWait'] == iWait)), out_cols[2]].mean()

            for iturn in [-4,-3,-2,-1,0,1,2,3]:

                tts_turn_lookup[iCity,iWait,iturn+4] = df_train.loc[((df_train['Turn'] == iturn) &

                       (df_train['iCity'] == iCity) &

                       (df_train['iWait'] == iWait)), out_cols[2]].mean()/ave_all_turns

    # replace any NaNs with 1.5:

    tts_turn_lookup[np.isnan(tts_turn_lookup)] = 1.5

        

    dtfs_turn_lookup = np.zeros([4,3,8])

    # For DTFS   out_cols[5]

    for iCity in range(4):

        for iWait in range(3):

            ##print("iCity,iWait = {},{}:".format(iCity,iWait))

            ave_all_turns = df_train.loc[((df_train['iCity'] == iCity) &

                       (df_train['iWait'] == iWait)), out_cols[5]].mean()

            for iturn in [-4,-3,-2,-1,0,1,2,3]:

                dtfs_turn_lookup[iCity,iWait,iturn+4] = df_train.loc[((df_train['Turn'] == iturn) &

                       (df_train['iCity'] == iCity) &

                       (df_train['iWait'] == iWait)), out_cols[5]].mean()/ave_all_turns

    # replace any NaNs with 1.0:

    dtfs_turn_lookup[np.isnan(dtfs_turn_lookup)] = 1.0

    

else:

    tts_turn_lookup = np.ones([4,3,3])

    dtfs_turn_lookup = np.ones([4,3,3])
# Look at the Lookup values

#   lookup_CMHWTV[iCity,iMonth,iHrWk,iWait,icol]



if SHOW_EDA:

    # Make plots of the values...

    # assign colors

    clr_wait = ['green','darkorange','pink']

    clr_month123 = ['red','darkorange','blue']

    iCity = 0

    for iCity in range(4):

        plt.figure(figsize=(10,4))

        for iMonth in [0,1,2]:

            for iWait in [2,1,0]:

                # Get an array of Total_p80 vs iHrWk

                array_p80 = np.zeros(48)

                array_p80  += (lookup_CMHWTV[iCity,iMonth,:,iWait,2] +

                                               lookup_CMHWTV[iCity,iMonth,:,iWait,5])

                plt.plot(array_p80, color=clr_month123[iMonth],marker='o',alpha=0.6)

        plt.xlabel("Hours (week day 0-23, weekend 24-47)")

        if ENTRYSECTIONS:

            plt.ylim(0,950)

        else:

            plt.ylim(0,600)

        plt.ylabel("TTS_p80 + DTFS_p80")

        plt.title(cities[iCity]+" -- Lookup table values vs HrWk, for iWait=0,1,2 and 'iMonth'=0,1,2")

    

        plt.savefig(out_dir+"/"+cities[iCity]+"_lookup_vs_HrWk_"+version_str+".png")

        plt.show()
# Compare the Lookup Predictions with actual p80 values



# Pick a City:  0,1,2,3  Atlanta, Boston, Chicago, Philly

iCity = 3

# Pick a month  0,1,2   +1 --> Month=1,2,3

iMonth = 0

# Will plot vs HrWk

# iHrWk

# Pick an iWait  0,1,2

iWait = 2



heading_col ='InHeading'



if SHOW_EDA:

    plt.figure(figsize=(14,7))

    

    # Get some of the actual values for these and overplot them

    select = ( (df_train['iCity'] == iCity) & (df_train['Month'] == iMonth+1) & 

              (df_train['iWait'] == iWait) )

    num_select = sum(select)

    df_select = df_train[select].copy()

    

    print("\nNumber of selected df_Train values = " + 

          "{},  average of {:.2f} entries/hour".format(num_select, num_select/48.0))

    

    hrwk_values = df_select['HrWk'].copy()

    # blur the HrWk values

    hrwk_values +=  + 0.15*np.random.randn(len(hrwk_values))

    

    tts_values = df_select['TotalTimeStopped_p80']

    dtfs_values = df_select['DistanceToFirstStop_p80']

    

    plt.plot(hrwk_values, np.log10(1.0+dtfs_values), linestyle='',

             marker='.',alpha=0.3,color=(0.7,0.7,0.5))

    

    if True:

        if True:

            # Get arrays for the lookup TTS and DTFS

            tts_lookup_values = np.zeros(48)

            dtfs_lookup_values = np.zeros(48)

            tts_lookup_values = lookup_CMHWTV[iCity,iMonth,:,iWait,2]

            dtfs_lookup_values = lookup_CMHWTV[iCity,iMonth,:,iWait,5]

            #

            plt.plot(np.log10(1.0+dtfs_lookup_values), color='gray',marker='o',alpha=0.8)





    # Overplot a single intersection's values

    inters_select = df_select['InterCode']

    inters_select = inters_select.unique()

    num_inters = len(inters_select)

    print("Number of unique (Entry-)Intersections selected = " + 

          "{},  average of {:.2f} values/inter-hour".format(num_inters, num_select/(48.0*num_inters)))

    

    

    # select the intersection  (middle one =  int(len(inters_select)/2) )

    iinter = int(len(inters_select)/2) + 17

    

    if True:

        # get its values:

        inter_hrwk = df_select.loc[df_select['InterCode'] == inters_select[iinter], 'HrWk'].values

        inter_dtfs = df_select.loc[df_select['InterCode'] == inters_select[iinter], 'DistanceToFirstStop_p80'].values

        inter_tts = df_select.loc[df_select['InterCode'] == inters_select[iinter], 'TotalTimeStopped_p80'].values

        inter_head = df_select.loc[df_select['InterCode'] == inters_select[iinter], heading_col].values

        head_clrs = ['black','red','orange','yellow','green','blue','purple','gray']

        # and plot them - offset HrWk by 0.3 hour to stand out better

        for ptind in range(len(inter_hrwk)):

            plt.plot([0.3+inter_hrwk[ptind]], [np.log10(1.0+inter_dtfs[ptind])], linestyle='', marker='o',

                     alpha=0.8, color=head_clrs[inter_head[ptind]])

        plt.text(4, 1.1, 'InterCode ='+str(inters_select[iinter]))

        # show the variable of the radius

        plt.text(10.0, 3.46, 'DTFS_p80', color='black')

    

    plt.xlabel("Hours (week day 0-23, weekend 24-47)")

    plt.xlim(-1,48)

    plt.ylabel("log10(  DTFS  )")

    plt.ylim(0.9, 3.6)

    plt.title(cities[iCity]+" -- Curve: DTFS Lookup Table values " +

              "(iCity={}, iMonth={}, iWait={} )".format(iCity,iMonth,iWait) +

             "   Color-code is {}".format(heading_col))

    

    plt.savefig(out_dir+"/"+cities[iCity]+"_DTFS_w_data_"+version_str+".png")

    plt.show()

    



if SHOW_EDA:

    # Plot the DTFS vs the Entry Heading

    # Can do it in polar coordinates ;-)

    use_polar = True

    

    if use_polar:

        plt.figure(figsize=(6,6))

        # Some grid lines, curves

        plt.plot([-4,4],[0,0,],color='gray')

        plt.plot([0,0],[-4,4],color='gray')

        plt.plot([-6,6],[-6,6,],color='gray')

        plt.plot([-6,6],[6,-6],color='gray')

        for radius in [1.0,2.0,3.0]:

            circx=[]

            circy=[]

            for thetaN in np.arange(0,2.0*np.pi+0.05,0.1):

                circx.append(radius*np.cos(thetaN))

                circy.append(radius*np.sin(thetaN))

            plt.plot(circx,circy,linestyle='-',color='gray')

        plt.text(0.9, 0.5, '10', color='black')

        plt.text(1.8, 1.0, '100', color='black')

        plt.text(2.7, 1.5, '1000', color='black')

        # show the variable of the radius

        plt.text(-2.5, 3.2, 'DTFS_p80', color='black')

        #

        for iplt in range(len(inter_head)):

            # Polar plot: (thetaN in radians: 0 pointing N and positive N to E)

            # blur theta and radius

            thetaN = (inter_head[iplt]+0.1*np.random.randn())*np.pi/4.0

            radius = np.log10(1.0+inter_dtfs[iplt]+np.abs(np.random.randn()))

            plt.plot([radius*np.sin(thetaN)], [radius*np.cos(thetaN)],

                     linestyle='', marker='o',alpha=0.5,color=head_clrs[inter_head[iplt]])

        plt.xlabel("West  - {} -  East".format(heading_col))

        plt.xlim(-3.6, 3.6)

        plt.ylabel("South  - {} -  North".format(heading_col))

        plt.ylim(-3.6, 3.6)

    else:

        plt.figure(figsize=(8,4))

        for iplt in range(len(inter_head)):

            # Linear plot:

            plt.plot([inter_head[iplt]+0.1*np.random.randn()], [np.log10(1.0+inter_dtfs[iplt])],

                     linestyle='', marker='o',alpha=0.5,color=head_clrs[inter_head[iplt]])

        plt.xlabel("Exit Heading code (0 - 7)")

        plt.xlim(-0.5,7.5)

        plt.ylabel("log10(  DTFS  )")

        plt.ylim(0.9, 3.6)

    



    plt.title(cities[iCity]+" " +

              "(iMonth={}, iWait={} )  InterCode = {}".format(

                  iMonth,iWait,str(inters_select[iinter])))

    

    plt.savefig(out_dir+"/"+cities[iCity]+"_DTFS_polar_"+version_str+".png")

    plt.show()
# Free up memory I don't need

del select, df_lookup, df_temp2

# collect any garbage?

gc_dummy = gc.collect()
# When the Test is the train we 'know' the answer,

# so this will give the best we can do on training.



if True and TEST_IS_TRAIN:

    # Set the "known" TEST NoWait and LowWait values based on the train values

    df_test['NoWait'] = df_train['NoWait']

    df_test['LoWait'] = df_train['LoWait']

    df_test['HiWait'] = df_train['HiWait']



    print("Test is Train, setting the Test Wait values from the known Train values:\n")

    print("Fraction of TEST rows assigned NoWait = {:.2f} %".format(100.0*df_test['NoWait'].mean()))

    print("Fraction of TEST rows assigned LoWait = {:.2f} %".format(100.0*df_test['LoWait'].mean()))

    print("Fraction of TEST rows assigned HiWait = {:.2f} %".format(100.0*df_test['HiWait'].mean()))
# Very simple ML to assign LoWait, HiWait for Test data



# Some sklearn routines to use

from sklearn.metrics import accuracy_score

# ML model(s) to use to 'learn' NoWait, LowWait from Xs

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



# ML_WAITS is set at the top in Preliminaries

if ML_WAITS:

    # Get the Features and Target(s)

    

    # The features, including LatOff and LongOff

    ml_x_cols = ['iCity','LatOff','LongOff','InHeading','EntryType',

               'DistToCenter','Center*InHead','UniqueExits','LocalDensity','EntryLength']

    # The features, without LatOff and LongOff

    ##ml_x_cols = ['iCity','InHeading','EntryType',

    ##           'DistToCenter','Center*InHead','UniqueExits','LocalDensity','EntryLength']



    

    X = df_train[ml_x_cols]

    Xtest = df_test[ml_x_cols]



    # The targets to learn

    ylo = df_train['LoWait']

    yhi = df_train['HiWait']

if ML_WAITS:

    # Look at the correlation between the X and y values

    X_temp = X.copy()

    X_temp['ylo'] = ylo

    X_temp['yhi'] = yhi



    corr_df = X_temp.corr()

    # In particular the correlations with ys

    print("\nTrain correlations with ylo:")

    print(corr_df.ylo)

    print("\nTrain correlations with yhi:")

    print(corr_df.yhi)

    

    del X_temp

    gc_dummy = gc.collect()
if ML_WAITS:

    t_ml = time()

    # Setup and Fit to Training data

    

    # RandomForestClassifier(

    # n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1,

    # min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0,

    # min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,

    # warm_start=False, class_weight=None)

    model_name = 'rfc'   # in case anyone asks

    # Parameters for each model:

    

    # The LoWait model is 'balanced' or 'none' depending on WAIT_CHOICE

    if WAIT_CHOICE == 0:

        # near 70% so use balanced with nominal threshold

        lo_weight = 'balanced'

        yloh_threshold = 0.50

    else:

        # near 50% so leave it None

        # and use adjusted threshold to reduce false positives

        lo_weight = None   # probably very similar to balanced

        yloh_threshold = 0.60

    #

    lo_params = {'n_estimators': 30,

              'max_depth': 14,

              'min_samples_leaf': 200,

              'min_impurity_decrease': 0.0,

              'n_jobs': 2,

              'class_weight': lo_weight

             }

    

    # The HiWait model is 'balanced' since HiWait = 1 for only 2-3%,

    # in addition the parameter yhih_threshold is used to reduce false positives

    # by setting a higher threshold probability to give a 1.

    hi_weight = 'balanced'

    yhih_threshold = 0.79

    hi_params = {'n_estimators': 30,

              'max_depth': 14,

              'min_samples_leaf': 200,

              'min_impurity_decrease': 0.0,

              'n_jobs': 2,

              'class_weight': hi_weight

             }

    lo_model_base = RandomForestClassifier(**lo_params)

    hi_model_base = RandomForestClassifier(**hi_params)



    

    # Do the 'learning'

    lo_fit_model = lo_model_base.fit(X,ylo)

    hi_fit_model = hi_model_base.fit(X,yhi)

    

    # Show the parameters

    ##print(lo_fit_model.get_params())

    ##print("")

    ##print(hi_fit_model.get_params())

    

    print("\nDoing the ML took {:.3f} seconds.".format(time() -  t_ml))

    # Doing the ML took 37.415 seconds.  Test is train, 30 estimators

    # Doing the ML took 114.396 seconds.               120 estimators

    # Doing the ML took 120.996 seconds.         120 estimators, 100 min...leaf
if ML_WAITS:

    feature_importance = lo_fit_model.feature_importances_

        

    # make importances relative to max importance

    max_import = feature_importance.max()

    feature_importance = 100.0 * (feature_importance / max_import)

    sorted_idx = np.argsort(feature_importance)

    pos = np.arange(sorted_idx.shape[0]) + 0.5



    plt.figure(figsize=(8, 5))

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    plt.yticks(pos, X.columns[sorted_idx])

    plt.xlabel(model_name.upper()+' -- Relative Importance')

    plt.title('           '+model_name.upper()+

              ' - LoWait -- Variable Importance                  max --> {:.3f} '.format(max_import))



    plt.savefig(out_dir+"/"+model_name.upper()+"-Lo_importance_"+version_str+".png")

    plt.show()
if ML_WAITS:

    feature_importance = hi_fit_model.feature_importances_

        

    # make importances relative to max importance

    max_import = feature_importance.max()

    feature_importance = 100.0 * (feature_importance / max_import)

    sorted_idx = np.argsort(feature_importance)

    pos = np.arange(sorted_idx.shape[0]) + 0.5



    plt.figure(figsize=(8, 5))

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    plt.yticks(pos, X.columns[sorted_idx])

    plt.xlabel(model_name.upper()+' -- Relative Importance')

    plt.title('           '+model_name.upper()+

              ' - HiWait -- Variable Importance                  max --> {:.3f} '.format(max_import))



    plt.savefig(out_dir+"/"+model_name.upper()+"-Hi_importance_"+version_str+".png")

    plt.show()
# Look at the predictions on Train and assign to Test



if ML_WAITS:

    # Training predictions (discrete, uses model prob of 0.5 as threshold)

    yloh = lo_fit_model.predict(X)

    yhih = hi_fit_model.predict(X)



    # Also get the continuous probabilities from the models:

    yloh_prob = lo_fit_model.predict_proba(X)

    yloh_prob = yloh_prob[:,1]

    yhih_prob = hi_fit_model.predict_proba(X)

    yhih_prob = yhih_prob[:,1]

    

    # Redefine the 0,1 results using a threshold for 1, default is 0.5

    yloh = (yloh_prob > yloh_threshold).astype(int)

    # Use a higher threshold for the HiWait to reduce false positives:

    # This is set above when the model is defined: yhih_threshold = 0.79

    yhih = (yhih_prob > yhih_threshold).astype(int)

    

    

    print("")

    lo_train_score = accuracy_score(ylo, yloh)

    print("Train accuracy: {:.2f} %,".format(100.0*lo_train_score) +

            "  LoWait fraction = {:.2f}%".format(100*yloh.mean()) +

         ", should be {:.2f}%".format(100*ylo.mean()))

    hi_train_score = accuracy_score(yhi, yhih)

    print("Train accuracy: {:.2f} %,".format(100.0*hi_train_score) +

            "  HiWait fraction = {:.2f}%".format(100*yhih.mean()) +

         ", should be {:.2f}%".format(100*yhi.mean()))



    

    # Make the Test predictions

    yloh_test = lo_fit_model.predict(Xtest)

    yhih_test = hi_fit_model.predict(Xtest)

    

    # Use thresholds for the yloh and yhih

    yloh_test_prob = lo_fit_model.predict_proba(Xtest)

    yloh_test_prob = yloh_test_prob[:,1]

    yloh_test = (yloh_test_prob > yloh_threshold).astype(int)

    yhih_test_prob = hi_fit_model.predict_proba(Xtest)

    yhih_test_prob = yhih_test_prob[:,1]

    yhih_test = (yhih_test_prob > yhih_threshold).astype(int)

    

    # Put those values in the test LoWait and HiWait:

    df_test['LoWait'] = yloh_test

    df_test['HiWait'] = yhih_test

    

    if TEST_IS_TRAIN:

        reminder_str = '[Note: TEST is train!]'

    else:

        reminder_str = ''

        

    print("\nTEST LoWait fraction = {:.2f}%  {}".format(100*yloh_test.mean(),reminder_str))

    print("TEST HiWait fraction = {:.2f}%  {}".format(100*yhih_test.mean(),reminder_str))

if ML_WAITS:

    # Don't need the models anymore

    del lo_fit_model, hi_fit_model, lo_model_base, hi_model_base

    del yloh_test, yhih_test

    gc_dummy = gc.collect()
# Use this routine to shown how the prediction is doing.

# This routine is taken from the file chirp_roc_lib.py in the github repo at: 

#   https://github.com/dan3dewey/chirp-to-ROC

# Some small modifications have been made here.



def y_yhat_plots(y, yh, title="y and y_score", y_thresh=0.5, ROC=True, plots_prefix=None):

    """Output plots showing how y and y_hat are related:

    the "confusion dots" plot is analogous to the confusion table,

    and the standard ROC plot with its AOC value.

    The yp=1 threshold can be changed with the y_thresh parameter.

    y and yh are numpy arrays (not series or dataframe.)

    """

    # The predicted y value with threshold = y_thresh

    y_pred = 1.0 * (yh > y_thresh)



    # Show table of actual and predicted counts

    crosstab = pd.crosstab(y, y_pred, rownames=[

                           'Actual'], colnames=['  Predicted'])

    print("\nConfusion matrix (y_thresh={:.3f}):\n\n".format(y_thresh),

        crosstab)



    # Calculate the various metrics and rates

    tn = crosstab[0][0]

    fp = crosstab[1][0]

    fn = crosstab[0][1]

    tp = crosstab[1][1]



    ##print(" tn =",tn)

    ##print(" fp =",fp)

    ##print(" fn =",fn)

    ##print(" tp =",tp)



    this_fpr = fp / (fp + tn)

    this_fnr = fn / (fn + tp)



    this_recall = tp / (tp + fn)

    this_precision = tp / (tp + fp)

    this_accur = (tp + tn) / (tp + fn + fp + tn)



    this_posfrac = (tp + fn) / (tp + fn + fp + tn)



    print("\nResults:\n")

    print(" False Pos = ", 100.0 * this_fpr, "%")

    print(" False Neg = ", 100.0 * this_fnr, "%")

    print("    Recall = ", 100.0 * this_recall, "%")

    print(" Precision = ", 100.0 * this_precision, "%")

    print("\n    Accuracy = ", 100.0 * this_accur, "%")

    print(" Pos. fract. = ", 100.0 * this_posfrac, "%")



    

    # Put them in a dataframe for plots and ROC

    # Reduce the number if very large:

    if len(y) > 100000:

        reduce_by = int(0.5+len(y)/60000)

        print("\nUsing 1/{} of the points for dots and ROC plots.".format(reduce_by))

        ysframe = pd.DataFrame([y[0: :reduce_by], yh[0: :reduce_by], 

                                y_pred[0: :reduce_by]], index=[

                           'y', 'y-hat', 'y-pred']).transpose()



    # If the yh is discrete (0 and 1s only) then blur it a bit

    # for a better visual dots plot

    if min(abs(yh - 0.5)) > 0.49:

        ysframe["y-hat"] = (0.51 * ysframe["y-hat"]

                            + 0.49 * np.random.rand(len(ysframe)))



    # Make a "confusion dots" plot

    # Add a blurred y column

    ysframe['y (blurred)'] = ysframe['y'] + 0.1 * np.random.randn(len(ysframe))



    # Plot the real y (blurred) vs the predicted probability

    # Note the flipped ylim values.

    ysframe.plot.scatter('y-hat', 'y (blurred)', figsize=(12, 5),

                         s=2, xlim=(0.0, 1.0), ylim=(1.8, -0.8), alpha=0.3)

    # show the "correct" locations on the plot

    plt.plot([0.0, y_thresh], [0.0, 0.0], '-',

        color='green', linewidth=5)

    plt.plot([y_thresh, y_thresh], [0.0, 1.0], '-',

        color='gray', linewidth=2)

    plt.plot([y_thresh, 1.0], [1.0, 1.0], '-',

        color='green', linewidth=5)

    plt.title("Confusion-dots Plot: " + title, fontsize=16)

    # some labels

    ythr2 = y_thresh/2.0

    plt.text(ythr2 - 0.03, 1.52, "FN", fontsize=16, color='red')

    plt.text(ythr2 + 0.5 - 0.03, 1.52, "TP", fontsize=16, color='green')

    plt.text(ythr2 - 0.03, -0.50, "TN", fontsize=16, color='green')

    plt.text(ythr2 + 0.5 - 0.03, -0.50, "FP", fontsize=16, color='red')



    if plots_prefix != None:

        plt.savefig(plots_prefix+"_dots.png")

    plt.show()



    # Go on to calculate and plot the ROC?

    if ROC == False:

        return 0

    

    

    # Make the ROC curve

    # 

    # Set the y-hat as the index and sort on it

    ysframe = ysframe.set_index('y-hat').sort_index()

    # Put y-hat back as a column (but the sorting remains)

    ysframe = ysframe.reset_index()



    # Initialize the counts for threshold = 0

    p_thresh = 0

    FN = 0

    TN = 0

    TP = sum(ysframe['y'])

    FP = len(ysframe) - TP



    # Assemble the fpr and recall values

    recall = []

    fpr = []

    # Go through each sample in y-hat order,

    # advancing the threshold and adjusting the counts

    for iprob in range(len(ysframe['y-hat'])):

        p_thresh = ysframe.iloc[iprob]['y-hat']

        if ysframe.iloc[iprob]['y'] == 0:

            FP -= 1

            TN += 1

        else:

            TP -= 1

            FN += 1

        # Recall and FPR:

        recall.append(TP / (TP + FN))

        fpr.append(FP / (FP + TN))



    # Put recall and fpr in the dataframe

    ysframe['Recall'] = recall

    ysframe['FPR'] = fpr



    # - - - ROC - - - could be separate routine

    zoom_in = False



    # Calculate the area under the ROC

    roc_area = 0.0

    for ifpr in range(1, len(fpr)):

        # add on the bit of area (note sign change, going from high fpr to low)

        roc_area += 0.5 * (recall[ifpr] + recall[ifpr - 1]

                           ) * (fpr[ifpr - 1] - fpr[ifpr])



    plt.figure(figsize=(6, 6))

    plt.title("ROC: " + title, size=16)

    plt.plot(fpr, recall, '-b')

    # Set the scales

    if zoom_in:

        plt.xlim(0.0, 0.10)

        plt.ylim(0.0, 0.50)

    else:

        # full range:

        plt.xlim(0.0, 1.0)

        plt.ylim(0.0, 1.0)



    # The reference line

    plt.plot([0., 1.], [0., 1.], '--', color='orange')



    # The point at the y_hat = y_tresh threshold

    if True:

        plt.plot([this_fpr], [this_recall], 'o', c='blue', markersize=15)

        plt.xlabel('False Postive Rate', size=16)

        plt.ylabel('Recall', size=16)

        plt.annotate('y_hat = {:.2f}'.format(y_thresh),

                            xy=(this_fpr+0.01 + 0.015,

                            this_recall), size=14, color='blue')

        plt.annotate(' Pos.Fraction = ' +

                        '  {:.0f}%'.format(100 * this_posfrac),

                        xy=(this_fpr + 0.03, this_recall - 0.045),

                        size=14, color='blue')



    # Show the ROC area (shows on zoomed-out plot)

    plt.annotate('ROC Area = ' + str(roc_area)

                 [:5], xy=(0.4, 0.1), size=16, color='blue')



    # Show the plot

    if plots_prefix != None:

        plt.savefig(plots_prefix+"_ROC.png")

    plt.show()



    return roc_area
# Plots of y - y-hat from these two classifiers

# Use this routine:

# y_yhat_plots(y, yh, title="y and y_score", y_thresh=0.5):

#    """Output plots showing how y and y_hat are related:

#    the "confusion dots" plot is analogous to the confusion table,

#    and the standard ROC plot with its AOC value.

#    The yp=1 threshold can be changed with the y_thresh parameter.

#    """



if True and ML_WAITS:

    

    # The ROC curves cn be shown or not

    show_roc = True

    

    # LoWait

    lo_roc_area = y_yhat_plots(ylo.values, yloh_prob, ROC=True,

                           title="     y and y-hat-prob    for    Low-Wait",

                               y_thresh=yloh_threshold,

                              plots_prefix=out_dir+"/"+model_name.upper()+"-Lo")



    # HiWait

    hi_roc_area = y_yhat_plots(yhi.values, yhih_prob, ROC=show_roc,

                           title="     y and y-hat-prob    for    High-Wait",

                               y_thresh=yhih_threshold,

                               plots_prefix=out_dir+"/"+model_name.upper()+"-Hi")

# Set the iWait values for Test from its LoWait and HiWait classifications

df_test['iWait'] = 1

df_test.loc[df_test['LoWait'] == 1, 'iWait'] = 0

df_test.loc[df_test['HiWait'] == 1, 'iWait'] = 2

#

# If ML did the Lo/Hi determination then there is the chance

# that an item was assigned both Lo and Hi,

# in this case set iWait to 1:

if True:

    select = (df_test['LoWait'] == 1) & (df_test['HiWait'] ==1)

    n_found = sum(select)

    if n_found > 0:

        print("\nOverlap between ML Lo and Hi: {} set to iWait=1 (medium)\n".format(n_found))

        df_test.loc[select, 'iWait'] = 1

    else:

        print("\nNo overlap between Lo and Hi.\n")
# If the Test is the real test data, then there is a LatOff for each City

# above-which we have little clue about the wait at the entry-sections...

# Can set the iWait status of those here

if TEST_IS_TRAIN == False:

    # Those offset thresholds are:

    lat_thresh = [5500, 5100, 5800, 5900]

    

    # Set them to an iWait value?  (-1=No, use the ML, 0,1,2=Yes)

    #

    ##northern_iwait = 1     # <-- * * * * * * * Set in beginning section

    

    

    # Look at / modify those

    print("\nTest Lo,Hi waits as ML-assigned above the latitude threshold:")

    for iCity in range(4):

        select_above = ((df_test['iCity'] == iCity) & 

                        (df_test['LatOff'] > lat_thresh[iCity]))

        fraction_above = sum(select_above)/len(df_test)

        mean_lo = df_test.loc[select_above, 'LoWait'].mean()

        mean_hi = df_test.loc[select_above, 'HiWait'].mean()

        print("   {:>14}: {:.2f}% above threshold, with {:>6.2f}% LoWait and {:.2f}% HiWait.".format(

                    cities[iCity], 100*fraction_above, 100*mean_lo, 100*mean_hi))

        # Set the iWait to a fixed value:

        if northern_iwait != -1:

            df_test.loc[select_above,'iWait'] = northern_iwait

    if northern_iwait != -1:

        print("\nThese were all set to have iWait = {}".format(northern_iwait))

    else:

        print("\nThese were assigned the usual iWait from Lo,Hi.")

else:

    print("\nTEST is Train, so there are no 'northern unknows' to adjust.")
# Apply the lookup to the test df

# Seems it should be possible to do this much faster...



# Doing lookup by going through df and getting the 6 lookup values;

# then put them in the df.

if True:

    # Now use that lookup array to fill the TEST output values

    t_testfill = time()

    # Put values in a list:

    TT20 = []

    DT20 = []

    TT50 = []

    DT50 = []

    TT80 = []

    DT80 = []

    # Go through the df rows

    for this_loc in df_test.index:

        ##print(df_test.loc[this_loc,'ExitStreetName'])

        # Get the values for this row's

        # iCity,iMonth,iHrWk,iLowWait,iTurn

        #

        this_row = df_test.loc[this_loc].astype(int)

        iCity = this_row['iCity']

        iMonth = this_row['Month'] - 1

        iHrWk = this_row['HrWk']

        iWait = this_row['iWait']

        # and for the turn corrections lookup

        iTurn = this_row['Turn'] + 4

        tts_corr = tts_turn_lookup[iCity,iWait,iTurn]

        dtfs_corr = dtfs_turn_lookup[iCity,iWait,iTurn]



        # Get the values...

        TT20.append(int(lookup_CMHWTV[iCity,iMonth,iHrWk,iWait,0]))

        TT50.append(int(lookup_CMHWTV[iCity,iMonth,iHrWk,iWait,1]))

        TT80.append(int(tts_corr*lookup_CMHWTV[iCity,iMonth,iHrWk,iWait,2]))

        DT20.append(int(lookup_CMHWTV[iCity,iMonth,iHrWk,iWait,3]))

        DT50.append(int(lookup_CMHWTV[iCity,iMonth,iHrWk,iWait,4]))

        DT80.append(int(dtfs_corr*lookup_CMHWTV[iCity,iMonth,iHrWk,iWait,5]))



    # Put the values in the df:

    df_test[out_cols[0]] = TT20

    df_test[out_cols[1]] = TT50

    df_test[out_cols[2]] = TT80

    df_test[out_cols[3]] = DT20

    df_test[out_cols[4]] = DT50

    df_test[out_cols[5]] = DT80

    

    print("Filling (w/loop) the TEST df took {:.3f} seconds.".format(time() -  t_testfill))

    # Filling (w/loop) the TEST df took 517.249 seconds. <-- Test is Train, my machine

    # Filling (w/loop) the TEST df took 1181.075 seconds. <-- actual Test, on Kaggle

    

    del TT20, DT20, TT50, DT50, TT80, DT80

    gc.collect()
# Look at some of the X and assigned target values

print("\nSome rows of Xtest:\n", (Xtest.iloc[0:len(Xtest):int(len(Xtest)/11)]))

print("\nThe assigned target values for these:\n")

# get the target values and show with shorter column names

some_targets = (df_test.iloc[0:len(df_test):int(len(df_test)/11)])[out_cols].copy()

some_targets.columns = ['TTS_p20','TTS_p50','TTS_p80','DTFS_p20','DTFS_p50','DTFS_p80']

print( (some_targets) )
if TEST_IS_TRAIN:

    # Calculate the RMS error between the test targets (6, given in out_cols) and the known train values.

    #

    total_se = 0.0

    for this_col in out_cols:

        # add-in the sum of squares of test-train for each target value:

        total_se += sum((df_test[this_col] - df_train[this_col])**2)



    rmse_testtrain = np.sqrt(total_se/(6*len(df_test)))

    print("\nThe overall RMSE for TEST-is-Train is {:.3f}".format(rmse_testtrain) + 

          "   Lo,Hi Thresholds = {}-{}-{}-{}, {}-{}-{}-{} \n".format(

        LoWait_Thresh[0], LoWait_Thresh[1], LoWait_Thresh[2], LoWait_Thresh[3],

        HiWait_Thresh[0], HiWait_Thresh[1], HiWait_Thresh[2], HiWait_Thresh[3]))



    

if TEST_IS_TRAIN:

    total_se = 0.0

    print("Sources of the error:      *** Only from DTFS_p80 ***")

    for iCity in range(4):

        print(" {} ".format(cities[iCity]))

        for iWait in range(3):

            partial_se = 0.0

            select = ((df_train['iCity'] == iCity) & (df_train['iWait'] == iWait))

            num_select = sum(select)

            # Just the DTFS_p80 value:

            for this_col in [out_cols[5]]:

                # add-in the sum of squares of test-train for each target value:

                partial_se += sum((df_test.loc[select,this_col] - df_train.loc[select,this_col])**2)

            print("  iWait={}:  {:>12.1f} x10^6   from {:>8}(x1) with ave = {:>8.1f}".format(

                                        iWait, partial_se/1.0e6, 

                                        int(num_select), (np.sqrt(partial_se/(1.0*num_select)))))

            total_se += partial_se



    # Calculate the overall average RMSE (per 6 columns)

    rmse_testtrain = np.sqrt(total_se/(6*len(df_test)))

    print("\n  The RMSE due to just DTFS_p80 is {:.3f}".format(rmse_testtrain))

    



#           Summary of some results here:



# Using Intersections (this case is just here for reference, Entry-sections do better.)



# - - - WAIT_CHOICE = 0   120, 280:     Lo: 71.39%, Hi: 4.63%

#   Known      ML(RFC)  Lo: 'balanced', 0.50 threshold;  Hi: 'balanced', 0.79 threshold

#   62.127     62.404   



# - - - WAIT_CHOICE = 1    18, 210:     Lo: 42.83%, Hi: 6.44%

#   Known      ML(RFC)  Lo: None, 0.60 threshold;  Hi: 'balanced', 0.79 threshold

#   63.189     63.424  





# Using Entry-sections



# - - - WAIT_CHOICE = 0    120-120-120-120, 500-300-500-500    Lo: 73.86%, Hi: 2.60%

#    49.839 <-- Known iWait

#           ML(RFC)  Lo: 'balanced', 0.50 threshold;  Hi: 'balanced', 0.79 threshold

#                With Lat/LongOff

#              51.657     with UniqueExits, LocalDensity, and EntryLength

#        --->  51.359     (as below) ",  ",  Integers in all X features.

#                No LatOff and LongOff:

#              52.248     with UniqueExits, LocalDensity, and EntryLength

#              52.247     ", Lo,Hi-n_estimators = 120

#              51.707     ", Lo,Hi-n_estimators = 120, Hi-min...leaf = 100

#        --->  51.944     ",  ",  Integers in all X features.



# - - - WAIT_CHOICE = 1   18-18-18-18  400-210-400-400  Lo: 50.05%, Hi: 3.05%

#    51.512 <-- Known iWait

#           ML(RFC)  Lo: None, 0.60 threshold;  Hi: 'balanced', 0.79 threshold

#                With Lat/LongOff

#              52.850   with UniqueExits, LocalDensity, and EntryLength

#        --->  52.315   (as above) ",  ",  Integers in all X features.

#                No LatOff and LongOff:

#              52.952   with UniqueExits, LocalDensity, and EntryLength

#        --->  52.686   (as above) ",  ",  Integers in all X features.

# Summarize some of the major parameters:

print("Some of the major choices:\n\nENTRYSECTIONS = {}".format(ENTRYSECTIONS))

print("WAIT_CHOICE = {}".format(WAIT_CHOICE))

if ML_WAITS:

    print("The ML features used:\n",Xtest.columns)

elif TEST_IS_TRAIN == True:

    print("Known iWait values used, no ML.")

    
# Output the predicted values from the df_test columns to submission.csv



# Only write-out the file if not TEST_IS_TRAIN and not REDUCED_SIZE:    

if ((TEST_IS_TRAIN == False) and (REDUCED_SIZE == False)):

    t_file_write = time()

    # Make a two column df from the first output value

    icol = 0

    df_out = df_test[[out_cols[icol]]].copy().reset_index()

    # rename the columns

    df_out.columns = ['TargetId','Target']

    # add the suffix

    df_out['TargetId'] = df_out['TargetId'].astype(str) + "_" + str(icol)

    # Use this first df as the final output one:

    df_outall = df_out.copy()

    #

    # Now go through the rest of the values and append them

    for icol in [1,2,3,4,5]:

        df_out = df_test[[out_cols[icol]]].copy().reset_index()

        df_out.columns = ['TargetId','Target']

        df_out['TargetId'] = df_out['TargetId'].astype(str) + "_" + str(icol)

        # append these to the outall:

        df_outall = df_outall.append(df_out)

    #

    df_outall.to_csv("submission.csv", index=False)

    #

    print("Writing the file took {:.3f} seconds.".format(time() -  t_file_write))

else:

    print("No submission file written.")
# that's all, take a look at it

print("\n   The beginning of submission.csv:")

!head -10 submission.csv
print("\n   The end of submission.csv:")

!tail -10 submission.csv
# show/confirm the random seed value

print("Used RANDOM_SEED = {}".format(RANDOM_SEED))