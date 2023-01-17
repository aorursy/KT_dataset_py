import pandas as pd  # Dataframe library
import numpy as np  # Array manipulation library
data_dir = "../input/alsa/clinicaltrials.csv"
trials_df = pd.read_csv(data_dir)
trials_df.head()
trials_df.columns
trials_df["Locations"].iloc[0]  # Specifically looking at the first record, to see what is going on
trials_df["Locations"].iloc[50]
record_location_0 = trials_df["Locations"].iloc[0]

locations = record_location_0.split("|")

print("Extracting each sentence...")
print("\n".join(locations))

print()
print("Extracting subdivison and country name...")
for location in locations:
    subdivision, country = location.split(", ")[-2:]
    
    print(subdivision.upper(), country.upper())  # converting strings to uppercase
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import _pickle as cPickle

geolocator = Nominatim(user_agent="specify_your_app_name_here", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
location_list = []

with open(r"location_list.pickle", "wb") as output_file:

    for idx, row in trials_df["Locations"].iteritems():

        if idx%50 == 0:
            print(idx)
    
        # Some records don't have a location, so in these cases I skip over them and leave a placeholder
        if pd.isna(row):
            location_list.append([idx, np.nan, np.nan, "NULL"])
            cPickle.dump(location_list, output_file)
            continue

        # Split multiple locations, as discussed above
        locations = list(row.split("|"))  

        # For each of those locations, extract the subdivision and country
        for location in locations: 
            _location = location.split(", ")[-2:]

            try:
                geo_location = geolocator.geocode(_location)

                if geo_location is None:
                    geo_location = geolocator.geocode(_location[-1])

            except: # sometimes the geopy library timesout, so this will just force it to try again.
                geo_location = geolocator.geocode(_location)

                if geo_location is None:
                    geo_location = geolocator.geocode(_location[-1])

            # Record the index, latitude, longitude, and string representation of the location
            location_list.append([idx, geo_location.latitude, geo_location.longitude, location])

# Convert the list into a DataFrame object        
locations_df = pd.DataFrame(location_list, columns=["TRIAL_ID", "LATITUDE", "LONGITUDE", "LOCATION"])
locations_df.head(10)
trials_df["Age"].head(10)
trials_df["AGE_LOWER"] = np.ones(trials_df.shape[0])
trials_df["AGE_UPPER"] = np.ones(trials_df.shape[0])

for idx, row in trials_df["Age"].iteritems():
    
    if (idx + 1)%100 == 0:
        print(idx)
    
    # Split the age information and extract all numbers
    age_range = [int(s) for s in row.split() if s.isdigit()]
    
    # If there is exactly one number (case 1 above)
    if len(age_range) == 1:
        trials_df["AGE_LOWER"].iloc[idx] = age_range[0]
        trials_df["AGE_UPPER"].iloc[idx] = 100  # assinging some default
    
    # If there are two numbers (case 2 above)
    elif len(age_range) == 2:
        trials_df["AGE_LOWER"].iloc[idx] = age_range[0]
        trials_df["AGE_UPPER"].iloc[idx] = age_range[1]
        
    # If there are no numbers (case 3 above)
    else:
        trials_df["AGE_LOWER"].iloc[idx] = 18  # assinging some default
        trials_df["AGE_UPPER"].iloc[idx] = 100  # assinging some default

trials_df[["AGE_LOWER", "AGE_UPPER"]].head()
trials_df.drop(columns=["Age"], inplace=True)  # We no longer need this column
trials_df.drop(columns=["Locations"],inplace=True)  # Or this one
merged_df = locations_df.join(trials_df, on="TRIAL_ID")

merged_df.head()