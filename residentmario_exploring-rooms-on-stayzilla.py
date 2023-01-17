import pandas as pd



stays = pd.read_csv("../input/stayzilla_com-travel_sample.csv")

pd.set_option('max_columns', None)

stays.head(3)
import numpy as np



# Map easy features.

stays = stays.assign(

    room_price=stays.room_price.map(

        lambda v: float(v[:v.find("per")]) if not pd.isnull(v) else np.nan

    ),

    acceptance_rate=stays.additional_info.map(

        lambda v: v.split("|")[0].split(":")[-1].split(" ")[0] if not pd.isnull(v) else np.nan

    ),

    response_time=stays.additional_info.map(

        lambda v: v.split("|")[-1].split(":")[-1] if not pd.isnull(v) else np.nan

    ),

    adult_occupancy=stays.occupancy.map(

        lambda v: float(v.split(" ")[0]) if not pd.isnull(v) else np.nan

    ),

    child_occupancy=stays.occupancy.map(

        lambda v: float(v.split(" ")[-2]) if not pd.isnull(v) else np.nan

    ),

)



# Clean up:

# -- Remove fields that are not useful.

# -- NaN-ify bad response and acceptance entries.

stays = (stays.drop(['sitename', 'uniq_id', 'query_time_stamp', 

                     'occupancy', 'country'], axis='columns'))

stays.response_time.loc[

    stays.response_time.map(lambda v: pd.notnull(v) and "~" in v)

] = np.nan

stays.acceptance_rate = stays.acceptance_rate.map(

    lambda v: v if pd.isnull(v) else v if v.isdigit() else np.nan

)



# Drop service values to Verified and Not verified.

stays = stays.assign(

    service_value=stays.service_value.map(

        lambda v: np.nan if pd.isnull(v) else v if v in ['Not Verified', 'Verified'] else np.nan

    )

)



# Create columns for the 12 most commonly offered amenities.

import itertools



top_amenities = pd.Series(

    list(itertools.chain(*(stays

                               .amenities

                               .fillna("")

                               .map(lambda f: [am.strip() for am in f.split("|")])

                               .values

                               .tolist())

                        ))

).value_counts().head(13).index.values

top_amenities = [am for am in top_amenities if am != ""]  # drop empty list



stays = stays.assign(

    temp=stays.amenities.fillna("").map(

        lambda f: [am.strip() for am in f.split("|")]

    )

)



for amenity in top_amenities:

    stays[amenity] = stays.temp.map(lambda l: amenity in l)

    

# Drop the temporary column.

stays = stays.drop('temp', axis='columns')
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")



f, axarr = plt.subplots(3, 2, figsize=(14, 12))

plt.suptitle('StayZilla Property Breakdown', fontsize=18)

f.subplots_adjust(hspace=0.5)



sns.kdeplot(stays.room_price, ax=axarr[0][0])

axarr[0][0].set_title("Room Price")



sns.kdeplot(stays.room_price.where(lambda v: v <= 5000), ax=axarr[0][1])

axarr[0][1].set_title("Room Price < 5000 Rupees (Detail)")



sns.countplot(stays.adult_occupancy, ax=axarr[1][0])

axarr[1][0].set_title("Adult Occupants Permitted")



sns.countplot(stays.child_occupancy, ax=axarr[1][1])

axarr[1][1].set_title("Child Occupants Permitted")



sns.countplot(stays.property_type, ax=axarr[2][0])

axarr[2][0].set_title("Property Types")



sns.countplot(stays.service_value, ax=axarr[2][1])

axarr[2][1].set_title("Verified Yes/No")



sns.despine()
import folium



stays_lat_long = stays.groupby('city').first().loc[:, ['longitude', 'latitude']].assign(

    n = stays.groupby('city').city.count()

)



m = folium.Map(

    location=[21.15, 79.09],

    zoom_start=4

)



max_n_stays = stays_lat_long.n.max()



stays_lat_long.apply(lambda ll: folium.Circle(radius=200000 * (ll.n / max_n_stays),

                                              location=[ll.latitude, ll.longitude],

                                              fill=True,

                                              color='indianred',

                                              popup=ll.name).add_to(m), axis='columns')



m
# Generate the count data.

amenity_counts = pd.Series(

    list(

        itertools.chain(

            *(stays

                  .amenities

                  .str

                  .split("|")

                  .map(lambda l: [] if isinstance(l, float) else l)

                  .tolist()

             )

        )

    )

).value_counts()



amenity_counts = amenity_counts.groupby(amenity_counts.index.str.strip()).sum()



# Plot it.

with sns.plotting_context("notebook"):

    amenity_counts.sort_values(ascending=False).head(10).plot.bar(

        fontsize=14, color='indianred'

    )

    sns.despine()

    

import matplotlib.pyplot as plt

plt.title('Top 10 Amenities Offered on StayZilla', fontsize=16)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")



f, axarr = plt.subplots(3, 4, figsize=(14, 12))

plt.suptitle('StayZilla Top 10 Amenities Price Factor Breakdown', fontsize=18)

f.subplots_adjust(hspace=0.5)



sns.boxplot('AC', 'room_price', data=stays.query('room_price < 5000'), ax=axarr[0][0])

axarr[0][0].set_title("Air Conditioning?")



sns.boxplot('Newspaper', 'room_price', data=stays.query('room_price < 5000'), ax=axarr[0][1])

axarr[0][1].set_title("Newspaper?")



sns.boxplot('Parking', 'room_price', data=stays.query('room_price < 5000'), ax=axarr[0][2])

axarr[0][2].set_title("Parking?")



sns.boxplot('Parking', 'room_price', data=stays.query('room_price < 5000'), ax=axarr[0][3])

axarr[0][3].set_title("WiFi?")



sns.boxplot('Parking', 'room_price', data=stays.query('room_price < 5000'), ax=axarr[1][0])

axarr[1][0].set_title("Card Payment?")



sns.boxplot('Elevator', 'room_price', data=stays.query('room_price < 5000'), ax=axarr[1][1])

axarr[1][1].set_title("Elevator?")



sns.boxplot('Pickup & Drop', 'room_price', data=stays.query('room_price < 5000'), ax=axarr[1][2])

axarr[1][2].set_title("Pickup & Dropoff?")



sns.boxplot('Free Breakfast', 'room_price', data=stays.query('room_price < 5000'), ax=axarr[1][3])

axarr[1][3].set_title("Free Breakfast")



sns.boxplot('Veg Only', 'room_price', data=stays.query('room_price < 5000'), ax=axarr[2][0])

axarr[2][0].set_title("Vegitarian Only")



sns.boxplot('Bar', 'room_price', data=stays.query('room_price < 5000'), ax=axarr[2][1])

axarr[2][1].set_title("Bar")



sns.despine()