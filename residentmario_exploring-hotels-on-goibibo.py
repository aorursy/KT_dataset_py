import pandas as pd

pd.set_option('max_columns', None)

hotels = pd.read_csv("../input/goibibo_com-travel_sample.csv")

hotels.head(3)
import numpy as np



# Helper functions for encoding.

def split_piped_list(srs, col):

    try:

        ret = [r.split("::")[-1] for r in srs[col].split("|")]

        ret = [float(r) if len(r) > 0 else np.nan for r in ret]

        return ret

    except AttributeError:

        return np.nan



# Encode the aggregated 5-star review categories into columns.

ratings = pd.DataFrame(data=hotels.apply(

                                lambda h: split_piped_list(h, 'site_stay_review_rating'), 

                                                           axis='columns').tolist(),

                       columns=['service_quality_rating', 'amenities_rating', 

                                'food_and_drinks_rating', 'value_for_money_rating', 

                                'location_rating', 'cleanliness_rating'])

hotels = hotels.join(ratings)



# Encode the reviews column into separate columns.

review_counts = pd.DataFrame(

    data=(

        hotels

            .apply(lambda h: split_piped_list(h, 'review_count_by_category'), axis='columns')

            .map(lambda v: [0, 0, 0] if isinstance(v, float) else v).tolist()

    ),

    columns=['positive_reviews_total', 'critical_reviews_total', 'reviews_with_images_total']

)

hotels = hotels.join(review_counts)



hotels = hotels.drop(['country', 'sitename', 'review_count_by_category', 

                      'site_stay_review_rating'], axis='columns')

hotels = hotels.drop(hotels['room_count'].argmax())  # bad entry
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")



f, axarr = plt.subplots(2, 2, figsize=(14, 8))

plt.suptitle('Goibibo Hotel Breakdown', fontsize=18)



sns.kdeplot(hotels['site_review_rating'], ax=axarr[0][0])

sns.kdeplot(hotels['site_review_count'], ax=axarr[0][1])

sns.countplot(hotels['hotel_star_rating'], ax=axarr[1][0])

sns.kdeplot(hotels['room_count'], ax=axarr[1][1])



sns.despine()
hotels.iloc[hotels['room_count'].argmax()]['pageurl']
sns.jointplot(hotels.hotel_star_rating, hotels.site_review_rating)
f, axarr = plt.subplots(1, 2, figsize=(14, 4))



sns.violinplot(hotels.positive_reviews_total / (hotels.positive_reviews_total + hotels.critical_reviews_total), ax=axarr[0], color='lightgreen')

sns.violinplot(hotels.reviews_with_images_total / (hotels.positive_reviews_total + hotels.critical_reviews_total), ax=axarr[1], color='lightgreen')



axarr[0].set_title("Ratio of Positive Reviews to Negative Ones")

axarr[1].set_title("Ratio of Reviews with Images")

plt.suptitle('Goibibo Hotel Review Ratios', fontsize=18, y=1.08)



sns.despine()
_ = (hotels[[col for col in hotels.columns if "_rating" in col]]

          .dropna()

          .sample(100))

_.columns = pd.Series([col for col in hotels.columns if "_rating" in col]).str.replace("_rating", "")



sns.pairplot(_, size=1.4)
hotels_lat_long = hotels.groupby('city').first().loc[:, ['longitude', 'latitude']].assign(

    n_hotels = hotels.groupby('city').area.count(),

    n_reviews = (hotels.assign(site_review_count=hotels.site_review_count.fillna(0))\

                 .groupby('city').site_review_count.sum())

)
import folium



m = folium.Map(

    location=[21.15, 79.09],

    zoom_start=4

)



max_n_hotels = hotels_lat_long.n_hotels.max()



hotels_lat_long.apply(lambda ll: folium.Circle(radius=200000 * (ll.n_hotels / max_n_hotels),

                                               location=[ll.latitude, ll.longitude],

                                               fill=True,

                                               color='black',

                                               popup=ll.name).add_to(m), axis='columns')

m
import itertools



top_amenities = pd.Series(

    list(itertools.chain(*hotels['room_facilities']\

                             .fillna("")\

                             .map(lambda f: [am.strip() for am in f.split("|")])\

                             .values\

                             .tolist()))).value_counts().head(12).index.values

temp = hotels.assign(amenities=hotels['room_facilities'].fillna("").map(

        lambda f: [am.strip() for am in f.split("|")]))



for amenity in top_amenities:

    temp[amenity] = temp.amenities.map(lambda l: amenity in l)
top_amenities
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")



f, axarr = plt.subplots(3, 4, figsize=(14, 8))

f.subplots_adjust(hspace=1)



sns.factorplot(x='Room Service', y='hotel_star_rating', data=temp.dropna(), ax=axarr[0][0])

axarr[0][0].set_title("Room Service?")



sns.factorplot(x='Basic Bathroom Amenities', y='site_review_rating', 

               data=temp.dropna(), ax=axarr[0][1])

axarr[0][1].set_title("Basic Bathroom Amenities?")



sns.factorplot(x='Hot / Cold Running Water', y='site_review_rating', 

               data=temp.dropna(), ax=axarr[0][2])

axarr[0][2].set_title("Hot / Cold Running Water?")



sns.factorplot(x='Housekeeping', y='site_review_rating', 

               data=temp.dropna(), ax=axarr[0][3])

axarr[0][3].set_title("Housekeeping?")



sns.factorplot(x='Ceiling Fan', y='site_review_rating', 

               data=temp.dropna(), ax=axarr[1][0])

axarr[1][0].set_title("Ceiling Fan?")



sns.factorplot(x='Air Conditioning', y='site_review_rating', 

               data=temp.dropna(), ax=axarr[1][1])

axarr[1][1].set_title("Air Conditioning?")



sns.factorplot(x='Cable / Satellite / Pay TV available', y='site_review_rating', 

               data=temp.dropna(), ax=axarr[1][2])

axarr[1][2].set_title("Cable / Satellite / Pay TV?")



sns.factorplot(x='Attached Bathroom', y='site_review_rating', 

               data=temp.dropna(), ax=axarr[1][3])

axarr[1][3].set_title("Attached Bathroom?")



sns.factorplot(x='Telephone', y='site_review_rating', 

               data=temp.dropna(), ax=axarr[2][0])

axarr[2][0].set_title("Telephone?")



sns.factorplot(x='Mirror', y='site_review_rating', 

               data=temp.dropna(), ax=axarr[2][1])

axarr[2][1].set_title("Mirror?")



sns.factorplot(x='TV', y='site_review_rating', 

               data=temp.dropna(), ax=axarr[2][2])

axarr[2][2].set_title("TV?")



sns.factorplot(x='Desk in Room', y='site_review_rating', 

               data=temp.dropna(), ax=axarr[2][3])

axarr[2][3].set_title("Desk in Room?")