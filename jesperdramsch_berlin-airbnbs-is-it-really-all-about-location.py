import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import folium
berlin_raw = pd.read_csv("../input/berlin-airbnb-data/listings.csv")
berlin_raw.neighbourhood_group = berlin_raw.neighbourhood_group.str.replace(" ", "")
berlin_raw.neighbourhood_group = berlin_raw.neighbourhood_group.str.replace("Charlottenburg-Wilm.", "Charlottenburg-Wilmersdorf")
berlin_airbnb = berlin_raw[berlin_raw.price > 0] # Price 0 seems buggy
berlin_airbnb = berlin_airbnb[berlin_airbnb.price < 1000] # To get around the long tail outliers
berlin_airbnb[["neighbourhood_group", "neighbourhood", "room_type"]] = berlin_airbnb[["neighbourhood_group", "neighbourhood", "room_type"]].astype("category")
berlin_airbnb.head()
berlin_airbnb.shape
berlin_airbnb.room_type.unique()
berlin_airbnb.neighbourhood_group.unique()
berlin_airbnb["price_norm"] = (berlin_airbnb.price-berlin_airbnb.price.min())/(berlin_airbnb.price.max()-berlin_airbnb.price.min())
berlin_lat = berlin_airbnb.latitude.mean()
berlin_long = berlin_airbnb.longitude.mean()
colors = ["#3333DD", "#B00000"]
berlin_map = folium.Map(location=[berlin_lat, berlin_long], zoom_start=11)

belin_boroughs = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
berlin_price = berlin_airbnb.groupby(by="neighbourhood_group").median().reset_index()

folium.Choropleth(
    geo_data=belin_boroughs,
    name='choropleth',
    data=berlin_price,
    columns=['neighbourhood_group', 'price'],
    key_on='feature.properties.name',
    fill_color='RdBu_r',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend = "Median Price (Euro)"
).add_to(berlin_map)

folium.LayerControl().add_to(berlin_map)

berlin_map

berlin_map = folium.Map(location=[berlin_lat, berlin_long], zoom_start=11)

belin_boroughs = "https://raw.githubusercontent.com/funkeinteraktiv/Berlin-Geodaten/master/berlin_bezirke.geojson"
berlin_price = berlin_airbnb.groupby(by="neighbourhood_group").mean().reset_index()

folium.Choropleth(
    geo_data=belin_boroughs,
    name='choropleth',
    data=berlin_price,
    columns=['neighbourhood_group', 'price'],
    key_on='feature.properties.name',
    fill_color='RdBu_r',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend = "Mean Average Price (Euro)"
).add_to(berlin_map)

folium.LayerControl().add_to(berlin_map)

berlin_map

sns.kdeplot(berlin_airbnb.price, shade=True, clip=(0, 500))
plt.title("Berlin AirBnB Price distribution")
sns.kdeplot(berlin_airbnb.price, bw=.1, shade=True, clip=(250, 500))
plt.title("Interesting Aside: At higher prices increments change to 50s")
berlin_airbnb.groupby(by="neighbourhood_group").id.count().plot(kind="bar", title="Number of offers per Neighbourhood")
features = ["neighbourhood_group", "room_type", "price", "number_of_reviews", "reviews_per_month"]
sns.pairplot(berlin_airbnb[features].sample(2000), hue="neighbourhood_group")

# Plotting the KDE Plot 
for hood in berlin_airbnb.neighbourhood_group.unique():
    sns.kdeplot(berlin_airbnb[berlin_airbnb.neighbourhood_group==hood].price, shade=False, clip=(0, 150), Label=hood)

  
plt.xlabel('Price') 
plt.ylabel('Probability Density') 
plt.title('AirBnB Price per Neighbourhood Berlin')
from folium import plugins
heatmap = folium.Map(location=[berlin_lat, berlin_long], zoom_start=11, tiles='Stamen Toner',)

# plot heatmap
heatmap.add_children(folium.plugins.HeatMap(berlin_airbnb[['latitude', 'longitude']].values, radius=15, cmap='viridis'))
heatmap
berlin_map = folium.Map(location=[berlin_lat, berlin_long], 
                        zoom_start=12, )

for index, berlin_row in berlin_airbnb[['latitude', 'longitude', 'neighbourhood_group', 'price']].query("price < 30").iterrows():
    label = folium.Popup(f'{index}\n{berlin_row["price"]:.2f} Euro')
    folium.CircleMarker(
        [berlin_row["latitude"], berlin_row["longitude"]],
        radius=5,
        popup=label,
        color="yellow",
        fill=True,
        fill_color="yellow",
        fill_opacity=0.7,
        parse_html=False).add_to(berlin_map)  
for index, berlin_row in berlin_airbnb[['latitude', 'longitude', 'neighbourhood_group', 'price']].query("price > 333").iterrows():
    label = folium.Popup(f'{index}\n{berlin_row["price"]:.2f} Euro')
    folium.CircleMarker(
        [berlin_row["latitude"], berlin_row["longitude"]],
        radius=5,
        popup=label,
        color="purple",
        fill=True,
        fill_color="purple",
        fill_opacity=0.7,
        parse_html=False).add_to(berlin_map)  

berlin_map
colors = ["purple", "green", "lightblue"]
berlin_airbnb.groupby(by="room_type").count().id.plot(kind="bar", color=colors)
sns.pairplot(berlin_airbnb[features].sample(2000), hue="room_type", palette=colors)
berlin_map = folium.Map(location=[berlin_lat, berlin_long], 
                        zoom_start=12, )

for index, berlin_row in berlin_airbnb[['latitude', 'longitude', 'neighbourhood_group', 'room_type', 'price']].sample(2222).iterrows():
    label = folium.Popup(f'{index}\n{berlin_row["room_type"]}\n{berlin_row["price"]:.2f} Euro')
    if "Entire" in berlin_row["room_type"]:
        color = colors[0]
    elif "Private" in berlin_row["room_type"]:
        color = colors[1]
    else:
        color = colors[2]
    folium.CircleMarker(
        [berlin_row["latitude"], berlin_row["longitude"]],
        radius=5,
        popup=label,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        parse_html=False).add_to(berlin_map)  

berlin_map
heatmap = folium.Map(location=[berlin_lat, berlin_long], zoom_start=12, tiles='Stamen Toner',)

# plot heatmap
heatmap.add_children(folium.plugins.HeatMap(berlin_airbnb[berlin_airbnb.room_type == "Entire home/apt"][['latitude', 'longitude']].values, radius=15))
heatmap

heatmap = folium.Map(location=[berlin_lat, berlin_long], zoom_start=12, tiles='Stamen Toner',)

# plot heatmap
heatmap.add_children(folium.plugins.HeatMap(berlin_airbnb[berlin_airbnb.room_type == "Private room"][['latitude', 'longitude']].values, radius=15))
heatmap

heatmap = folium.Map(location=[berlin_lat, berlin_long], zoom_start=12, tiles='Stamen Toner',)

# plot heatmap
heatmap.add_children(folium.plugins.HeatMap(berlin_airbnb[berlin_airbnb.room_type == "Shared room"][['latitude', 'longitude']].values, radius=15))
heatmap
# Plotting the KDE Plot 
for i, room in enumerate(berlin_airbnb.room_type.unique()):
    sns.kdeplot(berlin_airbnb[berlin_airbnb.room_type==room].price, shade=False, clip=(0, 200), Label=room, color=colors[i])
  
plt.xlabel('Price') 
plt.ylabel('Probability Density') 
plt.title('AirBnB Price per Room Type')
pd.pivot_table(berlin_airbnb, index="neighbourhood_group", columns="room_type", values='id', aggfunc='count').plot(kind = 'bar', color=colors)
df_train = pd.get_dummies(berlin_airbnb[["neighbourhood_group", "room_type"]])
target = berlin_airbnb["price"]
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
scores = []
mse = []
skf = StratifiedKFold(n_splits=10)
for train, test in skf.split(df_train, target):
    clf = LinearRegression()
    clf.fit(df_train.iloc[train], target.iloc[train])
    scores.append(clf.score(df_train.iloc[test], target.iloc[test]))
    mse.append(mean_squared_error(target.iloc[test], clf.predict(df_train.iloc[test])))

print("Average R2 score: \t", np.mean(scores),)
print("Average MSE: \t\t", np.mean(mse), "\n")

r = permutation_importance(clf, df_train.iloc[test], target.iloc[test],
                           n_repeats=30,
                           random_state=0)

print("Normalized importances")
sorted_idx = r.importances_mean.argsort()
max_i = sorted_idx[-1]

rel_max = r.importances_mean[max_i]

for i in sorted_idx[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{df_train.columns[i]:<50}"
              f"{r.importances_mean[i]/rel_max:.3f}"
              f" +/- {r.importances_std[i]/rel_max:.3f}")
scores = []
mse = []
skf = StratifiedKFold(n_splits=10)
for train, test in skf.split(df_train, target):
    clf = RandomForestRegressor()
    clf.fit(df_train.iloc[train], target.iloc[train])
    scores.append(clf.score(df_train.iloc[test], target.iloc[test]))
    mse.append(mean_squared_error(target.iloc[test], clf.predict(df_train.iloc[test])))

print("Average R2 score: \t", np.mean(scores),)
print("Average MSE: \t\t", np.mean(mse), "\n")

r = permutation_importance(clf, df_train.iloc[test], target.iloc[test],
                           n_repeats=30,
                           random_state=0)

print("Normalized importances")
sorted_idx = r.importances_mean.argsort()
max_i = sorted_idx[-1]

rel_max = r.importances_mean[max_i]

for i in sorted_idx[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{df_train.columns[i]:<50}"
              f"{r.importances_mean[i]/rel_max:.3f}"
              f" +/- {r.importances_std[i]/rel_max:.3f}")
mse = []
scores = []
skf = StratifiedKFold(n_splits=10)
enc = OrdinalEncoder()

df_rf = berlin_airbnb[["neighbourhood_group", "room_type"]]

enc.fit(df_rf)

for train, test in skf.split(df_rf, target):
    clf = RandomForestRegressor()
    clf.fit(enc.transform(df_rf.iloc[train]), target.iloc[train])
    scores.append(clf.score(enc.transform(df_rf.iloc[test]), target.iloc[test]))
    mse.append(mean_squared_error(target.iloc[test], clf.predict(enc.transform(df_rf.iloc[test]))))

print("Average R2 score: \t", np.mean(scores),)
print("Average MSE: \t\t", np.mean(mse), "\n")

r = permutation_importance(clf, enc.transform(df_rf.iloc[test]), target.iloc[test],
                           n_repeats=30,
                           random_state=0)

print("Normalized importances")
sorted_idx = r.importances_mean.argsort()
max_i = sorted_idx[-1]

rel_max = r.importances_mean[max_i]

for i in sorted_idx[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{df_rf.columns[i]:<50}"
              f"{r.importances_mean[i]/rel_max:.3f}"
              f" +/- {r.importances_std[i]/rel_max:.3f}")
