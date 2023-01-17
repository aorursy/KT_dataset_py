import pandas as pd

import folium
d = pd.read_csv('../input/folium/folium.csv')
d.head()
d.rename(columns = {'18810 Densmore Ave N': 'Locations'}, inplace = True)
d
d = d.iloc[:1001]
d
d = d.dropna().reset_index(drop = True)

d
m = folium.Map(location = [40.730610, -73.935242],zoom_start = 5)

for i in range(len(d.index)):

        folium.Marker(location=[d.iloc[i,1], d.iloc[i,2]], popup=d.iloc[i,0], tooltip = "Click for more").add_to(m)

m