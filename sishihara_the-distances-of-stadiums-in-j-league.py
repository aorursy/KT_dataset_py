!pip install japanize-matplotlib
from itertools import combinations



import pandas as pd

import numpy as np

import folium

import japanize_matplotlib

import seaborn as sns

sns.set()



import matplotlib

font = {'family' : 'IPAexGothic'}

matplotlib.rc('font', **font)
# https://www.jsgoal.jp/stadium/

# https://sportiva.shueisha.co.jp/clm/football/jleague_other/2020/07/03/fcfmfc/index_2.php

df = [

    ['札幌', 43.015019, 141.410005, '横浜FC、鹿島、湘南、仙台'],

    ['仙台', 38.319158, 140.881857, '湘南、横浜FC、柏'],

    ['鹿島', 35.991979, 140.64043, '川崎、浦和、湘南'],

    ['浦和', 35.903105, 139.717598, '仙台、FC東京、横浜FC'],

    ['柏', 35.848548, 139.975096, '川崎、浦和'],

    ['FC東京', 35.66427, 139.527151, '柏、横浜FM、札幌、鹿島'],

    ['川崎', 35.585808, 139.652722, 'FC東京、横浜FC、仙台'],

    ['横浜FM', 35.509946, 139.606394, '浦和、鹿島、札幌'],

    ['横浜FC', 35.469155, 139.603819, '柏、横浜FM'],

    ['湘南', 35.343579, 139.341179, '横浜FM、柏、川崎'],

    ['清水', 34.984656, 138.481246, 'Ｃ大阪、神戸、鳥栖'],

    ['名古屋', 35.084561, 137.17092, '清水、Ｃ大阪、大分、広島'],

    ['Ｇ大阪', 34.802964, 135.537886, '名古屋、清水、神戸'],

    ['Ｃ大阪', 34.61409, 135.51859, 'Ｇ大阪、広島、鳥栖'],

    ['神戸', 34.656811, 135.169623, '鳥栖、大分、Ｃ大阪'],

    ['広島', 34.440694, 132.394417, '神戸、鳥栖、Ｇ大阪'],

    ['鳥栖', 33.371674, 130.520248, '大分、名古屋'],

    ['大分', 33.200786, 131.6575, '広島、Ｇ大阪、清水']

]



df = pd.DataFrame(df)

df.columns = ['team', 'lat', 'lon', 'away']

df.head()
df.team
LAT = df.lat.mean()

LNG = df.lon.mean()



m = folium.Map(location=[LAT, LNG], zoom_start=10)

for i, (lat, lon) in enumerate(zip(df.lat, df.lon)):

    if i > 9:

        clr = 'red'

    else:

        clr = 'blue'

    folium.Marker(location=[lat, lon],

                  popup="/".join([str(lat), str(lon)]),

                  tooltip=str(lat) + "_" + str(lon),

                  icon=folium.Icon(color=clr)).add_to(m)

m
# https://github.com/nadare881/mynabi_x_signate_student_cup_nadare/blob/master/final_submission/final_scripts/preprocessing.py

def distance_processing(df):

    def hubeny(lng1, lat1, lng2, lat2):

        # http://www.trail-note.net/tech/calc_distance/

        # 座標から距離を計算する



        # WGS84

        Rx = 6378137.000

        Ry = 6356752.314140



        Dx = (lat1 - lat2)/360*2*np.pi

        Dy = (lng1 - lng2)/360*2*np.pi



        P = (lng1 + lng2)/360*np.pi



        E = np.sqrt((np.power(Rx, 2) - np.power(Ry, 2))/np.power(Rx, 2))

        W = np.sqrt(1-np.power(E, 2)*np.power(np.sin(P), 2))

        M = (Rx*(1-np.power(E, 2)))/(np.power(W, 3))

        N = Rx/W



        D = np.sqrt(np.power(Dy*M, 2) + np.power(Dx*N*np.cos(P), 2))

        return D



    results = []

    for A, B in combinations(df[["team", "lat", "lon"]].values, 2):

        dist = hubeny(A[1], A[2], B[1], B[2])

        results.append({"home_team": A[0],

                        "away_team": B[0],

                        "distance": dist * 2})

        results.append({"home_team": B[0],

                        "away_team": A[0],

                        "distance": dist * 2})



    distance_pair_df = pd.DataFrame(results)



    return distance_pair_df





distance_pair_df = distance_processing(df)

distance_pair_df
dists = []

for team, away in zip(df.team, df.away):

    aways = away.split('、')

    for a in aways:

        dists.append([team, a, distance_pair_df.query(f'home_team=="{team}" and away_team=="{a}"')['distance'].values[0]])



dists = pd.DataFrame(dists)

dists.columns = ['team', 'away', 'dist']

dists.head()
dists.pivot_table(values=['dist'], index=['team'], columns=['away'], aggfunc='sum').droplevel(None, axis=1).plot.bar(stacked=True, figsize=(10, 5), colormap='tab20').legend(loc='lower right', bbox_to_anchor=(1.2, 0))