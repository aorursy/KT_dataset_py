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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



data = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data.head()
round(data.isnull().sum() / len(data), 4)
data.corr().style.background_gradient(cmap='coolwarm')
data["neighbourhood_group"].value_counts()
bölge = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]



manhattan = data.loc[data["neighbourhood_group"] == bölge[0]]

brooklyn = data.loc[data["neighbourhood_group"] == bölge[1]]

queens = data.loc[data["neighbourhood_group"] == bölge[2]]

bronx = data.loc[data["neighbourhood_group"] == bölge[3]]

staten_island = data.loc[data["neighbourhood_group"] == bölge[4]]



ort_ücretler = [manhattan["price"].mean(), brooklyn["price"].mean(),

               queens["price"].mean(), bronx["price"].mean(),

               staten_island["price"].mean()]



ort_bölge = {

    

    "manhattan": round(ort_ücretler[0]),

    "brooklyn": round(ort_ücretler[1]),

    "queens": round(ort_ücretler[2]),

    "bronx": round(ort_ücretler[3]),

    "staten_island": round(ort_ücretler[4])

}



ort_bölge
sns.barplot(x = bölge, y = ort_ücretler)
sns.scatterplot(x = data["latitude"], y = data["longitude"], data = data,

               hue = data["neighbourhood_group"])
manh = []

manhattan_index = manhattan.index



for i in manhattan_index:

    

    if manhattan["price"][i] > 350:

        manh.append(1)

    else:

        manh.append(0)



manhattan["yuksek_ucret"] = manh

sns.scatterplot(x = manhattan["latitude"], y = manhattan["longitude"],

               data = manhattan, hue = manhattan["yuksek_ucret"])
br = []

brooklyn_index = brooklyn.index



for i in brooklyn_index:

    

    if brooklyn["price"][i] > 200:

        br.append(1)

    else:

        br.append(0)



brooklyn["yuksek_ucret"] = br

sns.scatterplot(x = brooklyn["latitude"], y = brooklyn["longitude"],

               data = brooklyn, hue = brooklyn["yuksek_ucret"])
que = []

queens_index = queens.index



for i in queens_index:

    

    if queens["price"][i] > 200:

        que.append(1)

    else:

        que.append(0)



queens["yuksek_ucret"] = que

sns.scatterplot(x = queens["latitude"], y = queens["longitude"],

               data = queens, hue = queens["yuksek_ucret"])
bro = []

bronx_index = bronx.index



for i in bronx_index:

    

    if bronx["price"][i] > 200:

        bro.append(1)

    else:

        bro.append(0)



bronx["yuksek_ucret"] = bro

sns.scatterplot(x = bronx["latitude"], y = bronx["longitude"],

               data = bronx, hue = bronx["yuksek_ucret"])
sta = []

staten_island_index = staten_island.index



for i in staten_island_index:

    

    if staten_island["price"][i] > 150:

        sta.append(1)

    else:

        sta.append(0)



staten_island["yuksek_ucret"] = sta

sns.scatterplot(x = staten_island["latitude"], y = staten_island["longitude"],

               data = staten_island, hue = staten_island["yuksek_ucret"])
df = pd.concat([manhattan, queens, bronx, staten_island, brooklyn], axis = 0)

sns.scatterplot(x = df["latitude"], y = df["longitude"],

               data = df, hue = df["yuksek_ucret"])
print("Ortalma yorum sayısı: ", round(data["number_of_reviews"].mean()))
data["number_of_reviews"].describe()
az_yorum = data.loc[data["number_of_reviews"] < 6]

cok_yorum = data.loc[data["number_of_reviews"] > 5]



yorum_ucret = [az_yorum["price"].mean(), cok_yorum["price"].mean()]

yorum = ["Az_Yorum", "Çok_Yorum"]



sns.barplot(x = yorum, y = yorum_ucret)
round(pd.pivot_table(data = data, values = ["price"], index = ["room_type"],

                         aggfunc = "mean"))



manhattan_room = round(pd.pivot_table(data = manhattan, values = ["price"], 

                                columns = ["neighbourhood"], index = ["room_type"], 

                                aggfunc = "mean", fill_value = 0))

manhattan_room
yabancı = [18497228,151728547,133425456,79269209,112439306,55197031,120047243,138349238,201403610,201403610,201403610,201403610,201403610,269078124,

269078124,269078124,232169947,42804325,122963021,196058543,196058543,196058543,196058543,196058543,

24755691,125492013,125492013,107234799,265422938,265422938,25915648,25915648,20630497,97044757,97044757,229278227,270327975,270327975,39394883,

224900105,253640919,236355800,43128266,43128266,43128266,69366248,61303786,20019653,32392762,16828799,204006071,51087754,215116298,

28996758,245014802,73655921,267933377,8905097,20985328,20985328,43264429,269297229,95642648,95642648,269308830,269822492,218511760,118839768,93315723,269837702,

104475208,16562278,129324501,224867226,130013537,141400974,115567902,6725061,7107154,23046080,4498389,26819014,56177749,157093664,157093664,

9352452,1495881,24740606,67211684,40317143,40317143,210779293,210779293,4439578,13422723,94820583,1510920,21653460,21653460,21940002,264618293,17898332,88827816,40219037,4665764,17075886,195803,1319462,1319462,51550484,51550484,250771864,

79489045,18217011,40225443,14729523,21492250,69131589,7252221,7252221,23684,20161989,67923313,3102648,21279472,123646786,123646786,34075689,34075689,34075689,21383358,40696448,52529356,29599415,49121324,

231707114,34556469,34556469,24707177,5953913,175166282,960013,10221694,127784473,10238618,11256721,254119216,36522433,7558452,7558452,88329273,19704368,12459436,12459436,163152195,85526404,3879852,152872412,

51329030]



yabancı_mı = []



for i in data["host_id"]:

    if i in yabancı:

        yabancı_mı.append(1)

    else:

        yabancı_mı.append(0)





df = data.copy()

df["yabancı_mı"] = yabancı_mı



print("yabancı ort: ", round(df.loc[df["yabancı_mı"] == 1]["price"].mean()),

      "yerli ort: ", round(df.loc[df["yabancı_mı"] == 0]["price"].mean()))

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor



data = data.drop(["id", "name", "host_name", "last_review"], axis = 1)

cols = ["neighbourhood_group", "neighbourhood", "room_type"]



le = LabelEncoder()

for i in cols:

    data[i] = le.fit_transform(data[i])



x = data.drop(["price"], axis = 1)

y = data["price"]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)



xgb = XGBRegressor()

xgb.fit(x_train, y_train)

xgb_tahmin = xgb.predict(x_test)



rmse = np.sqrt(mean_squared_error(y_test, xgb_tahmin))

r2 = r2_score(y_test, xgb_tahmin)



print("RMSE: ", round(rmse), "R2 Skor: ", round(r2,4))
sonuc = pd.DataFrame({'Gerçek Değerler': np.array(y_test).flatten(), 'Tahminler': xgb_tahmin.flatten()})

sonuc.head(10)
print("Teşekkürler ! :)")