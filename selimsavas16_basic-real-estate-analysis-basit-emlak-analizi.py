import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

data = pd.read_csv("../input/kc_house_data.csv")


plt.figure(figsize=(25,5))

data.grade.plot(kind="line", color="g", label="Value", linewidth=0.1, alpha=1,linestyle="-.")

plt.xlabel("x ")

plt.ylabel("y ")

plt.title("Line Plot")

plt.show()
# Banyo sayısının fiyat üzerindeki etkisi:

data.plot(kind="scatter",x="bathrooms", y="price", alpha=0.5, color="red")

plt.xlabel("Banyo Sayısı")

plt.ylabel("Ev Fiyatı")

plt.title("Banyo sayısına göre fiyat oranlaması")
# Evimizin yıla göre sıklığının hesaplanması:

data.lat.plot(kind="hist", bins=50,)

plt.xlabel("Lat (Yıl)")

plt.ylabel("Frequency (Sıklık)")

plt.show()
# Evin bulunduğu alanın yaşam standartlarının, fiyat üzerindeki etkisi

plt.figure(figsize=(9,5))

plt.scatter(x=data["price"],y=data["sqft_living"], alpha=0.5, color="darkgreen")

plt.xlabel("price ($)")

plt.ylabel("living standards")

plt.show()
# Yatak odasının, kat sayısına oranlığında ortalamanın altı ve üstü olduğunu gösteren tablo

bedoranfloors= sum(data.bedrooms)/sum(data.floors)

print("Yatak odasının, kat sayısına olan oranı: ",bedoranfloors)



data["the proportion"]= ["high" if i>bedoranfloors else "low" for i in data.bedrooms]

data.loc[:20,["id","bedrooms","the proportion"]]