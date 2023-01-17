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
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("../input/covidTR.csv")
data
data.info()
data.describe().T
plt.subplots(figsize = (20,10))
plt.plot(data.Tarih, data.TopVaka, color = "orange", linewidth = 2 , label = "Toplam Vaka Sayisi")
plt.plot(data.Tarih, data.TopIyiles, color = "green", linewidth = 2, label = "Toplam Iyilesen Sayisi")
plt.plot(data.Tarih, data.TestSay, color = "blue", linewidth = 2, label = "Gunluk Test Sayisi")
plt.plot(data.Tarih, data.TopOlen, color = "red", linewidth = 2, label = "Toplam Olen Sayisi" )
plt.legend()

plt.scatter(data.Tarih, data.TopVaka, color ="orange", alpha = 1)
plt.scatter(data.Tarih, data.TopIyiles, color ="green", alpha = 1)
plt.scatter(data.Tarih, data.TestSay, color ="blue", alpha = 1)
plt.scatter(data.Tarih, data.TopOlen, color ="red", alpha = 1)

plt.xticks(rotation= 75, fontsize = 12)
plt.title("Turkiyenin nCOV-19 Tablosu", fontsize = 16)
plt.xlabel("Tarih", fontsize = 14)
plt.ylabel("Kisi Sayisi", fontsize = 14)
plt.grid()
plt.subplots(figsize = (17,12))
sns.pointplot(x = data.Tarih, y = data.VakaSay, color = "orange", linewidth = 1, alpha = 1)

plt.xticks(rotation= 75, fontsize = 12)
plt.title("Vaka Sayisi (Gunluk)", fontsize = 16)
plt.xlabel("Tarih", fontsize = 14)
plt.ylabel("Kisi Sayisi", fontsize = 14)
plt.grid()
plt.subplots(figsize = (17,12))
sns.pointplot(x = data.Tarih, y = data.OlenSay, color = "red", linewidth = 1, alpha = 1)

plt.xticks(rotation= 75, fontsize = 12)
plt.title("Vefat Eden Kisi Sayisi (Gunluk)", fontsize = 16)
plt.xlabel("Tarih", fontsize = 14)
plt.ylabel("Kisi Sayisi", fontsize = 14)
plt.grid()
plt.subplots(figsize = (17,12))
sns.pointplot(x = data.Tarih, y = data.IyilesSay, color = "green", linewidth = 1, alpha = 1)

plt.xticks(rotation= 75, fontsize = 12)
plt.title("Iyilesen Kisi Sayisi (Gunluk)", fontsize = 16)
plt.xlabel("Tarih", fontsize = 14)
plt.ylabel("Kisi Sayisi", fontsize = 14)
plt.grid()
plt.subplots(figsize = (17,12))
sns.pointplot(x = data.Tarih, y = data.TestSay, color = "blue", linewidth = 1, alpha = 1)

plt.xticks(rotation= 75, fontsize = 12)
plt.title("Yapilan Test Sayisi (Gunluk)", fontsize = 16)
plt.xlabel("Tarih", fontsize = 14)
plt.ylabel("Kisi Sayisi", fontsize = 14)
plt.grid()
plt.subplots(figsize = (15,10))
plt.plot(data.Tarih, data.TopEntub, color = "yellow", linewidth = 2 , label = "Toplam Entube Sayisi")
plt.plot(data.Tarih, data.TopYB, color = "purple", linewidth = 2, label = "Toplam Yogun Bakim Sayisi"),
plt.legend()

plt.scatter(data.Tarih, data.TopEntub, color ="yellow", alpha = 1)
plt.scatter(data.Tarih, data.TopYB, color ="purple", alpha = 1)

plt.xticks(rotation= 75, fontsize = 12)
plt.title("Toplam Entube ve Yogun Bakimdaki Hasta Sayilari", fontsize = 16)
plt.xlabel("Tarih", fontsize = 14)
plt.ylabel("Kisi Sayisi", fontsize = 14)
plt.grid()
data["Aktif_Hasta"] = (data["TopVaka"] - (data["TopOlen"] + data["TopIyiles"]))
data["Pasif_Hasta"] = (data["TopIyiles"] + data["TopOlen"])

data
plt.subplots(figsize = (15,10))
plt.plot(data.Tarih, data.Aktif_Hasta, color = "pink", label = "Aktif Hasta", linewidth = 3)
plt.plot(data.Tarih, data.Pasif_Hasta, color = "yellow", label = "Pasif Hasta", linewidth = 3)
plt.legend()

plt.scatter(data.Tarih, data.Aktif_Hasta, color = "pink", alpha = 1)
plt.scatter(data.Tarih, data.Pasif_Hasta, color = "yellow", alpha = 1)

plt.xticks(rotation= 75, fontsize = 12)
plt.title("Atkif ve Pasif Hasta Sayisinin Degisim Frekansi", fontsize = 16)
plt.xlabel("Tarih", fontsize = 14)
plt.ylabel("Kisi Sayisi", fontsize = 14)
plt.grid()
plt.subplots(figsize = (17,12))
sns.pointplot(x = data.Tarih, y = data.TopTest, color = "black")

plt.xticks(rotation= 75, fontsize = 12)
plt.title("Toplam Yapilan Test Sayisi", fontsize = 16)
plt.xlabel("Tarih", fontsize = 14)
plt.ylabel("Kisi Sayisi", fontsize = 14)
plt.grid()