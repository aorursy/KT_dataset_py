
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Kurulum Tamamlandı.")


weatherww2_filepath = "../input/weatherww2/Summary of Weather.csv"
weatherww2_data = pd.read_csv(weatherww2_filepath,  index_col="Date", parse_dates=True)

weatherww2_data.describe()
weatherww2_data.head(10)

plt.figure(figsize=(12,6))
sns.lineplot(data=weatherww2_data["MeanTemp"])
plt.title("1940-1945 Yılları Arasındaki Günlük Ortalama Sıcaklıklar")

plt.figure(figsize=(12,6))
sns.lineplot(data=weatherww2_data["MaxTemp"])
plt.title("1940-1945 Yılları Arasındaki Günlük Maksimum Sıcaklıklar")

plt.figure(figsize=(12,6))
sns.lineplot(data=weatherww2_data["MinTemp"])
plt.title("1940-1945 Yılları Arasındaki Günlük Maksimum Sıcaklıklar")