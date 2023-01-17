import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px
df=pd.read_csv("../input/weather-history/weatherHistory.csv")
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df=df.set_index('Formatted Date')
df=df.resample('MS').mean()
apt=df['Apparent Temperature (C)']
hmd=df['Humidity']
plt.figure(figsize=(12,9))

plt.title("Analysis of Humidity and Temperature")

plt.plot(hmd,label="Average Humidity")

plt.plot(apt,label="Average Apparent Temperature")

plt.xlabel("Years")

plt.legend(loc=(1.01,0.8))

plt.show()
df1 = df[df.index.month==1]

hdt=df1['Humidity']

atm=df1['Apparent Temperature (C)']

plt.title("January Monthwise Stats",fontsize=17)

plt.plot(hdt,label="Average Humidity",marker=".")

plt.plot(atm,label="Average Apparent Temperature",marker='.')

plt.legend(loc=(1.02,0.8))

plt.show()
df1 = df[df.index.month==2]

hdt=df1['Humidity']

atm=df1['Apparent Temperature (C)']

plt.title("February Monthwise Stats",fontsize=17)

plt.plot(hdt,label="Average Humidity",marker=".")

plt.plot(atm,label="Average Apparent Temperature",marker=".")

plt.legend(loc=(1.02,0.8))

plt.show()
df1 = df[df.index.month==3]

hdt=df1['Humidity']

atm=df1['Apparent Temperature (C)']

plt.title("March Monthwise Stats",fontsize=17)

plt.plot(hdt,label="Average Humidity",marker=".")

plt.plot(atm,label="Average Apparent Temperature",marker=".")

plt.legend(loc=(1.02,0.8))

plt.show()
df1 = df[df.index.month==4]

hdt=df1['Humidity']

atm=df1['Apparent Temperature (C)']

plt.title("April Monthwise Stats",fontsize=17)

plt.plot(hdt,label="Average Humidity",marker=".")

plt.plot(atm,label="Average Apparent Temperature",marker='.')

plt.legend(loc=(1.02,0.8))

plt.show()
df1 = df[df.index.month==5]

hdt=df1['Humidity']

atm=df1['Apparent Temperature (C)']

plt.title("May Monthwise Stats",fontsize=17)

plt.plot(hdt,label="Average Humidity",marker=".")

plt.plot(atm,label="Average Apparent Temperature",marker='.')

plt.legend(loc=(1.02,0.8))

plt.show()
df1 = df[df.index.month==6]

hdt=df1['Humidity']

atm=df1['Apparent Temperature (C)']

plt.title("June Monthwise Stats",fontsize=17)

plt.plot(hdt,label="Average Humidity",marker=".")

plt.plot(atm,label="Average Apparent Temperature",marker='.')

plt.legend(loc=(1.02,0.8))

plt.show()
df1 = df[df.index.month==7]

hdt=df1['Humidity']

atm=df1['Apparent Temperature (C)']

plt.title("July Monthwise Stats",fontsize=17)

plt.plot(hdt,label="Average Humidity",marker=".")

plt.plot(atm,label="Average Apparent Temperature",marker='.')

plt.legend(loc=(1.02,0.8))

plt.show()
df1 = df[df.index.month==8]

hdt=df1['Humidity']

atm=df1['Apparent Temperature (C)']

plt.title("August Monthwise Stats",fontsize=17)

plt.plot(hdt,label="Average Humidity",marker=".")

plt.plot(atm,label="Average Apparent Temperature",marker='.')

plt.legend(loc=(1.02,0.8))

plt.show()
df1 = df[df.index.month==9]

hdt=df1['Humidity']

atm=df1['Apparent Temperature (C)']

plt.title("September Monthwise Stats",fontsize=17)

plt.plot(hdt,label="Average Humidity",marker=".")

plt.plot(atm,label="Average Apparent Temperature",marker='.')

plt.legend(loc=(1.02,0.8))

plt.show()
df1 = df[df.index.month==10]

hdt=df1['Humidity']

atm=df1['Apparent Temperature (C)']

plt.title("October Monthwise Stats",fontsize=17)

plt.plot(hdt,label="Average Humidity",marker=".")

plt.plot(atm,label="Average Apparent Temperature",marker='.')

plt.legend(loc=(1.02,0.8))

plt.show()
df1 = df[df.index.month==11]

hdt=df1['Humidity']

atm=df1['Apparent Temperature (C)']

plt.title("November Monthwise Stats",fontsize=17)

plt.plot(hdt,label="Average Humidity",marker=".")

plt.plot(atm,label="Average Apparent Temperature",marker='.')

plt.legend(loc=(1.02,0.8))

plt.show()
df1 = df[df.index.month==12]

hdt=df1['Humidity']

atm=df1['Apparent Temperature (C)']

plt.title("December Monthwise Stats",fontsize=17)

plt.plot(hdt,label="Average Humidity",marker=".")

plt.plot(atm,label="Average Apparent Temperature",marker='.')

plt.legend(loc=(1.02,0.8))

plt.show()