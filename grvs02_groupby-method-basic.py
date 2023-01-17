import pandas as pd

df = pd.read_csv(r'../input/weather_place.data.csv')

df.head()
df.drop(columns = 'Unnamed: 0', inplace=True)

df.head()
df.columns = ['Temperature','Date','Parameters','Place']

df.head()
tf = df[:12]

tf
gp = tf.groupby('Temperature')
for Temp, Index in gp:

    print("Temp:", Temp)

    print("\n")

    print("Index:",Index)
gp.get_group(0.0)
gp.max()
gp['Temperature'].mean()
gp['Temperature'].min()
gp.describe()
gp.size()