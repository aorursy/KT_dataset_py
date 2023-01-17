import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



properties = pd.read_csv(

    '../input/properties_2016.csv',

    usecols=['latitude','longitude']

)



locations = properties.sample(n=10000)[['latitude', 'longitude']].values

plt.plot(locations[:,0], locations[:,1], '.')