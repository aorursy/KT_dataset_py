# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings("ignore")





import geojson as geojson

import geopandas



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/directory.csv")

df.head(1)
df = df[df['Longitude'].notnull()]

df = df[df['Latitude'].notnull()]

df= df.fillna('00')
stores = zip(df['Longitude'], df['Latitude'], df['Store Number'],df['Store Name'],df['Ownership Type'],df['City'],df['State/Province'],df['Country'])

points = [geojson.Feature(geometry=geojson.Point((v[0], v[1])), properties={'Store Number': v[2],'Store Name': v[3],'Ownership Type': v[4],'City': v[5],'State/Province': v[6],'Country': v[7]}) for v in stores]



geo_collection = geojson.FeatureCollection(points)



print(points[0])
geo_collection
dump = geojson.dumps(geo_collection, sort_keys=True)



with open('Stores.geojson', 'w') as file:

    file.write(dump)