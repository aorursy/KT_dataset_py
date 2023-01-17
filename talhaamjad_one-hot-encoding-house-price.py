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
df = pd.read_csv("../input/homepricer/homeprices.csv")
df
dummies = pd.get_dummies(df.city)
dummies
merged = pd.concat([df,dummies], axis='columns')
merged
final = merged.drop(['bathrooms','yr_built','yr_renovated','sqft_basement','bedrooms','sqft_living','floors','waterfront',
                     'view','condition','date','street','city','country','statezip','Seattle','sqft_above','Algona','Auburn',
                     'Beaux Arts Village','Bellevue','Black Diamond','Bothell','Burien','Carnation','Sammamish','SeaTac',
                     'Skykomish','Snoqualmie','Snoqualmie Pass','Tukwila','Vashon','Woodinville','Yarrow Point',
                     'Clyde Hill','Covington','Des Moines','Duvall','Enumclaw','Fall City','Federal Way','Inglewood-Finn Hill',
                    'Issaquah','Kenmore','Kent','Kirkland','Lake Forest Park','Maple Valley','Medina','Mercer Island','Milton',
                     'Newcastle','Normandy Park','North Bend','Pacific','Preston','Ravensdale'
                    ],axis='columns')
final
final = final.drop(['Shoreline'], axis='columns')
final
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = final.drop('price',axis ='columns')
X
y = final.price
y
model.fit(X,y)
model.predict(X)
model.score(X,y)
model.predict([[9050,0,0]])