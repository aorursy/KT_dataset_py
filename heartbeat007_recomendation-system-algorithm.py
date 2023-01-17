import numpy as np

import pandas as pd
df = pd.read_csv("../input/recomend-data/rating_final.csv")

cuisine = pd.read_csv("../input/recomend-data/chefmozcuisine.csv")
df.head()
cuisine.head()
### recomendation based on counting

working_df = df[['userID','placeID','rating']]
working_df.head()
import matplotlib.pyplot as plt

r = pd.DataFrame(working_df.rating.value_counts())

labels = list(r.index)

data = list(r['rating'])

explode = (0, 0.1, 0)

fig1, ax1 = plt.subplots()

ax1.pie(data, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.legend()

plt.show()
## take the data and geoup the place(which in this case a resturant)

## then groupby the number of rating count

rating_count = pd.DataFrame(df.groupby('placeID')['rating'].count())
## and sort them based on the assending order

most_rated = rating_count.sort_values('rating',ascending=False)
top = most_rated.head()

top
top.plot(kind='bar')
## in order to merge the coumn you need

final = pd.merge(top,cuisine,on='placeID')
final
final2 = pd.DataFrame(final.groupby('Rcuisine')['rating'].count())
final2.plot(kind='bar')
### so we can see the maxican food get the most rating in the top resturant

### so we can recomend the mexican food
import numpy as np

import pandas as pd
df = pd.read_csv("../input/recomend-data/rating_final.csv")

cuisine = pd.read_csv("../input/recomend-data/chefmozcuisine.csv")

geodata = pd.read_csv("../input/recomend-data/geoplaces2.csv", engine='python')
df.head()
cuisine.head()
geodata.head()
geodata = geodata[['placeID','name']]
geodata.head()
new_rating = pd.DataFrame(df.groupby('placeID')['rating'].mean())
new_rating.head()
new_rating['rating_count'] = pd.DataFrame(df.groupby('placeID')['rating'].count())
new_rating
## sort the new_rating based on the assending value

final1 = new_rating.sort_values('rating_count',ascending=False).head()
final1
#### now merge the data to get the 

final = pd.merge(final1,geodata,on='placeID')
final
final = pd.merge(final,cuisine,on='placeID')
final
## doing a cross_tab analysis for the df data frame

df.head() ### this is the data frame we are doing crosstab analysis
places_crosstab = pd.pivot_table(data=df,values='rating',index='userID',columns='placeID')
places_crosstab
### this data actiually shows the raing is given by different user to different places

## there is null value because not all user give rating  to all places
places_crosstab.head()
first = final['placeID'][0] ## this is the first that has the highest rating
first
## so now from the crosstab we can find all the rating 

## of this place that user give

## and we can fin dit on clumn

## becase the cross tab the places is in the column

tortas_rating = places_crosstab[first]
tortas_rating
tortas_rating

## now remove the NAN

tortas_rating = tortas_rating[tortas_rating >= 0]

tortas_rating
tortas_rating.plot(kind="bar")
tortas_rating
similler_tortas = places_crosstab.corrwith(tortas_rating)

corr_tortus = pd.DataFrame(similler_tortas,columns=["pearson_r"])

corr_tortus.dropna(inplace=True)
corr_tortus
corr_tortus.plot(kind='bar')
## we slao need to know that the similler places rating count

corr_tortus['rating_count'] = new_rating['rating_count']
corr_smry_w_tortas=pd.DataFrame(corr_tortus)
## target

    ## we want to find the resturant that has best positive corrlation

    ## and has god rating [rating greater than 10]
similler = corr_smry_w_tortas[corr_smry_w_tortas['rating_count']>10].sort_values('pearson_r',ascending=False)
similler.head()
## for those in pearson r has 1 means that it has only one rating

## so we throw out them

final = pd.merge(similler,cuisine,on='placeID')

final2 = pd.merge(final,geodata,on='placeID')
final2.head()
final2[final2['Rcuisine']=='Fast_Food']
import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression
df = pd.read_csv("../input/recomend-data/bank_full_w_dummy_vars.csv")
df.head()
df.info()
df.columns
X = df.iloc[:,18:36]
X.head()
Y = df[['y_binary                    ']]
X.head()
Y.head()
Logreg1 = LogisticRegression()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y)
print(len(x_train))

print(len(y_train))

print(len(x_test))

print(len(y_test))
Logreg1.fit(x_train,y_train)
Logreg1.score(x_test,y_test)
x_test.values[0]
Logreg1.predict([x_test.values[0]])
import numpy as np

import pandas as pd

import sklearn

from sklearn.neighbors import NearestNeighbors
df = pd.read_csv("../input/recomend-data/mtcars.csv")
df.head()
X = df[['mpg','cyl','disp','hp','drat','wt','qsec']]
nbrc = NearestNeighbors(n_neighbors=1).fit(X)
nbrc
nbrc.kneighbors([[15.0 ,9,260.0 ,90 ,2.90 ,1.620 ,20.46]])
df[df.index==3]
## so we can see that it finds the related property car