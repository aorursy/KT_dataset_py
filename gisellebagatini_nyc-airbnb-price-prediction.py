# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import statsmodels.api as sm

from scipy import stats

import altair as alt

alt.data_transformers.disable_max_rows()
df_nyc = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv', index_col = 0)

df_nyc.columns
df_nyc.reset_index(drop = True, )

df_nyc.head(3)
df_nyc.info()
df_nyc.isna().sum()
df_nyc["name"].fillna("Unknown", inplace = True)

df_nyc["host_name"].describe()
df_nyc["host_name"].fillna("Unknown", inplace = True)
df_nyc["last_review"].describe()
df_nyc["reviews_per_month"].describe()
df_nyc["reviews_per_month"].isna().sum()
for i in df_nyc.columns:

    display(pd.DataFrame(df_nyc[i].isnull().value_counts()))
np.mean(df_nyc.isna(), axis = 0)
df_nyc["reviews_per_month"].fillna(0, inplace = True)
df_nyc["last_review"] = pd.to_datetime(df_nyc["last_review"])
df_nyc["last_review"].fillna(max(df_nyc["last_review"]), inplace = True)
df_nyc.isna().sum()
sns.pairplot(df_nyc[["price","number_of_reviews","minimum_nights","neighbourhood_group"]], hue = "neighbourhood_group");
data = df_nyc



alt.Chart(df_nyc).mark_circle().encode(

    x='number_of_reviews:Q',

    y='price:Q',

    color = "neighbourhood_group"

)
data = df_nyc



alt.Chart(df_nyc).mark_circle().encode(

    x='number_of_reviews:Q',

    y = "neighbourhood_group",

    color= "room_type"

)
data = df_nyc



alt.Chart(df_nyc).mark_circle().encode(

    x='price:Q',

    y = "neighbourhood_group",

    color= "room_type"

)
sns.distplot(df_nyc["calculated_host_listings_count"], bins = 6)
price_min = df_nyc.loc[df_nyc["price"] <= 300]

price_max = df_nyc.loc[df_nyc["price"] > 300]

price_min[["price"]].count()/df_nyc[["price"]].count()
df_nyc.groupby("neighbourhood_group").agg(["mean"])
sns.distplot(price_min["price"])
alt.Chart(price_min).mark_rect().encode(

    alt.X('price:Q', bin=alt.Bin(maxbins=60)),

    alt.Y('number_of_reviews:Q', bin=alt.Bin(maxbins=40)),

    alt.Color('neighbourhood_group'),

)

alt.Chart(price_max).mark_rect().encode(

    alt.X('price:Q', bin=alt.Bin(maxbins=70)),

    alt.Y('number_of_reviews:Q', bin=alt.Bin(maxbins=40)),

    alt.Color('neighbourhood_group'),

)
alt.Chart(df_nyc).mark_bar().encode(

    alt.X('mean(price):Q'),

    alt.Y('neighbourhood_group'),

)
sns.boxplot(x = "room_type", y = "price", data = price_min);

sns.boxplot(x = "room_type", y = "price", data = price_max);
sns.boxplot(x = "neighbourhood_group", y = "price", data = price_min);
plt.figure(figsize=(10.0,8.0))



corr=price_min.corr(method='pearson')

sns.heatmap(corr, annot=True, fmt=".2f", vmax=.3, center=0,

            square=True, linewidths= 0.5, cbar_kws={"shrink": 0.5}).set(ylim=(11, 0))

plt.xticks(rotation=45)

plt.title("Correlation Matrix",size=15, weight='bold')
plt.figure(figsize=(10,6))

sns.scatterplot(df_nyc.longitude,df_nyc.latitude,hue=df_nyc.room_type)

plt.ioff()
plt.figure(figsize=(10,6))

sns.scatterplot(df_nyc.longitude,df_nyc.latitude,hue=df_nyc.availability_365)

plt.ioff()
price_min["host_id"].value_counts()
price_min.drop(['name', 'host_id', 'host_name','last_review'], axis = 1, inplace = True)
#neighbourhood threshold:

price_min["neighbourhood"].value_counts()



neighbourhood_threshhold = 300





binary_value_counts = price_min["neighbourhood"].value_counts() > neighbourhood_threshhold

neighbourhood_to_keep =list(binary_value_counts[binary_value_counts == True].index)

neighbourhood_to_keep
price_min.loc[~price_min["neighbourhood"].isin(neighbourhood_to_keep),'neighbourhood'] = "other"
price_min["neighbourhood"].value_counts().sort_values().plot.barh()
price_min.shape
price_model= pd.get_dummies(price_min,drop_first = True)
price_model.shape
target = price_model["price"]
price_model.drop("price", axis = 1, inplace = True)
price_model["price"] = target

price_model.head()
X = price_model[price_model.columns[:-1]]

y = price_model["price"]



X.drop(["neighbourhood_Upper East Side","neighbourhood_East Village", "neighbourhood_Ditmars Steinway", "neighbourhood_Kips Bay","neighbourhood_Crown Heights","neighbourhood_Bedford-Stuyvesant","neighbourhood_Upper West Side","reviews_per_month"], axis = 1, inplace = True)

X_withconstant = sm.add_constant(X)

linear_model = sm.OLS(y,X_withconstant).fit()

linear_model.summary()
coefficients = linear_model.summary().tables[1]

coefficients_model = pd.DataFrame(data = coefficients.data[1:],columns = coefficients.data[0])

coefficients_model.rename(mapper = {"": "Data"}, axis = 1, inplace = True)

coefficients_model["coef"] = coefficients_model["coef"].astype(float)

coefficients_model["P>|t|"] = coefficients_model["P>|t|"].astype(float)
coefficients_model[coefficients_model["P>|t|"] > 0.05].sort_values("P>|t|", ascending = False)
coefficients2 = linear_model.summary().tables[1]

coefficients_model2 = pd.DataFrame(data = coefficients2.data[1:],columns = coefficients2.data[0])

coefficients_model2.rename(mapper = {"": "Data"}, axis = 1, inplace = True)

coefficients_model2["coef"] = coefficients_model2["coef"].astype(float)

coefficients_model2["P>|t|"] = coefficients_model2["P>|t|"].astype(float)

coefficients_model2[coefficients_model2["P>|t|"] > 0.05].sort_values("P>|t|", ascending = False)
coefficients_model2["abs_coef"] =np.abs(coefficients_model2["coef"])

coefficients_by_magnitude = coefficients_model2.sort_values("abs_coef", ascending = False)

coefficients_by_magnitude
predictions = linear_model.predict(X_withconstant)

price_model["predictions"] = predictions

price_model.head(10)
plt.figure()

plt.scatter(linear_model.fittedvalues, linear_model.resid)

plt.show()
plt.figure()





ax1 = sns.distplot(price_model['price'], hist=False, color="r", label="Actual Value")

sns.distplot(predictions, hist=False, color="b", label="Fitted Values" , ax=ax1)





plt.title('Actual vs Fitted Values for Prices')

plt.xlabel('Price')

plt.ylabel('Proportion of Airbnbs')



plt.show()

plt.close()