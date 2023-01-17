# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Reading the data in Dataframe



df = pd.read_csv('../input/bengaluru-house-price-data/Bengaluru_House_Data.csv')

df.head()

## Column information



df.info()
df.isnull().sum()
for feature in df.columns:

    if df[feature].isnull().sum() > 1:

        print("{} Feature has {}% Missing values ".format(feature,round(df[feature].isnull().mean()*100,1)))
## Copy the data from orginal DF to duplicate DF1



df1 = df.copy()

df1.head()
## Update the null vales with respective value in society



df1['society'].fillna("Info Not available",inplace = True)

df1.head()
df1['size'].unique()
## We have 0.1% null values in size feature, so we can update with respective values



df1['size'].fillna('0',inplace = True)
## Update the null vales with respective value in bathroom feature



df1['bath'].fillna(1.0,inplace = True)
## Update the null vales with respective value in balcony feature



df1['balcony'].fillna(0.0,inplace = True)
## Find out varies values in total_sqft feature



def is_float(x):

    try:

        float(x)

    except:

        return False

    return True





df1[~df1['total_sqft'].apply(is_float)]
def convert_sqft_to_num(x):

    tokens = x.split('-')

    if len(tokens) == 2:

        return (float(tokens[0])+float(tokens[1]))/2 ## take a mean value for range value

    try:

        return float(x) ## Directly return float value if it is same

    except:

        return None ## otherwise make Null
df1.total_sqft.isnull().sum() ## Before executing the function
## pass the total_sqft value to the convert_sqft_to_num function



df1.total_sqft = df1.total_sqft.apply(convert_sqft_to_num) 
df1.total_sqft.isnull().sum() ## After executing the function
## Remove the records if the total_sqft column has null value based on index comparision



df1.total_sqft.dropna(axis='index',inplace=True)
df1.total_sqft.isnull().sum()
df1


df1 = df1.astype({'bath':np.int32, 'balcony':np.int32})
df1.info()
## Creating new size column which can only have numerical value alone



df1['bhk'] = df1['size'].apply(lambda x : int(x.split()[0]))

df1
## Creating new price_per_sqr column which can evoluate the price per Sqr feet



df1['price_per_sqr'] = round(df1['price'] * 100000 / df1['total_sqft'],2) 



df1
## Finding unique location



df1.location.unique()
len(df1.location.unique())
## Locaton count value pair



location_stats = df1['location'].value_counts() 

location_stats
location_stats.values.sum()
## Locaton data point count > 10



len(location_stats[location_stats > 10] )
## Locaton data point count < 10



len(location_stats[location_stats <= 10])
## Identify the location which has below 10 data point



below_10_dp = location_stats[location_stats <= 10]

below_10_dp
## Replace the location name by others which is present in "below_10_dp" list



df1['location'] = df1['location'].apply(lambda x : 'Others' if x in below_10_dp else x)

df1
len(df1.location.unique())
df2 = df1.copy()
df2[(df2.total_sqft/df2.bhk) < 300].head()
l = len(df2[df2.total_sqft/df2.bhk<300])

print("Around {}% of records become otliers with this condition".format(round(l/len(df) * 100,2)))
df2.shape
df3 = df2[~(df2.total_sqft/df2.bhk < 300)]

df3.shape
df3.price_per_sqr.describe()
def remove_pps_outliers(df):

    df_out = pd.DataFrame()

    for key, subdf in df.groupby('location'):

        m = np.mean(subdf.price_per_sqr)

        st = np.std(subdf.price_per_sqr)

        reduced_df = subdf[(subdf.price_per_sqr>(m-st)) & (subdf.price_per_sqr<=(m+st))]

        df_out = pd.concat([df_out,reduced_df],ignore_index=True)

    return df_out

df4 = remove_pps_outliers(df3)

df4.shape
def plot_scatter_chart(df,location):

    bhk2 = df[(df.location==location) & (df.bhk==2)]

    bhk3 = df[(df.location==location) & (df.bhk==3)]

    fig = plt.figure(figsize=(12,8))

    fig, plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)

    fig, plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)

    plt.xlabel("Total Square Feet Area")

    plt.ylabel("Price (Lakh Indian Rupees)")

    plt.title(location)

    plt.legend()

    

plot_scatter_chart(df4,"Rajaji Nagar") ## Display only for Rajaji nagar
## For Hebbal



plot_scatter_chart(df4,"Hebbal")
def remove_bhk_outliers(df):

    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):

        bhk_stats = {}

        for bhk, bhk_df in location_df.groupby('bhk'):

            bhk_stats[bhk] = {

                'mean': np.mean(bhk_df.price_per_sqr),

                'std': np.std(bhk_df.price_per_sqr),

                'count': bhk_df.shape[0]

            }

        for bhk, bhk_df in location_df.groupby('bhk'):

            stats = bhk_stats.get(bhk-1)

            if stats and stats['count']>5:

                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqr<(stats['mean'])].index.values)

    return df.drop(exclude_indices,axis='index')

df5 = remove_bhk_outliers(df4)

df5.shape
plot_scatter_chart(df5,"Rajaji Nagar")
plot_scatter_chart(df5,"Hebbal")
import matplotlib



plt.hist(df5.price_per_sqr,rwidth=0.8)

plt.xlabel("Price Per Square Feet",size = 13)

plt.ylabel("Count", size = 13)

plt.title("Price per sqft distribution", size = 20)
df5.bath.unique()
df5[df5.bath>10]
## Bedroom < bathroom

l = len(df5[df5.bath > df5.bhk + 1]) ## Each house may have one additional bathroom for guest

bath_bed = df5[df5.bath > df5.bhk + 1]

bath_bed
l
df6 = df5[~(df5.bath > df5.bhk + 1)]

df6.shape
## removing unnecessary columns



df7 = df6.drop(['area_type','availability','size','society','price_per_sqr'],axis='columns')

df7
dummies = pd.get_dummies(df7.location)

dummies.head()
## Combining latested DF & Dummies(which have location value in the form of numeric)



df8 = pd.concat([df7,dummies.drop('Others',axis='columns')],axis='columns')

df8.head()
## Droping the location column 



df8 = df8.drop('location',axis='columns')

df8.head()
def LinearEquationPlot(df7,location):

    xy = df8[(df7.location==location)]

    fig = plt.figure(figsize=(20,10))

    sns.regplot(x='total_sqft', y='price', data=xy,ci = 68)
## Linear line and plots for Hebbal location



LinearEquationPlot(df7,'Hebbal')
df8.shape
## Indipendent features



X = df8.drop(['price'],axis='columns')



X.head()
## Dependent feature



y = df8.price

y.head()
## Seperate the data for training & testing



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
## Accuracy rate using LinearRegression algorithm



from sklearn.linear_model import LinearRegression

lr_clf = LinearRegression()

lr_clf.fit(X_train,y_train)

lr_clf.score(X_test,y_test)
## cross validation to measure accuracy of our LinearRegression model



from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)



cross_val_score(LinearRegression(), X, y, cv=cv)
def predict_price(location,sqft,bath,bhk):    

    loc_index = np.where(X.columns==location)[0][0]



    x = np.zeros(len(X.columns))

    x[0] = sqft

    x[1] = bath

    x[2] = bhk

    if loc_index >= 0:

        x[loc_index] = 1



    return round(lr_clf.predict([x])[0],2)
## Predecting house price by giving Location, sqft, Bathroom and Bedroom as a input to the "predict_price" function



amount = predict_price('Indira Nagar',1000, 2, 2)



print("Rs.{} Lakhs".format(amount))
## Coeffecient of Linear equation



lr_clf.coef_ 
## Intercept



lr_clf.intercept_ 