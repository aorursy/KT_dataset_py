import pandas as pd

import numpy as np
df=pd.read_csv("../input/bengaluru-house-price-data/Bengaluru_House_Data.csv")
df.head()
df.groupby("area_type")["area_type"].agg("count")
df.shape
df.drop(["area_type","availability","society","balcony"],axis=1,inplace=True)
df.isnull().sum()
df.dropna(inplace=True)    #drops all the Rows which has NA values
df.shape    
df["size"].unique()    #BHK and bedroooms are same
# Now we will be needing the first numeric value only



df["bhk"]=df["size"].apply(lambda x: int(x.split(" ")[0]))  
df.head()
df["bhk"].unique()
df[df.bhk>20]    # its not possible to have house with 43 bedroooms in just area of 2400 sqft
df.total_sqft.unique()   # here you can see a sqft in range-> so we will take avg of min and max  
#this function will check whether the value is convertable to float or not

#the values which have range or string in it they cannot be converted float -> so the func returns flase for it



def is_float(x):

    try:

        float(x)

    except:

        return False

    return True
df.total_sqft.apply(is_float)   #returns Boolean value
df[df.total_sqft.apply(is_float)].head() #returns the rows with only True boolean values-> but we need the opposite
df[~df.total_sqft.apply(is_float)].tail(10)   #putting the negate symbol-> converts Flase to True and returns them
def convert_to_sqft(x):

    tokens =x.split("-")

    if len(tokens)==2:

        return (float(tokens[0])+float(tokens[1]))/2

    try:

        return float(x)

    except: 

        return None
convert_to_sqft("4000 - 4450")
convert_to_sqft("2000")
print(convert_to_sqft("200yard"))  #returns none
df["sqft"]=df["total_sqft"].apply(convert_to_sqft)
df.head()
df.drop(["total_sqft","size"],axis=1,inplace=True)
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
#adding price_per_sqft



df["price_per_sqft"]= (df["price"]*100000)/df["sqft"]
df.head(3)
# working loaction feature



len(df.location.unique())   #a lot of location so we will define a Other category
df["location"].apply(lambda x: x.strip())
loc_count=df.groupby("location")["location"].agg("count")

len(loc_count)
other=loc_count[loc_count<=10]

len(other)
##applying it 



df.location=df.location.apply(lambda x: "other" if x in other else x)
len(df.location.unique())
df.tail()    #2nd location is converted now to other
#on seeing the data you can say that sqft per room must be around 600



df[df["sqft"]/df["bhk"]<300].head()  #these are anomalies -> we need to remove them
df=df[~(df["sqft"]/df["bhk"]<300)]   # Negate~ will filter out the outliers
df.price_per_sqft.describe()   #min value is very low and unlikely , same case with max
a=[]

for key ,subdf in df.groupby("location"): 

    a.append(np.mean(subdf.price_per_sqft))

len(a)
def remove_outliers(var):

    df_out=pd.DataFrame()

    for key, subdf in var.groupby("location"):

        m=np.mean(subdf.price_per_sqft)

        st=np.std(subdf.price_per_sqft)

        reduced_df= subdf[(subdf.price_per_sqft> (m-st)) & (subdf.price_per_sqft< (m+st))]

        df_out=pd.concat([df_out,reduced_df],ignore_index=True)

    return df_out
df1=remove_outliers(df)
df1.shape   #around 2000 outliers have been removed
df.shape
from matplotlib import pyplot as plt

%matplotlib inline

import matplotlib 

matplotlib.rcParams["figure.figsize"] = (20,10)


def plot_scatter_chart(df,location):

    bhk2 = df[(df.location==location) & (df.bhk==2)]

    bhk3 = df[(df.location==location) & (df.bhk==3)]

    matplotlib.rcParams['figure.figsize'] = (15,10)

    plt.scatter(bhk2.sqft,bhk2.price,color='blue',label='2 BHK', s=50)

    plt.scatter(bhk3.sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)

    plt.xlabel("Total Square Feet Area")

    plt.ylabel("Price (Lakh Indian Rupees)")

    plt.title(location)

    plt.legend()

    

plot_scatter_chart(df1,"Rajaji Nagar")



#between 1600 and 1800 we can see a vertical line which shows for same area price for 2 bedroom is higher than of 3
def remove_bhk_outliers(df):

    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):

        bhk_stats = {}

        for bhk, bhk_df in location_df.groupby('bhk'):

            bhk_stats[bhk] = {

                'mean': np.mean(bhk_df.price_per_sqft),

                'std': np.std(bhk_df.price_per_sqft),

                'count': bhk_df.shape[0]

            }

        for bhk, bhk_df in location_df.groupby('bhk'):

            stats = bhk_stats.get(bhk-1)

            if stats and stats['count']>5:

                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)

    return df.drop(exclude_indices,axis='index')
df2 = remove_bhk_outliers(df1)

df2.shape
plot_scatter_chart(df2,"Rajaji Nagar") #you can notice the diff between this and previous
matplotlib.rcParams["figure.figsize"] = (20,10)

plt.hist(df2.price_per_sqft,rwidth=0.8)

plt.xlabel("Price Per Square Feet")

plt.ylabel("Count")



#lokks perfect #bell curve
df2[df2.bath>df2.bhk+2]
df3=df2[df2.bath<df2.bhk+2]   #removing the outliers
df3.shape
df3.drop("price_per_sqft",axis=1,inplace=True)
len(df3)
#One hot encoding



dummies=pd.get_dummies(df3.location)

dummies.head()

len(dummies)
df4=pd.concat([df3,dummies.drop("other",axis=1)],axis=1)   #remember to specify the axis

df4.head()
df4.drop("location",axis=1,inplace=True)
X=df4.drop("price",axis=1)

y=df4.price
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(X_train,y_train)

lr.score(X_test,y_test)
from sklearn.model_selection import GridSearchCV,ShuffleSplit



from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor



def find_best_model_using_gridsearchcv(X,y):

    algos = {

        'linear_regression' : {

            'model': LinearRegression(),

            'params': {

                'normalize': [True, False]

            }

        },

        'lasso': {

            'model': Lasso(),

            'params': {

                'alpha': [1,2],

                'selection': ['random', 'cyclic']

            }

        },

        'decision_tree': {

            'model': DecisionTreeRegressor(),

            'params': {

                'criterion' : ['mse','friedman_mse'],

                'splitter': ['best','random']

            }

        }

    }

    scores = []

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    for algo_name, config in algos.items():

        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)

        gs.fit(X,y)

        scores.append({

            'model': algo_name,

            'best_score': gs.best_score_,

            'best_params': gs.best_params_

        })



    return pd.DataFrame(scores,columns=['model','best_score','best_params'])



find_best_model_using_gridsearchcv(X,y)
from sklearn.linear_model import LinearRegression



lr=LinearRegression(normalize=True)

lr.fit(X_train,y_train)

lr.score(X_test,y_test)
import joblib



joblib.dump(lr,"Bangalore_House_prices")
import json

columns = {

    'data_columns' : [col.lower() for col in X.columns]

}

with open("columns.json","w") as f:

    f.write(json.dumps(columns))