import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder

from sklearn.linear_model import LinearRegression



#import libaries to transform our features 

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer



df = pd.DataFrame({

    'Dog Breed':["Beagle", "Bulldog", "Poodle", "German Shepard","Beagle", "Bulldog", "Poodle", "German Shepard"], 

    'Age':[1, 4, 7, 9,12,22,10,3],

    'Weight':[60,32,70,100,66,44,24,66],

    "Sex":["Male","Female","Male","Female","Male","Female","Male","Female"],

    "Height":[15,50,77,110,55,66,77,88],

    "Intelligence":["Average","Below Average","Above Average","Average","Average","Below Average","Above Average","Average"],

    

    "Life Expentancy":[10,20,7,15,10,20,7,15]



})

df.head(10)

try:

    y=df["Life Expentancy"]

    X=df.drop(columns=["Life Expentancy"])

    model=LinearRegression()

    model.fit(X,y)

except Exception as e: 

    print(e)
ordinalEncodingScheme = ['Below Average','Average','Above Average']

ordinalEncoder = OrdinalEncoder(categories=[ordinalEncodingScheme])





results=ordinalEncoder.fit_transform(X[["Intelligence"]])

print(f"Here are my categories:\n{ordinalEncoder.categories_}")

results=np.reshape(results, (-1,1)).flatten()



print(results)

dataframe=pd.DataFrame({

    "Intelligence":df["Intelligence"].array,

    "Transformed Intelligence":results



})

print(dataframe.head())

BinaryEncodingScheme = ['Male','Female']

ordinalEncoder = OrdinalEncoder(categories=[BinaryEncodingScheme])





results=ordinalEncoder.fit_transform(X[["Sex"]])

print(f"Here are my categories:\n{ordinalEncoder.categories_}")

results=np.reshape(results, (-1,1)).flatten()



print(results)

dataframe=pd.DataFrame({

    "Sex":X["Sex"].array,

    "Binary Encoded results":results



})

print(dataframe.head())

encoder = OneHotEncoder()

results=encoder.fit_transform(X[["Dog Breed"]]).toarray()

print(f"Here are my categories:\n{encoder.categories_}\n")

print(results)

print(pd.get_dummies(X[["Dog Breed"]]))

scaler = StandardScaler()

print(scaler.fit_transform(X[["Age","Weight"]]))
X["Height ^2"]=np.power(X[["Height"]],2)

X["BMI"]=X["Weight"]/X["Height ^2"]

X["Difference between Height and Weight"]=X["Height"]- X["Weight"]

X["Sum between Height and Weight"]=X["Height"]+ X["Weight"]

X.head()

BinaryEncodingScheme = ['Male','Female']

binaryEncoder = OrdinalEncoder(categories=[BinaryEncodingScheme])



ordinalEncodingScheme = ['Below Average','Average','Above Average']

ordinalEncoder = OrdinalEncoder(categories=[ordinalEncodingScheme])



ord_attribs=["Intelligence"]

bin_attribs=["Sex"]

one_h_attribs=["Dog Breed"]

num_attribs=["Age","Weight"]

num_pipline=make_pipeline(StandardScaler())

full_pipeline=make_column_transformer(

    (num_pipline,num_attribs),

    (OneHotEncoder(),one_h_attribs),

    (binaryEncoder,bin_attribs),

    (ordinalEncoder,ord_attribs)



)





full_pipeline=full_pipeline.fit(df)

X=full_pipeline.transform(df)

model.fit(X,y)

model.predict(X)
df = pd.DataFrame(

        {

            'Date': pd.date_range(start='2015-01-01', end='2020-12-31', freq='D')

        }

    )



size=(df.count())

np.random.seed(0)

data = np.random.randint(0,1000,size=size)

df["Data"]=data

print(df.info())

print(df.head())
df['year'] = df["Date"].dt.year

df['month'] = df["Date"].dt.month

df['day'] = df["Date"].dt.day

df['week'] = df["Date"].dt.week

df['dayofweek_name']=df['Date'].dt.day_name()



df['isWeekend'] = (df["Date"].dt.dayofweek >=5).astype(int)

df['isMonthStart'] = df["Date"].dt.is_month_start.astype(int)

df.head(10)

df["Days since since today"]=(datetime.datetime.today() - df["Date"]).dt.days

df.head()
df["Lag_1"]=df["Data"].shift(1)

df.head()
df['rolling_mean'] = df['Data'].rolling(window=7).mean()

df.tail(10)
df['Expanding_Mean'] = df['Data'].expanding(2).mean()

df.tail(10)