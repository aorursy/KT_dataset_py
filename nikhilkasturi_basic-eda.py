import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.offline as py

import plotly.graph_objs as go
train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

building_meta = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

sample_sub = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")
train.head(5)
building_meta.head(5)
test.head(5)
weather_train.head(5)
weather_test.head(5)
train = train.merge(building_meta, left_on = "building_id", right_on = "building_id", how = "left")
train.head(5)
train = train.merge(weather_train, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")
test = test.merge(building_meta, left_on = "building_id", right_on = "building_id", how = "left")
test = test.merge(weather_test, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")
print ("Train: ",train.shape[0]," and ",train.shape[1],"features")

print ("Test:  ",test.shape[0]," and ",test.shape[1],"features")
%%time

def reduce_mem_usage(df):

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df

train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
train.info()
pd.isnull(train).any()



#most of the columns have null values 
train.describe()
sns.set_style("whitegrid")

missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()



#floor_count has most missing values
#Separate categorical and numerical columns

cat_column = train.dtypes[train.dtypes == 'object']

num_column = train.dtypes[train.dtypes != 'object']
for col in list(cat_column.index):

    print(f"--------------------{col.title()}-------------------------")

    total= train[col].value_counts()

    percent = total / train.shape[0]

    df = pd.concat([total,percent],keys = ['total','percent'],axis = 1)

    print(df)

    print('\n')
plt.style.use('seaborn-whitegrid')

plt.figure(figsize=(20,5)) 

sns.countplot(y="meter", data=train);

 

#0: electricity, 1: chilledwater, 2: steam, hotwater: 3
plt.style.use('seaborn-colorblind')

plt.figure(figsize=(20,8)) 

sns.countplot(y="primary_use", data=train);



#major of the building are education, Office, Public assembly, Residental and Public services
fig = plt.figure(figsize = (12,10))



sns.heatmap(train[list(num_column.index)].corr(),annot = True,square = True);



#no features are highly corelated except few of them like site_id to building_id, air_temprature to dew_temperature
plt.style.use('seaborn-colorblind')

plt.figure(figsize=(20,40)) 

sns.countplot(y="year_built", data=train);



#the data contains building from 1900 to 2019

# Highest number of building were built in 1976
count = building_meta["floor_count"].value_counts()

trace = go.Bar(

    x=count.index,

    y=count.values,

    marker=dict(

        color="green",

    ),

)

layout = go.Layout(

    title=go.layout.Title(

        text="Floor Count",

        x=0.5

    ),

    font=dict(size=14),

    width=1000,

    height=500,

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="meter")



#There are many buildings 1 floor and as floor counts increases, the number of buildings decreases. 

#Only 1 building exists in our data with 26 floors
#Work in Progress !!! 

#Please upvote if you like it !!!