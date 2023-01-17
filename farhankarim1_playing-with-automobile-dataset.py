import pandas as pd
df = pd.read_csv("../input/Automobile_data.csv")
print("Done")
df.head()
import numpy as np
df.replace("?",np.nan,inplace=True)
df.head()
missing_data = df.isnull()
missing_data.head(5)
df.isnull().sum()
#df['normalized-losses'].astype("float").mean()
df['normalized-losses'].replace(np.nan,122.0,inplace=True)
df.head()
bore_mean= df['bore'].astype("float").mean()
df['bore'].replace(np.nan,bore_mean,inplace=True)
df.head()
stroke_mean = df['stroke'].astype('float').mean()
df['stroke'].replace(np.nan,stroke_mean,inplace=True)
df.head()
avg_4=df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_4, inplace= True)
avg_5=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_5, inplace= True)
df['num-of-doors'].value_counts()
df['num-of-doors'].value_counts().idxmax()
#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace = True)
df.dropna(subset=['price'],axis=0,inplace=True)
df.head()
df.isnull().sum()
df.dtypes

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
print("Done")
df.dtypes
df["horsepower"]=df["horsepower"].astype(float, copy=True)
#Divide data in 3 equally sized bins
binwidth = (max(df["horsepower"])-min(df["horsepower"]))/4
binwidth
#1st bin 48-101 2nd bin 101.5-155 and 3rd bin 155-208.5 that's why in previous cell divided by 4
bins = np.arange(min(df["horsepower"]), max(df["horsepower"]), binwidth)
bins
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names,include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot

a = (0,1,2)

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3,color=['green'],rwidth=0.75)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]
# replace (origianl value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()
#dummy_variable_1 = pd.get_dummies(df["fuel-type"])
#dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
#df = pd.concat([df, dummy_variable_1], axis=1)
#df.drop("fuel-type", axis = 1, inplace=True)
df.head()
#MY MISTAKE SORRY TRIED TO RUN THE DROP STATMENT WITHOUT INPLACE AND THEN ADDED INPLACE RESULTED IN DUPLICATE COLUMNS
df = df.loc[:,~df.columns.duplicated()]
df
dummy_variable_2 = pd.get_dummies(df['aspiration'])
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
df = pd.concat([df, dummy_variable_2], axis=1)
df.drop('aspiration', axis = 1, inplace=True)
df.head()
df.to_csv('clean_automobile_data.csv')
import numpy as np
dfclean = pd.read_csv('clean_automobile_data.csv')
dfclean.head(20)
dfclean.dtypes
dfclean.drop(['Unnamed: 0'], axis=1,inplace=True)
dfclean
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
dfclean.corr()
dfclean[['bore','stroke' ,'compression-ratio','horsepower','price']].corr()
dfclean[["engine-size", "price"]].corr()
dfclean['drive-wheels'].value_counts()
sns.boxplot(x='drive-wheels',y='price',data=dfclean)
plt.scatter(x=dfclean['engine-size'],y=dfclean['price'])
