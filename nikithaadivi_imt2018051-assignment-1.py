import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import sklearn as sklearn

import sklearn.linear_model as linear_model

from sklearn.model_selection import train_test_split
alcdata = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/alcoholism/student-mat.csv")

fifadata = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/fifa18/data.csv")

accidata1 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2005_to_2007.csv")

accidata2 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2009_to_2011.csv")

accidata3 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2012_to_2014.csv")
alcdata.info()
alcdata.isnull().sum().sum()
alcdata.head()
alcdata["Average_G"] = alcdata[["G1","G2","G3"]].mean(axis = 1)

alcdata.drop(["G1","G2","G3"],axis = 1, inplace = True)
#cat_alcdata is alcdata without encoding categorical features

cat_alcdata = alcdata.copy()



alcdata.head()
female_alcdata = alcdata.groupby("sex").get_group("F")

male_alcdata = alcdata.groupby("sex").get_group("M")
sns.distplot(female_alcdata.Average_G,bins = 20)

sns.distplot(male_alcdata.Average_G,bins = 20)
sns.boxplot(x='sex',y='Average_G',data=alcdata)
sns.boxplot(x='famsize',y='Average_G',data = alcdata)
sns.boxplot(x='school',y='Average_G',data = alcdata)
alcdata.describe(include='all').loc['unique', :]
cat_attributes = alcdata.select_dtypes(include = ['object'])



for col in cat_attributes:    

    dumm = pd.get_dummies(alcdata[col])

    alcdata = pd.concat([alcdata, dumm], axis = 1)

    alcdata.drop(columns=[col], inplace = True)
print(alcdata["famrel"].unique())

sns.catplot(x = 'famrel', y = 'Average_G', data = alcdata)
sns.catplot(x = 'Pstatus', y = 'Average_G', data = cat_alcdata)

#One point is way too below. Can be outlier
sns.catplot(x = 'famrel', y = 'Average_G', hue = "Pstatus", kind = "box", data = cat_alcdata, aspect = 2)
numerical_attributes = cat_alcdata.select_dtypes(include = ['int', 'float'])
numerical_attributes[["absences", "Average_G"]].hist(figsize = (12,5), bins = 50)
sns.catplot(x = 'absences', data = numerical_attributes[["absences"]], aspect = 2, kind ='box')
# numerical_attributes[["absences"]] = numerical_attributes[["absences"]].replace({0: None}).dropna()

# # transformed_absences.replace(0, "nan").dropna(axis=1,how="all")

# print(numerical_attributes[["absences"]])



def ztransform(x):

    return (x - numerical_attributes.absences.mean())/numerical_attributes.absences.std()



sns.distplot(numerical_attributes[["absences"]].apply(lambda x: np.log(ztransform(x)+0.001)), bins = 30)



print("Skew before: " + str(numerical_attributes[["absences"]].apply(lambda x: ztransform(x).skew())))

print("Skew after: " + str(numerical_attributes[["absences"]].apply(lambda x: np.log(ztransform(x)+0.0001).skew())))
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

fifadata.info()

fifadata.head()
def PreProcess(i):

    if(isinstance(i,str)):

        if(i[-1]=='M'):

            return(float(i.lstrip('€').rstrip('M'))*1000000)

        elif(i[-1]=='K'):

            return(float(i.lstrip('€').rstrip('K'))*1000)

        else:

            return(float(i.replace('€','')))

        

for col in ['Wage', 'Value', 'Release Clause']:

    fifadata[col] = fifadata[col].apply(lambda x: PreProcess(x))
#We have summed up value and wage of all players of a particular club and sorted to get the most economical club



club_Wage = fifadata['Wage'].groupby(fifadata['Club']).apply(lambda x : x.sum())

club_Value = fifadata['Value'].groupby(fifadata['Club']).apply(lambda x : x.sum())
(club_Value - club_Wage).sort_values(ascending = False)
print(fifadata["Potential"].isnull().sum())

print(fifadata["Value"].isnull().sum())

print(fifadata["SprintSpeed"].isnull().sum())
fifadata['SprintSpeed'] = fifadata['SprintSpeed'].fillna(fifadata['SprintSpeed'].mean())
# fig, axes = plt.subplots(figsize=(7,5))

# axes.set_ylabel('Potential')

# fifadata[["Age", "Potential"]].plot(x = 'Age',ax = axes, y = 'Potential', kind = 'scatter')



sns.lmplot(x = 'Age',y = 'Potential', order = 2, data = fifadata, aspect = 1.5)

sns.lmplot(x = 'Age',y = 'Value', order = 2, data = fifadata, aspect = 1.5)

sns.lmplot(x = 'Age',y = 'SprintSpeed', order = 2, data = fifadata, aspect = 1.5)
for col in fifadata.iloc[:,54:87]:

    if(np.abs(fifadata.corr()['Potential'][col]) > 0.4):

        print( str(col) + " is related to " + 'Potential and hence might be helpful in deciding Potential' )

    



# plt.figure(figsize=(10,10))

# sns.heatmap(fifadata[54:87].corr(), vmin=-1, cmap="coolwarm", annot=True)
# sns.lmplot(x = 'Wage',y = 'SprintSpeed', order = 2, data = fifadata)

fifadata.plot(kind = 'scatter',y='Wage',x='SprintSpeed',figsize=(10,10))

fifadata.plot(kind = 'scatter',y='Wage',x='Overall',figsize=(10,10))
fifadata.plot(kind = 'scatter',y='Wage',x='Potential',figsize=(10,10))
age_dist = fifadata["Age"].groupby(fifadata["Club"])

list_values = ["min", "max", "median", "mean"]

print(age_dist.agg(list_values))



club_age = fifadata['Age'].groupby(fifadata['Club']).apply(lambda x : (x<=20).sum())

club_age.sort_values(ascending = False)
print(accidata1.shape)

print(accidata2.shape)

print(accidata3.shape)
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

accidata = pd.concat([accidata1, accidata2, accidata3])

accidata.info()
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

fig, axes = plt.subplots(figsize=(10,5))

axes.set_ylabel('Casualities')



casualties = accidata[["Number_of_Casualties","Day_of_Week"]].groupby(["Day_of_Week"], as_index = False).sum().sort_values(by = 'Number_of_Casualties',ascending = False)

casualties["Day_of_Week"] = casualties["Day_of_Week"].map({1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday",7:"Sunday"})



casualties.plot(x = 'Day_of_Week', y = 'Number_of_Casualties', ax = axes,kind='bar',color= 'red')
speed_limit_data = accidata[["Speed_limit", "Day_of_Week"]].groupby("Day_of_Week")

list_values = ["min", "max", "median", "mean"]

speed_limit_data.agg(list_values)
sns.boxplot(x ="Day_of_Week", y = 'Speed_limit', data = accidata)
plt.figure(figsize=(20,5))

sns.countplot(data=accidata[["Weather_Conditions","Accident_Severity"]], x= 'Weather_Conditions', hue = 'Accident_Severity')
fig, axes = plt.subplots(figsize=(7,5))

axes.set_ylabel('Accident_Severity')

severity_light = accidata[['Accident_Severity','Weather_Conditions']].groupby(["Weather_Conditions"], as_index = False).mean().sort_values(by = "Accident_Severity")

print(severity_light)

severity_light.plot(x = 'Weather_Conditions', y = 'Accident_Severity', ax = axes, kind='bar',color= 'deepskyblue',ylim = [1,3], yticks = np.arange(0, 3, step=0.2))
plt.figure(figsize=(20,5))

sns.countplot(data=accidata[["Light_Conditions","Accident_Severity"]], x= 'Light_Conditions', hue = 'Accident_Severity')
fig, axes = plt.subplots(figsize=(7,5))

axes.set_ylabel('Accident_Severity')

severity_light = accidata[['Accident_Severity','Light_Conditions']].groupby(["Light_Conditions"], as_index = False).mean().sort_values(by = "Accident_Severity")

print(severity_light)

severity_light.plot(x = 'Light_Conditions', y = 'Accident_Severity', ax = axes, kind='bar',color= 'deepskyblue',ylim = [1,3], yticks = np.arange(0, 3, step=0.2))
accidata.isnull().sum()
accidata.drop(columns = ['Junction_Detail','Junction_Control', 'LSOA_of_Accident_Location'], inplace = True)
def split_date_get_month(d):

    if(isinstance(d,str)):

        return(int(d.split('/')[1]))

    else:

        return 1

    

def split_date_get_day(d):

    if(isinstance(d,str)):

        return(int(d.split('/')[0]))

    else:

        return 1



accidata['Day'] = accidata['Date'].apply(lambda x: split_date_get_day(x))

accidata['Month'] = accidata['Date'].apply(lambda x: split_date_get_month(x))

accidata[['Accident_Severity','Day','Month','Year']].head()
plt.figure(figsize=(8,5))

sns.heatmap(accidata[['Accident_Severity','Day','Month','Year']].corr(), vmin=-1, cmap="coolwarm", annot=True)
accidata.drop(columns = ['Day','Month','Year', 'Time'], inplace = True)
numerical_attributes = accidata.select_dtypes(include = ['int', 'float'])

plt.figure(figsize=(25,15))

sns.heatmap(numerical_attributes.corr(), vmin=-1, cmap="coolwarm", annot=True)
accidata.drop(columns = ['Location_Easting_OSGR','Location_Northing_OSGR','Police_Force'], inplace = True)
accidata.columns
y=accidata['Accident_Severity']

x=accidata.drop(columns=['Accident_Severity'],inplace= False)
model=linear_model.LogisticRegression(class_weight='balanced',C=0.0005)