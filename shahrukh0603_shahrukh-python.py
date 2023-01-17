# Create List and print it

grocery = ["orange", "Chilies", "apple", "banana"]

print(grocery)
# Get a value from List

print(grocery[2])
# Get the length of list

# Method Used

# len()

length_of_grocery = len(grocery)

print("The length of grocery ",length_of_grocery)
import numpy as np
# define an array

arr = np.array([1,2,4,6,7,8])

print(arr)

print("The length of the array is ", len(arr))

print("The dim of the array is ",(arr.ndim))

print("The shape of the array is ", arr.shape)
#A complex array

arr_com = np.array([

    [2,4,6,8],

    [1,2,3,4]

])

print(arr_com)

print("The length of the array is ", len(arr_com))

print("The dim of the array is ", arr_com.ndim)

print("The shape of the array is ", arr_com.shape)
#An even more complex array

arr_even_complex = np.array([

    [[2,4,6,8],[1,2,3,4],[1,2,3,4],[1,2,3,4]],

    [[0,9,4,8],[7,1,3,5],[1,2,3,4],[1,2,3,4]],

    [[0,9,4,8],[7,1,3,5],[1,2,3,4],[1,2,3,4]]

])

print(arr_even_complex)

print("The length of the array is ", len(arr_even_complex))

print("The dim of the array is ", arr_even_complex.ndim)

print("The shape of the array is ", arr_even_complex.shape)

print("The number of bytes to be jumped to go to next entry ", arr_even_complex.strides)

print("The size of the array is ", arr_even_complex.size)

print("The item size(in bytes) of the array is ", arr_even_complex.itemsize)
# Create an array

arr_crea = np.arange(60)

print(arr_crea)
# Create an array with custom values

# linspace(start, end, num-points)

arr_crea_cus = np.linspace(0,1,5, endpoint = False)

print(arr_crea_cus)
# Create an array with custom values

# arange(start, end, Steps)

arr_crea_cus2 = np.arange(0,20,5)

print(arr_crea_cus2)
#Only ones

arr_crea_cus3 = np.eye(7)

print(arr_crea_cus3)
arr_to_test1 = np.arange(1,10,1)

arr_to_test2 = np.arange(1,10,1)

arr_to_test = np.array([arr_to_test1, arr_to_test2])

print(arr_to_test)

print("The length of the array is ", len(arr_to_test))

print("The dim of the array is ", arr_to_test.ndim)

print("The shape of the array is ", arr_to_test.shape)

print("The number of bytes to be jumped to go to next entry ", arr_to_test.strides)

print("The size of the array is ", arr_to_test.size)

print("The item size(in bytes) of the array is ", arr_to_test.itemsize)
#A complex array for indexing

arr_complex_for_indexing = np.array([

    [np.arange(1,5,1),np.arange(5,10,1),np.arange(10,15,1)],

    [np.arange(15,20,1),np.arange(20,25,1),np.arange(25,30,1)],

    [np.arange(30,35,1),np.arange(35,40,1),np.arange(40,45,1)],

    [np.arange(55,50,1),np.arange(50,55,1),np.arange(55,60,1)]

])



print(arr_complex_for_indexing.shape)

print("===============Everything========================")

#Print everything

print(arr_complex_for_indexing[:])



#Print Second Row

print("=================From Third Row=========================")

print(arr_complex_for_indexing[2:])





print("====================A particular array======================")

print((arr_complex_for_indexing[2:,2])[0])



print("====================A particular Value======================")

print((arr_complex_for_indexing[2:,2])[0][1])



#print("=================Third Row, Second Column, Second Value=========================")

#print(arr_complex_for_indexing[])

#A complex array for indexing

arr_complex_for_indexing_ii = np.array([[

    [np.arange(1,5,1),np.arange(5,10,1),np.arange(10,15,1)],

    [np.arange(15,20,1),np.arange(20,25,1),np.arange(25,30,1)],

    [np.arange(30,35,1),np.arange(35,40,1),np.arange(40,45,1)],

    [np.arange(55,50,1),np.arange(50,55,1),np.arange(55,60,1)]

]])



print(((arr_complex_for_indexing_ii[0:,1])[0:,1])[0][1])
#A complex array for indexing

arr_complex_for_indexing_iii = np.array(

    [

        [

            [

                [np.arange(1,5,1),np.arange(5,10,1),np.arange(10,15,1)],

                [np.arange(15,20,1),np.arange(20,25,1),np.arange(25,30,1)],

                [np.arange(30,35,1),np.arange(35,40,1),np.arange(40,45,1)],

                [np.arange(55,50,1),np.arange(50,55,1),np.arange(55,60,1)]

            ]

        ]

    ]

)



print(arr_complex_for_indexing_iii.ndim)

# what is ndim - to reach the innermost array you have to use this much pair of square brackets + 1 --lol

print((((arr_complex_for_indexing_iii[0:,0])[0:,0])[0:,0])[0][1])
#A complex array for indexing

arr_complex_for_indexing_iii = np.array(

    [

        [

            [

                [

                    [np.arange(1,5,1),np.arange(5,10,1),np.arange(10,15,1)],

                    [np.arange(15,20,1),np.arange(20,25,1),np.arange(25,30,1)],

                    [np.arange(30,35,1),np.arange(35,40,1),np.arange(40,45,1)],

                    [np.arange(55,50,1),np.arange(50,55,1),np.arange(55,60,1)]

                ]

            ]

        ]

    ]

)



print("The  array is of ", arr_complex_for_indexing_iii.ndim, " dimensions.")

# what is ndim - to reach the innermost array you have to use this much pair of square brackets + 1--lol

print((((arr_complex_for_indexing_iii[0:,0])[0:,0])[0:,0])[0:,0][0][3])
arr_to_test = np.arange(1,10,2)

print(arr_to_test)



#Multiple with 2

arr_mul_2 = arr_to_test * 2

print(arr_mul_2)



#adding to other

arr_mul_3 = arr_mul_2 * arr_mul_2

print(arr_mul_3)



#Multiply by other - To do the operation two numpy array they have to be of same shape

arr_mul_4 = arr_mul_2 * np.arange(1,6,1)

print(arr_mul_4)
# Greater than 30.

print(arr_mul_4 > 30)

#To get the values

print(arr_mul_4[arr_mul_4 > 30])
#mean

print("The mean is ", np.mean(arr_mul_4))



#median

print("The mean is ", np.median(arr_mul_4))
dict_1 = {

    "shahrukh":23,

    "zenab":28

}

print(dict_1["shahrukh"])
dict_lvl1 = {

    "dict_lvl2":{

        "dict_lvl3_1":{

            "shahrukh":23,

            "asad":28

        },

        "dict_lvl3_2":{

            "shahrukh":23,

            "zenab":28

        }

    }

}



print(((dict_lvl1["dict_lvl2"])["dict_lvl3_1"]))

print("variable Type: %s" %type (dict_lvl1))
dict_test_2 = {

    "name": "shahrukh",

    "age": 25

}

#dict_test_2.items()



dict_test_3 = dict([('name','shahrukh'),('age',25)])

print("The keys are %s" %dict_test_3.keys())

print("The values are %s" %dict_test_3.values())

print("The str is %s" %str(dict_test_3))
# Broadcasting is numpy technique to reduce code and make it efficent.

# C = A + B

# a11 + b11, a12 + b12, a13 + b13

# C = (a21 + b21, a22 + b22, a23 + b23)



A = np.array([[1, 2, 3], [1, 2, 3]])



b = np.array([1, 2, 3])



C = A + b

print(C)
import pandas as pd

from bokeh.plotting import figure, output_file, show
# Load the data

# Use pd.read_csv

football_data = pd.read_csv("../input/Football_CompleteDataset.csv",na_values = 0, index_col=False)
# Get information about the data

football_data.info()
# Get the shape of data

football_data.shape
# Get the first five

football_data.head()
# Type of Data

type(football_data)
#Check the columns

football_data.columns
# Check few values

football_data.head()
# Dropping irrelevant coulmn

football_data = football_data.drop(['Unnamed: 0', 'Flag', 'Club Logo', 'Photo'], axis=1)

football_data
football_data.columns
import seaborn as sns

import matplotlib.pyplot as plt
sns.distplot(football_data['Age'], hist = True)
sns.boxplot(football_data['Age'])
#selecting relevant columns (according to me)

football_data = football_data[['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',

       'Wage', 'Special', 'Acceleration', 'Aggression', 'Agility', 'Balance',

       'Ball control', 'Crossing', 'Curve', 'Dribbling',

       'Finishing', 'Free kick accuracy', 'Heading accuracy',

       'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking',

       'Penalties', 'Short passing', 'Shot power',

       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',

       'Strength', 'Vision']]
football_data.groupby(['Club'])['Name'].count().sort_values(ascending = False)
#Get Nationalities count in each club

football_data.groupby(['Club','Nationality'])['Name'].count().sort_values()
data_to_test = (football_data[(football_data['Strength']) == '70+1'])
# Searching special characters in the columns.

col = ['Name', 'Nationality', 'Club', 'Value',

       'Wage', 'Acceleration', 'Aggression', 'Agility', 'Balance',

       'Ball control', 'Crossing', 'Curve', 'Dribbling',

       'Finishing', 'Free kick accuracy', 'Heading accuracy',

       'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking',

       'Penalties', 'Short passing', 'Shot power',

       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',

       'Strength', 'Vision']



# Write a function to check special characters in columns and same it a different files.

    

mask = np.column_stack([football_data[col].str.contains(r'[+-]', na=False) for col in football_data[col]])

football_clean = (football_data.loc[~mask.any(axis=1)])
col = ['Name', 'Nationality', 'Club', 'Value',

       'Wage', 'Acceleration', 'Aggression', 'Agility', 'Balance',

       'Ball control', 'Crossing', 'Curve', 'Dribbling',

       'Finishing', 'Free kick accuracy', 'Heading accuracy',

       'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking',

       'Penalties', 'Short passing', 'Shot power',

       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',

       'Strength', 'Vision']

for col in data_to_test[col]:

    data_to_test[col] = data_to_test[col].str.replace('€','').all()



data_to_test
# Convert all the character values to numeric.

columns_to_convert = ['Acceleration', 'Aggression', 'Agility', 'Balance',

       'Ball control', 'Crossing', 'Curve', 'Dribbling',

       'Finishing', 'Free kick accuracy', 'Heading accuracy',

       'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking',

       'Penalties', 'Short passing', 'Shot power',

       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',

       'Strength', 'Vision']

football_clean[columns_to_convert] = football_clean[columns_to_convert].apply(pd.to_numeric)
import itertools as it

import math as mt
rows = np.arange(0, mt.floor((mt.sqrt(len(columns_to_convert)))))

columns = rows

iter_rows_cols = it.product(rows, columns)
iterat_for_plot = list(iter_rows_cols)
f, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=False)

colm_count = 0;

for i in iterat_for_plot:

    column_to_draw = columns_to_convert[colm_count]

    sns.boxplot(football_clean[column_to_draw], color = 'g', ax = axes[i])

    colm_count = colm_count + 1
# Replace all the euros icons

col = ['Name', 'Nationality', 'Club', 'Value',

       'Wage', 'Acceleration', 'Aggression', 'Agility', 'Balance',

       'Ball control', 'Crossing', 'Curve', 'Dribbling',

       'Finishing', 'Free kick accuracy', 'Heading accuracy',

       'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking',

       'Penalties', 'Short passing', 'Shot power',

       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',

       'Strength', 'Vision']

for col in ['Value','Wage']:

    football_clean[col] = football_clean[col].str.replace('€','')
# Covert wages and values to numeric. Write a function

def convert_unites(val_to_convert,conversion = 'numeric'):

    

    if(conversion == 'numeric'):

        units_numeric = dict([

            ('K', 1000),

            ('M', 1000000),

            ('B', 1000000000)

        ])

        mask_contain_k = np.column_stack([val_to_convert.str.contains('K', na=False)])

        mask_contain_m = np.column_stack([val_to_convert.str.contains('M', na=False)])

        val_to_convert[mask_contain_m.flatten()] = (val_to_convert[mask_contain_m.flatten()].str.replace('M','')).astype(float)* units_numeric['M']

        val_to_convert[mask_contain_k.flatten()] = (val_to_convert[mask_contain_k.flatten()].str.replace('K','')).astype(float)* units_numeric['K']

        return val_to_convert

football_clean.apply(lambda x: (convert_unites(x)) if x.name in ['Value', 'Wage'] else x, axis=0)
cols_to_num = ['Value', 'Wage']

football_clean[cols_to_num] = football_clean[cols_to_num].apply(pd.to_numeric)
f, axes = plt.subplots(2,1,figsize=(20, 15))

#axes[0,0].set(xscale="log"

plot_value = sns.boxplot(football_clean['Value'], ax = axes[0], color='g')

plot_value.tick_params(labelsize=14)

plot_value.set_xlabel("Value", fontsize = 14)



plot_wage = sns.boxplot(football_clean['Wage'], ax = axes[1], color='g')

plot_wage.tick_params(labelsize=14)

plot_wage.set_xlabel("Wage", fontsize = 14)

plt.show()
f, axes = plt.subplots(figsize=(20,10))

sns.set(style = 'white')

football_corr = football_clean.corr(method='pearson')

cmap = sns.diverging_palette(220, 10, as_cmap=True)

plot_corr = sns.heatmap(football_corr, cmap=cmap, vmax=.4, center=0,

            square=True, linewidths=2, cbar_kws={"shrink": 1})

plot_corr.set_title('Correlation among players features', fontsize = 20, pad = 20)

plt.show()
cm = sns.light_palette("green", as_cmap=True)

football_corr.style.background_gradient(cmap=cm)
f, axes = plt.subplots(2,2,figsize=(20,20))

sns.scatterplot(x=football_clean['Wage'], y=football_clean['Value'],alpha=0.5, ax = axes[0,0])

sns.scatterplot(x=football_clean['Interceptions'], y=football_clean['Marking'], alpha=0.5, ax = axes[0,1])

sns.scatterplot(x=football_clean['Stamina'], y=football_clean['Acceleration'], alpha = 0.5, ax = axes[1,0])

sns.scatterplot(x=football_clean['Penalties'], y=football_clean['Finishing'], alpha = 0.5, ax = axes[1,1])

plt.show()
# Put in the clustering algo in the datasets

from sklearn.cluster import KMeans
def calculate_distortion(dtf):

    distortions = []

    K = range(1,10)

    for k in K:

        kmeanModel = KMeans(n_clusters=k)

        kmeanModel.fit(dtf)

        distortions.append(kmeanModel.inertia_)

    return distortions
val_dt = pd.DataFrame(football_clean['Value'])

dist_value = calculate_distortion(val_dt)
plt.figure(figsize=(16,8))

plt.plot(range(1,10), dist_value, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
kmeanModel = KMeans(n_clusters=5)

kmeanModel.fit(val_dt)
y_means_val=kmeanModel.predict(val_dt)

val_dt['y_means_val'] = y_means_val
fig, axes = plt.subplots(figsize=(16,8))

plt.scatter(val_dt.iloc[:,0], val_dt.iloc[:,1], c=y_means_val, s=50, cmap='viridis')

plt.show()
dt_stamin_Acc = football_clean[['Stamina','Acceleration']]

dist_stm_acc = calculate_distortion(dt_stamin_Acc)
plt.figure(figsize=(16,8))

plt.plot(range(1,10), dist_stm_acc, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
kmeanModel = KMeans(n_clusters=3)

kmeanModel.fit(dt_stamin_Acc)
y_means_st_acc=kmeanModel.predict(dt_stamin_Acc)
fig, axes = plt.subplots(figsize=(16,8))

plt.scatter(dt_stamin_Acc.iloc[:,0], dt_stamin_Acc.iloc[:,1], c=y_means_st_acc, s=50, cmap='viridis', alpha = 0.7)

plt.show()
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics
football_clean.describe()
football_clean.isnull().any()
print("The number of observations with no clubs is", football_clean['Club'].isnull().sum())
independent_val_1 = football_clean[['Acceleration', 'Aggression', 'Agility', 'Balance',

       'Ball control', 'Crossing', 'Curve', 'Dribbling',

       'Finishing', 'Free kick accuracy', 'Heading accuracy',

       'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking',

       'Penalties', 'Short passing', 'Shot power',

       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',

       'Strength', 'Vision']].values

independent_val = football_clean[['Wage']].values
dependent_value = football_clean['Value'].values
independent_val_train, independent_val_test, dependent_value_train, dependent_value_test = train_test_split(independent_val, dependent_value, test_size = 0.2, random_state=0)
independent_val = football_clean[[ 'Ball control']].values

dependent_value = football_clean['Dribbling'].values

independent_val_train, independent_val_test, dependent_value_train, dependent_value_test = train_test_split(independent_val, dependent_value, test_size = 0.2, random_state=0)



# Train the Model



regressor = LinearRegression()

regressor.fit(independent_val_train, dependent_value_train)



cm = sns.light_palette("green", as_cmap=True)



#coeff_df = pd.DataFrame(regressor.coef_, football_clean[['Wage']].columns, columns=['Coefficient'])  

coeff_df.sort_values('Coefficient').style.background_gradient(cmap=cm)



val_predict = regressor.predict(independent_val_test)

df = pd.DataFrame({'Actual': dependent_value_test, 'Predicted': val_predict})

df.head(25)

#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(dependent_value_test, val_predict)))