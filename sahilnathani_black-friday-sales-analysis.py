import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("=../BlackFriday.csv", encoding='ISO-8859-1')

df = df.drop(['Product_ID', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3'], axis=1)
df.isnull().sum()

df.columns
df.fillna(0, inplace=True)

len(df)
sns.set_palette("husl")

cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']

n=0

plt.figure(figsize=(30, 20))

for each in cols:

    n+=1

    plt.subplot(3, 2, n)

    sns.countplot(each, data=df)   

    

#Young People shop the most i.e. 18-35.

#People new to the cities are the heaviest spenders.
#Hued Plots

n=0

plt.figure(figsize=(30, 20))

for each in cols:

    n+=1

    plt.subplot(3, 2, n)

    sns.countplot(each, data=df, hue='City_Category')

    

#Occupation 4 occupies mainly unmarried people.

#Occupation 9 occupies more women than men.

#Occupation 10, 17 mostly live in C category cities.
#Product Plots

plt.figure(figsize=(18, 6))

sns.countplot('Product_Category_1', data=df, hue='Marital_Status')

    

#In Category_1, (1, 5, 8) are popular.

#Product_Category_1 9, 14, 17 while 1, 5, 8 are the most Popular.

#Product_Category_1 9, 17, 18 seem like mens' product only.
x = pd.DataFrame(df.groupby(['User_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years',

                 'Marital_Status'])['Purchase'].sum())



x = x['Purchase'][0:]
df = pd.DataFrame(x).reset_index()

df.columns = ['User_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years',

              'Marital_Status', 'Purchase']
out_data = pd.DataFrame(df)

out_data.columns= df.columns

out_data.to_excel('Black_Friday_Sales_for_Tableau.xlsx', header=True)
df_purchase_wise = df['Purchase'].agg(['mean', 'max', 'min', 'median', 'std'])
df = df.drop(['User_ID', 'Purchase'], 1)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])

df['Age'] = le.fit_transform(df['Age'])

df['City_Category'] = le.fit_transform(df['City_Category'])

stay = []

for each in df['Stay_In_Current_City_Years']:

    if each=='4+':

        stay.append(4)

    else:

        stay.append(each)



df['Stay_In_Current_City_Years'] = stay  

df['Stay_In_Current_City_Years'] = [int(x) for x in df['Stay_In_Current_City_Years']]
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



for each in ['Age', 'Gender', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']:

    x = np.array(df.drop([each], 1))

    y = np.array(df[each])

    print('Predicting '+each)

    from sklearn.cross_validation import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=47)

    '''

    names_r = ['RandomForestRegressor', 'AdaBoostRegressor', 'DecisionTreeRegressor']

    Regressors = [RandomForestRegressor(n_estimators=500), AdaBoostRegressor(), DecisionTreeRegressor()]



    for n, c in zip(names, Regressors):

        c.fit(x_train, y_train)

        score = c.score(x_test, y_test)

        print('Accuracy achieved by ', n, 'is ', score*100)

    '''  



    names_c = ['RandomForestClassifier', 'AdaBoostClassifier', 'DecisionTreeClassifier', 'MLPClassifier']

    classifiers = [RandomForestClassifier(n_estimators=100), AdaBoostClassifier(), DecisionTreeClassifier(), MLPClassifier()]

    for n, c in zip(names_c, classifiers):

        c.fit(x_train, y_train)

        score = c.score(x_test, y_test)

        print('Accuracy achieved by ', n, 'is ', score*100)

    print('____________________________________________________________________________________________')    
df['Stay_In_Current_City_Years'].unique()