import requests

import os
# This function will be used to acquire the data from the UCI website

def aquire_data(path_to_data, data_urls):

    if not os.path.exists(path_to_data):

        os.mkdir(path_to_data)

        

    for url in data_urls:

        data = requests.get(url).content

        filename = os.path.join(path_to_data, os.path.basename(url))

        with open(filename, 'wb') as file: 

            file.write(data)
data_urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",

             "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",

             "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"]



aquire_data('data', data_urls)
# Check the success of accessing the data

print('Output n° {}\n'.format(1))

! find data
column_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", 

                "Martial Status", "Occupation", "Relationship", "Race", "Sex", 

                "Capital-Gain", "Capital-Loss", "Hours-per-week", "Country", "Income"] 

import pandas as pd

import numpy as np
train = pd.read_csv('data/adult.data', names=column_names, sep=' *, *', na_values='?', 

                   engine='python')

test = pd.read_csv('data/adult.test', names=column_names, sep=' *, *', skiprows=1, 

                   engine='python', na_values='?')
test.Income.unique() 
train.Income.unique()
test.Income = np.where(test.Income == '<=50K.', '<=50K', '>50K')
# Concatenate train and test. We will split it before the training phase 

df = pd.concat((train, test), axis=0)
df.Income.unique()
print('Output n° {}\n'.format(2))



'''

First 5 observations

'''

df.head()
print('Output n° {}\n'.format(3))



'''

Last 5 observations

'''

df.tail()
print('Output n° {}\n'.format(4))



print('Our data contains {} observations and {} columns.'.format(df.shape[0],

                                                                df.shape[1]))
print('Output n° {}\n'.format(5))

print(df.isnull().sum())
print('Output n° {}\n'.format(6))

print(df.dtypes)
# Workclass  

print('Output n° {}\n'.format(7))

print('Number of missing values: {}'.format(len(df['Workclass'].unique())))

print(df['Workclass'].unique())
# Occupation  

print(print('Output n° {}\n'.format(8)))

print('Number of missing values: {}'.format(len(df['Occupation'].unique())))

print(df['Occupation'].unique())
# Country  

print('Output n° {}\n'.format(9))

print('Number of missing values: {}'.format(len(df['Country'].unique())))

print(df['Country'].unique())

import statistics as stat
def fill_categorical_missing(data, column):

    data.loc[data[column].isnull(), column] = stat.mode(data[column])
cols_to_fill = ['Workclass', 'Occupation', 'Country']



for col in cols_to_fill:

    fill_categorical_missing(df, col)



print('Output n° {}\n'.format(10))



# Check the final data if there is any missing values 

print(df.isnull().sum())
df_cp = df.copy()
df_cp.head()
df_cp.describe()
import seaborn as sns 

import numpy as np

import matplotlib.pyplot as plt
# Age 

sns.boxplot(y='Age', data=df_cp)

plt.show()
def ten_to_ten_percentiles(data, column):

    for i in range(0,100,10):

        var = data[column].values

        var = np.sort(var, axis=None)

        print('{} percentile value is {}'.format(i, var[int(len(var) * (float(i)/100))]))

    print('100 percentile value is {}'.format(var[-1]))
ten_to_ten_percentiles(df_cp, 'Age')
#calculating column values at each percntile 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100

def percentiles_from_90(data, column):

    for i in range(90,100):

        var = data[column].values

        var = np.sort(var, axis=None)

        print('{} percentile value is {}'.format(i, var[int(len(var) * (float(i)/100))]))

    print('100 percentile value is {}'.format(var[-1]))
#calculating colunm values at each percntile 99.0,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100

def percentiles_from_99(data, column):

    for i in np.arange(0.0, 1.0, 0.1):

        var =data[column].values

        var = np.sort(var,axis = None)

        print("{} percentile value is {}".format(99+i,var[int(len(var)*(float(99+i)/100))]))

    print("100 percentile value is ",var[-1])
# Education-Num

sns.boxplot(y='Education-Num', data=df_cp)

plt.show()
ten_to_ten_percentiles(df_cp, 'Education-Num')
# Capital-Gain

sns.boxplot(y='Capital-Gain', data=df_cp)

plt.show()
ten_to_ten_percentiles(df_cp, 'Capital-Gain')
percentiles_from_90(df_cp, 'Capital-Gain')
percentiles_from_99(df_cp, 'Capital-Gain')
# Removing the outliers based on 99.5th percentile of Capital-Gain

df_cp = df_cp[df_cp['Capital-Gain']<=34095]
# Capital-Gain

sns.boxplot(y='Capital-Gain', data=df_cp)

plt.show()
# Capital-Loss

sns.boxplot(y='Capital-Loss', data=df_cp)

plt.show()
ten_to_ten_percentiles(df_cp, 'Capital-Loss')
percentiles_from_90(df_cp, 'Capital-Loss')
percentiles_from_99(df_cp, 'Capital-Loss')
# Hours-per-week

sns.boxplot(y='Hours-per-week', data=df_cp)

plt.show()
ten_to_ten_percentiles(df_cp, 'Hours-per-week')
def remove_outliers(data):

    a = data.shape[0]

    print("Number of salary records = {}".format(a))

        

    temp_data = data[data['Capital-Gain']<=34095]

    b = temp_data.shape[0]

    

    print('Number of outliers from the Capital-Gain column= {}'.format(a - b))

        

    data = data[(data['Capital-Gain']<=34095)]

    

    print('Total outlies removed = {}'.format(a-b))

    print('-----'*10)

    return data
print('Removing all the outliers from the data')

print('-----'*10)

df_no_outliers = remove_outliers(df)



proportion_remaing_data = float(len(df_no_outliers)) / len(df)

print('Proportion of observation that remain after removing outliers = {}'.format(proportion_remaing_data))
df_no_outliers.Income.unique()
palette = {"<=50K":"r", ">50K":"g"}

sns.countplot(x="Income", data=df_no_outliers, hue="Income", palette=palette)
df_no_outliers.describe()
# Age  

df_no_outliers.Age.plot(kind='kde', title='Density plot for Age', color='c')
# Capital-Gain  

df_no_outliers['Capital-Gain'].plot(kind='kde', title='Density plot for Capital-Gain', color='c')
# Capital-Loss  

df_no_outliers['Capital-Loss'].plot(kind='kde', title='Density plot for Capital-Loss', color='c')
# Capital-Loss  

df_no_outliers['Hours-per-week'].plot(kind='kde', title='Density plot for Hours-per-week', color='c')
# Capital-Gain and Education-Num 

# use scatter plot for bi-variate distribution

df_no_outliers.plot.scatter(x='Education-Num', y='Capital-Gain', color='c', title='scatter plot : Education-Num vs Capital-Gain');
# Hours-per-week and Education-Num 

# use scatter plot for bi-variate distribution

df_no_outliers.plot.scatter(x='Education-Num', y='Hours-per-week', color='c', title='scatter plot : Education-Num vs Hours-per-week');
# Capital-Gain and Hours-per-week

# use scatter plot for bi-variate distribution

df_no_outliers.plot.scatter(x='Hours-per-week', y='Capital-Gain', color='c', title='scatter plot : Hours-per-week vs Capital-Gain');
# Capital-Gain and Capital-Loss

# use scatter plot for bi-variate distribution

df_no_outliers.plot.scatter(x='Capital-Gain', y='Capital-Loss', color='c', title='scatter plot : Capital-Loss vs Capital-Gain');
numerical_cols = ['int64']  

plt.figure(figsize=(10, 10))

sns.heatmap( 

            df_no_outliers.select_dtypes(include=numerical_cols).corr(),

            cmap=plt.cm.RdBu, 

            vmax=1.0,

            linewidths=0.1,

            linecolor='white', 

            square=True,

            annot=True

)
df_no_outliers.head()
df_no_outliers['Country'].unique()
south_df = df_no_outliers[df_no_outliers['Country']=='South']

a = south_df.shape[0]

b = df_no_outliers.shape[0]



print('{} rows corresponds to South, which represents {}% of the data'.format(a, (1.0*a/b)*100))
south_index = south_df.index 

df_no_outliers.drop(south_index, inplace=True)
# Changing the corresponding values.

df_no_outliers.loc[df_no_outliers['Country']=='Outlying-US(Guam-USVI-etc)', 'Country'] = 'Outlying-US'

df_no_outliers.loc[df_no_outliers['Country']=='Trinadad&Tobago', 'Country'] = 'Trinadad-Tobago'

df_no_outliers.loc[df_no_outliers['Country']=='Hong', 'Country'] = 'Hong-Kong'
# Check if the process worked

df_no_outliers['Country'].unique()
asia = ['India', 'Iran', 'Philippines', 'Cambodia', 'Thailand', 'Laos', 'Taiwan', 

       'China', 'Japan', 'Vietnam', 'Hong-Kong']  



america = ['United-States', 'Cuba', 'Jamaica', 'Mexico', 'Puerto-Rico', 'Honduras', 

           'Canada', 'Columbia', 'Ecuador', 'Haiti', 'Dominican-Republic', 

           'El-Salvador', 'Guatemala', 'Peru', 'Outlying-US', 'Trinadad-Tobago', 

           'Nicaragua', '']  



europe = ['England', 'Germany', 'Italy', 'Poland', 'Portugal', 'France', 'Yugoslavia', 

          'Scotland', 'Greece', 'Ireland', 'Hungary', 'Holand-Netherlands'] 
# Now, create a dictionary to map each country to a Corresponding continent. 

continents = {country: 'Asia' for country in asia}

continents.update({country: 'America' for country in america})

continents.update({country: 'Europe' for country in europe})
# Then use Pandas map function to map continents to countries  

df_no_outliers['Continent'] = df_no_outliers['Country'].map(continents)
df_no_outliers['Continent'].unique()
def Occupation_VS_Income(continent):

    choice = df_no_outliers[df_no_outliers['Continent']==continent] 

    countries = list(choice['Country'].unique())



    for country in countries:

        pd.crosstab(choice[choice['Country']==country].Occupation, choice[choice['Country']==country].Income).plot(kind='bar', 

                                                                                                                       title='Income VS Occupation in {}'.format(country))
Occupation_VS_Income('Asia')
Occupation_VS_Income('America')
Occupation_VS_Income('Europe')
def Workclass_VS_Income(continent):

    choice = df_no_outliers[df_no_outliers['Continent']==continent] 

    countries = list(choice['Country'].unique())



    for country in countries:

        pd.crosstab(choice[choice['Country']==country].Workclass, choice[choice['Country']==country].Income).plot(kind='bar', 

                                                                                                                       title='Income VS Workclass in {}'.format(country))
Workclass_VS_Income('Asia')
Workclass_VS_Income('America')
Workclass_VS_Income('Europe')
def MaritalStatus_VS_Income(continent):

    choice = df_no_outliers[df_no_outliers['Continent']==continent] 

    countries = list(choice['Country'].unique())



    for country in countries:

        pd.crosstab(choice[choice['Country']==country]['Martial Status'], choice[choice['Country']==country].Income).plot(kind='bar', 

                                                                                                                       title='Income VS Workclass in {}'.format(country))
MaritalStatus_VS_Income('Asia')
# reset_index(): to convert to aggregation result to a pandas dataframe.

agg_df = df_no_outliers.groupby(['Continent','Country', 'Martial Status'])['Capital-Gain'].mean().reset_index()
agg_df['Mean_Capital_Gain'] = agg_df['Capital-Gain']

agg_df.drop('Capital-Gain', axis=1, inplace=True)
agg_df.head()
import seaborn as sns
def Mean_TotCapital_VS_Marital_Status(continent):

    choice = agg_df[agg_df['Continent']==continent] 

    countries = list(choice['Country'].unique())



    for country in countries:

        df_c = choice[choice['Country']==country]

        ax = sns.catplot(x='Martial Status', y='Mean_Capital_Gain', 

                         kind='bar', data=df_c)



        ax.fig.suptitle('Country: {}'.format(country))

        ax.fig.autofmt_xdate()
Mean_TotCapital_VS_Marital_Status('Asia')
Mean_TotCapital_VS_Marital_Status('America')
Mean_TotCapital_VS_Marital_Status('Europe')
edu = df_no_outliers.Education.unique()

eduNum = df_no_outliers['Education-Num'].unique()

print('Education: \nTotal category:{}\nValues: {}\n'.format(len(edu),list(edu)))

print('Education Num: \nTotal Education-Num:{}\nValues: {}'.format(len(eduNum),

                                                                  list(eduNum)))
ax = sns.catplot(x='Education', y='Education-Num', kind='bar', data=df_no_outliers)

ax.fig.suptitle('Numerical Representation of Educations')

ax.fig.autofmt_xdate()
# Finally remove the Education column  

df_no_outliers.drop('Education', axis=1, inplace=True)
df_no_outliers['Capital-State'] = df_no_outliers['Capital-Gain'] - df_no_outliers['Capital-Loss']
# Then remove Capital-Gain and Capital-Loss. 

df_no_outliers.drop(['Capital-Gain', 'Capital-Loss'], axis=1, inplace=True)
'''

Let not forget to drop the 'Continent' column we added for 

visualization purpose. 

'''

df_no_outliers.drop('Continent', axis=1, inplace=True)
df_no_outliers.head(3)
# AgeState based on Age

df_no_outliers['AgeState'] = np.where(df_no_outliers['Age'] >= 18, 'Adult', 'Child')
# AgeState Counts  

df_no_outliers['AgeState'].value_counts()
sns.countplot(x='AgeState', data=df_no_outliers)
df_no_outliers.drop('fnlwgt', axis=1, inplace=True)
df_no_outliers.head()
# Information about our data

df_no_outliers.info()
# Columns: Workclass, Martial Status Occupation, Relationship, Race, Sex, Country, AgeState

df_no_outliers = pd.get_dummies(df_no_outliers, columns=['Workclass', 'Martial Status', 'Occupation', 

                                 'Relationship', 'Race', 'Sex', 'Country', 'AgeState'])
df_no_outliers['Income'].unique()
'''

1: For those who make more than 50K 

0: For those who don't

'''

df_no_outliers['Income'] = np.where(df_no_outliers['Income'] =='>50K', 1, 0)
# Reorder columns : In order to have 'Income' as last feature.

columns = [column for column in df_no_outliers.columns if column != 'Income']

columns = columns + ['Income'] 

df = df_no_outliers[columns]
# Information about our data

df.info()
y = df.Income.ravel()

X = df.drop('Income', axis=1).as_matrix().astype('float')
print('X shape: {} | y shape: {}'.format(X.shape, y.shape))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('X train shape: {} | y shape: {}'.format(X_train.shape, y_train.shape))

print('X test shape: {} | y shape: {}'.format(X_test.shape, y_test.shape))
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=0)
# Train the model 

dummy_clf.fit(X_train, y_train)
print('Score of baseline model : {0:.2f}'.format(dummy_clf.score(X_test, y_test)))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
lr_clf = LogisticRegression(random_state=0)

parameters = {'C':[1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty' : ['l1','l2']}

lr_clf = GridSearchCV(lr_clf, param_grid=parameters, cv=3)
lr_clf.fit(X_train, y_train)
lr_clf.best_params_
print('Best score : {0:.2f}'.format(lr_clf.best_score_))
print('Score for logistic regression - on test : {0:.2f}'.format(lr_clf.score(X_test, y_test)))