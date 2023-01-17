import pandas as pd
#Load the data set

hdf = pd.read_csv('../input/housing.csv')
import numpy  as np
hdf.columns
# Find the datatypes of the data

[i + ' '+ str(hdf[i].dtype) for i in hdf.columns]
#Let's have a look at the numeric variables. 

hdf.describe()
#Transform the data inside the dataframe

hdf['avg_rooms'] = hdf.total_rooms/hdf.households

hdf['avg_bedrooms'] = hdf.total_bedrooms/hdf.households

hdf['avg_non_bedrooms'] = hdf.avg_rooms - hdf.avg_bedrooms

hdf['household_size'] = hdf.population/hdf.households

hdf['median_house_value_100000'] = hdf.median_house_value/100000
#Now let's check the values

hdf.describe()
#First let's have a look what categories are stored in this column. 

hdf.ocean_proximity.unique()
#Now we need to append our new dummy variables to the main data frame. 

dummy_cols = pd.get_dummies(hdf.ocean_proximity)

hdf = pd.concat([hdf, dummy_cols ],axis=1)
#Save records with missing data in its own dataframe. 

miss_df = hdf[hdf.isnull().T.any().T]

hdf = hdf[~hdf.isnull().T.any().T]

print('There are {} missing records.'.format(str(len(miss_df))))
hdf.info()
import matplotlib.pyplot as plt
hdf.iloc[:,:3].boxplot(figsize=(10,5))

plt.title('Box and Whisker Plots for Housing Data')

plt.show()

hdf[['median_income', 'median_house_value_100000']].boxplot(figsize=(10,5))

plt.show()

hdf.iloc[:,-10:-6].boxplot(figsize=(10,5))

plt.show()
hdf.head()
bar_df = hdf[['ocean_proximity', 'median_house_value']]

bar_df.columns = ['ocean_proximity', 'count']

bar_df.groupby('ocean_proximity').count().plot(kind='bar')

plt.title('Count of Geographic Categories')
hdf=hdf[['longitude', 'latitude', 'housing_median_age', 'population', 

         'median_income', 'avg_rooms','avg_bedrooms', 'avg_non_bedrooms', 'household_size', '<1H OCEAN',

        'INLAND', 'ISLAND','NEAR BAY', 'NEAR OCEAN', 'median_house_value']]
hdf.hist(figsize=(20,15))

plt.tight_layout()
import seaborn as sea    
corr_matrix = hdf.corr()

fig, ax = plt.subplots(figsize=(15,15))

sea.heatmap(corr_matrix, annot= True)

plt.title('Correlation Matrix: California Housing Data')
ft_list = abs(corr_matrix.iloc[:-1,-1]).sort_values()

ft_list
ft_list.plot(kind='barh', figsize=(30,15))

plt.title('Feature Importance Chart: California Housing Data', fontsize = 20)

plt.xlabel('Correlation', fontsize=18)

plt.ylabel('Variable', fontsize=18)
corr_matrix.iloc[:-1,-1]
#Let's look and see if there are still values that look unreal in the dataset. 

hdf.describe()
hdf.sort_values(by='avg_non_bedrooms', ascending=False).head(10)
#Look at the average non-bedrooms variable for potential outliers. 

rom_out= hdf[~(np.abs(hdf.avg_non_bedrooms-hdf.avg_non_bedrooms.mean())

               <= (3*hdf.avg_non_bedrooms.std()))]

hdf_no_ouliers= hdf[(np.abs(hdf.avg_non_bedrooms-hdf.avg_non_bedrooms.mean())

                               <= (3*hdf.avg_non_bedrooms.std()))]

print('''There are possible {} outling points in the avg_non_bedrooms variable. Lets make an attempt at modeling the data before we take further action.'''.format(str(len(rom_out))))
rom_out= hdf_no_ouliers[~(np.abs(hdf_no_ouliers.household_size-hdf_no_ouliers.household_size.mean())

                          <= (3*hdf_no_ouliers.household_size.std()))]

hdf_no_ouliers = hdf_no_ouliers[(np.abs(hdf_no_ouliers.household_size-hdf_no_ouliers.household_size.mean())

                               <= (3*hdf_no_ouliers.household_size.std()))]

print('''There are possible {} outling points in the household_size variable. Lets make an attempt at modeling the data before

we take further action.'''.format(str(len(rom_out))))
def stdize_data(df):

    mean = df.mean()

    std = df.std()

    std_df = (mean - df)/std

    return mean, std, std_df
def unstdize_data(npar, mean, std):

    new_values = ((npar*std)-mean)*-1

    return new_values
hdf_dummies = hdf[['INLAND', '<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN']]
m, s, new_df = stdize_data(hdf[['median_income', 

                                'avg_non_bedrooms', 'housing_median_age', 

                                'avg_bedrooms', 'household_size', 'latitude', 'median_house_value' ]])
#Let's inspect the new values. 

new_df = pd.concat([hdf_dummies, new_df], axis=1)

new_df.head()
from sklearn.model_selection import train_test_split
x= new_df.iloc[:,:-1].values

x.shape
y= new_df.iloc[:,-1].values

y.shape
y
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 1234, test_size=.3)

x_train.shape, y_train.shape, x_test.shape, y_test.shape
#Setting up the packages to build our models

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam, SGD
def keras_model_lreg(var_cnt=2):

    model= Sequential()

    model.add(Dense(1, input_shape= (var_cnt,)))

    model.compile(SGD(lr=.0001), 'mean_squared_error')

    return model
reg= keras_model_lreg(var_cnt=10)

reg.fit(x_train, y_train, epochs = 30)
y_pred = reg.predict_classes(x_test)
from sklearn.metrics import mean_squared_error
#Unstandardize the Data

y_test = unstdize_data(y_test, m[-1], s[-1])

y_pred = unstdize_data(y_pred, m[-1], s[-1])
rmse = np.sqrt(mean_squared_error(y_pred, y_test))

mpe  = np.mean((y_test- y_pred)/y_test*100)

mape = np.mean(abs(y_test- y_pred)/y_test*100)

me = np.mean((y_test- y_pred))
print('RMSE = {}'.format(rmse))

print('MPE = {}'.format(mpe))

print('MAPE = {}'.format(mape))

print('ME = {}'.format(me))
pd.DataFrame(y_pred).drop_duplicates()
from sklearn.linear_model import LinearRegression
hdf_no_ouliers.head()
x = hdf_no_ouliers[['INLAND', '<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN','median_income', 

                    'avg_non_bedrooms', 'housing_median_age', 'household_size' ]].values
y= hdf_no_ouliers.iloc[:,-1].values
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 1234, test_size=.3)
l_reg = LinearRegression()
l_reg.fit(x_train, y_train)
y_pred = l_reg.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))

mpe  = np.mean((y_test- y_pred)/y_test*100)

mape = np.mean(abs(y_test- y_pred)/y_test*100)

me = np.mean((y_test- y_pred))
print('RMSE = {}'.format(rmse))

print('MPE = {}'.format(mpe))

print('MAPE = {}'.format(mape))

print('ME = {}'.format(me))
y_pred = l_reg.predict(x_train)
rmse = np.sqrt(mean_squared_error(y_pred, y_train))

mpe  = np.mean((y_train- y_pred)/y_train*100)

mape = np.mean(abs(y_train- y_pred)/y_train*100)

me = np.mean((y_train- y_pred))
print('RMSE = {}'.format(rmse))

print('MPE = {}'.format(mpe))

print('MAPE = {}'.format(mape))

print('ME = {}'.format(me))