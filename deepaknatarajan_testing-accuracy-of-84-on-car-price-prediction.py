import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
data = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')
data.head()
new_data = data
new_data.columns
#re-naming the columns
new_data.rename({'odometer': 'mileage'}, axis= 1,inplace= True)
# Lets drop the columns that are not useful for the car price prediction.
drop_columns = ['id','url','region','region_url','lat','long','image_url','vin','county','state']
new_data = new_data.drop(drop_columns, axis =1)
new_data.shape
#lets drop the columns where price of the car is more than 2.5 million dollars because high end used cars like ferrari, porsche costs less than that
new_data = new_data.drop(new_data[new_data.price > 250000].index)
new_data.shape
#Lets drop the columns where year is less than 1990 and greater than 2019
new_data = new_data[(new_data['year'] > 1990) & (new_data['year'] < 2020)]
new_data.index = range(len(new_data))
new_data.shape
# finding the length of description of each car
new_data.dropna(subset = ['description'],inplace = True) # dropping the null values in 'desc' column
new_data['word_len'] = new_data.description.apply(lambda x: len(str(x.lower().split())))
new_data.index = range(len(new_data))
del new_data['description']
#finding the age of car
from datetime import datetime

current_year = datetime.now().year
new_data['Age'] = current_year - new_data['year']
del new_data['year']
# removing the 'cylinders' in the columns and converting into float.
new_data.cylinders = new_data.cylinders.apply(lambda x: x if str(x).lower()[-1] == 'o' or str(x).lower()[-1] == 'n' else str(x).lower().replace('cylinders', ''))
new_data.cylinders = pd.to_numeric(new_data.cylinders, errors = 'coerce')
new_data.cylinders.fillna(new_data.cylinders.median(), inplace = True)
# Lets drop the rows where has more than 1 NaN values.
new_data.dropna(thresh = 13, axis = 0, inplace = True)
new_data.shape
#dropping the duplicates in the data
new_data.drop_duplicates(keep = 'first', inplace = True)
new_data.index = range(len(new_data))
new_data.shape
# fixing the spelling errors
new_data.manufacturer = new_data.manufacturer.apply(lambda x: x.replace('porche', 'porsche') if x == 'porche' else x)
# creating new dataframe to find the size of each car in order to fill the null values in the size feature of same car
new = new_data.sort_values(by= ['size','manufacturer','type'])
new.drop_duplicates(subset = ['model','type'], keep = 'first', inplace = True)
new.index = range(len(new))
# dropping the null values
new.dropna(subset = ['size','model'], inplace= True)
new.index = range(len(new))
# dictionary that contain size of each car
size = dict(zip(new['model'], new['size']))
for i in range(len(new_data)):
    if str(new_data['size'][i]).lower()[0] == 'n' and (new_data['model'][i] in size.keys()):
        new_data['size'][i] = size[new_data['model'][i]]
# Adding the manufacturers name to list from the dataset and adding few more manufacturer after looking the make column in dataset
manuf = []
for i in new_data.manufacturer.value_counts().index:
    manuf.append(i)
manuf.append('Tesla')
manuf.append('Rolls-Royce')
manuf.append('genesis')
# Replace the nan values in the manufacturer column based on the make column 
# from make column we can identify manufacturer of the car as it sometimes contains the manufacturer name in it
d = new_data['manufacturer']
m = new_data['model']
for i in range(len(new_data)):
    if str(d[i]).lower()[0] == 'n':
        for x in str(m[i]).lower().split():
            for mm in manuf:
                if (len(x) > 4) & (x[:4] == mm.lower()[:4]):
                    new_data['manufacturer'][i] = mm
                elif (x[:3] == mm.lower()[:3]) & (x[:-1] == mm.lower()[:-1]):
                    new_data['manufacturer'][i] = mm
#dropping the null values
new_data.dropna(subset = ['transmission','type','manufacturer','model','fuel','mileage','title_status','paint_color','drive','size','condition'],inplace = True)
new_data.index = range(len(new_data))
new_data.shape
# Function to remove the outliers in the data
def outlier_removal(x):
    for i in x:
        z_score_od = np.abs(stats.zscore(new_data[i]))
        outliers = np.where(z_score_od > 3)
        new_data.drop(new_data.index[[i for i in outliers[0]]],inplace= True)
        new_data.index = range(len(new_data))  
        
outlier_removal(new_data._get_numeric_data().columns)
#price of car cant be zero so we are eliminating the price less than 30 dollars. Some people mention very less price and they will revealthe price directly to customer while buying and bargain later.
new_data = new_data[new_data['price'] > 50]
new_data.index = range(len(new_data))
new_data.shape
#shifting the 'age' to new place
new_data.insert(1, 'age', new_data['Age'])
del new_data['Age']
# creating the dictionary for integer labelling
cond_dict = {'new':10, 'like new':9, 'excellent':8, 'good':7, 'fair':5, 'salvage':3}
title_dict = {'clean': 6, 'lien': 4, 'rebuilt':3, 'salvage': 2, 'parts only': 1, 'missing': 0}
columns = list(['condition', 'title_status'])
dictionary = list([cond_dict, title_dict])
#Function that does the integer labelling.
def labelling(columns, dictionary):
    for i in range(len(columns)):
        new_data[columns[i]] = new_data[columns[i]].map(dictionary[i])

labelling(columns, dictionary)
# Finding the models of car which has fewer data
other_models = new_data.model.value_counts().index[new_data.model.value_counts().values < 5]
# Function replaces the less frequent models of car to other_models category
def model_edit(model_list, data):
    for i in range(len(data)):
        if data[i] in model_list:
            data[i] = 'other_models'

model_edit(other_models, new_data['model'])
#changing the type from float to int.
new_data['age'] = new_data['age'].astype(int)
new_data.head()
new_data.dtypes
new_data.isnull().sum() # Now data is clean and it has no null values
new_data.describe()
#saving the cleaned data to new csv file
new_data.to_csv('Final_data3', index= False)
Final_data = pd.read_csv('Final_data3')
Final_data.columns
plt.figure(figsize = (12,5))
sns.countplot(x = 'type', order = Final_data['type'].value_counts().index, data = Final_data)
plt.figure(figsize = (10,5))
sns.countplot(x = 'paint_color', order = Final_data['paint_color'].value_counts().index,data = Final_data)
plt.figure(figsize = (10,5))
sns.countplot(x = 'manufacturer', order = Final_data['manufacturer'].value_counts().index, data = Final_data)
plt.xticks(rotation = 90)
plt.figure(figsize = (10,5))
sns.countplot(x = 'age', order = Final_data['age'].value_counts().index, data = Final_data)
Final_data.boxplot('price')
sns.distplot(Final_data['price'])
plt.figure(figsize = (10,5))
corr = Final_data.corr()
sns.heatmap(corr, annot = True)
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
F1 = pd.get_dummies(Final_data, drop_first= True)
F1.head()
F1.shape
X = F1.iloc[:, 1:]
y = F1.iloc[:,0:1]
X = np.array(X)
y = np.array(y).reshape(-1)
# scaling the inputs
scaler = MinMaxScaler().fit(X)
scaledX = scaler.transform(X)
# Algorithms
lasso = Lasso()
ridge = Ridge()
gradboost = GradientBoostingRegressor()
rfreg100 = RandomForestRegressor(n_estimators = 100)
# parameters
kfold = KFold(n_splits = 5)
scoring = 'r2'
algo_list = list([rfreg100,lasso,ridge,gradboost])
algo_name = list(['rfreg100','lasso','ridge','gradboost'])
# Cross validation on various algorithms
def model_building(algo,X,y,fold,scoring):
    algo_score = []
    for i in algo:
        score = cross_val_score(i, X, y, cv=fold, scoring=scoring).mean()
        algo_score.append(score)
    return algo_score
result = model_building(algo_list, scaledX, y, kfold, scoring)
Final_score = dict(zip(algo_name,result))
Final_score 