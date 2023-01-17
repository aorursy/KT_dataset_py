import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv' ,encoding= 'ISO-8859-1')  

print(f'Data Shape is {data.shape}')

data.head()
def drop(feature) :

    global data

    data.drop([feature],axis=1, inplace=True)

    data.head()



def unique(feature) : 

    global data

    print(f'Number of unique vaure are {len(list(data[feature].unique()))} which are : \n {list(data[feature].unique())}')



def unique_all(show_value = True) : 

    global data

    for col in data.columns : 

        print(f'Length of unique data for   {col}   is    {len(data[col].unique())} ')

        if show_value == True  : 

            print(f'unique values ae {data[col].unique()}' )

            print('-----------------------------')



def drop_nulls(percentage = 0.3) : 

    global data

    for col in data.columns : 

        ratio =  data[col].isna().sum()/data.shape[0]

        if ratio >= percentage : 

            data.drop([col],axis=1, inplace=True)

            print(f'Column {col} has been dropped since nulls percentage is {round(ratio *100)} %')



def count_nulls() : 

    global data

    for col in data.columns : 

        if not data[col].isna().sum() == 0 : 

            print(f'Column {col} has been number of nulls {data[col].isna().sum()}')





def fillna(feature , val = 'none') : 

    global data

    data[feature].fillna(val, inplace=True)



def cplot(feature) : 

    global data

    sns.countplot(x=feature, data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))



def spie(series) : 

    global data

    plt.pie(series.values,labels=list(series.index),autopct ='%1.2f%%',labeldistance = 1.1,explode = [0.05 for i in range(len(series.values))])

    plt.show()



def pie(feature) : 

    global data

    plt.pie(data[feature].value_counts(),labels=list(data[feature].value_counts().index),

        autopct ='%1.2f%%' , labeldistance = 1.1,explode = [0.05 for i in range(len(data[feature].value_counts()))] )

    plt.show()



def make_xy(feature) : 

    global data

    X = data.drop([feature], axis=1, inplace=False)

    y = data[feature]

    return X , y

   

def encoder(feature , new_feature, drop = True) : 

    global data

    enc  = LabelEncoder()

    enc.fit(data[feature])

    data[new_feature] = enc.transform(data[feature])

    if drop == True : 

        data.drop([feature],axis=1, inplace=True)

    

def max_counts(feature, number, return_rest = False) : 

    global data

    counts = data[feature].value_counts()

    values_list = list(counts[:number].values)

    rest_value =  sum(counts.values) - sum (values_list)

    index_list = list(counts[:number].index)



    if return_rest : 

        values_list.append(rest_value )

        index_list.append('rest items')



    result = pd.Series(values_list, index=index_list)



    if len(data[feature]) <= number : 

        result = None

    return result



def remove_zero(feature , val = 0) :

    global data

    data = data[data[feature] != val]

    

def show_details() : 

    global data

    for col in data.columns : 

        print(f'for feature : {col}')

        print(f'Number of Nulls is   {data[col].isna().sum()}')

        print(f'Number of Unique values is   {len(data[col].unique())}')

        print(f'random Value {data[col][0]}')

        print(f'random Value {data[col][10]}')

        print(f'random Value {data[col][20]}')

        print('--------------------------')

data.shape
drop_nulls()
data.shape
drop('eventid')

drop('country')

drop('region')

drop('attacktype1')

drop('targtype1')

drop('targsubtype1')

drop('natlty1')

drop('weaptype1')

drop('weapsubtype1')
data.head()
unique_all(False)
count_nulls()
show_details()
unique('iyear')
cplot('iyear')
unique('imonth')
data.shape[0]
remove_zero('imonth')
data.shape[0]
unique('imonth')
cplot('imonth')
unique('iday')
remove_zero('iday')
unique('iday')
data.shape[0]
unique('extended')
data['latitude'].isna().sum()
fillna('latitude',0)

remove_zero('latitude')
data['latitude'].isna().sum()
data.shape[0]
fillna('longitude',0)

remove_zero('longitude')

data['longitude'].isna().sum()
data.shape[0]
data['specificity'].isna().sum()
data['vicinity'].isna().sum()
data['doubtterr'].isna().sum()
unique('doubtterr')
fillna('doubtterr',33)

remove_zero('doubtterr',33)

data['doubtterr'].isna().sum()
data['multiple'].isna().sum()
unique('multiple')
fillna('multiple',33)

remove_zero('multiple',33)

data['multiple'].isna().sum()
data['guncertain1'].isna().sum()
unique('guncertain1')
fillna('guncertain1',33)

remove_zero('guncertain1',33)

data['guncertain1'].isna().sum()
pie('guncertain1')
unique('nkill')
fillna('nkill',999999)

remove_zero('nkill',999999)

data['nkill'].isna().sum()
victims = max_counts('nkill',10, True)

victims
spie(victims)
data['nwound'].isna().sum()
unique('nwound')
fillna('nwound',999999)

remove_zero('nwound',999999)

data['nwound'].isna().sum()
wounded = max_counts('nwound',10, True)

wounded
spie(wounded)
data['ishostkid'].isna().sum()
unique('ishostkid')
fillna('ishostkid',33)

remove_zero('ishostkid',33)

data['ishostkid'].isna().sum()
data.shape
count_nulls()
fillna('provstate','other')

fillna('city','other')

fillna('targsubtype1_txt','other')

fillna('corp1','other')

fillna('target1','other')

fillna('natlty1_txt','other')

fillna('weapsubtype1_txt','other')
count_nulls()
encoder('provstate','provstate_code',True)

encoder('city','city_code',True)

encoder('targsubtype1_txt','targsubtype_code',True)

encoder('corp1','corp_code',True)

encoder('target1','target_code',True)

encoder('natlty1_txt','natlty_code',True)

encoder('weapsubtype1_txt','weapsubtype_code',True)



encoder('country_txt','country_code',True)    

encoder('region_txt','region_code',True)    

encoder('attacktype1_txt','attacktype_code',True)    

encoder('targtype1_txt','targtype_code',True)    

encoder('gname','gname_code',True)    

encoder('weaptype1_txt','weaptype_code',True)    

encoder('dbsource','dbsource_code',True)    
show_details()
data.head()
data.info()
X , y = make_xy('success')
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44, shuffle =True)

print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
cplot('success')
from sklearn.ensemble import GradientBoostingClassifier



GBCModel = GradientBoostingClassifier(n_estimators=100,max_depth=3,random_state=33) 

GBCModel.fit(X_train, y_train)
print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))

print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))
y_pred = GBCModel.predict(X_test)

y_pred_prob = GBCModel.predict_proba(X_test)

print('Predicted Value for GBCModel is : ' , y_pred[:10])

print('Prediction Probabilities Value for GBCModel is : ' , y_pred_prob[:10])