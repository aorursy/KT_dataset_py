import pandas as pd



data = pd.read_csv("../input/Melbourne_housing_extra_data.csv")

data_cv = data.loc[data['Price'].notnull()][:300]

data.drop(data_cv.index,inplace = True)

data.reset_index(inplace=True)

data_cv.reset_index(inplace=True)
data
def miss(x):

    return(sum(x.isnull()))

data.apply(miss)
def address(stri):

    lo = len(stri)

    for i in range(lo):

        if (stri[i] ==" "):

            name = stri[i+1:]

            break

    return name       



k = []

l=[]

for i in data['Address']:

    k.append(address(i))    

data['name'] = pd.DataFrame(k)

for i in data_cv['Address']:

    l.append(address(i))    

data_cv['name'] = pd.DataFrame(l)
prize_land_pred = data.loc[data['Landsize'].notnull()]



col = ['Bedroom2', 'Bathroom' ,'Car','Landsize','BuildingArea','Type','Price']

data2 = prize_land_pred[col]



buliding_area_notnull = data2[data2['BuildingArea'].notnull()]

buliding_area_isnull = data2[data2['BuildingArea'].isnull()]







buliding_area_notnull.reset_index(inplace = True)

buliding_area_isnull.reset_index(inplace = True)
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

fig, ax = plt.subplots(figsize=(5,5)) 

sns.heatmap(buliding_area_notnull.corr(), annot=True)

plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

plt.scatter(buliding_area_notnull['BuildingArea'],buliding_area_notnull['Landsize'])
sns.lmplot(x='BuildingArea',y='Landsize',data = buliding_area_notnull, size=10,

           scatter_kws={"s": 50, "alpha": 1})

x= buliding_area_notnull.drop(['BuildingArea'],axis = 1)

y = buliding_area_notnull['BuildingArea']
filled = buliding_area_notnull.groupby(['Bedroom2', 'Bathroom' ,'Car','Type'], as_index =False).mean()

col2 = ['Bedroom2', 'Bathroom', 'Car', 'Type']

data_merge = data.merge(filled, on=col2, how='left')





data['Landsize_y'] = data_merge['Landsize_y']

data.loc[data['Landsize'].isnull(),'Landsize'] = data.loc[data['Landsize'].isnull()].Landsize_y

data['BuildingArea_y'] = data_merge['BuildingArea_y']

data.loc[data['BuildingArea'].isnull(),'BuildingArea'] = data.loc[data['BuildingArea'].isnull()].BuildingArea_y



data_cv
data_cv.reset_index(inplace =True)

data['age'] = 2017 - data['YearBuilt']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

var_mod = ['Suburb','Type','name']

le = LabelEncoder()

for i in var_mod:

    data[i] = le.fit_transform(data[i])

for i in var_mod:

    data_cv[i] = le.fit_transform(data_cv[i])
fig, ax = plt.subplots(figsize=(8,8)) 

sns.boxplot(x='Rooms',y='Price',data=data)
fig, ax = plt.subplots(figsize=(10,10)) 

sns.boxplot(x='Regionname',y='Price',data=data)
col1 =['Bedroom2', 'Bathroom' ,'Car','Type','Landsize','Price']



data[col]
colf = ['Bedroom2', 'Bathroom' ,'Car','Type','Landsize','Suburb','name','Price']

data2 =data[colf]

data2 = data2.dropna(axis=0)
x= data2.drop('Price',axis = 1)

y= data2['Price']



from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators = 10)



y
clf.fit(x,y)
colf2 = ['Bedroom2', 'Bathroom' ,'Car','Type','Landsize','Suburb','name']

pred = data.loc[data['Price'].isnull(),colf2]

to_cv = data_cv[colf2]
data_cv
to_cv.dropna(axis = 0,inplace =True)
data_cv = data_cv[['Bedroom2', 'Bathroom' ,'Car','Type','Landsize','Suburb','name','Price']]

data_cv.dropna(axis = 0,inplace =True)

data_cv.reset_index(inplace =True)
pred_price =  clf.predict(to_cv)



data_cv['pred'] = pd.DataFrame(pred_price)

data_cv['diff'] = data_cv['Price'] - data_cv['pred'] 



x=np.array(data_cv['pred']).shape

y=np.array(data_cv['Price']).shape



sns.regplot(data_cv['Price'],data_cv['diff'])
sns.regplot(data_cv['pred'],data_cv['diff'])