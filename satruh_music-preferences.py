import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("../input/responses.csv")
data.head()
print(data.info())
data_music = data.iloc[:,0:19]
data_use = data_music
data_use["Gender"] = data["Gender"]
data_use["Age"] = data["Age"]
data_use["Alcohol"] = data["Alcohol"]
data_use["Education"] = data["Education"]
print(data_use.info())
print(data_use.isnull().sum())
data_use.dropna(inplace = True)
data_use.reset_index(drop=True,inplace=True)
row = len(data_use.index)
data_use.info()
for each in range(0,23) :
    if type(data_use.iloc[1,each]) == np.float64 :
        data_use[data_use.columns[each]] = data_use[data_use.columns[each]].astype(int)
    else :
        data_use[data_use.columns[each]] = data_use[data_use.columns[each]]
data_use.boxplot(column='Music')
plt.show()
print(data_use['Music'].value_counts(dropna =False))
filtre = data_use.Music < 5
filt_list = list(data_use[filtre].index)
i=0
for each in filt_list:
    data_use.drop(data_use.index[each-i], inplace=True)
    i=i+1
data_use.reset_index(drop=True,inplace=True)
row = len(data_use.index)
print(data_use['Music'].value_counts(dropna =False))
data_use.drop(['Music'], axis=1,inplace = True)
plt.show()
data_use.Gender.unique()
for sex in range(0,row) :
    if data_use.loc[sex,'Gender'] == 'female' :
        data_use.loc[sex,'Gender'] = 0
    else :
        data_use.loc[sex,'Gender'] = 1
data_use['Gender'].head()
data_use.Alcohol.unique()
for drink in range(0,row) :
    if data_use.loc[drink,'Alcohol'] == 'never' :
        data_use.loc[drink,'Alcohol'] = 0
    elif data_use.loc[drink,'Alcohol'] == 'social drinker':
        data_use.loc[drink,'Alcohol'] = 1
    else :
        data_use.loc[drink,'Alcohol'] = 2
data_use['Alcohol'].head()
data_use.Education.unique()
for edu in range(0,row) :
    if data_use.loc[edu,'Education'] == 'currently a primary school pupil' :
        data_use.loc[edu,'Education'] = 0
    elif data_use.loc[edu,'Education'] == 'primary school':
        data_use.loc[edu,'Education'] = 1
    elif data_use.loc[edu,'Education'] == 'secondary school':
        data_use.loc[edu,'Education'] = 2
    elif data_use.loc[edu,'Education'] == 'college/bachelor degree':
        data_use.loc[edu,'Education'] = 3
    elif data_use.loc[edu,'Education'] == 'masters degree':
        data_use.loc[edu,'Education'] = 4
    else :
        data_use.loc[edu,'Education'] = 5
data_use['Education'].tail(7)
data_use.info()
data_use.corr()
f,ax = plt.subplots(figsize=(25, 20))
sns.heatmap(data_use.corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)
plt.show()
data_use.columns
# create dataframe
x2 = data_use.corr()
value = pd.DataFrame([np.zeros])
x = pd.DataFrame(data_use.columns.copy()).T
i=0
for each in data_use.columns :
    value.loc[0,i] = x2.loc[each,'Gender']
    i=i+1
    
# and drop itself from data
plot_values= value
clmn = int(np.where(plot_values>= 1)[1])
plot_values.drop([clmn], axis = 1, inplace = True)
x.drop([clmn], axis = 1, inplace = True)
# plot the values
plt.stem(x.loc[0,:], plot_values.loc[0,:])
plt.xticks(rotation=-90)
plt.plot(x.loc[0,0:21], np.linspace(0, 0, 21),'r-',linewidth=2)
plt.grid()
plt.show()
# create dataframe
x2 = data_use.corr()
value = pd.DataFrame([np.zeros])
x = pd.DataFrame(data_use.columns.copy()).T
i=0
for each in data_use.columns :
    value.loc[0,i] = x2.loc[each,'Classical music']
    i=i+1
# and drop itself from data
plot_values= value
clmn = int(np.where(plot_values>= 1)[1])
plot_values.drop([clmn], axis = 1, inplace = True)
x.drop([clmn], axis = 1, inplace = True)
# plot the values
plt.stem(x.loc[0,:], plot_values.loc[0,:])
plt.xticks(rotation=-90)
plt.plot(x.loc[0,0:21], np.linspace(0, 0, 21),'r-',linewidth=2)
plt.grid()
plt.show()
# create dataframe
x2 = data_use.corr()
value = pd.DataFrame([np.zeros])
x = pd.DataFrame(data_use.columns.copy()).T
i=0
for each in data_use.columns :
    value.loc[0,i] = x2.loc[each,'Rock']
    i=i+1
# and drop itself from data
plot_values= value
clmn = int(np.where(plot_values>= 1)[1])
plot_values.drop([clmn], axis = 1, inplace = True)
x.drop([clmn], axis = 1, inplace = True)
# plot the values
plt.stem(x.loc[0,:], plot_values.loc[0,:])
plt.xticks(rotation=-90)
plt.plot(x.loc[0,0:21], np.linspace(0, 0, 21),'r-',linewidth=2)
plt.grid()
plt.show()