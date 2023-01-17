import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

plt.style.use('seaborn')

sns.set(style='white', context='notebook', palette='deep')







crimes1 = pd.read_csv('../input/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)

crimes2 = pd.read_csv('../input/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)

crimes3 = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)

crimes = pd.concat([crimes1, crimes2, crimes3], ignore_index=False, axis=0)



crimes.head()
crime_count = pd.DataFrame(crimes.groupby('Primary Type').size().sort_values(ascending=False).rename('Count').reset_index())

crime_count
crime_count[:20].plot(x='Primary Type',y='Count',kind='bar')
crime_count = pd.DataFrame(crimes.groupby('Location Description').size().rename('Count').sort_values(ascending=False).reset_index())

crime_count[:20]

crime_count = pd.DataFrame(crimes.groupby('District').size().rename('Count').sort_values(ascending=False).reset_index())

crime_count

crimes[['X Coordinate', 'Y Coordinate']] = crimes[['X Coordinate', 'Y Coordinate']].replace(0, np.nan)

crimes.dropna()

crimes.plot(kind='scatter',x='X Coordinate', y='Y Coordinate', c='District', cmap=plt.get_cmap('jet'))
plt.figure(figsize=(15,15))

sns.jointplot(x=crimes['X Coordinate'].values, y=crimes['Y Coordinate'].values, size=10, kind='hex')

plt.ylabel('Longitude', fontsize=12)

plt.xlabel('Latitude', fontsize=12)

plt.show()
plt.figure(figsize=(12,12))

sns.lmplot(x='X Coordinate', y='Y Coordinate', size=10, hue='Primary Type', data=crimes, fit_reg=False)

plt.ylabel('Longitude', fontsize=15)

plt.xlabel('Latitude', fontsize=15)

plt.show()
topk = crimes.groupby(['District', 'Primary Type']).size().reset_index(name='counts').groupby('District').apply(lambda x: x.sort_values('counts',ascending=False).head(3))

topk[:51]
g =sns.factorplot("Primary Type", y='counts', col="District", col_wrap=4,

                   data=topk, kind='bar')

for ax in g.axes:

    plt.setp(ax.get_xticklabels(), visible=True, rotation=30, ha='right')



plt.subplots_adjust(hspace=0.4)
g =sns.factorplot("Arrest", col="District", col_wrap=4, legend_out=True,

                   data=crimes, orient='h',

                    kind="count")

for ax in g.axes:

    plt.setp(ax.get_xticklabels(), visible=False)
df_theft = crimes[crimes['Primary Type'] == 'NARCOTICS']

plt.figure(figsize = (15, 7))

sns.countplot(y = df_theft['Description'],order=df_theft['Description'].value_counts().index[:20])
Beat = pd.DataFrame(crimes.groupby('Beat').size().rename('Count').sort_values(ascending=False).reset_index())

Beat[:20]
crimes.Date = pd.to_datetime(crimes.Date, format='%m/%d/%Y %I:%M:%S %p')

crimes.index = pd.DatetimeIndex(crimes.Date)



crimes.resample('M').size().plot(legend=True)

plt.title('Number of crimes per month (2001 - 2018)')

plt.xlabel('Months')

plt.ylabel('Number of crimes')

plt.show()
crimes['day_of_week']=crimes['Date'].dt.weekday_name

crimes['month']=crimes['Date'].dt.month

crimes['day']=crimes['Date'].dt.day

crimes['hour']=crimes['Date'].dt.hour

crimes['minute']=crimes['Date'].dt.minute





crimes.head()

crime_ = pd.DataFrame(crimes.groupby('day').size().rename('Count').reset_index())

crime_.plot(x='day',y='Count',kind='bar')

crime_ = pd.DataFrame(crimes.groupby('month').size().rename('Count').reset_index())

crime_.plot(x='month',y='Count',kind='bar')

crime_ = pd.DataFrame(crimes.groupby('day_of_week').size().rename('Count').reset_index())

crime_.plot(x='day_of_week',y='Count',kind='bar')

crime_ = pd.DataFrame(crimes.groupby('hour').size().rename('Count').reset_index())

crime_.plot(x='hour',y='Count',kind='bar')

crime_ = pd.DataFrame(crimes.groupby('minute').size().rename('Count').reset_index())

crime_.plot(x='minute',y='Count',kind='bar')
crimes_count_date = crimes.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index=crimes.index.date, fill_value=0)

crimes_count_date.index = pd.DatetimeIndex(crimes_count_date.index)

plo = crimes_count_date.rolling(365).sum().plot(figsize=(12, 30), subplots=True, layout=(-1, 2), sharex=False, sharey=False)
crimes['day_of_week']=crimes['Date'].dt.weekday

crimes.head()



crimes=crimes.dropna()

crimes.isnull().sum(axis = 0)
crimes=crimes.drop('Case Number', axis=1)

crimes=crimes.drop('ID', axis=1)

crimes=crimes.drop('FBI Code', axis=1)

crimes=crimes.drop('Date', axis=1)

crimes=crimes.drop('Block', axis=1)

crimes=crimes.drop('Updated On', axis=1)

crimes=crimes.drop('Location', axis=1)

crimes=crimes.drop('Longitude', axis=1)

crimes=crimes.drop('Latitude', axis=1)

crimes=crimes.drop('IUCR', axis=1)



x=crimes['X Coordinate'].mean()

y=crimes['Y Coordinate'].mean()

print(x,y)

categories_type = {c:i for i,c in enumerate(crimes['Primary Type'].unique())}

categories_description = {c:i for i,c in enumerate(crimes['Description'].unique())}

categories_location_des = {c:i for i,c in enumerate(crimes['Location Description'].unique())}

categories_Arrest = {c:i for i,c in enumerate(crimes['Arrest'].unique())}

categories_Domestic = {c:i for i,c in enumerate(crimes['Domestic'].unique())}

#categories_IUCR = {c:i for i,c in enumerate(crimes['IUCR'].unique())}



crimes['Primary_Type_Num'] = [float(categories_type[t]) for t in crimes['Primary Type']]

crimes['Description_Num'] = [float(categories_description[t]) for t in crimes['Description']]

crimes['Location_Des_Num'] = [float(categories_location_des[t]) for t in crimes['Location Description']]

crimes['Arrest_Num'] = [float(categories_Arrest[t]) for t in crimes['Arrest']]

crimes['Domestic_Num'] = [float(categories_Domestic[t]) for t in crimes['Domestic']]

#crimes['IUCR_Num'] = [float(categories_IUCR[t]) for t in crimes['IUCR']]



crimes=crimes.drop('Primary Type', axis=1)

crimes=crimes.drop('Description', axis=1)

crimes=crimes.drop('Location Description', axis=1)

crimes=crimes.drop('Arrest', axis=1)

crimes=crimes.drop('Domestic', axis=1)

#crimes=crimes.drop('IUCR', axis=1)



crimes.head()
from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder as le

from sklearn.preprocessing import MultiLabelBinarizer



crimes['X Coordinate'] = preprocessing.scale(list(map(lambda x: x-1164537.87395, crimes['X Coordinate'])))

crimes['Y Coordinate'] = preprocessing.scale(list(map(lambda x: x-1885607.09892, crimes['X Coordinate'])))

crimes.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.ensemble import RandomForestClassifier



print(crimes.shape)

X_train, X_test, y_train, y_test = train_test_split(crimes.loc[:, crimes.columns != 'Primary_Type_Num'], crimes['Primary_Type_Num'], test_size = 0.2, random_state = 0)

clf = RandomForestClassifier(max_features="log2", max_depth=16, n_estimators=25,

                             min_samples_split=600, oob_score=False,n_jobs=4).fit(X_train,y_train)

y = clf.predict(X_test)

print(X_train.shape,X_test.shape)



print(recall_score(y,y_test, average='micro'))

print(recall_score(y,y_test, average='macro'))

print(recall_score(y,y_test, average='weighted'))



print(precision_score(y,y_test, average='micro'))

print(precision_score(y,y_test, average='macro'))

print(precision_score(y,y_test, average='weighted'))



print(f1_score(y,y_test, average='micro'))

print(f1_score(y,y_test, average='macro'))

print(f1_score(y,y_test, average='weighted'))

crime_count = pd.DataFrame(crimes.groupby('Primary_Type_Num').size().sort_values(ascending=False).rename('Count').reset_index())

crime_count
from imblearn.over_sampling import SMOTE



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.ensemble import RandomForestClassifier



X_train, X_test, y_train, y_test = train_test_split(crimes.loc[:, crimes.columns != 'Primary_Type_Num'], crimes['Primary_Type_Num'], test_size = 0.2, random_state = 0)



print(X_train.shape,y_train.shape,'before')

sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print(X_train_res.shape,y_train_res.shape,'after')



clf = RandomForestClassifier(max_features="log2", max_depth=20, n_estimators=25,

                             min_samples_split=600, oob_score=False,n_jobs=4).fit(X_train_res,y_train_res.ravel())

y = clf.predict(X_test)









print(recall_score(y,y_test, average='micro'))

print(recall_score(y,y_test, average='macro'))

print(recall_score(y,y_test, average='weighted'))



print(precision_score(y,y_test, average='micro'))

print(precision_score(y,y_test, average='macro'))

print(precision_score(y,y_test, average='weighted'))



print(f1_score(y,y_test, average='micro'))

print(f1_score(y,y_test, average='macro'))

print(f1_score(y,y_test, average='weighted'))

from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression()

lgr.fit(X_train,y_train)

y = lgr.predict(X_test)





print(recall_score(y,y_test, average='micro'))

print(recall_score(y,y_test, average='macro'))

print(recall_score(y,y_test, average='weighted'))



print(precision_score(y,y_test, average='micro'))

print(precision_score(y,y_test, average='macro'))

print(precision_score(y,y_test, average='weighted'))



print(f1_score(y,y_test, average='micro'))

print(f1_score(y,y_test, average='macro'))

print(f1_score(y,y_test, average='weighted'))
Image(filename='xai.jpg')
Image(filename='house.png')
Image(filename='final.png')
Image(filename='anchor.png')
import sklearn

import sklearn.datasets

import sklearn.ensemble

import numpy as np

import lime

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

import lime.lime_tabular





predict_fn_logreg = lambda x: lgr.predict_proba(x).astype(float)

predict_fn_rf = lambda x: clf.predict_proba(x).astype(float)









explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values ,feature_names = list(crimes.loc[:, crimes.columns != 'Primary_Type_Num'].columns)

,class_names=list(crimes.Primary_Type_Num.unique()))







observation_1 = 4

# Get the explanation for Logistic Regression

exp = explainer.explain_instance(X_test.values[observation_1], predict_fn_logreg, num_features=6)

exp.show_in_notebook(show_all=False)



# Get the explanation for RandomForest

exp = explainer.explain_instance(X_test.values[observation_1], predict_fn_rf, num_features=6)

exp.show_in_notebook(show_all=False)





print('Actual class of our observation --->  ',y_test[observation_1])









observation_2 = 124

# Get the explanation for Logistic Regression

exp = explainer.explain_instance(X_test.values[observation_2], predict_fn_logreg, num_features=6)

exp.show_in_notebook(show_all=False)



# Get the explanation for RandomForest

exp = explainer.explain_instance(X_test.values[observation_2], predict_fn_rf, num_features=6)

exp.show_in_notebook(show_all=False)





print('Actual class of our observation --->  ',y_test[observation_2])