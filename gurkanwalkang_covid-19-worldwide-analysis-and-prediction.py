import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn import metrics

from sklearn.metrics import classification_report
data = pd.DataFrame(pd.read_csv("../input/covid19-refined-dataset/Covid-19_dataset.csv"))

data.head(2)
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

data.head(2)
data.shape
data.drop(data.columns[[0,1,3,8,9,10,11,12,17,18,19]], axis = 1, inplace = True)

data['reporting_date'] = pd.to_datetime(data.reporting_date)



data.head()
data.shape
data.columns
data.describe()
data.info()
print('Number of Null values in Columns')

data.isnull().sum()
refined_data = data.dropna(subset=['gender', 'age', 'from_wuhan'])
print('Number of Null values in Columns')

refined_data.isnull().sum()
refined_data.head(5)
refined_data.shape
refined_data.columns
refined_data.describe()
refined_data.info()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

refined_data[refined_data.columns[1]] = labelencoder.fit_transform(refined_data[refined_data.columns[1]])
labelencoder = LabelEncoder()

refined_data[refined_data.columns[2]] = labelencoder.fit_transform(refined_data[refined_data.columns[2]])
labelencoder = LabelEncoder()

refined_data[refined_data.columns[3]] = labelencoder.fit_transform(refined_data[refined_data.columns[3]])
refined_data.head(5)
refined_data.info()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



X = refined_data.iloc[:,1:7]  #independent columns

y = refined_data.iloc[:,7]    #target column i.e Death
#apply SelectKBest class to extract top 5 best features

bestfeatures = SelectKBest(score_func=chi2, k=5)

fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

featureScores
print(featureScores.nlargest(6,'Score'))
labels = 'age', 'location', 'country', 'from_wuhan', 'visiting_wuhan', 'gender'

sizes = [444.595027, 407.528544, 81.670406, 53.350788, 9.143794, 3.567871]



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=190)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
refined_data.sample(5)
refined_data.shape
y= refined_data["death"]

y
x= refined_data["age"]

y= refined_data["death"]

plt.bar(x,y)

plt.title("Number of Patients Died based on their age")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.show()

x= refined_data["age"]

y= refined_data["recovered"]

plt.bar(x,y)

plt.title("Number of Patients Recovered based on their age")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.show()
print("Current count of patients:", refined_data.shape[0])

print("Recovered:",len(refined_data[refined_data.recovered == 1]))

print("Dead:",len(refined_data[refined_data.death == 1]))

print("Number of Patients receiving treatment:", refined_data.shape[0] - len(refined_data[refined_data.recovered == 1]) 

      - len(refined_data[refined_data.death == 1]))



y = np.array([len(refined_data[refined_data.recovered == 1]),len(refined_data[refined_data.death == 1])])

x = ["Recovered","Dead"]

plt.bar(x,y)

plt.title("Patient Status")

plt.xlabel("Patients")

plt.ylabel("Frequency")

plt.show()
print("Male:",len(refined_data[refined_data.gender == 1][refined_data.death == 1]))

print("Female:",len(refined_data[refined_data.gender == 0][refined_data.death == 1]))



y = np.array([len(refined_data[refined_data.gender == 1][refined_data.death == 1]),

              len(refined_data[refined_data.gender == 0][refined_data.death == 1])])

x = ["Male","Female"]

plt.bar(x,y)

plt.title("Number of Patients Died")

plt.xlabel("Patients")

plt.ylabel("Frequency")

plt.show()
print("Male:",len(refined_data[refined_data.gender == 1][refined_data.recovered == 1]))

print("Female:",len(refined_data[refined_data.gender == 0][refined_data.recovered == 1]))



y = np.array([len(refined_data[refined_data.gender == 1][refined_data.recovered == 1]),

              len(refined_data[refined_data.gender == 0][refined_data.recovered == 1])])

x = ["Male","Female"]

plt.bar(x,y)

plt.title("Number of Patients Recovered")

plt.xlabel("Patients")

plt.ylabel("Frequency")

plt.show()
print("From Wuhan :",len(refined_data[refined_data.from_wuhan == 1][refined_data.death == 1]))

print("Not From Wuhan:",len(refined_data[refined_data.from_wuhan == 0][refined_data.death == 1]))



y = np.array([len(refined_data[refined_data.from_wuhan == 1][refined_data.death == 1]),

              len(refined_data[refined_data.from_wuhan == 0][refined_data.death == 1])])

x = ["From Wuhan","Not From Wuhan"]

plt.bar(x,y)

plt.title("Number of Patients Died")

plt.xlabel("Patients")

plt.ylabel("Frequency")

plt.show()
print("From Wuhan :",len(refined_data[refined_data.from_wuhan == 1][refined_data.recovered == 1]))

print("Not From Wuhan:",len(refined_data[refined_data.from_wuhan == 0][refined_data.recovered == 1]))



y = np.array([len(refined_data[refined_data.from_wuhan == 1][refined_data.recovered == 1]),

              len(refined_data[refined_data.from_wuhan == 0][refined_data.recovered == 1])])

x = ["From Wuhan","Not From Wuhan"]

plt.bar(x,y)

plt.title("Number of Patients Recovered")

plt.xlabel("Patients")

plt.ylabel("Frequency")

plt.show()
group = data.groupby('country').size()

group.head()
x= ['Afghanistan','Algeria','Australia','Austria','Bahrain','Belgium',

'Cambodia','Canada','China','Croatia','Egypt','Finland','France',

'Germany','Hong Kong','India','Iran','Israel','Italy','Japan',

'Kuwait','Lebanon','Malaysia','Nepal','Philippines','Russia',

'Singapore','South Korea','Spain','Sri Lanka','Sweden','Switzerland',

'Taiwan','Thailand','UAE','UK','USA', 'Vietnam']

y= group



plt.title("Patients identified at different locations")

plt.xlabel("Location")

plt.ylabel("Number of Covid Patients")

plt.xticks(rotation=90)

plt.bar(x,y)
# WORLD MAP SHOWING LOCATIONS WITH COVID-19 PATIENTS

import plotly.express as px

fig = px.choropleth(data, locations="country", locationmode='country names', 

                    hover_name="country", title='PATIENTS IDENTIFIED AT DIFFERENT LOCATIONS', 

                    color_continuous_scale=px.colors.sequential.Magenta)

fig.update(layout_coloraxis_showscale=False)

fig.show()
# Over the time



fig = px.choropleth(data, locations="country", locationmode='country names', 

                    hover_name="country", animation_frame=data["reporting_date"].dt.strftime('%Y-%m-%d'),

                    title='OVER THE TIME PATIENTS IDENTIFIED BASED ON THEIR LOCATION', 

                    color_continuous_scale=px.colors.sequential.Magenta)

fig.update(layout_coloraxis_showscale=False)

fig.show()
X = refined_data[refined_data.columns[1:7]] #(location, country, gender, age, visiting wuhan, from wuhan)

y = refined_data[refined_data.columns[[7]]] #death
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



reg=LogisticRegression()

reg.fit(X_train,y_train)
reg.score(X_train,y_train)
pdt = reg.predict(X_test)

pdt
#CONFUSION MATRIX

cm = metrics.confusion_matrix(y_test, pdt)

print(cm)
from sklearn.metrics import mean_squared_error

from math import sqrt



rms = sqrt(mean_squared_error(y_test,pdt))

rms
from sklearn.metrics import classification_report

print(classification_report(y_test, pdt))
X = refined_data[refined_data.columns[1:7]] #(location, country, gender, age, visiting wuhan, from wuhan)

y = refined_data[refined_data.columns[[8]]] #recovered
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



reg=LogisticRegression()

reg.fit(X_train,y_train)
reg.score(X_train,y_train)
pdt = reg.predict(X_test)

pdt
#CONFUSION MATRIX

cm = metrics.confusion_matrix(y_test, pdt)

print(cm)
from sklearn.metrics import mean_squared_error

from math import sqrt



rms = sqrt(mean_squared_error(y_test,pdt))

rms
from sklearn.metrics import classification_report

print(classification_report(y_test, pdt))