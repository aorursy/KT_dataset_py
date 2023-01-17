# Reference notebooks

# Good Viz - https://www.kaggle.com/lesibius/crime-scene-exploration-and-model-fit
# Import packages



# visualizations

%matplotlib inline

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



# Stats

from scipy import stats as ss

import numpy as np



# datetime

from datetime import tzinfo, timedelta, datetime



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



# Random Forest

from sklearn.ensemble import RandomForestClassifier
# Read train and test

train_data = pd.read_csv('../input/train.csv')

# shape of the dataset

print(train_data.shape)
# Structure of the data

train_data.head()
# Read a snapshot of the test dataset

# Description and resolution is not present in the test dataset

#test_dataset = pd.read_csv('../input/test.csv')

test_data = pd.read_csv('../input/test.csv')

test_data.head()
# Type of the dataset

train_data.info()
# Distribution of crime category in the train dataset

number_of_crimes = train_data["Category"].value_counts()

number_of_crimes.head()
crime_plot = sns.barplot(x = number_of_crimes.index, y = number_of_crimes)

crime_plot.set_xticklabels(number_of_crimes.index,rotation = 90)
pareto_crime = number_of_crimes/ sum(number_of_crimes)

pareto_crime = pareto_crime.cumsum()

_pareto_crime_plot = sns.tsplot(data=pareto_crime)

_pareto_crime_plot.set_xticklabels(pareto_crime.index,rotation=90)

_pareto_crime_plot.set_xticks(np.arange(len(pareto_crime)))

Main_Crime_Categories = list(pareto_crime[0:8].index)

print("The following categories :")

print(Main_Crime_Categories)

print("make up to {:.2%} of the crimes".format(pareto_crime[8]))
# Unique levels in the dataset 

#train_data["Category"].unique()
# Weekdays 

train_data['DayOfWeek'].value_counts()
# Relative Time Scale

origin_date = datetime.strptime('2003-01-01 00:00:00','%Y-%m-%d %H:%M:%S')



def delta_origin_date(dt):

    _ = datetime.strptime(dt,'%Y-%m-%d %H:%M:%S') - origin_date

    return(_.days+(_.seconds/86400))



delta_origin_date(train_data.loc[1,"Dates"])



tmp = train_data.loc[:,["Dates","Category"]]

tmp["RelativeDates"]=train_data.Dates.map(delta_origin_date)

tmp.head()
# At this stage, it can be interesting to see how the number of crimes per ~ quarter evolved 

# to see if the RelativeDates variable makes sense. To proceed, I'll cut my variables by 

# buckets of roughly 90 days and plot it as a stacked area plot. 

# This will allow to see at once both the increase/decrease of total crimes and 

# the split of crime categories over time.
tmp["QuarterBucket"] = tmp.RelativeDates.map(lambda d: int(d/90.0))

tmp.head()
pt = pd.pivot_table(tmp,index="QuarterBucket",columns="Category",aggfunc=len,fill_value=0)

pt = pt["Dates"]

pt[Main_Crime_Categories].iloc[:49,:].cumsum(1).plot()

pt.head()

# There's a lot of noise in this graph. I'll take a 3Q-smoothed average of it to make trends easier to see.

#pd.rolling_mean(pt[Main_Crime_Categories],3).iloc[2:49,:].plot()

pt.iloc[2:49,0:9].rolling(3).mean().plot()
# Function to calculate correlation between categorical variables

# Source: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9



def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
# Correlation between Category and Description

# Ignore Description since it is not present in the test dataset

cramers_v(train_data['Category'],train_data['Descript'])
# Correlation between Category and Resolution

# Ignore Resolution since it is not present in the test dataset

cramers_v(train_data['Category'],train_data['Resolution'])
# Corr b/w Category and Weekday

cramers_v(train_data['Category'],train_data['DayOfWeek'])
# Corr b/w Category and PdDistrict

cramers_v(train_data['Category'],train_data['PdDistrict'])
# Correlation b/w month and category

cramers_v(train_data['Category'],train_data['Address'])
# 10 unique districts

train_data['PdDistrict'].unique()
# Unique address in the dataset

len(train_data['Address'].unique())
train_data.columns
# Convert object data type to datetime type

# https://chrisalbon.com/machine_learning/preprocessing_dates_and_times/break_up_dates_and_times_into_multiple_features/



train_data['Dates'] = pd.to_datetime(train_data['Dates'], format='%Y%m%d %H:%M:%S')

train_data.info()
# Create year and month column out of date

train_data['year'] = train_data['Dates'].dt.year

train_data['month'] = train_data['Dates'].dt.month
train_data['year'].value_counts(sort=False)
train_data.head()
train_data.info()
X = train_data.drop(['Category','Dates','Descript','Resolution','Address','X','Y'],axis = 1)

X = pd.get_dummies(X)

X.head()
y = pd.get_dummies(train_data['Category'])

y.head()
# For logistic regression

y = train_data['Category']

y.head()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.2, random_state = 42)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Import random forest classifier

rf_model = RandomForestClassifier(n_estimators=10)

rf_model.fit(X_train,y_train)



# Predict the result

predictions = rf_model.predict(X_test)



# Accuracy score

score = metrics.accuracy_score(y_test,predictions)

print(score)
# Logistic regresssion multi classifier

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='lbfgs',multi_class = 'multinomial')



logreg.fit(X_train,y_train)
# Predict the result

predictions = logreg.predict(X_test)



# Accuracy score

score = metrics.accuracy_score(y_test,predictions)

print(score)
## Binary Relevance - methods to solve mutli class classifier

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



# integer encode

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(y.values)

print("Label Encoder:" ,integer_encoded)



# onehot encode

onehot_encoder = OneHotEncoder(sparse=False,categories='auto')

integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print("OneHot Encoder:", onehot_encoded)
from sklearn.model_selection import train_test_split

x_train,x_test, y_train,y_test = train_test_split(X, onehot_encoded, test_size=0.33, random_state=42)

# using binary relevance

from skmultilearn.problem_transform import BinaryRelevance

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier



# initialize binary relevance multi-label classifier

# with a from sklearn.ensemble import RandomForestClassifier bayes base classifier

classifier = BinaryRelevance(GaussianNB())



# train

classifier.fit(x_train, y_train)



# predict

predictions = classifier.predict(x_test)

print(metrics.accuracy_score(y_test,predictions))
####### Label powerset

# using Label Powerset

from skmultilearn.problem_transform import LabelPowerset

from sklearn.naive_bayes import GaussianNB



# initialize Label Powerset multi-label classifier

# with a gaussian naive bayes base classifier

classifier = LabelPowerset(GaussianNB())



# train

classifier.fit(x_train, y_train)



# predict

predictions = classifier.predict(x_test)



metrics.accuracy_score(y_test,predictions)

# 0.01604447864
# multi-label version of kNN is represented by MLkNN



from skmultilearn.adapt import MLkNN

classifier = MLkNN(k=5)



# train

classifier.fit(x_train, y_train)



# predict

predictions = classifier.predict(x_test)



metrics.accuracy_score(y_test,predictions)
