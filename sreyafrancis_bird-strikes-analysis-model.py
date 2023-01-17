# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import matplotlib

%matplotlib inline

import seaborn as sns

from collections import Counter

from sklearn.metrics import mean_squared_error

from pandas import concat, Series, DataFrame



# machine learning

from sklearn.preprocessing import scale

from sklearn import metrics

from sklearn.cross_validation import cross_val_score, train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



pd.set_option('display.max_columns', 100)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
bird = pd.read_csv("../input/Bird Strikes Test.csv", low_memory=False, thousands=',')



# only drop rows that are all NA:

bird = bird.dropna(how='all')

import os

cwd = os.getcwd()

print (cwd)
bird.head()
bird.info()
bird.drop(['Record ID'], axis= 1).describe()
#Time of strike





# month variable

bird['Flight Month'] = pd.DatetimeIndex(bird['FlightDate']).month

# year variable

bird['Flight Year'] = pd.DatetimeIndex(bird['FlightDate']).year



# count over flight month and year

count_time = DataFrame({'count' : bird.groupby( ['Flight Month', 'Flight Year'] ).size()}).reset_index()

# reshape frame

count_time_p=count_time.pivot("Flight Month", "Flight Year", "count")

# plot the frequency over month and year in a heat map

plt.figure(figsize=(8, 7))

heat_time = sns.heatmap(count_time_p);

heat_time.set_title('The Frequency of All Strikes Over Flight Year and Month');
# add damage index

bird['Damage'] = 0

bird.loc[(bird['Effect: Indicated Damage'] != 'No damage') | 

                    (bird['Cost: Total $'] > 0) ,'Damage'] = 1



# define independent and dependent variables

X = ['Aircraft: Number of engines?',

     'Wildlife: Size',

     'When: Phase of flight','Feet above ground','Miles from airport','Speed (IAS) in knots',

     'Flight Month','Flight Year','When: Time (HHMM)',

     'Pilot warned of birds or wildlife?']

Y = ['Damage']



# clean missing data, keep those with values on key metrics

bird_keep = bird[np.concatenate((X,Y))].dropna(how='any')

# list of damage indices

damage_index = np.array(bird_keep[bird_keep["Damage"]==1].index)



# getting the list of normal indices from the full dataset

normal_index = bird_keep[bird_keep["Damage"]==0].index



No_of_damage = len(bird_keep[bird_keep["Damage"]==1])



# choosing random normal indices equal to the number of damaging strikes

normal_indices = np.array( np.random.choice(normal_index, No_of_damage, replace= False) )



# concatenate damaging index and normal index to create a list of indices

undersampled_indices = np.concatenate([damage_index, normal_indices])



# define training and testing sets

# choosing random indices equal to the number of damaging strikes

train_indices = np.array( np.random.choice(undersampled_indices, No_of_damage, replace= False) )

test_indices = np.array([item for item in undersampled_indices if item not in train_indices])
# add dummy variables for categorical variables

wildlife_dummies = pd.get_dummies(bird_keep['Wildlife: Size'])

bird_keep = bird_keep.join(wildlife_dummies)



phase_dummies = pd.get_dummies(bird_keep['When: Phase of flight'])

bird_keep = bird_keep.join(phase_dummies)



warn_dummies = pd.get_dummies(bird_keep['Pilot warned of birds or wildlife?'])

bird_keep = bird_keep.join(warn_dummies)



#  convert engine number to numeric

bird_keep['Aircraft: Number of engines?'] = pd.to_numeric(bird_keep['Aircraft: Number of engines?'])



# scale variables before fitting our model to our dataset

# flight year scaled by subtracting the minimum year

bird_keep["Flight Year"] = bird_keep["Flight Year"] - min(bird_keep["Flight Year"])

# scale time by dividing 100 and center to the noon

bird_keep["When: Time (HHMM)"] = bird_keep["When: Time (HHMM)"]/100-12

# scale speed

bird_keep["Speed (IAS) in knots"] = scale( bird_keep["Speed (IAS) in knots"], axis=0, with_mean=True, with_std=True, copy=False )

# use the undersampled indices to build the undersampled_data dataframe

undersampled_bird = bird_keep.loc[undersampled_indices, :]



# drop original values after dummy variables added

bird_use = undersampled_bird.drop(['Wildlife: Size','When: Phase of flight',

     'Pilot warned of birds or wildlife?'],axis=1)
# define training and testing sets

# choosing random indices equal to the number of damaging strikes

X_train = bird_use.drop("Damage",axis=1).loc[train_indices,]

Y_train = bird_use.loc[train_indices,'Damage']

X_test = bird_use.drop("Damage",axis=1).loc[test_indices,]

Y_test = bird_use.loc[test_indices,'Damage']

X_test.head()
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

print('Training Accuracy:')

logreg.score(X_train, Y_train)
# generate evaluation metrics

logreg_t = metrics.accuracy_score(Y_test, Y_pred)

print('Testing Accuracy:')

logreg_t
# evaluate the model using 10-fold cross-validation

scores_lr = cross_val_score(logreg, X_train, Y_train, scoring='accuracy', cv=10)

print('Cross-Validation Accuracy:')

print (scores_lr.mean())

x=zip(X_train.columns, np.transpose(logreg.coef_))

x1=pd.DataFrame(list(x))

x1.head()
# get Correlation Coefficient for each feature using Logistic Regression

logreg_df = pd.DataFrame(list(zip(X_train.columns, np.transpose(logreg.coef_))))

logreg_df.columns = ['Features','Coefficient Estimate']

logreg_df['sort'] = logreg_df['Coefficient Estimate'].abs()



# get top 10 most influential coefficient estimates

logreg_df.sort_values(['sort'],ascending=0).drop('sort',axis=1).head(10)