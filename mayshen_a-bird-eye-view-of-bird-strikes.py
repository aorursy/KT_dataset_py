# import libraries

import numpy as np

import pandas as pd 

from matplotlib import pyplot as plt

import matplotlib

%matplotlib inline

import seaborn as sns

from collections import Counter

from sklearn.metrics import mean_squared_error

from pandas import concat

from pandas import Series, DataFrame

import statsmodels.api as sm



# machine learning

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.cross_validation import cross_val_score

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



pd.set_option('display.max_columns', 100)
# import in the bird strikes data set

bird = pd.read_csv("../input/Bird Strikes Test.csv", low_memory=False, thousands=',')

# only drop rows that are all NA:

bird = bird.dropna(how='all')
# take a look at the first 5 rows of data

bird.head()
# check the number of entries and data type for each variable

bird.info()
# get a quick description of non-null numeric values in the data

bird.drop(['Record ID'], axis=1).describe()
# subset the data with any damage or negative impact to the flight

bird_dmg = bird.loc[(bird['Effect: Indicated Damage'] != 'No damage') | 

                    (bird['Cost: Total $'] > 0) ]
# get a table of number of strikes across aircraft type and aircraft engine numbers

count_air_type = DataFrame({'count' : bird.groupby( ['Aircraft: Number of engines?'] ).size()}).reset_index()

count_air_type.sort_values(['count'], ascending=0)
# set abnormal entries for Aircraft: Number of engines? to be NaN

bird.loc[(bird['Aircraft: Number of engines?'] == 'S'),'Aircraft: Number of engines?'] = np.nan



# update bird_dmg as well

bird_dmg = bird.loc[(bird['Effect: Indicated Damage'] != 'No damage') | 

                    (bird['Cost: Total $'] > 0) ]



# re-generate count table

count_air_type = DataFrame({'count' : bird.groupby( ['Aircraft: Number of engines?'] ).size()}).reset_index()

# plot the frequency of Aircraft: Number of engines?

fig_air_type = sns.barplot(x=u'Aircraft: Number of engines?', y='count', data=count_air_type)

fig_air_type.set(xlabel='Aircraft: Number of Engines', ylabel='Counts - All Strikes');

fig_air_type.set_title('The Frequency of All Strikes Over Aircraft Number of Engines');
# table for damaging stikes

count_air_type0 = DataFrame({'count' : bird_dmg.groupby( [ 'Aircraft: Number of engines?'] ).size()}).reset_index()

count_air_type0['All Strikes Counts'] = count_air_type['count']

count_air_type0['Damage Rate'] = count_air_type0['count']/count_air_type0['All Strikes Counts']

# plot the frequency of Aircraft: Number of engines?

fig_air_type0 = sns.barplot(x=u'Aircraft: Number of engines?', y='count', data=count_air_type0)

fig_air_type0.set(xlabel='Aircraft: Number of Engines', ylabel='Counts - Damaging Strikes');

fig_air_type0.set_title('The Frequency of Damaging Strikes Over \n Aircraft Number of Engines');
# plot the damage rate of Aircraft: Number of engines?

fig_air_type01 = sns.barplot(x=u'Aircraft: Number of engines?', y='Damage Rate', data=count_air_type0)

fig_air_type01.set(xlabel='Aircraft: Number of Engines', ylabel='Damage Rate');

fig_air_type01.set_title('Damage Rate Over Aircraft Number of Engines');
count_air_n_eng = DataFrame({'count' : bird.groupby( ['Aircraft: Airline/Operator'] ).size()}).reset_index()

count_air_n_eng.sort_values(['count'], ascending=0).head(10)
count_air_n_eng0 = DataFrame({'count' : bird_dmg.groupby( ['Aircraft: Airline/Operator'] ).size()}).reset_index()

count_air_n_eng0.sort_values(['count'], ascending=0).head(10)
count_air_make = DataFrame({'count' : bird.groupby( ['Aircraft: Make/Model'] ).size()}).reset_index()

count_air_make.sort_values(['count'], ascending=0).head(10)
count_air_make0 = DataFrame({'count' : bird_dmg.groupby( ['Aircraft: Make/Model'] ).size()}).reset_index()

count_air_make0.sort_values(['count'], ascending=0).head(10)
# top 10 bird species - all strikes

count_species = DataFrame({'count' : bird.groupby( ['Wildlife: Species'] ).size()}).reset_index()

count_species.sort_values(['count'], ascending=0).head(10)
# top 10 bird species causing damages

count_species0 = DataFrame({'count' : bird_dmg.groupby( ['Wildlife: Species'] ).size()}).reset_index()

count_species0.sort_values(['count'], ascending=0).head(10)
# count of strikes by bird size 

count_bird = DataFrame({'count' : bird.groupby( ['Wildlife: Size'] ).size()}).reset_index()

# plot the frequency of all strikes over Wildlife: Size

fig_bird = sns.barplot(x=u'Wildlife: Size', y='count', data=count_bird)

fig_bird.set(ylabel='Count - All Strikes',xlabel='Wildlife Size');

fig_bird.set_title('The Frequency of All Strikes over Wildlife Size');
# count of strikes by bird size

count_bird0 = DataFrame({'count' : bird_dmg.groupby( ['Wildlife: Size'] ).size()}).reset_index()

count_bird0['All Strikes Counts'] = count_bird['count']

count_bird0['Damage Rate'] = count_bird0['count']/count_bird0['All Strikes Counts']



# plot the frequency of damaging strikes Wildlife: Number struck and Wildlife: Size

fig_bird0 = sns.barplot(x=u'Wildlife: Size', y='count', data=count_bird0)

fig_bird0.set(ylabel='Count - Damaging Strikes',xlabel='Wildlife Size');

fig_bird0.set_title('The Frequency of Damaging Strikes over Wildlife Size');
# plot damage rate over Wildlife: Size

fig_bird01 = sns.barplot(x=u'Wildlife: Size', y=u'Damage Rate', data=count_bird0)

fig_bird01.set(xlabel='Wildlife: Size', ylabel='Damage Rate');

fig_bird01.set_title('Damage Rate over Wildlife Size');
# There are a lot of missing data in these variables, but since the phase of the flight is available 

# some remedy is done here by filling in reasonalble values

bird.loc[ (bird['Miles from airport'].isnull()) & ( (bird['When: Phase of flight'] == 'Take-off run') |

          (bird['When: Phase of flight'] == 'Parked') | (bird['When: Phase of flight'] == 'Taxi') |

          (bird['When: Phase of flight'] == 'Landing Roll') ),'Miles from airport'] = 0

bird.loc[ (bird['Feet above ground'].isnull()) & ( (bird['When: Phase of flight'] == 'Take-off run') |

          (bird['When: Phase of flight'] == 'Parked') | (bird['When: Phase of flight'] == 'Taxi') |

          (bird['When: Phase of flight'] == 'Landing Roll') ),'Feet above ground'] = 0
DataFrame({'count' : bird.groupby( ['Altitude bin'] ).size()}).reset_index()
DataFrame({'count' : bird_dmg.groupby( ['Altitude bin'] ).size()}).reset_index()
# histogram of aircraft altitude information

hist_altitude = sns.distplot(bird['Feet above ground'].dropna(),kde=False);

hist_altitude.set_title('The Frequency of All Strikes over Aircraft Altitude');

hist_altitude.set(ylabel='Count - All Strikes');
# histogram of aircraft altitude information

hist_altitude0 = sns.distplot(bird_dmg['Feet above ground'].dropna(),kde=False);

hist_altitude0.set_title('The Frequency of Damaging Strikes over Aircraft Altitude');

hist_altitude0.set(ylabel='Count - Damaging Strikes');
# rate of aircraft below 1000 and 5000 ft for all strikes

rate_1000 = len( bird.loc[bird['Altitude bin']=='< 1000 ft','Altitude bin'] ) / len( bird.loc[(bird['Altitude bin']=='< 1000 ft') | (bird['Altitude bin']=='> 1000 ft'),'Altitude bin'] )

rate_5000 = len( bird.loc[bird['Feet above ground']<5000,'Feet above ground'] ) / len( bird.loc[~(bird['Feet above ground'].isnull()),'Feet above ground'] )

# rate of aircraft below 1000 and 5000 ft for damaging strikes

rate_1000 = len( bird_dmg.loc[bird_dmg['Altitude bin']=='< 1000 ft','Altitude bin'] ) / len( bird_dmg.loc[(bird_dmg['Altitude bin']=='< 1000 ft') | (bird_dmg['Altitude bin']=='> 1000 ft'),'Altitude bin'] )

rate_5000 = len( bird_dmg.loc[bird_dmg['Feet above ground']<5000,'Feet above ground'] ) / len( bird_dmg.loc[~(bird_dmg['Feet above ground'].isnull()),'Feet above ground'] )
count_phase = bird['When: Phase of flight'].value_counts()

fig_count = sns.barplot(x=count_phase.index, y=count_phase)

fig_count.set_xticklabels(labels=count_phase.index,rotation=30);

fig_count.set(xlabel='Phase of Flight', ylabel='Counts - All Strikes');

fig_count.set_title('The Frequency of All Strikes over Flight Status');
count_phase0 = bird_dmg['When: Phase of flight'].value_counts()

fig_count0 = sns.barplot(x=count_phase0.index, y=count_phase0)

fig_count0.set_xticklabels(labels=count_phase0.index,rotation=30);

fig_count0.set(xlabel='Phase of Flight', ylabel='Counts - Damaging Strikes');

fig_count0.set_title('The Frequency of Damaging Strikes over Flight Status');
flight_altitude = sns.boxplot(x="When: Phase of flight", y="Feet above ground", data=bird)

flight_altitude.set_xticklabels(flight_altitude.get_xticklabels(), rotation=30);

flight_altitude.set_title('Flight Altitude across Flight Status among All Strikes');
# one point stands out as the aircraft being 1200 miles from the airport in the Approach phase

# which is unlikely and could be a data entry error, the 'Miles from airport' in this row is thus

# replaced with NA, the boxplot is redrawn after the replacement

bird.loc[bird['Miles from airport'] > 1200,'Miles from airport'] = np.nan



# update bird_dmg as well

bird_dmg = bird.loc[(bird['Effect: Indicated Damage'] != 'No damage') | 

                    (bird['Cost: Total $'] > 0) ]



# re-draw the box plot

flight_miles1 = sns.boxplot(x="When: Phase of flight", y="Miles from airport", data=bird) 

flight_miles1.set_xticklabels(labels=flight_miles1.get_xticklabels(),rotation=30);

flight_miles1.set_title('Airplane-Airport Distance across Flight Status among All Strikes');
bird1 = bird.loc[(~bird['Miles from airport'].isnull()) &

                        (~bird['Feet above ground'].isnull()) ]

plt.scatter(x='Miles from airport', y='Feet above ground', 

              data= bird1);

plt.xlabel('Miles from Airport');

plt.ylabel('Feet above Ground');

plt.title('Flight Altitude over Airplane-Airport Distance among All Strikes');
bird10 = bird_dmg.loc[(~bird_dmg['Miles from airport'].isnull()) &

                        (~bird_dmg['Feet above ground'].isnull()) ]

plt.scatter(x='Miles from airport', y='Feet above ground', 

              data= bird10);

plt.xlabel('Miles from Airport');

plt.ylabel('Feet above Ground');

plt.title('Flight Altitude over Airplane-Airport Distance among Damaging Strikes');
# histogram of speed

# the current record of airplane is 6082.834 knots, any entry higher than that is set as NA

bird.loc[bird['Speed (IAS) in knots'] > 6100,'Speed (IAS) in knots'] = np.nan

speed = sns.distplot(bird['Speed (IAS) in knots'].dropna(),kde=False);

speed.set(xlabel='Speed in Knots', ylabel='Counts - All Strikes');

speed.set_title('The Frequency of All Strikes over Flight Speed');
# histogram of speed

speed0 = sns.distplot(bird_dmg['Speed (IAS) in knots'].dropna(),kde=False);

speed0.set(xlabel='Speed in Knots', ylabel='Counts - Damaging Strikes');

speed.set_title('The Frequency of Damaging Strikes over Flight Speed ');
# top 10 airports among all strikes

df_location = pd.DataFrame({'count' : bird.groupby( ['Airport: Name'] ).size()}).reset_index()

df_location.sort_values(['count'], ascending=False).head(10)
# top 10 airports among all damaging strikes

df_airport0 = pd.DataFrame({'count' : bird_dmg.groupby( ['Airport: Name'] ).size()}).reset_index()

df_airport0.sort_values(['count'], ascending=False).head(10)
# month variable

bird['Flight Month'] = pd.DatetimeIndex(bird['FlightDate']).month

# year variable

bird['Flight Year'] = pd.DatetimeIndex(bird['FlightDate']).year



# subset the data with any damage or negative impact to the flight

bird_dmg = bird.loc[(bird['Effect: Indicated Damage'] != 'No damage') | 

                    (bird['Cost: Total $'] > 0) ]
# count over flight month and year

count_time = DataFrame({'count' : bird.groupby( ['Flight Month', 'Flight Year'] ).size()}).reset_index()

# reshape frame

count_time_p=count_time.pivot("Flight Month", "Flight Year", "count")

# plot the frequency over month and year in a heat map

plt.figure(figsize=(8, 7))

heat_time = sns.heatmap(count_time_p);

heat_time.set_title('The Frequency of All Strikes Over Flight Year and Month');
# count over flight month and year

count_time0 = DataFrame({'count' : bird_dmg.groupby( ['Flight Month', 'Flight Year'] ).size()}).reset_index()

# reshape frame

count_time_p0=count_time0.pivot("Flight Month", "Flight Year", "count")

# plot the frequency over month and year in a heat map

plt.figure(figsize=(8, 7))

heat_time0 = sns.heatmap(count_time_p0);

heat_time0.set_title('The Frequency of Damaging Strikes Over Flight Year and Month');
# histogram of time information

fig_time = sns.distplot(bird['When: Time (HHMM)'].dropna(),kde=False);

fig_time.set(ylabel='Counts - All Strikes');

fig_time.set_title('The Frequency of All Strikes Over Time of the Day');
# histogram of time information

fig_time0 = sns.distplot(bird_dmg['When: Time (HHMM)'].dropna(),kde=False);

fig_time0.set(ylabel='Counts - Damaging Strikes');

fig_time.set_title('The Frequency of Damaging Strikes Over Time of the Day');
# cost histogram

cost = sns.distplot(np.log10(bird.loc[bird['Cost: Total $']>0,'Cost: Total $']),kde=False);

cost.set(xlabel='Log 10 of Total Cost in Dollar', ylabel='Counts - All Strikes');

cost.set_title('The Frequency of All Strikes Over Log Cost');
# cost histogram

cost0 = sns.distplot(np.log10(bird_dmg.loc[bird_dmg['Cost: Total $']>0,'Cost: Total $']),kde=False);

cost0.set(xlabel='Log 10 of Total Cost in Dollar', ylabel='Counts - Damaging Strikes');

cost0.set_title('The Frequency of Damaging Strikes Over Log Cost');
# damage count table

DataFrame({'count' : bird.groupby( ['Effect: Indicated Damage'] ).size()}).reset_index()
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
# scale the X_train and X_test

X_use = bird_use.drop("Damage",axis=1)

standard_scaler = StandardScaler().fit(X_use)

X_use1 = standard_scaler.transform(X_use) # Xs is the scaled matrix but has lost the featuren names

X_use2 = pd.DataFrame(X_use1, columns=X_use.columns) # Add feature names
# define training and testing sets

# choosing random indices equal to the number of damaging strikes

train_indices = np.array( np.random.choice(X_use2.index, int((X_use2.shape[0]/2)), replace= False) )

test_indices = np.array([item for item in X_use2.index if item not in train_indices])



# choosing random indices equal to the number of damaging strikes

bird_use = bird_use.reset_index()

X_train = X_use2.loc[train_indices,]

Y_train = bird_use.loc[train_indices,'Damage']

X_test = X_use2.loc[test_indices,]

Y_test = bird_use.loc[test_indices,'Damage']
# Logistic Regression using Scikit-learn

logreg = LogisticRegression(class_weight='balanced')

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

print('Training Accuracy: %1.3f.' % logreg.score(X_train, Y_train))
# generate evaluation metrics

logreg_t = metrics.accuracy_score(Y_test, Y_pred)

print('Testing Accuracy: %1.3f.' % logreg_t)
# evaluate the model using 10-fold cross-validation

scores_lr = cross_val_score(logreg, X_train, Y_train, scoring='accuracy', cv=10)

print('Cross-Validation Accuracy: %1.3f.' % scores_lr.mean())
# ROC AUC on train set

Y_prob_train = logreg.predict_proba(X_train)

lr_auc_train = metrics.roc_auc_score(Y_train, Y_prob_train[:, 1])

print ("ROC AUC on train set: %1.3f." % lr_auc_train)



# Predict on validation set

Y_prob_test = logreg.predict_proba(X_test)

lr_auc_test = metrics.roc_auc_score(Y_test, Y_prob_test[:, 1])

print ("ROC AUC on validation set: %1.3f." % lr_auc_test)
# logistic regression using statsmodels

logit = sm.Logit(bird_use["Damage"].reset_index(drop=True), X_use2)

result = logit.fit()

result.summary()
# check significance of the features

features_coefs = result.params.sort_values(ascending=False)

selectSignificant = result.pvalues[result.pvalues <= 0.05].index

selectSignificant
# Support Vector Machines

svc = SVC(class_weight='balanced',probability=True)

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

print('Training Accuracy: %1.3f.' % svc.score(X_train, Y_train))
# generate evaluation metrics

svc_t = metrics.accuracy_score(Y_test, Y_pred)

print('Testing Accuracy: %1.3f.' % svc_t)
# evaluate the model using 10-fold cross-validation

scores_svc = cross_val_score(svc, X_train, Y_train, scoring='accuracy', cv=10)

print('Cross-Validation Accuracy: %1.3f.' % scores_svc.mean())
# ROC AUC on train set

Y_prob_train = svc.predict_proba(X_train)

svc_auc_train = metrics.roc_auc_score(Y_train, Y_prob_train[:, 1])

print ("ROC AUC on train set: %1.3f." % svc_auc_train)



# Predict on validation set

Y_prob_test = svc.predict_proba(X_test)

svc_auc_test = metrics.roc_auc_score(Y_test, Y_prob_test[:, 1])

print ("ROC AUC on validation set: %1.3f." % svc_auc_test)
# Random Forests

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)

print('Training Accuracy: %1.3f.' % rf.score(X_train, Y_train))
# generate evaluation metrics

rf_t = metrics.accuracy_score(Y_test, Y_pred)

print('Testing Accuracy: %1.3f.' % rf_t)
# evaluate the model using 10-fold cross-validation

scores_rf = cross_val_score(rf, X_train, Y_train, scoring='accuracy', cv=10)

print('Cross-Validation Accuracy: %1.3f.' % scores_rf.mean())
# ROC AUC on train set

Y_prob_train = rf.predict_proba(X_train)

rf_auc_train = metrics.roc_auc_score(Y_train, Y_prob_train[:, 1])

print ("ROC AUC on train set: %1.3f." % rf_auc_train)



# Predict on validation set

Y_prob_test = rf.predict_proba(X_test)

rf_auc_test = metrics.roc_auc_score(Y_test, Y_prob_test[:, 1])

print ("ROC AUC on validation set: %1.3f." % rf_auc_test)
# KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

print('Training Accuracy:')

knn.score(X_train, Y_train)
# generate evaluation metrics

knn_t = metrics.accuracy_score(Y_test, Y_pred)

print('Testing Accuracy:')

knn_t
# evaluate the model using 10-fold cross-validation

scores_knn = cross_val_score(knn, X_train, Y_train, scoring='accuracy', cv=10)

print('Cross-Validation Accuracy:')

print(scores_knn.mean())
# ROC AUC on train set

Y_prob_train = knn.predict_proba(X_train)

knn_auc_train = metrics.roc_auc_score(Y_train, Y_prob_train[:, 1])

print ("ROC AUC on train set: %1.3f." % knn_auc_train)



# Predict on validation set

Y_prob_test = knn.predict_proba(X_test)

knn_auc_test = metrics.roc_auc_score(Y_test, Y_prob_test[:, 1])

print ("ROC AUC on validation set: %1.3f." % knn_auc_test)
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

print('Training Accuracy:')

gaussian.score(X_train, Y_train)
# generate evaluation metrics

gaussian_t = metrics.accuracy_score(Y_test, Y_pred)

print('Testing Accuracy:')

gaussian_t
# evaluate the model using 10-fold cross-validation

scores_gaussian = cross_val_score(gaussian, X_train, Y_train, scoring='accuracy', cv=10)

print('Cross-Validation Accuracy:')

print (scores_gaussian.mean())
# ROC AUC on train set

Y_prob_train = gaussian.predict_proba(X_train)

gaussian_auc_train = metrics.roc_auc_score(Y_train, Y_prob_train[:, 1])

print ("ROC AUC on train set: %1.3f." % gaussian_auc_train)



# Predict on validation set

Y_prob_test = gaussian.predict_proba(X_test)

gaussian_auc_test = metrics.roc_auc_score(Y_test, Y_prob_test[:, 1])

print ("ROC AUC on validation set: %1.3f." % gaussian_auc_test)
train_acc = [logreg.score(X_train, Y_train), svc.score(X_train, Y_train), rf.score(X_train, Y_train),

             knn.score(X_train, Y_train), gaussian.score(X_train, Y_train)]

test_acc = [logreg_t, svc_t, rf_t, knn_t, gaussian_t]

cross_val_acc = [scores_lr.mean(), scores_svc.mean(), scores_rf.mean(), scores_knn.mean(), scores_gaussian.mean()]

train_auc = [lr_auc_train, svc_auc_train, rf_auc_train, knn_auc_train, gaussian_auc_train]

test_auc = [lr_auc_test, svc_auc_test, rf_auc_test, knn_auc_test, gaussian_auc_test]

models = DataFrame({'Training Accuracy': train_acc, 'Testing Accuracy': test_acc, 

                    "Cross-Validation Accuracy": cross_val_acc,'Training AUC': train_auc,

                   'Testing AUC': test_auc})

models.index = ['Logistic Regression','Support Vector Machines ','Random Forests','K-Nearest Neighbors','Gaussian Naive Bayes']

models
models1 = DataFrame({'Accuracy' : models.unstack()}).reset_index()

# plot accuracies

plt.figure(figsize=(8, 7))

fig_models = sns.barplot(x='level_0', y='Accuracy', hue='level_1', data=models1);

fig_models.set(xlabel='Accuracy Metric', ylabel='Accuracy');

fig_models.set_title('The Accuracy of All Models Over Five Metrics');
x=zip(X_train.columns, np.transpose(logreg.coef_))

x1=pd.DataFrame(list(x))

x1.head()
# get Correlation Coefficient for each feature using Logistic Regression

logreg_df = pd.DataFrame(list(zip(X_train.columns, np.transpose(logreg.coef_))))

logreg_df.columns = ['Features','Coefficient Estimate']

logreg_df['sort'] = logreg_df['Coefficient Estimate'].abs()

logreg_df.sort_values(['sort'],ascending=0).drop('sort',axis=1).head(10)