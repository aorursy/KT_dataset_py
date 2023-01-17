import pandas as pd

import numpy as np

import seaborn as sns

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.utils import resample, shuffle

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split

from sklearn.metrics import f1_score, confusion_matrix ,accuracy_score

from sklearn.metrics import precision_score, recall_score
data = pd.read_csv('../input/KaggleV2-May-2016.csv')

data.head()
n_rows = data.shape[0]

n_cols = data.shape[1]

print('The dataset has', n_rows, 'rows and', n_cols, 'features')
data['No-show'] = data['No-show'].replace({'No':0, 'Yes':1})
# engineer feature missed_appointment before

missed_appointment = data.groupby('PatientId')['No-show'].sum()

missed_appointment = missed_appointment.to_dict()

data['missed_appointment_before'] = data.PatientId.map(lambda x: 1 if missed_appointment[x]>0 else 0)

data['missed_appointment_before'].corr(data['No-show'])
patients = data.drop_duplicates(subset=['PatientId'])

n_patients = patients.shape[0]

count = patients.Gender.value_counts()

n_women = count.values[0]

n_men = count.values[1]

print('Proportion of the dataset that are women: {0:2.1f}%'.format(100*n_women/n_patients))

print('Proportion of the dataset that are men: {0:2.1f}%'.format(100*n_men/n_patients))
#separate in 2 groups

show = data[data['No-show']==0]

no_show = data[data['No-show']==1]

n_show = show.shape[0]

n_no_show = no_show.shape[0]

print('Percentage of persons that do not miss appointments:{0: 2.2f}%'.format(100*n_show/n_rows))

print('Percentage of persons that miss appointments:{0: 2.2f}%'.format(100*n_no_show/n_rows))
def map_waiting_interval_to_days(x):

    '''

    Receives an integer representing the number of days until an appointment and

    returns the category it is in.

    '''

    if x ==0 :

        return 'Less than 15 days'

    elif x > 0 and x <= 2:

        return 'Between 1 day and 2 days'

    elif x > 2 and x <= 7:

        return 'Between 3 days and 7 days'

    elif x > 7 and x <= 31:

        return 'Between 7 days and 31 days'

    else:

        return 'More than 1 month'
d = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

data['mapped_AppointmentDay'] = data['AppointmentDay'].map(lambda x: datetime.strptime(x,"%Y-%m-%dT%H:%M:%SZ"))

data['mapped_ScheduledDay'] = data['ScheduledDay'].map(lambda x: datetime.strptime(x,"%Y-%m-%dT%H:%M:%SZ"))

data['waiting_interval'] = abs(data['mapped_ScheduledDay'] - data['mapped_AppointmentDay'])

data['waiting_interval_seconds'] = data['waiting_interval'].map(lambda x: x.seconds)

data['waiting_interval_days'] = data['waiting_interval'].map(lambda x: x.days)

data['waiting_interval_days'] = data['waiting_interval_days'].map(lambda x: map_waiting_interval_to_days(x))



data['ScheduledDay_month'] = data['mapped_ScheduledDay'].map(lambda x: x.month)

data['ScheduledDay_day'] = data['mapped_ScheduledDay'].map(lambda x: x.day)

data['ScheduledDay_weekday'] = data['mapped_ScheduledDay'].map(lambda x: x.weekday())



data['AppointmentDay_month'] = data['mapped_AppointmentDay'].map(lambda x: x.month)

data['AppointmentDay_day'] = data['mapped_AppointmentDay'].map(lambda x: x.day)

data['AppointmentDay_weekday'] = data['mapped_AppointmentDay'].map(lambda x: x.weekday())

data['AppointmentDay_weekday'] = data['AppointmentDay_weekday'].replace(d)



# separate in 2 groups

show = data[data['No-show']==0]

no_show = data[data['No-show']==1]

n_show = show.shape[0]

n_no_show = no_show.shape[0]
levels = ['Less than 15 days','Between 1 day and 2 days','Between 3 days and 7 days',

'Between 7 days and 31 days','More than 1 month']



grouped = show.groupby(by='waiting_interval_days')

count_days1 = grouped.waiting_interval_days.count().reindex(index = levels)

count_days1 = 100*count_days1/show.shape[0]



grouped = no_show.groupby(by='waiting_interval_days')

count_days2 = grouped.waiting_interval_days.count().reindex(index = levels)

count_days2 = 100*count_days2/no_show.shape[0]



sns.set_style("whitegrid")

f, ax = plt.subplots(1, 2,figsize=(12, 4),sharey=True)

g1 = sns.barplot(x=count_days1.index, y=count_days1.values, 

                color='lightblue',ax=ax[0])

g1.set_xticklabels(levels, rotation=90);

g1.set_xlabel('');

g1.set_title('Show');

g2 = sns.barplot(x=count_days2.index, y=count_days2.values, 

                color='salmon',ax=ax[1])

g2.set_xticklabels(levels, rotation=90);

g2.set_xlabel('');

g2.set_title('No-Show');
def map_age(x):

    '''

    Receives an integer and returns the age category that this age is in.

    '''

    if x < 12:

        return 'Child'

    elif x > 12 and x < 18:

        return 'Teenager'

    elif x>=20 and x<25:

        return 'Young Adult'

    elif x>=25 and x<60:

        return 'Adult'

    else:

        return 'Senior'

data['mapped_Age'] = data['Age'].map(lambda x: map_age(x))

patients['mapped_Age'] = patients['Age'].map(lambda x: map_age(x))
ages = ['Child','Teenager','Young Adult','Adult','Senior']

n_patients = patients.shape[0]

grouped = patients.groupby(by='mapped_Age')

count_ages = grouped.Age.count().reindex(index = ages)

g = sns.barplot(x=count_ages.index, y=count_ages.values*(100/n_patients), color='lightblue');

g.set_title('Age percentage');

g.set_xlabel('');
show = data[data['No-show']==0]

no_show = data[data['No-show']==1]

n_show = show.shape[0]

n_no_show = no_show.shape[0]

ages = ['Child','Teenager','Young Adult','Adult','Senior']



# count ages for group which didnt miss appointment

grouped = show.groupby(by='mapped_Age')

count_ages1 = grouped.Age.count().reindex(index = ages)

count_ages1 = count_ages1*(100/show.shape[0])



# count ages for group which missed appointment

grouped = no_show.groupby(by='mapped_Age')

count_ages2 = grouped.Age.count().reindex(index = ages)

count_ages2 = count_ages2*(100/no_show.shape[0])



sns.set_style("whitegrid")

f, ax = plt.subplots(1, 2,figsize=(12, 4),sharey=True)

g1 = sns.barplot(x=count_ages1.index, y=count_ages1.values, 

            color='lightblue',ax=ax[0])

g1.set_xlabel('');

g1.set_title('Show');

g2 = sns.barplot(x=count_ages2.index, y=count_ages2.values, 

            color='salmon',ax=ax[1]);

g2.set_xlabel('');

g2.set_title('No-Show');
patients = data.drop_duplicates(subset=['PatientId'])

patients[['Hipertension','Diabetes','Alcoholism','Handcap']].sum(axis=0)/patients.shape[0]
data['haveDisease'] = data.Alcoholism | data.Handcap | data.Diabetes | data.Hipertension
fig, ax = plt.subplots(figsize=[15,10])

data = data.drop(columns=['AppointmentID', 'PatientId'])

cor=data.corr()

mask = np.zeros_like(cor)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

ax = sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns, 

            annot=True, cmap=cmap, mask=mask);
# Check for missing values

data.isnull().sum().any()
def process_data(data):

    '''

    Receives the dataset, clean data and engineer new features. 

    Return cleaned dataset with features that will be used for training model.

    '''

    d = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

    data['mapped_AppointmentDay'] = data['AppointmentDay'].map(lambda x: datetime.strptime(x,"%Y-%m-%dT%H:%M:%SZ"))

    data['mapped_ScheduledDay'] = data['ScheduledDay'].map(lambda x: datetime.strptime(x,"%Y-%m-%dT%H:%M:%SZ"))

    data['waiting_interval'] = abs(data['mapped_ScheduledDay'] - data['mapped_AppointmentDay'])

    data['waiting_interval_days'] = data['waiting_interval'].map(lambda x: x.days)

    data['waiting_interval_days'] = data['waiting_interval_days'].map(lambda x: map_waiting_interval_to_days(x))

    

    data['ScheduledDay_month'] = data['mapped_ScheduledDay'].map(lambda x: x.month)

    data['ScheduledDay_day'] = data['mapped_ScheduledDay'].map(lambda x: x.day)

    data['ScheduledDay_weekday'] = data['mapped_ScheduledDay'].map(lambda x: x.weekday())

    data['ScheduledDay_weekday'] = data['ScheduledDay_weekday'].replace(d)



    data['AppointmentDay_month'] = data['mapped_AppointmentDay'].map(lambda x: x.month)

    data['AppointmentDay_day'] = data['mapped_AppointmentDay'].map(lambda x: x.day)

    data['AppointmentDay_weekday'] = data['mapped_AppointmentDay'].map(lambda x: x.weekday())

    data['AppointmentDay_weekday'] = data['AppointmentDay_weekday'].replace(d)

    

    data['No-show'] = data['No-show'].replace({'Yes':1, 'No':0})

   

    missed_appointment = data.groupby('PatientId')['No-show'].sum()

    missed_appointment = missed_appointment.to_dict()

    data['missed_appointment_before'] = data.PatientId.map(lambda x: 1 if missed_appointment[x]>0 else 0)

    data['mapped_Age'] = data['Age'].map(lambda x: map_age(x))

    data['Gender'] = data['Gender'].replace({'F':0, 'M':1})

    data['haveDisease'] = data.Alcoholism | data.Handcap | data.Diabetes | data.Hipertension



    data = data.drop(columns=['waiting_interval', 'AppointmentDay', 'ScheduledDay',

                             'PatientId','Age', 'mapped_ScheduledDay',

                             'mapped_AppointmentDay', 'AppointmentID', 

                              'Alcoholism','Handcap','Diabetes','Hipertension'])

    

    return data
def one_hot_encode(data):

    return pd.get_dummies(data)
data = pd.read_csv('../input/KaggleV2-May-2016.csv')

processed_data = process_data(data)

processed_data.head()
encoded_data = one_hot_encode(processed_data)

encoded_data.head()
print('All data - Naive predictor accuracy: {:2.2f}%'.format(100 - (100*encoded_data['No-show'].sum()/encoded_data.shape[0])))
# row: true label ; columns: predictions

tn, fp, fn, tp = confusion_matrix(encoded_data['No-show'], np.zeros(encoded_data.shape[0])).ravel()

(tn, fp, fn, tp)
X = encoded_data.drop(columns='No-show')

y = encoded_data['No-show']
scaler = MinMaxScaler()

X_std = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.30, random_state=42)



print('Train size:{}'.format(X_train.shape))

print('Test size:{}'.format(X_test.shape))
clf1 = RandomForestClassifier(random_state = 0)

clf1.fit(X_train, y_train)

y_preds = clf1.predict(X_test)
print('RF - Accuracy: {:2.2f}%'.format(accuracy_score(y_test, y_preds) * 100))

print('RF - Precision score: {:2.2f}%'.format(precision_score(y_test, y_preds)*100))

print('RF - Recall score: {:2.2f}%'.format(recall_score(y_test, y_preds)*100))

print('RF - F1-score: {:2.2f}%'.format(f1_score(y_test, y_preds) * 100))
tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()

confusion_matrix(y_test, y_preds)
clf1 = GradientBoostingClassifier(random_state = 0)

clf1.fit(X_train, y_train)

y_preds = clf1.predict(X_test)

print('GB - Accuracy: {:2.2f}%'.format(accuracy_score(y_test, y_preds) * 100))

print('GB - Precision score: {:2.2f}%'.format(precision_score(y_test, y_preds)*100))

print('GB - Recall score: {:2.2f}%'.format(recall_score(y_test, y_preds)))

print('GB - F1-score: {:2.2f}%'.format(f1_score(y_test, y_preds) * 100))
tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()

confusion_matrix(y_test, y_preds)
print(processed_data.shape)

print(encoded_data.shape)



# separates data in two groups

show = data[data['No-show']==0]

no_show = data[data['No-show']==1]

n_show = show.shape[0]

n_no_show = no_show.shape[0]

print('Proportion of minority class: {:2.2f}%'.format(100*no_show.shape[0]/n_rows))

print('Proportion of majority class: {:2.2f}%'.format(100*show.shape[0]/n_rows))
no_show.shape[0]/show.shape[0] # 1 positive no-show for 4 negatives no-show
# Calculates how many samples we resample from the majority class

downsampling_factor = 4

n_samples = round(show.shape[0]/downsampling_factor)

n_samples
# separate classes

df_majority = encoded_data[encoded_data['No-show']==0]

df_minority = encoded_data[encoded_data['No-show']==1]

 

# downsample without replacement majority class

df_majority_downsampled = resample(df_majority, replace=False, n_samples=n_samples, random_state=0)  

 

# save the index to use when perform the upweight of the samples

resampled_index = df_majority_downsampled.index



# combine minority class with downsampled majority class

df_downsampled = pd.concat([df_majority_downsampled, df_minority])



# shuffle dataframe

df_downsampled = shuffle(df_downsampled)



# take a look into proportion after resampling

df_majority = df_downsampled[df_downsampled['No-show']==0]

df_minority = df_downsampled[df_downsampled['No-show']==1]



print('Proportion of minority class after downsampling: {:2.2f}%'.format(100*df_minority.shape[0]/df_downsampled.shape[0]))

print('Proportion of majority class after downsampling: {:2.2f}%'.format(100*df_majority.shape[0]/df_downsampled.shape[0]))
# create column of original index before resampling

df_downsampled = df_downsampled.reset_index()



# get the weights

weight_factor = round(downsampling_factor)

weights = np.ones(df_downsampled.shape[0])



# iforiginal index is in resampled_index change weight

cols = df_downsampled.columns

df_downsampled.columns = ['original_index'] + list(cols)[1:]

df_downsampled['weight'] = df_downsampled.original_index.map(lambda x: weight_factor if x in resampled_index else 1).values
X = df_downsampled.drop(columns='No-show')

y = df_downsampled['No-show']
scaler = MinMaxScaler()

X_to_scale = X.drop(columns=['original_index', 'weight'])

cols = X_to_scale.columns

X_std = scaler.fit_transform(X_to_scale)

X_std = pd.DataFrame(X_std)



# reappend 'original_index', 'weight' to the dataframe

X_std = X_std.join(X[['original_index', 'weight']])

X_std.columns = list(cols) + ['original_index', 'weight']
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.30, random_state=0)



print('Train size:{}'.format(X_train.shape))

print('Test size:{}'.format(X_test.shape))
weights = X_train.weight

X_train = X_train.drop(columns=['original_index','weight'])

X_test = X_test.drop(columns=['original_index','weight'])
clf1 = RandomForestClassifier(random_state = 0)

clf1.fit(X_train, y_train, sample_weight=weights)

y_preds = clf1.predict(X_test)

print('RF - Accuracy: {:2.2f}%'.format(accuracy_score(y_test, y_preds) * 100))

print('RF - Precision score: {:2.2f}%'.format(precision_score(y_test, y_preds)*100))

print('RF - Recall score: {:2.2f}%'.format(recall_score(y_test, y_preds)*100))

print('RF - F1-score: {:2.2f}%'.format(f1_score(y_test, y_preds) * 100))
tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()

confusion_matrix(y_test, y_preds)
clf = GradientBoostingClassifier(random_state=0)

clf.fit(X_train, y_train)

y_preds = clf.predict(X_test)

acc = accuracy_score(y_test, y_preds) * 100

precision = precision_score(y_test, y_preds)*100

recall = recall_score(y_test, y_preds)*100

f_score = f1_score(y_test, y_preds) * 100



print('GB - Accuracy: {:2.2f}%'.format(acc))

print('GB - Precision score: {:2.2f}%'.format(precision))

print('GB - Recall score: {:2.2f}%'.format(recall))

print('GB - F1-score: {:2.2f}%'.format(f_score))
tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()

confusion_matrix(y_test, y_preds)
#http://benalexkeen.com/bar-charts-in-matplotlib/

%matplotlib inline



x = ['Acc', 'Precision','Recall', 'F1']

metrics = [acc, precision, recall, f_score]

plt.gca().yaxis.grid(True)

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, metrics, color='lightblue')

plt.xlabel("Metrics")

plt.ylabel("Percentage")

plt.xticks(x_pos, x)

plt.show()