import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
%matplotlib inline
# i will work with both files at the same time in order to transform
# the data at once. in the ML part i'll splite the data in:
# Train, validation file and test

train_df =  pd.read_csv("../input/train.csv", parse_dates=['DateTime'])
test_df =  pd.read_csv('../input/test.csv', parse_dates=['DateTime'])
test_df['OutcomeType'] = ''

train_df = train_df.drop(['AnimalID','OutcomeSubtype'] , axis=1)
test_df = test_df.drop('ID', axis=1)

full_data = pd.concat([train_df, test_df], axis=0)
full_data.info()
del train_df , test_df
#check population of cat and dog balance
bal_population = full_data.groupby('AnimalType')['DateTime'].count()
bal_population.plot.bar()
# Check Missing Data:
def num_missing(x):
    
    return sum(x.isnull())

#Applying per column:

print ('Missing data per column\n', full_data.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column)

#Applying per row:
# "'if are there rows with a lot of missing data they should be pulled out of the calculation (althogu in this case the most important columns haven't empty data )'"

# print ('Missing data per row',df_shelter.apply(num_missing, axis=1).sort_values(ascending=False)) #axis=1 defines that function is to be applied on each row)
#clean empty rows (rows without Datatime, AnimalType and Age aren't usefull)
mask = ~full_data['DateTime'].isna() & ~full_data['AnimalType'].isna()& ~full_data['AgeuponOutcome'].isna()
#full_data.loc[mask == False]
# print(full_data.shape)
full_data = full_data.loc[mask, :]
# print(full_data.shape)
# #another way to check rows with empty values 
# def count_missing(vec):
#     """counts the number of missing values in a vector
#     """
#     null_vec = pd.isnull(vec) # vector of True/False
#     null_count = np.sum(null_vec) # True has a value of 1
#     return null_count

# df_shelter.apply(count_missing, axis=1).value_counts()
#step 1
full_data['AgeuponOutcome'].unique() # how do i sorted values here?.sort()
# step 2 & 3
# Calculating the total animal's age:
full_data['AgeuponOutcome_Time']= full_data['AgeuponOutcome'].str.split(' ').str.get(1) #split the data 
full_data['AgeuponOutcome_Time']=full_data['AgeuponOutcome_Time'].replace( ['year', 'years', 'week', 'weeks', 'month','months', 'days','day'], 
                                                                [365, 365, 7, 7,30.5, 30.5, 1,1]) #replace by the total days 
#fill the empty values in order to be able to do math calculations
full_data['AgeuponOutcome_Time'] = full_data['AgeuponOutcome_Time'].fillna(0).astype(float)

#transform the age data to numbers that we can manipulate with math tecniques
full_data['AgeuponOutcome_Age']= full_data['AgeuponOutcome'].str.split(' ').str.get(0).fillna(0).astype(float)

#calculate the age in days 
full_data['AgeuponOutcome_InDays']= full_data['AgeuponOutcome_Age']* full_data['AgeuponOutcome_Time']

#calculate the age in years
full_data['AgeuponOutcome_InYears']= full_data['AgeuponOutcome_InDays']/365


#split the gender of  the animal 
def General_gender(x):
    if x == 'Female':
        return 'Female'
    elif x == 'Male':
        return 'Male'
    else:
        return 'unknow'
    
full_data['Gender']= full_data['SexuponOutcome'].str.split(' ').str.get(1) #separate general gender
full_data['Gender'] = full_data.Gender.apply(General_gender) 

# data.loc[data.Gender == 'unknow'].count()
# data.loc[data.Gender.isnull()].count()

from numpy import where  #why do i need to import "where "again?

Ages = full_data.groupby(['AnimalType','AgeuponOutcome_InYears'])['AgeuponOutcome_InYears'].count().reset_index(name='Qty')
a_min, a_max = 0, 3000
y_min, y_max = 0, 20
Ages.plot('Qty', 'AgeuponOutcome_InYears', kind='scatter',
          xlim=[a_min, a_max], ylim=[y_min, y_max],
          c=where(full_data.AnimalType =='Dog', 'blue', 'green'),legend = True, 
          label = 'Dog' , s=50)
def CategoryAge(x):
    if x <= 1.5:
        return 'NewBorn'
    elif 1.5 < x <= 3 :
        return 'Adolescence'
    elif 3 < x <= 7 :
        return 'Adulthood '
    else:
        return 'Senior'
    
full_data['AgeCategory'] = full_data.AgeuponOutcome_InYears.apply(CategoryAge)
sns.factorplot(x="AnimalType", hue="AgeCategory", data=full_data, kind="count",
                    size=4, aspect=.7)

#Drill down into each Animaltype and current OutcomeType

dogs = full_data.loc[(full_data['AnimalType']=='Dog' )]
cats = full_data.loc[(full_data['AnimalType']=='Cat')]
#Check the segmentation of the cases based on all dog's data, 
#AgeCategory and OutcomeType

sns.factorplot(x="OutcomeType", hue="AgeCategory", data=dogs, kind="count",
                    size=10, aspect=.7)

#Check the segmentation of the cases based on all cat's data, 
#AgeCategory and OutcomeType

sns.factorplot(x="OutcomeType", hue="AgeCategory", data=cats, kind="count",
                    size=10, aspect=.7)
full_data['Day_ofWeek'] = full_data['DateTime'].dt.weekday_name
 
def WeekEnd(x):
    if x in ('Saturday', 'Sunday'):
        return 'Weekend'
    else:
        return 'NoWeekend'
    
full_data['WeekEnd'] = full_data.Day_ofWeek.apply(WeekEnd)

sns.factorplot(x="AgeCategory", hue="WeekEnd", data=full_data, kind="count",
                    size=10, aspect=.7)
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
cal = calendar()
dr = full_data.DateTime
holidays = cal.holidays(start=dr.min(), end=dr.max())
full_data['Holiday'] = full_data['DateTime'].isin(holidays)

# Check status
full_data.groupby(['Holiday'])['DateTime'].count()
sns.factorplot(x="AnimalType", hue="Gender", data=full_data, kind="count",
                    size=5, aspect=.7)
full_data['Name_missing'] = full_data['Name'].isna()
full_data.groupby(['AnimalType'])['Name_missing'].count()
full_data = full_data[full_data['Name_missing']==True]
# sns.factorplot(x="OutcomeType", hue="AnimalType", data=name_missing, kind="count",
#                     size=4, aspect=4)
#color
           
full_data['Color1'] = full_data.Color.str.split('/').str.get(1)  
full_data['Color2'] =full_data.Color.str.split('/').str.get(2)  
full_data['unicolor'] = full_data.Color1.apply(lambda x: 1 if x else 0) 

# full_data['unicolor'].isna()      
#checking if the breed is mix or not
def get_mix(x):
    x = str(x)
    if x.find('Mix') >= 0: return 'mix'
    return 'not'
full_data['Mix'] = full_data.Breed.apply(get_mix)

# full_data.columns
from sklearn.model_selection import train_test_split

#splite the files test and train again

X_test_general =  full_data[full_data['OutcomeType']==''] 
X_train_general = full_data[full_data['OutcomeType']!='']

#create the Target for train
X_train = X_train_general.drop('OutcomeType', axis = 1)
X_Target = X_test_general.drop('OutcomeType', axis = 1)
y_train = X_train_general.OutcomeType

#create file for the frist cheking
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,train_size=0.7, random_state=451816)


# Choose relevant features for start the prediction 

X_train = X_train[['AnimalType','SexuponOutcome',
                      'AgeuponOutcome_InYears', 'Gender',
                      'AgeCategory','Day_ofWeek','WeekEnd',
                      'Holiday','Name_missing','unicolor']]
X_test = X_test[['AnimalType','SexuponOutcome',
                      'AgeuponOutcome_InYears', 'Gender',
                      'AgeCategory','Day_ofWeek','WeekEnd',
                      'Holiday','Name_missing','unicolor']]

X_train_dm= pd.get_dummies(X_train)
X_test_dm= pd.get_dummies(X_test)

# X_train.head(2)
# X_test.head(2)
# X_train_dm.head(2)
# y_train.head(2)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

#fit the model (X_train_dm,y_train )
shelter_dt = DecisionTreeClassifier(max_depth=5).fit(X_train_dm, y_train)
# Assess the model
cm = confusion_matrix(y_true=y_train,
                      y_pred=shelter_dt.predict(X_train_dm))
pd.DataFrame(cm,
             index=shelter_dt.classes_,
             columns=shelter_dt.classes_)
#Accuracy
print('Accuracy Validation', accuracy_score(y_true=y_train,
                      y_pred=shelter_dt.predict(X_train_dm)))
#improve the accuraccy by max_dep
shelter_dt_3 = DecisionTreeClassifier(max_depth=3).fit(X_train_dm, y_train)
shelter_dt_5 = DecisionTreeClassifier(max_depth=5).fit(X_train_dm, y_train)
shelter_dt_20 = DecisionTreeClassifier(max_depth=20).fit(X_train_dm, y_train)
acc_3 = accuracy_score(y_true=y_train,
                      y_pred=shelter_dt_3.predict(X_train_dm))
acc_4 = accuracy_score(y_true=y_train,
                      y_pred=shelter_dt.predict(X_train_dm))
acc_5 = accuracy_score(y_true=y_train,
                      y_pred=shelter_dt_5.predict(X_train_dm))
acc_20 = accuracy_score(y_true=y_train,
                      y_pred=shelter_dt_20.predict(X_train_dm))
Max_dep_t = ['max_dep3', 'max_dep4','max_dep5', 'max_dep20']
Max_dep_r = [acc_3, acc_4, acc_5, acc_20]
best_accurracy = dict(zip(Max_dep_t,Max_dep_r))
best_accurracy
#fit the model (X_train_dm,y_train )
# shelter_dt = DecisionTreeClassifier(max_depth=5).fit(X_train_dm, y_train)
# Assess the model
cm = confusion_matrix(y_true=y_train,
                      y_pred=shelter_dt_20.predict(X_train_dm))
pd.DataFrame(cm,
             index=shelter_dt_20.classes_,
             columns=shelter_dt_20.classes_)
#validating the model with the validation file from train
clf = DecisionTreeClassifier(max_depth=20)

clf.fit(X_train_dm, y_train)
print(clf.score(X_train_dm, y_train))
print(clf.score(X_test_dm, y_test))

# DateTime (during the day)
interval = pd.to_datetime(full_data['DateTime'])-pd.to_datetime('2013-01-01')
full_data.loc[:, 'DateTime_day'] = interval /np.timedelta64(1, 'D')
interval = ((interval/np.timedelta64(1, 'D'))%1*24)
interval[(interval>=0) & (interval)<6] = 0 #'midnight'
interval[(interval>=6) & (interval<12)] = 1 #'morning'
interval[(interval>=12) & (interval<18)] = 2 #'afternoon'
interval[(interval>=18) & (interval<22)] = 3 #'evening'
interval[(interval>=22) & (interval<24)] =0 # 'midnight'
full_data.loc[:, 'DateTime_intday'] = interval
# SexuponOutcome - increace the feature definition
full_data.loc[:,['SexuponOutcome']] = full_data['SexuponOutcome'].fillna(full_data['SexuponOutcome'].mode()[0])
full_data['Sex'] = full_data['SexuponOutcome'].map({'Intact Female':0,'Spayed Female':0,'Intact Male':1,'Neutered Male':1,'Unknown':2})
full_data['IsIntact'] = full_data['SexuponOutcome'].map({'Intact Female':0,'Intact Male':0, 'Neutered Male':1,'Spayed Female':1,'Unknown':2})
full_data[['SexuponOutcome','Sex','IsIntact']].head()
#splite the files test and train again

X_test_general_1 =  full_data[full_data['OutcomeType']==''] 
X_train_general_1 = full_data[full_data['OutcomeType']!='']

#create the Target for train
X_train_1 = X_train_general_1.drop('OutcomeType', axis = 1)
X_Target_1 = X_test_general_1.drop('OutcomeType', axis = 1)
y_train_1 = X_train_general_1.OutcomeType


#create file for the frist cheking
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_train_1,y_train_1,train_size=0.7, random_state=451816)

# Choose relevant features for start the prediction 

X_train_1 = X_train_1[['DateTime_intday','AnimalType',
                       'SexuponOutcome','Sex','IsIntact',
                      'AgeuponOutcome_InYears', 'Gender',
                      'AgeCategory','Day_ofWeek','WeekEnd',
                      'Holiday','Name_missing','unicolor']]
X_test_1 = X_test_1[['DateTime_intday','AnimalType',
                       'SexuponOutcome','Sex','IsIntact',
                      'AgeuponOutcome_InYears', 'Gender',
                      'AgeCategory','Day_ofWeek','WeekEnd',
                      'Holiday','Name_missing','unicolor']]

X_train_dm_1= pd.get_dummies(X_train_1)
X_test_dm_1= pd.get_dummies(X_test_1)

# X_train.head(2)
# X_test.head(2)
# X_train_dm.head(2)
# y_train.head(2)
#fit the model (X_train_dm,y_train )
shelter_dt_1 = DecisionTreeClassifier(max_depth=5).fit(X_train_dm_1, y_train_1)
# Assess the model
cm = confusion_matrix(y_true=y_train_1,
                      y_pred=shelter_dt_1.predict(X_train_dm_1))
#Accuracy
print('Accuracy with more features',accuracy_score(y_true=y_train,
                      y_pred=shelter_dt_1.predict(X_train_dm_1)))

# Try to keep improving the data
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

steps = [ ('scale', StandardScaler()), ('reduce_dim', PCA(2)), ('clf', RandomForestClassifier()) ]
pipe = Pipeline(steps)

pipe.fit(X_train_dm_1, y_train_1)
train_pred = pipe.predict(X_train_dm_1)
print('Accuracy with Pipeline Steps', accuracy_score(train_pred, y_train_1))

#Validating the model :

# pipe.fit(X_train_dm_1, y_train_1)
train_pred = pipe.predict(X_test_dm_1)
accuracy_score(train_pred, y_test_1)


# classifiers 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = SVC()

classifiers = [('LR', clf1), ('DT', clf2), ('SVM', clf3)]

results = y_train_1.to_frame()
for clf_name, clf in classifiers:
    clf.fit(X_train_dm_1, y_train_1)
    results[clf_name] = clf.predict(X_train_dm_1)
    print("{:3} classifier:\n \
        \ttrain accuracy: {:.2f}\n \
        \ttest accuracy: {:.2f}"\
        .format(clf_name, 
                clf.score(X_train_dm_1, y_train_1), 
                clf.score(X_test_dm_1, y_test_1)))

  
#checking for overfit
from sklearn.model_selection import cross_val_score

for n in range (1,5):
    pipe.set_params(reduce_dim__n_components = n)
    scores = cross_val_score(pipe,X_train_dm_1, y_train_1,cv=5)
    print(f'{n} components')
    print(f'mean: {scores.mean()}')
    print(f'std: {scores.std()}\n')
#upload resoults
# X_Target_1.head(2)

X_target_1_upload = X_Target_1[['DateTime_intday','AnimalType',
                       'SexuponOutcome','Sex','IsIntact',
                      'AgeuponOutcome_InYears', 'Gender',
                      'AgeCategory','Day_ofWeek','WeekEnd',
                      'Holiday','Name_missing','unicolor']]

# Target prediction 
X_target_1_upload = pd.get_dummies(X_target_1_upload)
probability = pipe.predict_proba(X_target_1_upload)

#file preparation
columns = pipe.classes_
results = pd.DataFrame(data = X_target_1_upload, columns = columns )
#each result has their corresponding probabilistic value
results["Adoption"] = probability[:,0]
results["Died"] = probability[:,1]
results["Euthanasia"] = probability[:,2]
results["Return_to_owner"] = probability[:,3]
results["Transfer"] = probability[:,4]
results.head(2)
results.to_csv("submition.csv", index= True, index_label='ID')
