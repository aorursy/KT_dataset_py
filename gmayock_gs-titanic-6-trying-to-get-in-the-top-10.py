import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd 
import os
print(os.listdir("../input"))
['train.csv', 'gender_submission.csv', 'test.csv']
# Step 1 is to import both data sets
training_data = pd.read_csv("../input/train.csv")
testing_data = pd.read_csv("../input/test.csv")

# Step two is to create columns which I will add to the respective datasets, in order to know which row came from which dataset when I combine the datasets
training_column = pd.Series([1] * len(training_data))
testing_column = pd.Series([0] * len(testing_data))

# Now we append them by creating new columns in the original data. We use the same column name
training_data['is_training_data'] = training_column
testing_data['is_training_data'] = testing_column
# Now we can merge the datasets while retaining the key to split them later
combined_data = training_data.append(testing_data, ignore_index=True, sort=False)

# Encode gender (if == female, True)
combined_data['female'] = combined_data.Sex == 'female'

# Split out Title
title = []
for i in combined_data['Name']:
    period = i.find(".")
    comma = i.find(",")
    title_value = i[comma+2:period]
    title.append(title_value)
combined_data['title'] = title

# Replace the title values with an aliased dictionary
title_arr = pd.Series(title)
title_dict = {
    'Mr' : 'Mr', 
    'Mrs' : 'Mrs',
    'Miss' : 'Miss',
    'Master' : 'Master',
    'Don' : 'Formal',
    'Dona' : 'Formal',
    'Rev' : 'Religious',
    'Dr' : 'Academic',
    'Mme' : 'Mrs',
    'Ms' : 'Miss',
    'Major' : 'Formal',
    'Lady' : 'Formal',
    'Sir' : 'Formal',
    'Mlle' : 'Miss',
    'Col' : 'Formal',
    'Capt' : 'Formal',
    'the Countess' : 'Formal',
    'Jonkheer' : 'Formal',
}
cleaned_title = title_arr.map(title_dict)
combined_data['cleaned_title'] = cleaned_title

# Fill NaN of Age - first create groups to find better medians than just the overall median and fill NaN with the grouped medians
grouped = combined_data.groupby(['female','Pclass', 'cleaned_title']) 
combined_data['Age'] = grouped.Age.apply(lambda x: x.fillna(x.median()))

#add an age bin
age_bin_conditions = [
    combined_data['Age'] == 0,
    (combined_data['Age'] > 0) & (combined_data['Age'] <= 16),
    (combined_data['Age'] > 16) & (combined_data['Age'] <= 32),
    (combined_data['Age'] > 32) & (combined_data['Age'] <= 48),
    (combined_data['Age'] > 48) & (combined_data['Age'] <= 64),
    combined_data['Age'] > 64
]
age_bin_outputs = [0, 1, 2, 3, 4, 5]
combined_data['age_bin'] = np.select(age_bin_conditions, age_bin_outputs, 'Other').astype(int)

# Fill NaN of Embarked
combined_data['Embarked'] = combined_data['Embarked'].fillna("S") 

# Fill NaN of Fare, adding flag for boarded free, binning other fares
combined_data['Fare'] = combined_data['Fare'].fillna(combined_data['Fare'].mode()[0]) 
combined_data['boarded_free'] = combined_data['Fare'] == 0 
fare_bin_conditions = [
    combined_data['Fare'] == 0,
    (combined_data['Fare'] > 0) & (combined_data['Fare'] <= 7.9),
    (combined_data['Fare'] > 7.9) & (combined_data['Fare'] <= 14.4),
    (combined_data['Fare'] > 14.4) & (combined_data['Fare'] <= 31),
    combined_data['Fare'] > 31
]
fare_bin_outputs = [0, 1, 2, 3, 4]
combined_data['fare_bin'] = np.select(fare_bin_conditions, fare_bin_outputs, 'Other').astype(int)

# Fill NaN of Cabin with a U for unknown. Not sure cabin will help.
combined_data['Cabin'] = combined_data['Cabin'].fillna("U") 

# Counting how many people are riding on a ticket
from collections import Counter
tickets_count = pd.DataFrame([Counter(combined_data['Ticket']).keys(), Counter(combined_data['Ticket']).values()]).T
tickets_count.rename(columns={0:'Ticket', 1:'ticket_riders'}, inplace=True)
tickets_count['ticket_riders'] = tickets_count['ticket_riders'].astype(int)
combined_data = combined_data.merge(tickets_count, on='Ticket')

# Finding survival rate for people sharing a ticket
# Note that looking at the mean automatically drops NaNs, so we don't have an issue with using the combined data to calculate survival rate as opposed to just the training data
combined_data['ticket_rider_survival'] = combined_data['Survived'].mean()

# # Finding survival rate for people sharing a ticket (cont'd)
# This groups the data by ticket
# And then if the ticket group is greater than length 1 (aka more than 1 person rode on the ticket)
# it looks at the max and min of the _other_ rows in the group (by taking the max/min after dropping the current row)
# and if the max is 1, it replaces the default survival rate of .3838383 (the mean) with 1. This represents there being
# at least one known member of the ticket group which survived. If there is no known survivor on that ticket, but there  
# is a known fatality, the value is replaced with 0, representing there was at least one known death in that group. If
# neither, then the value remains the mean. 
for ticket_group, ticket_group_df in combined_data[['Survived', 'Ticket', 'PassengerId']].groupby(['Ticket']):
    if (len(ticket_group_df) != 1):
        for index, row in ticket_group_df.iterrows():
            smax = ticket_group_df.drop(index)['Survived'].max()
            smin = ticket_group_df.drop(index)['Survived'].min()
            if (smax == 1.0):
                combined_data.loc[combined_data['PassengerId'] == row['PassengerId'], 'ticket_rider_survival'] = 1
            elif (smin==0.0):
                combined_data.loc[combined_data['PassengerId'] == row['PassengerId'], 'ticket_rider_survival'] = 0

# Finding survival rate for people with a shared last name (same method as above basically)
combined_data['last_name'] = combined_data['Name'].apply(lambda x: str.split(x, ",")[0])  
combined_data['last_name_group_survival'] = combined_data['Survived'].mean()

for last_name_group, last_name_group_df in combined_data[['Survived', 'last_name', 'PassengerId']].groupby(['last_name']):
    if (len(last_name_group_df) != 1):
        for index, row in last_name_group_df.iterrows():
            smax = last_name_group_df.drop(index)['Survived'].max()
            smin = last_name_group_df.drop(index)['Survived'].min()
            if (smax == 1.0):
                combined_data.loc[combined_data['PassengerId'] == row['PassengerId'], 'last_name_group_survival'] = 1
            elif (smin==0.0):
                combined_data.loc[combined_data['PassengerId'] == row['PassengerId'], 'last_name_group_survival'] = 0

# Finding survival rate for people with a shared last name _and_ fare
combined_data['last_name_fare_group_survival'] = combined_data['Survived'].mean()

for last_name_fare_group, last_name_fare_group_df in combined_data[['Survived', 'last_name', 'Fare', 'PassengerId']].groupby(['last_name', 'Fare']):
    if (len(last_name_fare_group_df) != 1):
        for index, row in last_name_fare_group_df.iterrows():
            smax = last_name_fare_group_df.drop(index)['Survived'].max()
            smin = last_name_fare_group_df.drop(index)['Survived'].min()
            if (smax == 1.0):
                combined_data.loc[combined_data['PassengerId'] == row['PassengerId'], 'last_name_fare_group_survival'] = 1
            elif (smin==0.0):
                combined_data.loc[combined_data['PassengerId'] == row['PassengerId'], 'last_name_fare_group_survival'] = 0
                
# Finding cabin group
cabin_group = []
for i in combined_data['Cabin']:
    cabin_group.append(i[0])
combined_data['cabin_group'] = cabin_group

# Adding a family_size feature as it may have an inverse relationship to either of its parts
combined_data['family_size'] = combined_data.Parch + combined_data.SibSp + 1

# Mapping ports to passenger pickup order
port = {
    'S' : 1,
    'C' : 2,
    'Q' : 3
}
combined_data['pickup_order'] = combined_data['Embarked'].map(port)

# Encode childhood
combined_data['child'] = combined_data.Age < 16

# One-Hot Encoding the titles
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['cleaned_title'], prefix="C_T")], axis = 1)

# One-Hot Encoding the Pclass
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['Pclass'], prefix="PClass")], axis = 1)

# One-Hot Encoding the  cabin group
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['cabin_group'], prefix="C_G")], axis = 1)

# One-Hot Encoding the ports
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['Embarked'], prefix="Embarked")], axis = 1)
new_train_data=combined_data.loc[combined_data['is_training_data']==1]
new_test_data=combined_data.loc[combined_data['is_training_data']==0]
# here is the expanded model set and metric tools
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

k_fold = KFold(n_splits = 10, shuffle=True, random_state=0) 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

# Here are the features
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'female', 'child', 'C_T_Master', 'C_T_Miss', 'C_T_Mr', 'C_T_Mrs',
            'C_T_Formal','C_T_Academic', 'C_T_Religious','C_G_A', 'C_G_B', 'C_G_C', 'C_G_D', 'C_G_E', 'C_G_F', 'C_G_G', 
            'C_G_T', 'C_G_U', 'family_size', 'ticket_riders', 'ticket_rider_survival', 'last_name_group_survival', 'last_name_fare_group_survival']
target = 'Survived'
cvs_train_data = new_train_data[features]
cvs_test_data = new_test_data[features]
cvs_target = new_train_data['Survived']
cvs_train_data.shape
# Define the classifiers I will use
classifiers = [
    RandomForestClassifier(n_estimators=10, random_state=0),
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=6,
            min_weight_fraction_leaf=0.0, n_estimators=35, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False),
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=9, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False),
    DecisionTreeClassifier(random_state=0),
    LogisticRegression(solver='liblinear'),
    KNeighborsClassifier(n_neighbors=15),
    SVC(gamma='auto'),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier(),
    XGBClassifier(),
    GaussianNB(),
    LinearSVC()]
# Fit and use cross_val_score and k_fold to score accuracy
clf_scores = [['Score', 'Classifier']]
i = 0
for clf in classifiers:
    name = clf.__class__.__name__
    acc_score = round(np.mean(cross_val_score(clf, cvs_train_data, cvs_target, cv=k_fold, n_jobs=1, scoring='accuracy'))*100,2)
    print(acc_score, "%", name);
    clf_scores.append([acc_score, name])
    clf.fit(cvs_train_data, cvs_target)
    prediction = clf.predict(cvs_test_data)
    submission = pd.DataFrame({
        "PassengerId" : new_test_data['PassengerId'],
        "Survived" : prediction.astype(int)
    })
    submission_name = 'run1_{}_{}.csv'.format(i, name)
    submission.to_csv(submission_name, index=False)
    i += 1
clf_scores_temp = clf_scores.copy()
clf_scores_1 = pd.DataFrame(clf_scores_temp, columns=clf_scores_temp.pop(0))
clf_scores_1
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
scaled_train_data = std_scaler.fit_transform(cvs_train_data)
scaled_test_data = std_scaler.transform(cvs_test_data)
scaled_target = cvs_target
clf_scores = [['Score', 'Classifier']]
i = 0
for clf in classifiers:
    name = clf.__class__.__name__
    acc_score = round(np.mean(cross_val_score(clf, scaled_train_data, scaled_target, cv=k_fold, n_jobs=1, scoring='accuracy'))*100,2)
    print(acc_score, "%", name);
    clf_scores.append([acc_score, name])
    clf.fit(scaled_train_data, scaled_target)
    prediction = clf.predict(scaled_test_data)
    submission = pd.DataFrame({
        "PassengerId" : new_test_data['PassengerId'],
        "Survived" : prediction.astype(int)
    })
    submission_name = 'run2_{}_{}.csv'.format(i, name)
    submission.to_csv(submission_name, index=False)
    i += 1

clf_scores_temp = clf_scores.copy()
clf_scores_2 = pd.DataFrame(clf_scores_temp, columns=clf_scores_temp.pop(0))
clf_scores_2
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler()
mms_train_data = mm_scaler.fit_transform(cvs_train_data)
mms_test_data = mm_scaler.transform(cvs_test_data)
mms_target = cvs_target
clf_scores = [['Score', 'Classifier']]
i = 0
for clf in classifiers:
    name = clf.__class__.__name__
    acc_score = round(np.mean(cross_val_score(clf, mms_train_data, mms_target, cv=k_fold, n_jobs=1, scoring='accuracy'))*100,2)
    print(acc_score, "%", name);
    clf_scores.append([acc_score, name])
    clf.fit(mms_train_data, mms_target)
    prediction = clf.predict(mms_test_data)
    submission = pd.DataFrame({
        "PassengerId" : new_test_data['PassengerId'],
        "Survived" : prediction.astype(int)
    })
    submission_name = 'run3_{}_{}.csv'.format(i, name)
    submission.to_csv(submission_name, index=False)
    i += 1

clf_scores_temp = clf_scores.copy()
clf_scores_3 = pd.DataFrame(clf_scores_temp, columns=clf_scores_temp.pop(0))
clf_scores_3
from sklearn.feature_selection import chi2, SelectKBest
cvs_train_data.shape
fit_test = SelectKBest(score_func=chi2, k='all')
fit_test.fit(
    mms_train_data, 
    mms_target)
np.set_printoptions(precision=3)

fit_test_scores_1 = pd.DataFrame(fit_test.scores_)
fit_test_scores_1['feature_name'] = cvs_train_data.columns
fit_test_scores_1.columns = ['chi_squared', 'feature_name']
fit_test_scores_1 = fit_test_scores_1.sort_values('chi_squared', ascending=False)
fit_test_scores_1
from scipy.stats import chi2
chi2.ppf(0.95, len(fit_test_scores_1)-1)
# Here are the features
features_2 = ['female', 'C_T_Mr', 'C_T_Mrs', 'C_T_Miss', 'Pclass',
       'ticket_rider_survival', 'C_G_B', 'C_G_U', 'C_G_D', 'C_G_E',
       'child', 'last_name_fare_group_survival',
       'last_name_group_survival', 'C_G_C', 'fare_bin']
target = 'Survived'
lcs_train_data = new_train_data[features_2]
lcs_test_data = new_test_data[features_2]
lcs_target = new_train_data['Survived']
lcs_train_data.shape, lcs_test_data.shape, lcs_target.shape
mm_scaler = MinMaxScaler()
mms_lcs_train_data = mm_scaler.fit_transform(lcs_train_data)
mms_lcs_test_data = mm_scaler.transform(lcs_test_data)
mms_lcs_target = new_train_data['Survived']
mms_lcs_train_data.shape, mms_lcs_test_data.shape, mms_lcs_target.shape
from sklearn.feature_selection import chi2, SelectKBest
fit_test2 = SelectKBest(score_func=chi2, k='all')
fit_test2.fit(
    mms_lcs_train_data, 
    mms_lcs_target)
np.set_printoptions(precision=3)

fit_test_scores_2 = pd.DataFrame(fit_test2.scores_)
fit_test_scores_2['feature_name'] = lcs_train_data.columns
fit_test_scores_2.columns = ['chi_squared', 'feature_name']
fit_test_scores_2 = fit_test_scores_2.sort_values('chi_squared', ascending=False)
fit_test_scores_2
from scipy.stats import chi2
chi2.ppf(0.95, len(fit_test_scores_2)-1)


from sklearn.preprocessing import StandardScaler

ttrd = new_train_data[features_2]
ttsd = new_test_data[features_2]

std_scaler = StandardScaler()
ssca_train_data = std_scaler.fit_transform(ttrd)
ssca_test_data = std_scaler.transform(ttsd)

f2_target = new_train_data['Survived']
clf_scores = [['Score', 'Classifier']]
i = 0
for clf in classifiers:
    name = clf.__class__.__name__
    acc_score = round(np.mean(cross_val_score(clf, ssca_train_data, f2_target, cv=k_fold, n_jobs=1, scoring='accuracy'))*100,2)
    print(acc_score, "%", name);
    clf_scores.append([acc_score, name])
    clf.fit(ssca_train_data, f2_target)
    prediction = clf.predict(ssca_test_data)
    submission = pd.DataFrame({
        "PassengerId" : new_test_data['PassengerId'],
        "Survived" : prediction.astype(int)
    })
    submission_name = 'run4_{}_{}.csv'.format(i, name)
    submission.to_csv(submission_name, index=False)
    i += 1

clf_scores_temp = clf_scores.copy()
clf_scores_4 = pd.DataFrame(clf_scores_temp, columns=clf_scores_temp.pop(0))
clf_scores_4
clf_score_temp = clf_scores_1.copy()
clf_score_cons = pd.DataFrame(clf_score_temp['Classifier'])
clf_score_cons['run4'] = clf_scores_4['Score']
clf_score_cons['run3'] = clf_scores_3['Score']
clf_score_cons['run2'] = clf_scores_2['Score']
clf_score_cons['run1'] = clf_scores_1['Score']
clf_score_cons
import matplotlib.pyplot as plt
import seaborn as sns

cm = sns.light_palette('green', as_cmap=True)

s = clf_score_cons.style.background_gradient(cmap=cm, low=0, high=1, axis=1)
s
# since I started scaling the variables, KNN seems to be functioning real nice. Let's see if GridSearchCV can help us there.
from sklearn.model_selection import GridSearchCV
n_neighbors = [14, 16, 17, 18, 19, 20, 22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(10,30,1))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 
                cv=k_fold, scoring = "accuracy")
gd.fit(mms_train_data, mms_target)
print(gd.best_score_)
print(gd.best_estimator_)
gd.best_estimator_.fit(mms_train_data, mms_target)
prediction = gd.best_estimator_.predict(mms_test_data)
submission = pd.DataFrame({
    "PassengerId" : new_test_data['PassengerId'],
    "Survived" : prediction.astype(int)
})
submission_name = 'run5_knn.csv'
submission.to_csv(submission_name, index=False)
# New record! .80382! 
forest_params = dict(     
    max_depth = [n for n in range(5, 20)],     
    min_samples_split = [n for n in range(3, 13)], 
    min_samples_leaf = [n for n in range(1, 7)],     
    n_estimators = [n for n in range(10, 60, 2)],
)

# # Going to reduce the number of attempts for a while to see if this works better
# forest_params = dict(     
#     max_depth = [n for n in range(8, 12)],     
#     min_samples_split = [n for n in range(5, 9)], 
#     min_samples_leaf = [n for n in range(2, 4)],     
#     n_estimators = [n for n in range(10, 60, 25)],
# )


forest_gscv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=forest_params, cv=k_fold) 
forest_gscv.fit(ssca_train_data, cvs_target)


print("Best score: {}".format(forest_gscv.best_score_))
print("Optimal params: {}".format(forest_gscv.best_estimator_))


forest_gscv.best_estimator_.fit(ssca_train_data, cvs_target)
prediction = forest_gscv.best_estimator_.predict(ssca_test_data)
submission = pd.DataFrame({
    "PassengerId" : new_test_data['PassengerId'],
    "Survived" : prediction.astype(int)
})
submission_name = 'run6_forest_gscv.csv'
submission.to_csv(submission_name, index=False)
