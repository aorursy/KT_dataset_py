import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import sklearn
import matplotlib.pyplot as plt
%matplotlib inline
pd.options.display.max_columns = 200
data = pd.read_csv('../input/Interview.csv')
data.shape
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
# Lets make the columns names easier to work with, by adding '_' where there's spaces. That way we invoke them as an object member
cols = data.columns
new_cols = []
for item in cols:
    new = item.replace(' ', '_')
    new_cols.append(new)
print(new_cols)
data.columns = new_cols
data = data.drop('Date_of_Interview', axis=1).drop('Name(Cand_ID)', axis=1)
# Now lets see where we have missing values
data.isnull().sum()
# But we also have some Nan values that are strings - 'Na', 'NA'
data.replace(to_replace=['Na', 'NA'], value=np.nan, inplace=True)
data.isnull().sum()
data[data['Are_you_clear_with_the_venue_details_and_the_landmark.'].isnull()]
# lets check what will happen if we drop the rows with Nan for the column that has the most Nan.
data[~data['Are_you_clear_with_the_venue_details_and_the_landmark.'].isnull()].isnull().sum()
# Well that was easy....
data = data[~data['Are_you_clear_with_the_venue_details_and_the_landmark.'].isnull()]
# Lets drop the last row where we have Nan
data = data[~data['Hope_there_will_be_no_unscheduled_meetings'].isnull()]
data.isnull().sum()
# Now lets clean out the data a little bit.....
data.nunique()
# Our label column Observed_Attendance has too many values - 'Yes' 'yes' 'YES etc...
data.Observed_Attendance.unique()
data.Observed_Attendance = data.Observed_Attendance.apply(lambda a: 1 if a in ['Yes', 'yes', 'yes '] else 0)
print(data.Observed_Attendance.unique())
# Same issue with Expected Attendance
data.Expected_Attendance.unique()
data.Expected_Attendance = data.Expected_Attendance.apply(lambda a: 'Yes' if a in ['11:00 AM', '10.30 Am', 'yes'] else a)
data.Expected_Attendance.unique()
data.Has_the_call_letter_been_shared.unique()
data.Has_the_call_letter_been_shared = data.Has_the_call_letter_been_shared.apply(lambda a: 'Yes' if a in ['Yes', 'yes'] else ('No' if a in ['No', 'no'] else 'Maybe'))
data.nunique()
data['Are_you_clear_with_the_venue_details_and_the_landmark.'].unique()
data['Are_you_clear_with_the_venue_details_and_the_landmark.'] = data['Are_you_clear_with_the_venue_details_and_the_landmark.'].apply(lambda a: 'Yes' if a in ['Yes', 'yes'] else 'No')
data['Have_you_taken_a_printout_of_your_updated_resume._Have_you_read_the_JD_and_understood_the_same'].unique()
data['Have_you_taken_a_printout_of_your_updated_resume._Have_you_read_the_JD_and_understood_the_same'] = data['Have_you_taken_a_printout_of_your_updated_resume._Have_you_read_the_JD_and_understood_the_same'].apply(lambda a: 'Yes' if a in ['Yes', 'yes'] else ('Not Yet' if a in ['Not Yet', 'Not yet'] else 'No'))
print(data['Have_you_taken_a_printout_of_your_updated_resume._Have_you_read_the_JD_and_understood_the_same'].unique())
data['Can_I_have_an_alternative_number/_desk_number._I_assure_you_that_I_will_not_trouble_you_too_much'].unique()
data['Can_I_have_an_alternative_number/_desk_number._I_assure_you_that_I_will_not_trouble_you_too_much'] = data['Can_I_have_an_alternative_number/_desk_number._I_assure_you_that_I_will_not_trouble_you_too_much'].apply(lambda a: 'Yes' if a in ['Yes', 'yes'] else 'No')
data['Can_I_have_an_alternative_number/_desk_number._I_assure_you_that_I_will_not_trouble_you_too_much'].unique()
data['Can_I_Call_you_three_hours_before_the_interview_and_follow_up_on_your_attendance_for_the_interview'].unique()
data['Can_I_Call_you_three_hours_before_the_interview_and_follow_up_on_your_attendance_for_the_interview'] = data['Can_I_Call_you_three_hours_before_the_interview_and_follow_up_on_your_attendance_for_the_interview'].apply(lambda a: 'Yes' if a in ['Yes', 'yes'] else 'No')
print(data['Can_I_Call_you_three_hours_before_the_interview_and_follow_up_on_your_attendance_for_the_interview'].unique())
data['Hope_there_will_be_no_unscheduled_meetings'].unique()
data['Hope_there_will_be_no_unscheduled_meetings'] = data['Hope_there_will_be_no_unscheduled_meetings'].apply(lambda a: 'Yes' if a in ['Yes', 'yes'] else ('No' if a == 'No' else 'Maybe'))
data['Hope_there_will_be_no_unscheduled_meetings'].unique()
data['Have_you_obtained_the_necessary_permission_to_start_at_the_required_time'].unique()
data['Have_you_obtained_the_necessary_permission_to_start_at_the_required_time'] = data['Have_you_obtained_the_necessary_permission_to_start_at_the_required_time'].apply(lambda a: 'Yes' if a in ['Yes', 'yes'] else 'No')
data['Have_you_obtained_the_necessary_permission_to_start_at_the_required_time'].unique()
data.nunique()
for col in data:
    print('%s: ' % col, data[col].unique())
# Editing Industry
data.Industry.unique()
data.Industry = data.Industry.apply(lambda a: 'IT' if a in ['IT Services', 'IT Products and Services', 'IT'] else a)
data.Industry.unique()
# Editing Location
data.Location.unique()
data.Location = data.Location.apply(lambda a: 'Gurgaon' if a in ['Gurgaon', 'Gurgaonr'] else ('Cochin' if a == '- Cochin- ' else a))
data.Location.unique()
data.Location = data.Location.apply(lambda a: a.upper())
data.Location.unique()
data.Location = data.Location.apply(lambda a: 'CHENNAI' if a in ['CHENNAI', 'CHENNAI '] else a)
data.Location.unique()
data.Nature_of_Skillset = data.Nature_of_Skillset.apply(lambda a: a.upper())
data.Nature_of_Skillset = data.Nature_of_Skillset.apply(lambda a: 'ANALYTICAL_R&D' if a in ['ANALYTICAL R & D', 'ANALYTICAL R&D'] else ('TECHNICAL_LEAD' if a in ['TECHNICAL LEAD', 'TL'] else a))
data.Nature_of_Skillset = data.Nature_of_Skillset.apply(lambda a: 'LENDING_LIABILITY' if a in ['LENDING AND LIABILITIES', 'LENDING & LIABILITY', 'L & L', 'LENDING&LIABLITIES'] else a)
data[data.Nature_of_Skillset.isin(['10.00 AM', '9.00 AM', '11.30 AM', '9.30 AM'])]
# Lets drop these 4 lines....
data = data[~data.Nature_of_Skillset.isin(['10.00 AM', '9.00 AM', '11.30 AM', '9.30 AM'])]
# Grouping all the JAVA related entries
data.Nature_of_Skillset = data.Nature_of_Skillset.apply(lambda a: 'JAVA' if 'JAVA' in a else a)
# Same for SCCM
data.Nature_of_Skillset = data.Nature_of_Skillset.apply(lambda a: 'SCCM' if 'SCCM' in a else a)
data.Nature_of_Skillset = data.Nature_of_Skillset.apply(lambda a: 'AML/KYC/CDD' if a in ['AML/KYC/CDD', 'CDD KYC'] else a)
data.Nature_of_Skillset = data.Nature_of_Skillset.apply(lambda a: 'TECH_LEAD' if a in ['TECH LEAD-MEDNET', 'TECH LEAD- MEDNET', 'TECHNICAL_LEAD'] else a)
data.Nature_of_Skillset = data.Nature_of_Skillset.apply(lambda a: 'COTS' if 'COTS' in a else a) 
# Enough with that....
data.nunique()
data.Interview_Type.unique()
# I assumed that a scheudle walkin is a like a scheduled interview....cause what ta hell is a scheduled walkin?
data.Interview_Type = data.Interview_Type.apply(lambda a: 'Scheduled' if 'Scheduled' in a else 'Walkin')
data.Candidate_Current_Location.unique()
data.Candidate_Current_Location = data.Candidate_Current_Location.apply(lambda a: (a.upper()).strip())
data.Candidate_Current_Location.unique()
print(data.Candidate_Job_Location.unique())
data.Candidate_Job_Location = data.Candidate_Job_Location.apply(lambda a: (a.upper()).strip())
print(data.Candidate_Job_Location.unique())
print(data.Interview_Venue.unique())
data.Interview_Venue = data.Interview_Venue.apply(lambda a: (a.upper()).strip())
print(data.Interview_Venue.unique())
data.Candidate_Native_location = data.Candidate_Native_location.apply(lambda a: (a.upper()).strip())
data.Candidate_Native_location.unique()
# A little bit of feature engineering
data['is_interview_in_native_town'] = data.apply(lambda a: True if a.Candidate_Native_location == a.Interview_Venue else False, axis=1)
data['is_interview_in_current_town'] = data.apply(lambda a: True if a.Candidate_Current_Location == a.Interview_Venue else False, axis=1)
data['is_interview_in_current_job_town'] = data.apply(lambda a: True if a.Candidate_Job_Location == a.Interview_Venue else False, axis=1)
data['is_job_in_native_town'] = data.apply(lambda a: True if a.Location == a.Candidate_Native_location else False, axis=1)
data['is_job_in_current_twon'] = data.apply(lambda a: True if a.Candidate_Current_Location == a.Location else False, axis=1)
data['is_job_in_current_job_town'] = data.apply(lambda a: True if a.Candidate_Job_Location == a.Location else False, axis=1)
data.shape
features, labels = data.loc[:, data.columns != 'Observed_Attendance'], data.loc[:, 'Observed_Attendance']
print(features.shape)
print(labels.shape)
# We're all done with cleaning out the data, now lets one hot our categorical columns (which is all of them)
features_one_hot = pd.get_dummies(data=features)
features_one_hot.shape
features_one_hot = features_one_hot.applymap(lambda a: 1 if a == True else (0 if a == False else 0))
from sklearn.model_selection import train_test_split
# Lets create our train, dev, and test sets
x_train, x_dev, y_train, y_dev = train_test_split(features_one_hot, labels, test_size=80, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=80, random_state=42)
# Im gonna go with an AdaBoost Classifier using DecisionTree stumps (depth=1)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix
model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=110, learning_rate=0.6)
model.fit(X=x_train, y=y_train)
cross_validate(model, features_one_hot, labels, cv=5)
test_preds = model.predict(x_test)
confusion_matrix(test_preds, y_test)
# Lets try Bagging our model
from sklearn.ensemble import BaggingClassifier
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=110, learning_rate=0.6)
bagger = BaggingClassifier(base_estimator=clf, verbose=1, n_estimators=50)
bagger.fit(X=x_train, y=y_train)
bagger.score(X=x_test, y=y_test)
preds = bagger.predict(x_test)
confusion_matrix(preds, y_test)