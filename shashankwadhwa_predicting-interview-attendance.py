import warnings
warnings.filterwarnings('ignore')
import datetime
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

seed = 1
# Load the data
df = pd.read_csv('../input/Interview.csv')
df.shape
df.head()
df.tail()
# We can see that the last row consists of all nulls, so remove that row
df.drop(1233, inplace=True)
df.info()
df.drop(['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27'], axis=1, inplace=True)
df.describe()
def get_cleaned_date(date):
    """
    Rteurn datetime object from a string
    """
    date = date.strip()
    
    if '&' in date:
        date = date.split('&')[0].strip()
    
    cleaned_date = None
    
    # Since there are a lot of formats in the data, need to handle all the possible options
    date_formats = [
        '%d.%m.%Y', '%d.%m.%y', '%d.%m.%y', '%d-%m-%Y', '%d/%m/%y', '%d/%m/%Y', '%d %b %y', '%d-%b -%y',
        '%d – %b-%y', '%d -%b -%y'
    ]
    
    for date_format in date_formats:
        try:
            return datetime.datetime.strptime(date, date_format)
        except ValueError:
            pass
cleaned_interview_dates = df['Date of Interview'].apply(get_cleaned_date)
# Check the min and max dates to see if the dates have been converted properly

print(cleaned_interview_dates.min())
print(cleaned_interview_dates.max())
cleaned_interview_dates.sort_values(ascending=False)[:10]
df.iloc[438:445, :]['Date of Interview'] = '12.04.2016'
# Apply the function again, and check the min and max dates now
cleaned_interview_dates = df['Date of Interview'].apply(get_cleaned_date)
print(cleaned_interview_dates.min())
print(cleaned_interview_dates.max())
# Since the data looks fine now, replace the column with this new Series
df['Date of Interview'] = cleaned_interview_dates
df['Client name'].value_counts()
# Some clients are written in different names, so combine them
replace_dict = {
    'Standard Chartered Bank Chennai': 'Standard Chartered Bank',
    'Hewitt': 'Aon Hewitt',
    'Aon hewitt Gurgaon': 'Aon Hewitt'
}
df['Client name'].replace(replace_dict, inplace=True)
df['Client name'].value_counts()
def merge_categories(column_name, threshold, merged_name='Others'):
    """
    Will merge those categories which have count below a certain threshold
    """
    column_counts = df[column_name].value_counts()
    to_merge = column_counts[column_counts < threshold].index
    df.loc[df[column_name].isin(to_merge), column_name] = merged_name
merge_categories('Client name', 50)
df['Client name'].value_counts()
df['Industry'].value_counts()
merge_categories('Industry', 50, 'IT')
df['Industry'].value_counts()
df['Position to be closed'].value_counts()
replace_dict = {
    'Dot Net': 'Routine',
    'Trade Finance': 'Niche',
    'AML': 'Niche',
    'Selenium testing': 'Routine',
    'Production- Sterile': 'Routine'
}
df['Position to be closed'].replace(replace_dict, inplace=True)
df['Nature of Skillset'].value_counts()
nature_of_skillset = df['Nature of Skillset']

def clean_nature_of_skillset(x):
    x = x.lower()
    if 'java' in x:
        return 'java'
    elif 'oracle' in x:
        return 'oracle'
    elif 'testing' in x:
        return 'testing'
    elif 'aml' in x or 'kyc' in x or 'cdd' in x:
        return 'cdd'
    else:
        return x
cleaned_nature_of_skillset = nature_of_skillset.apply(clean_nature_of_skillset)
cleaned_nature_of_skillset.value_counts()
df['Nature of Skillset'] = cleaned_nature_of_skillset
merge_categories('Nature of Skillset', 50)
df['Nature of Skillset'].value_counts()
df['Interview Type'].value_counts()
replace_dict = {
    'Scheduled Walk In': 'Scheduled Walkin',
    'Sceduled walkin': 'Scheduled Walkin',
    'Walkin ': 'Walkin'
}
df['Interview Type'].replace(replace_dict, inplace=True)
location_columns = [
    'Candidate Current Location', 'Candidate Job Location', 'Interview Venue',
    'Candidate Native location'
]

def clean_location(s):
    s = s.translate(str.maketrans({key: None for key in string.punctuation})) # remove punctuations
    s = s.lower().strip()
    
    if 'delhi' in s or 'ncr' in s or 'gurgaon' in s or 'noida' in s:
        return 'ncr'
    else:
        return s
    
for col in location_columns:
    df[col] = df[col].apply(clean_location)
df['interview_venue_same_as_current_location'] = df['Candidate Current Location'] == df['Interview Venue']
df['interview_venue_same_as_native_location'] = df['Candidate Native location'] == df['Interview Venue']
merge_categories('Candidate Current Location', 35)
merge_categories('Interview Venue', 35)
merge_categories('Candidate Native location', 40)
# Rename the long question columns

columns_rename_dict = {
    'Have you obtained the necessary permission to start at the required time': 'question_obtained_necessary_permission',
    'Hope there will be no unscheduled meetings': 'question_no_unscheduled_meetings',
    'Can I Call you three hours before the interview and follow up on your attendance for the interview': 'question_can_follow_up',
    'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much': 'question_alternate_number',
    'Have you taken a printout of your updated resume. Have you read the JD and understood the same': 'question_taken_printout',
    'Are you clear with the venue details and the landmark.': 'question_clear_with_venue_details',
    'Has the call letter been shared': 'question_call_letter_shared'
}
df.rename(columns=columns_rename_dict, inplace=True)
question_columns = [col for col in df.columns if col.startswith('question')]

def clean_question_answers(a):
    yes_answers = ['yes']
    not_known_answers = ['cant say', 'yet to confirm', 'need to check', 'na', 'not sure']
    no_answers = [
        'no', 'no- i need to check', 'not yet', 'no i have only thi number', 'no dont', 'havent checked',
        'yet to check', 'no- will take it soon'
    ]
    
    if pd.isna(a):
        return 'not_known'
    
    a = a.lower().strip()
    if a in yes_answers:
        return 'yes'
    elif a in not_known_answers:
        return 'not_known'
    elif a in no_answers:
        return 'no'

for col in question_columns:
    df[col] = df[col].apply(clean_question_answers)
df['Expected Attendance'].value_counts()
def clean_expected_attendance(x):
    yes_list = ['yes', '11:00 am', '10.30 am']
    not_known_list = ['uncertain']
    no_list = ['no']
    
    if pd.isna(x):
        return 'not_known'
    
    x = x.lower().strip()
    if x in yes_list:
        return 'yes'
    elif x in not_known_list:
        return 'not_known'
    elif x in no_list:
        return 'no'

df['Expected Attendance'] = df['Expected Attendance'].apply(clean_expected_attendance)
df['Observed Attendance'] = df['Observed Attendance'].apply(lambda x: x.lower().strip())
df['Observed Attendance'].value_counts()
# Create new columns for interview date, month and day of week

df['interview_date'] = df['Date of Interview'].apply(lambda x: x.day)
df['interview_month'] = df['Date of Interview'].apply(lambda x: x.month)
df['interview_day'] = df['Date of Interview'].apply(lambda x: x.dayofweek)
sns.set(rc={'figure.figsize': (9, 6)})
sns.set_style('white')
df['Observed Attendance'].replace({'no': 0, 'yes': 1}, inplace=True)
def plot_categorical_column(column_name):
    f, (ax1, ax2) = plt.subplots(2, figsize=(9, 12))
    sns.countplot(x=column_name, data=df, ax=ax1)
    sns.pointplot(x=column_name, y='Observed Attendance', data=df, ax=ax2)
plot_categorical_column('interview_month')
plot_categorical_column('interview_day')
plot_categorical_column('Client name')
plot_categorical_column('Industry')
plot_categorical_column('Position to be closed')
plot_categorical_column('Nature of Skillset')
plot_categorical_column('Interview Type')
plot_categorical_column('Gender')
plot_categorical_column('Marital Status')
plot_categorical_column('Candidate Current Location')
sns.pointplot(x='Interview Venue', y='Observed Attendance', data=df)
sns.pointplot(x='Candidate Native location', y='Observed Attendance', data=df)
plot_categorical_column('interview_venue_same_as_current_location')
plot_categorical_column('interview_venue_same_as_native_location')
f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, figsize=(9, 16))
f.tight_layout()
axes = {
    'ax1': ax1,
    'ax2': ax2,
    'ax3': ax3,
    'ax4': ax4,
    'ax5': ax5,
    'ax6': ax6,
    'ax7': ax7
}
for ctr, col in enumerate(question_columns):
    ax_number = 'ax%s' % (ctr+1)
    sns.pointplot(x=col, y='Observed Attendance', data=df, ax=axes[ax_number])
plot_categorical_column('Expected Attendance')
sns.countplot(x='Observed Attendance', data=df)
unrequired_columns = [
    'Date of Interview', 'Location', 'Name(Cand ID)', 'Candidate Job Location',
]
df.drop(unrequired_columns, axis=1, inplace=True)
# Create dummy variables for the categorical columns

categorical_columns = [
    'Client name', 'Industry', 'Nature of Skillset', 'Interview Type', 'Candidate Current Location',
    'Interview Venue', 'Candidate Native location', 'Expected Attendance'
]
categorical_columns += question_columns
for categorical_column in categorical_columns:
    dummy_df = pd.get_dummies(df[categorical_column], prefix=categorical_column)
    df = pd.concat([df, dummy_df], axis=1)
    df.drop([categorical_column], axis=1, inplace=True)
# replace binary text values with numbers

binary_columns_replace_dict = {
    'Position to be closed': {
        'Routine': 0,
        'Niche': 1
    },
    'Gender': {
        'Female': 0,
        'Male': 1
    },
    'interview_venue_same_as_current_location': {
        False: 0,
        True: 1
    },
    'interview_venue_same_as_native_location': {
        False: 0,
        True: 1
    },
    'Marital Status': {
        'Single': 0,
        'Married': 1
    },
}

binary_columns = [
    'Position to be closed', 'Gender', 'interview_venue_same_as_current_location',
    'interview_venue_same_as_native_location', 'Marital Status'
]
for binary_col in binary_columns_replace_dict:
    df[binary_col].replace(binary_columns_replace_dict[binary_col], inplace=True)
# We can see in the pointplots that, observed attendance is low in months of March and April and high in May.
# This could be because the salary hike takes place in these months, so people tend not to take leaves from
# current company during this period and switch after taking increment.

df['month_march_or_april'] = df['interview_month'].apply(lambda x: 1 if x in [3, 4] else 0)
df['month_may'] = df['interview_month'].apply(lambda x: 1 if x == 5 else 0)
sns.pointplot(x='month_march_or_april', y='Observed Attendance', data=df)
sns.pointplot(x='month_may', y='Observed Attendance', data=df)
# People prefer going for interviews on weekends, as they don't have to take a leave from office.

df['is_weekend'] = df['interview_day'].apply(lambda x: 1 if x in [5, 6] else 0)
# Also, we can see from plots that attendance is quite low on fridays, so create a feature for that also
df['day_friday'] = df['interview_day'].apply(lambda x: 1 if x == 4 else 0)
sns.pointplot(x='is_weekend', y='Observed Attendance', data=df)
sns.pointplot(x='day_friday', y='Observed Attendance', data=df)
# There are 7 columns which are in a column form and all of them have 3 possible answers: yes, not_known and no.
# Instead of keeping all the 7 columns, we can create a single column by assigning weights to answers.

df['questions_score'] = (df['question_obtained_necessary_permission_not_known']*0.5) + \
(df['question_obtained_necessary_permission_yes']*1) + \
(df['question_no_unscheduled_meetings_not_known']*0.5) + \
(df['question_no_unscheduled_meetings_yes']*1) + \
(df['question_can_follow_up_not_known']*0.5) + \
(df['question_can_follow_up_yes']*1) + \
(df['question_alternate_number_not_known']*0.5) + \
(df['question_alternate_number_yes']*1) + \
(df['question_taken_printout_not_known']*0.5) + \
(df['question_taken_printout_yes']*1) + \
(df['question_clear_with_venue_details_not_known']*0.5) + \
(df['question_clear_with_venue_details_yes']*1) + \
(df['question_call_letter_shared_not_known']*0.5) + \
(df['question_call_letter_shared_yes']*1)
target_variable = 'Observed Attendance'
exclude_list = [
    'Observed Attendance',
    'question_obtained_necessary_permission_no',
    'question_obtained_necessary_permission_not_known',
    'question_obtained_necessary_permission_yes',
    'question_no_unscheduled_meetings_no',
    'question_no_unscheduled_meetings_not_known',
    'question_no_unscheduled_meetings_yes', 'question_can_follow_up_no',
    'question_can_follow_up_not_known', 'question_can_follow_up_yes',
    'question_alternate_number_no', 'question_alternate_number_not_known',
    'question_alternate_number_yes', 'question_taken_printout_no',
    'question_taken_printout_not_known', 'question_taken_printout_yes',
    'question_clear_with_venue_details_no',
    'question_clear_with_venue_details_not_known',
    'question_clear_with_venue_details_yes',
    'question_call_letter_shared_no',
    'question_call_letter_shared_not_known',
    'question_call_letter_shared_yes'
]
features = [x for x in df.columns if x not in exclude_list]
# Scale the data using min max scaling

scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df[features]))
scaled_df.columns = features
scaled_df[target_variable] = df[target_variable]
def evaluate_models(features):
    results = {}
    models = [
        ('lr', LogisticRegression(random_state=seed)),
        ('lda', LinearDiscriminantAnalysis()),
        ('svm', SVC(random_state=seed)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('nb', GaussianNB()),
        ('dt', DecisionTreeClassifier(random_state=seed)),
        ('rf', RandomForestClassifier(random_state=seed, n_estimators=100)),
        ('et', ExtraTreesClassifier(random_state=seed, n_estimators=100)),
    ]

    for model_name, model in models:
        # Since data size is not too big, we'll get 1/6th of data for testing
        cross_val_scores = cross_val_score(model, scaled_df[features], scaled_df[target_variable], cv=6)
        results[model_name] = (model, cross_val_scores.mean())
        
    sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
    for model_name, (model, accuracy) in sorted_results:
        print(model_name, accuracy)
        
    return results
results_using_all_features = evaluate_models(features)
reduced_features = [
    'questions_score', 'Position to be closed', 'month_may', 'month_march_or_april', 'day_friday',
    'Interview Type_Walkin', 'Expected Attendance_no', 'Expected Attendance_yes',
    'Interview Type_Scheduled ', 'Candidate Current Location_hyderabad'
]
results_using_reduced_features = evaluate_models(reduced_features)
