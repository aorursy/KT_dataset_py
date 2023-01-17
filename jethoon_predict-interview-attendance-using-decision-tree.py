import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/Interview.csv')
#df.info()
print("Dataset Length : ", len(df))
print("Dataset Shape : ", df.shape)
print(df.isnull().sum())
print("Dataset : \n", df.head())
print("Dataset tail \n", df.tail())
# Drop last row as it has all NaN
#df = df[:-1] 
df = df.drop(df.tail(1).index)
# drop irrelevant columns 
df.drop(['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Name(Cand ID)'], 
                  axis=1, inplace=True)
print("Dataset Length : ", len(df))
print("Dataset Shape : ", df.shape)
# change column name to shorter names
df=df.rename(columns={'Date of Interview':'weekday',
                          'Client name':'client',
                      'Gender':'gender',
                         'Industry':'industry',
                          'Location':'location',
                          'Position to be closed':'job_skills_required',
                          'Nature of Skillset':'candidate_skills',
                          'Interview Type':'interview_type',
                          'Candidate Current Location':'candidate_loc',
                          'Candidate Job Location':'job_location',
                          'Interview Venue':'venue',
                          'Candidate Native location':'native_loc',
                          'Have you obtained the necessary permission to start at the required time':'permission',
                          'Hope there will be no unscheduled meetings':'hope',
                          'Can I Call you three hours before the interview and follow up on your attendance for the interview':'3_hour_call',
                          'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much':'alt_number',
                          'Have you taken a printout of your updated resume. Have you read the JD and understood the same':'resume_printout',
                          'Has the call letter been shared':'share_letter',
                          'Are you clear with the venue details and the landmark.':'knows_location',
                          'Expected Attendance':'expected_attendance',
                          'Observed Attendance':'observed_attendance'
                         })
print(df.columns)
print(df.isnull().sum())
%matplotlib inline
sns.countplot(df['hope'])
sns.countplot(df['expected_attendance'])
df.describe()
#lets automate the plots
fig,axes = plt.subplots(len(df.columns),figsize=(10,5*len(df.columns)))
plt.tight_layout()
for idx,col in enumerate(df.columns):
    sns.countplot(df[col],ax=axes[idx])

df.gender.unique()
df.job_skills_required.unique()
# group them into skilled and routine(production-Sterile and Routine)
def rename_job_skill(data):
    if data == 'Routine' or data == 'Production- Sterile':
        return 'routine'
    else:
        return 'skilled'
    
df.job_skills_required = df.job_skills_required.apply(rename_job_skill)
print("Total count",df.job_skills_required.count())
print("Unique categories:\n",df.job_skills_required.value_counts())
df.candidate_skills.unique()
df.share_letter.unique()
df.drop(['candidate_skills','share_letter'], 
                  axis=1, inplace=True)
print("Dataset Length : ", len(df))
print("Dataset Shape : ", df.shape)
print("list of columns",df.columns)
df.interview_type.unique()
#We will re-classify them into walkin, scheduled and scheduled_walkin
def rename_interview_type(data):
    interview_type=data.rstrip().lower()
    if interview_type=='walkin':
        return 'walkin'
    elif interview_type =='scheduled':
        return 'scheduled'
    else:
        return 'scheduled_walkin'
df.interview_type=df.interview_type.apply(rename_interview_type)
print("Total count",df.interview_type.count())
print("Unique categories:\n",df.interview_type.value_counts())
# Industry
df.industry.unique()
# Group IT in one category
def rename_industry(data):
    if 'IT' in data:
        return 'IT'
    else:
        return data
df.industry=df.industry.apply(rename_industry)
print("Total count",df.industry.count())
print("Unique categories:\n",df.industry.value_counts())
df[(pd.isnull(df.knows_location)) | (df.knows_location == 'na') | (df.knows_location == 'Na')]
def rename_know_location(data):
    if pd.isnull(data) or data =='na' or data =='Na':
        return np.nan
    knows=data.rstrip().lower()
    if knows=='yes':
        return 'yes'
    else:
        return 'no'
df.knows_location=df.knows_location.apply(rename_know_location)
print("Total count",df.knows_location.count())
print("Unique categories:\n",df.knows_location.value_counts())
#Let's replace missing values with the mode
most_freq_loc=df.knows_location.mode().iloc[0]
# set missing value with the mode
df.knows_location=df.knows_location.apply(lambda x:most_freq_loc if pd.isnull(x) else x)
print("Total count",df.knows_location.count())
print("Unique categories:\n",df.knows_location.value_counts())
print("Unique Expected Attendance",df.expected_attendance.unique())
print("Unique Observed Attendance", df.observed_attendance.unique())
def rename_expected_attendance(data):
    if pd.isnull(data):
        return np.nan
    attendance=data.rstrip().lower()
    if attendance =='no' or attendance == 'uncertain':
        return attendance
    else:
        return 'yes'
    
df.expected_attendance=df.expected_attendance.apply(rename_expected_attendance)
print("Total count",df.expected_attendance.count())
print("Unique categories:\n",df.expected_attendance.value_counts())
#expected_attendance is missing some value , let's replace them with mode
most_freq_loc=df.expected_attendance.mode().iloc[0]
# set missing value with the mode
df.expected_attendance=df.expected_attendance.apply(lambda x:most_freq_loc if pd.isnull(x) else x)
print("Total count",df.expected_attendance.count())
print("Unique categories:\n",df.expected_attendance.value_counts())
# We should not care if candidate is not expected to come so lets drop those rows
df=df[df.expected_attendance !='no']
print("Dataset Length : ", len(df))
print("Dataset Shape : ", df.shape)

def rename_hope(data):
    if pd.isnull(data):
        return np.nan
    value=data.rstrip().lower()
    if value == 'unsure' or value == 'not sure' or value == 'cant say' or value == 'nan' or value == 'na':
        return 'no'
    else:
        return 'yes'
df.hope = df.hope.fillna('unsure')
df.hope=df.hope.apply(rename_hope)

print("Total count",df.hope.count())
print("Unique categories:\n",df.hope.value_counts())
def rename_permission(data):
    if pd.isnull(data):
        return np.nan
    value=data.rstrip().lower()
    if value == 'not yet' or value == 'na' or value =='no':
        return 'no'
    elif value == 'yet to confirm' or value == 'yes':
        return 'yes'
    else:
        return data
    
df.permission = df.permission.fillna('no')
df.permission=df.permission.apply(rename_permission)

print("Total count",df.permission.count())
print("Unique categories:\n",df.permission.value_counts())
def rename_observed_attendance(data):
    if pd.isnull(data):
        return np.nan
    attendance=data.rstrip().lower()
    if attendance =='no':
        return attendance
    else:
        return 'yes'
    
df.observed_attendance=df.observed_attendance.apply(rename_observed_attendance)
print("Total count",df.observed_attendance.count())
print("Unique categories:\n",df.observed_attendance.value_counts())
from datetime import datetime

#function to check if a character is between a-z or 1-9
def is_myalnum(char):
    return (ord(char.lower()) in range(ord('a'), ord('z')+1)
            or char.isdigit())

#extracts the day, month and year from the data
def parse_string(date_str):

    date = [] #contain [day, month, year]
    val = ""
    
    counter = 0
    str_len = len(date_str)
    
    while (len(date) < 3):
        char = date_str[counter]
        #print(counter, str_len, char, is_myalnum(char))
        
        if is_myalnum(char):
            val += char
        
        elif not is_myalnum(char) and not val == "":
            date.append(val)
            val = ""

        if counter == (str_len - 1) and not val == "":
            date.append(val)
            val = ""
        
        counter += 1
    return date
    
#converts the date into a weekday
def convert_date(data):

    [day, month, year]= parse_string(data)

    year = int(year)
    day = int(day)
    if month.isdigit(): 
        date = datetime(year, int(month), day)
                        
    else:
        month = int(datetime.strptime(month, "%b").strftime("%m"))
        date = datetime(year, month, day)

    return date.strftime('%A')
        
df.weekday = df.weekday.apply(convert_date)
print("Total count",df['weekday'].count())
print("Unique categories:\n",df['weekday'].value_counts())
df['3_hour_call'] = df['3_hour_call'].fillna('no')
for i,v in enumerate(df['3_hour_call']):
    value = v.lower()
    if value == 'no dont' or value == 'na':
        df['3_hour_call'].iloc[i] = 'no'
    else:
         df['3_hour_call'].iloc[i] = value

print("Total count",df['3_hour_call'].count())
print("Unique categories:\n",df['3_hour_call'].value_counts())
def rename_altnum(data):
    if pd.isnull(data):
        return np.nan
    value=data.rstrip().lower()
    if value == 'no i have only thi number' or value == 'na':
        return 'no'
    else:
        return 'yes'
df.alt_number = df.alt_number.fillna('no')
df.alt_number=df.alt_number.apply(rename_altnum)

print("Total count",df.alt_number.count())
print("Unique categories:\n",df.alt_number.value_counts())
def rename_print(data):
    if pd.isnull(data):
        return np.nan
    value=data.rstrip().lower()
    if value == 'no- will take it soon' or value == 'not yet' or value == 'na' or value=='no':
        return 'no'
    elif value == 'yes':
        return 'yes'

df.resume_printout = df.resume_printout.fillna('NA')
df.resume_printout=df.resume_printout.apply(rename_print)

print("Total count",df.resume_printout.count())
print("Unique categories:\n",df.resume_printout.value_counts())

print(df.columns)
for col in df.columns:
    print("Unique ",col,"\n",df[col].unique())
df.describe()
#lets plot the clean data(subset)
fig,axes = plt.subplots(len(df.columns),figsize=(10,5*len(df.columns)))
plt.tight_layout()
for idx,col in enumerate(df.columns):
    sns.countplot(df[col],ax=axes[idx])
# Lets drop native location and 'Marital Status' columns as well
df.drop(['native_loc','Marital Status'], 
                  axis=1, inplace=True)
df.head()

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
df.head()
encoder = LabelEncoder()
encoder.fit(df['observed_attendance'])
df.observed_attendance = encoder.fit_transform(df['observed_attendance'])
for col in df.drop(['observed_attendance'],axis=1).columns :
    encoder.fit(df[col])
    df[col] = encoder.transform(df[col])
df.head()
y=df.pop("observed_attendance")
# Split Data into training and test set
X_train1, X_test, y_train1, y_test = train_test_split( df, y, test_size = 0.3, random_state = 100)

# Further split the Training set into training and validation set
X_train, X_val, y_train, y_val = train_test_split( X_train1, y_train1, test_size = 0.3, random_state = 100)


print("Test set",df.shape, y.shape)
print("Training set",X_train.shape, y_train.shape)
print("Val set", X_val.shape, y_val.shape)
print("Test set",X_test.shape, y_test.shape)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
max_depth=None, min_samples_leaf=4)
clf_gini=clf_gini.fit(X_train1,y_train1)
clf_gini
dotfile=open("dtmax_depth8.dot",'w')
tree.export_graphviz(clf_gini,out_file=dotfile,feature_names=df.columns)
dotfile.close()
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=None, min_samples_leaf=5)
clf_entropy.fit(X_train1, y_train1)
clf_entropy
dotfile=open("dtmax_entropy.dot",'w')
tree.export_graphviz(clf_entropy,out_file=dotfile,feature_names=df.columns)
dotfile.close()
y_pred = clf_gini.predict(X_test)
print("Accuracy is uning gini index", accuracy_score(y_test,y_pred)*100)
y_pred_en = clf_entropy.predict(X_test)
print("Accuracy is using information gain ", accuracy_score(y_test,y_pred_en)*100)
from sklearn.feature_selection import RFE
from sklearn import linear_model
regr=linear_model.LinearRegression()
rfe=RFE(regr, 3)
fit=rfe.fit(df,y)
print("coeficient ie. scores is  ",fit.ranking_)

print ("Features sorted by their rank:")
print (sorted(zip(map(lambda X: round(X, 4), fit.ranking_), df.columns)))
df1=df[['candidate_loc', 'expected_attendance', 'job_skills_required', 'location', 'knows_location',
       'interview_type', 'alt_number', 'venue', 'job_location', 'industry']]

X_train1, X_test, y_train1, y_test = train_test_split( df1, y, test_size = 0.3, random_state = 100)
X_train, X_val, y_train, y_val = train_test_split( X_train1, y_train1, test_size = 0.3, random_state = 100)
print("Test set",df1.shape, y.shape)
print("Training set",X_train1.shape, y_train1.shape)
print("Test set",X_test.shape, y_test.shape)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
max_depth=None, min_samples_leaf=4)
clf_gini=clf_gini.fit(X_train1,y_train1)
dotfile=open("dtmax_depth8_top8.dot",'w')
tree.export_graphviz(clf_gini,out_file=dotfile,feature_names=df1.columns)
dotfile.close()
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=None, min_samples_leaf=5)
clf_entropy.fit(X_train1, y_train1)
clf_entropy
dotfile=open("dtmax_entropy_top8.dot",'w')
tree.export_graphviz(clf_entropy,out_file=dotfile,feature_names=df1.columns)
dotfile.close()
# Predict whole set with gini model
y_pred = clf_gini.predict(X_test)
#y_pred
print("Accuracy is uning gini index", accuracy_score(y_test,y_pred)*100)
y_pred_en = clf_entropy.predict(X_test)
#y_pred_en
print("Accuracy is using information gain ", accuracy_score(y_test,y_pred_en)*100)