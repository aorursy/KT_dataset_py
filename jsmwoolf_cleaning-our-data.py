import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/Interview.csv')
df.head()
# If we don't run the code in a try block, jupyter will stop executing
try:
    pd.to_datetime(df["Date of Interview"],format='%d.%m.%Y')
except ValueError as e:
    print(e)
df["Date of Interview"] = df["Date of Interview"].str.replace(".","-")

# Extract the year and add 20 to the beginning of each string
# We only need to extract the last four digits.
tmp = df["Date of Interview"].str.extract(".*-(\d+)", expand=False)
tmp = np.array([("20" + date)[-4:] if type(date) == str else date for date in tmp ])
# Remodify the Series
df["Date of Interview"] = df["Date of Interview"].str.extract("(\d+-\d+-)\d+", expand=False) + tmp

# Now do the conversion
df["Date of Interview"] = pd.to_datetime(df["Date of Interview"],format="%d-%m-%Y")
df.head()
retainColumns = df.columns.values[:-5]

# This is to prevent removing useful columns
if len(retainColumns) == 23:
    df = df[retainColumns]
    
print(len(df))
df.head()
def getUniqueValueFormatter(column, limit = 10):
    values = df[column].unique()
    # Make sure that there the limit is capped at the max number of unique values
    if limit > len(values):
        limit = len(values)
    if len(values) > limit:
        values = values[:limit]
    print("Column {} contains values:".format(column))
    for value in values:
        print("\t *{}".format(value))

for column in df.columns[1:]:
    getUniqueValueFormatter(column)
for column in df.columns[1:]:
    print("{} record(s) missing for column {}".format(df[column].isnull().sum(), column))
df.tail(3)
df = df[0:1233]
df.tail(3)
column = 'Client name'
getUniqueValueFormatter('Client name')
df.loc[df[column].str.contains("Standard Chartered Bank"), column] = "Standard Chartered Bank"
df.loc[df[column].str.contains("Aon"), column] = "Aon Hewitt"

getUniqueValueFormatter(column)
getUniqueValueFormatter('Industry')
df.loc[df['Industry'].str.contains("IT*"), 'Industry'] = "IT"
getUniqueValueFormatter('Industry')
getUniqueValueFormatter('Location')
df['Location'] = df['Location'].str.strip('- ')
df['Location'] = df['Location'].str.lower()

getUniqueValueFormatter('Location')
getUniqueValueFormatter('Position to be closed')
column = 'Nature of Skillset'
getUniqueValueFormatter(column, 1000)
df.loc[df[column].str.contains("Java", case=False), column] = "Java"
df.loc[df[column].str.contains("SCCM", case=False), column] = "SCCM"
df.loc[df[column].str.contains("Analytical R & D", case=False), column] = "Analytical R&D"
df.loc[df[column].str.contains("Lending", case=False), column] = "Lending & Liability"
df.loc[df[column].str.contains("L & L", case=False), column] = "Lending & Liability"
df.loc[df[column].str.contains("Tech lead", case=False), column] = "Tech Lead - Mednet"
df.loc[df[column].str.contains("production", case=False), column] = "Production"

getUniqueValueFormatter(column, 1000)
getUniqueValueFormatter('Interview Type')
df['Interview Type'] = df['Interview Type'].str.strip()
df.loc[df['Interview Type'].str.contains('Sc*W*'), 'Interview Type'] = "Scheduled"

getUniqueValueFormatter('Interview Type')
if 'Name(Cand ID)' in df.columns:
    df = df.drop('Name(Cand ID)', axis=1)

if 'Name(Cand ID)' not in df.columns:
    print("Column removed")
getUniqueValueFormatter('Gender')
getUniqueValueFormatter('Candidate Current Location')
column = 'Candidate Current Location'
df[column] = df[column].str.strip('- ')
df[column] = df[column].str.lower()

getUniqueValueFormatter(column)
column = 'Have you obtained the necessary permission to start at the required time'
getUniqueValueFormatter(column)
df[column] = df[column].fillna('Not yet')
df.loc[df[column].str.contains('Na'), column] = 'No'
df.loc[df[column].str.contains('Yet to confirm'), column] = 'Not yet'
df[column] = df[column].str.lower()

getUniqueValueFormatter(column)
column = 'Hope there will be no unscheduled meetings'
getUniqueValueFormatter(column)
df[column] = df[column].fillna('Not Sure')
df.loc[df[column].str.contains('Na'), column] = 'No'
df.loc[df[column].str.contains('cant'), column] = 'Not Sure'
df[column] = df[column].str.lower()

getUniqueValueFormatter(column)
column = "Can I Call you three hours before the interview and follow up on your attendance for the interview"
getUniqueValueFormatter(column)
df[column] = df[column].fillna('Not sure')
df.loc[df[column].str.contains('Na'), column] = 'No'
df.loc[df[column].str.contains('No'), column] = 'No'
df[column] = df[column].str.lower()

getUniqueValueFormatter(column)
column = "Can I have an alternative number/ desk number. I assure you that I will not trouble you too much"
getUniqueValueFormatter(column)
df[column] = df[column].fillna('No')
df.loc[df[column].str.contains('na'), column] = 'No'
df.loc[df[column].str.contains('No'), column] = 'No'
df[column] = df[column].str.lower()

getUniqueValueFormatter(column)
column = "Have you taken a printout of your updated resume. Have you read the JD and understood the same"
getUniqueValueFormatter(column)
df[column] = df[column].fillna('Not Yet')
df.loc[df[column].str.contains('No-'), column] = 'No'
df.loc[df[column].str.contains('Na', case=False), column] = 'no'
df[column] = df[column].str.lower()

getUniqueValueFormatter(column)
column = "Are you clear with the venue details and the landmark."
getUniqueValueFormatter(column)
df[column] = df[column].fillna('I need to check')
df.loc[df[column].str.contains('No-'), column] = 'I need to check'
df.loc[df[column].str.contains('Na', case=False), column] = 'no'
df[column] = df[column].str.lower()

getUniqueValueFormatter(column)
column = "Has the call letter been shared"
getUniqueValueFormatter(column)
df[column] = df[column].fillna('Need To Check')
df.loc[df[column].str.contains('H'), column] = 'Need To Check'
df.loc[df[column].str.contains('Yet'), column] = 'Need To Check'
df.loc[df[column].str.contains('Not'), column] = 'Need To Check'
df.loc[df[column].str.contains('na', case=False), column] = 'No'
df[column] = df[column].str.lower()

getUniqueValueFormatter(column)
column = "Expected Attendance"
getUniqueValueFormatter(column)
df[column] = df[column].fillna('Uncertain')
df.loc[df[column].str.contains('1'), column] = 'Yes'
df[column] = df[column].str.lower()

getUniqueValueFormatter(column)
column = "Observed Attendance"
getUniqueValueFormatter(column)
df[column] = df[column].str.strip()
df[column] = df[column].str.lower()

getUniqueValueFormatter(column)
column = "Marital Status"
getUniqueValueFormatter(column)
df[column] = df[column].str.lower()
df.to_csv("Interview-Cleaned.csv")