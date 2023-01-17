#Importing the required libraries
import pandas as pd
import re
import datetime as dt
#fetching our dataset using pandas
df = pd.read_csv('../input/Automatic email content store.csv')
#creating columns in our dataset
new_df = df['Date$to $raw'].apply(lambda x: pd.Series(x.split('$')))
new_df.columns = ['Date', 'to', 'raw']
#Take a look at the content
new_df
#TASK 1
#Getting unique email addresses from our dataset
unq_mail = list(set(new_df.to))
print(unq_mail)
#TASK 2
#Getting last date of conversation with all email addresses
dates_dict = {}
for mail in unq_mail:
    temp = new_df[new_df['to'] == mail]
    dates = str(temp.Date)
    dates = re.findall(r"[\d]{1,2}/[\d]{1,2}/[\d]{2}", dates)
    latest_date = str(pd.to_datetime(dates, format='%d/%m/%y').max())
    match = re.search('\d{4}-\d{2}-\d{2}', latest_date)
    date = dt.datetime.strptime(match.group(), '%Y-%m-%d').date()
    dates_dict[mail] = date
    del latest_date, date
keys = []
values = []
for email in dates_dict:
    keys.append(email)
    values.append(dates_dict[email])
    dict = {'Email': keys, 'Date': values}
    last_convo_date = pd.DataFrame(dict)
last_convo_date
#TASK 4
#Determining the number of days after the last conversation
def elapsed_time():
    today = dt.date(2018, 6, 19)
    No_of_days = pd.DataFrame(dict, columns=['Email', 'Last Conversation'])
    No_of_days['Last Conversation'] = today - last_convo_date['Date']
    return No_of_days
NumberOfDays = elapsed_time()
NumberOfDays
#TASK 5
#Matching the column "raw" with contents of check_list
def matching_words():
    check_list = ["OpenCV", "Python"]  
    all_content = new_df.raw
    new_df["Matched"] = ""     #new column created
    count = 0
    for i in all_content:
        list_words = i.split()
        for j in list_words:
            for k in check_list:
                if j in check_list and j==k:
                    new_df["Matched"].iloc[count] = k
                    common_list.append(k)
                else:
                    pass
            if j not in check_list:
                new_df["Matched"].iloc[count] = "No Match"
        count += 1  
    return new_df    
MatchedWords = matching_words()
MatchedWords
