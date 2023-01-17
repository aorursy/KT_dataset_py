#import pandas
import pandas as pd 
train_df= pd.read_csv('../input/train.csv')
train_df.head()
test_df= pd.read_csv('../input/test.csv')
test_df.head()
gender_df = pd.read_csv('../input/gender_submission.csv')
gender_df.head()
test_size= len(test_df)
train_size = len(train_df)
print(test_size, train_size)

#get the % of total we are training with + testing with
total= test_size + train_size
print('test percent ', test_size/total)
print('train percent ', train_size/total)
train_passenger = set(train_df.PassengerId)
test_passenger = set(test_df.PassengerId)
len(set.intersection(train_passenger,test_passenger))
for x in train_df.columns: print(x)
train_df.Name[0:10]
train_df.Name.str.split(',', expand=True)[0][0:5]

first_name= train_df.Name.str.split(',', expand=True)[1]
title=first_name.str.split('.', expand=True)[0]
titles= title.unique()

for title in titles: print(title)
first_name.str.split('.', expand=True)[1]
#get all content in parentheses
first_name.str.extract(r"\((.*?)\)", expand=False)[0:10]

#get all content in quotations
first_name.str.extract(r"\"(.*?)\"", expand=False)[0:10]
first_name = train_df.Name.str.split(',', expand=True)[1]
first_name = first_name.str.split('.', expand=True)[1]


first_name = first_name.str.replace(r"\(.*\)","")
first_name.str.replace(r"\".*\"","")[0:5]
#take a dataframe as an input
def name_process (df):
    last_name = df.Name.str.split(',', expand=True)[0]
    first_name = df.Name.str.split(',', expand=True)[1]
    
    woman_name = first_name.str.extract(r"\((.*?)\)", expand=False)
    
    title = first_name.str.split('.', expand=True)[0]
    first_name = first_name.str.split('.', expand=True)[1]
    
    nick_name = first_name.str.extract(r"\"(.*?)\"", expand=False)
    
    first_name = first_name.str.replace(r"\(.*\)","")
    first_name = first_name.str.replace(r"\".*\"","")
    
    df['first_name'] = first_name
    df['last_name']=last_name
    df['title'] = title
    df['nick_name'] = nick_name
    df['woman_name'] = woman_name
    
    return df

name_process(train_df)
name_process(test_df)

train_df.head() 
test_df.head()