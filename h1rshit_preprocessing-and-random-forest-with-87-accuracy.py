import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('Set2')
import matplotlib.pyplot as plt
%matplotlib inline
response=pd.read_csv('../input/mental-heath-in-tech-2016_20161114.csv')
#response.describe()#(include='all')
#response.info()
#response.get_index()
features=(list(response))
print(len(features))
response.shape
# for val in features:
#     print(val)
#     print('\n')
count=0
major_not_ans=[]
for col in features:
    #print(col,sum(pd.isnull(response[col])))
    if(sum(pd.isnull(response[col]))>721):
        count=count+1
        major_not_ans.append(col)
        #response.drop([col],axis=1,inplace=True)
print(len(major_not_ans))
response.drop([i for i in major_not_ans],axis=1,inplace=True)
extra_feature=['What US state or territory do you work in?','Why or why not?','Why or why not?.1','What US state or territory do you live in?']
one_more=['Which of the following best describes your work position?']
response.drop([i for i in extra_feature],axis=1,inplace=True)
response.drop(one_more,axis=1,inplace=True)
response.drop(['What country do you live in?'],axis=1,inplace=True)
response.drop(['Are you self-employed?'],axis=1,inplace=True)
response.drop(['Do you have previous employers?'],axis=1,inplace=True)
real_features=(list(response))
print(len(real_features))

# REMOVING SELF EMPLOYED 287 NAN
count=0
for index,col in enumerate(real_features):
    idx=response.index[response[col].isnull()]
    #response.drop(idx,inplace=True)
    if(len(idx)==287):
        #print(index,idx)
        k=idx
        count+=1
#print(count)    12
#print(k)
#idx=response.index[response['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].isnull()]
response.drop(k,inplace=True)
response.shape
# Group of people not answering the same questions So removing them
count=0
for index,col in enumerate(real_features):
    idx=response.index[response[col].isnull()]
    #response.drop(idx,inplace=True)
    if(len(idx)==131):
        #print(index,idx)
        k=idx
        count+=1
# print(count) 11
# print(k)
#idx=response.index[response['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].isnull()]
response.drop(k,inplace=True)
response.shape
# clean the genders by grouping the genders into 3 categories: Female, Male, Genderqueer/Other
response['What is your gender?'] = response['What is your gender?'].replace([
    'male', 'Male ', 'M', 'm', 'man', 'Cis male',
    'Male.', 'Male (cis)', 'Man', 'Sex is male',
    'cis male', 'Malr', 'Dude', "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
    'mail', 'M|', 'male ', 'Cis Male', 'Male (trans, FtM)',
    'cisdude', 'cis man', 'MALE'], 'Male')
response['What is your gender?'] = response['What is your gender?'].replace([
    'female', 'I identify as female.', 'female ',
    'Female assigned at birth ', 'F', 'Woman', 'fm', 'f',
    'Cis female', 'Transitioned, M2F', 'Female or Multi-Gender Femme',
    'Female ', 'woman', 'female/woman', 'Cisgender Female', 
    'mtf', 'fem', 'Female (props for making this a freeform field, though)',
    ' Female', 'Cis-woman', 'AFAB', 'Transgender woman',
    'Cis female '], 'Female')
response['What is your gender?'] = response['What is your gender?'].replace([
    'Bigender', 'non-binary,', 'Genderfluid (born female)',
    'Other/Transfeminine', 'Androgynous', 'male 9:1 female, roughly',
    'nb masculine', 'genderqueer', 'Human', 'Genderfluid',
    'Enby', 'genderqueer woman', 'Queer', 'Agender', 'Fluid',
    'Genderflux demi-girl', 'female-bodied; no feelings about gender',
    'non-binary', 'Male/genderqueer', 'Nonbinary', 'Other', 'none of your business',
    'Unicorn', 'human', 'Genderqueer'], 'Genderqueer/Other')
# clean the ages by replacing the weird ages with the mean age
# min age was 3 and max was 393 
response.loc[(response['What is your age?'] > 90), 'What is your age?'] = 34
response.loc[(response['What is your age?'] < 10), 'What is your age?'] = 34
# replace the one null with Male, the mode gender, so we don't have to drop the row
response['What is your gender?'] = response['What is your gender?'].replace(np.NaN, 'Male')
response['What is your gender?'].value_counts()

response.fillna(method='ffill',inplace=True)
response.fillna(value='Yes', limit=1,inplace=True)
#print response['Do you know the options for mental health care available under your employer-provided coverage?']
g = sns.countplot(x='Do you know the options for mental health care available under your employer-provided coverage?',data=response)
for index,val in enumerate(real_features):
    p=response[val].unique()
    print(index,val)
    print(p)
    print('\n')
    #print(response[val].isnull().sum())
    #print("\n")
country=(response[real_features[40]].unique())
num_rep=[]    #numeric representation with there index
alp_rep=[]    # name of country
#print(type(country))
for index,val in enumerate(country):
    num_rep.append(index)
    alp_rep.append(val)
print(len(num_rep),len(alp_rep))
response[real_features[40]].replace(alp_rep, num_rep,inplace=True)  # Replacing country name with the index
response['Does your employer provide mental health benefits as part of healthcare coverage?'] = response['Does your employer provide mental health benefits as part of healthcare coverage?'].replace('Not eligible for coverage / N/A','No')
g = sns.countplot(x='Does your employer provide mental health benefits as part of healthcare coverage?',data=response)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace('1-5', 5)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace('6-25',25)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace('26-100', 100)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace('100-500',500)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace('500-1000',1000)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace('More than 1000',5000)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace(np.nan,5)
# Replacing NaN values wd range 26-100
# USED DIRECTLY
# #response[real_features[7]]
# g = sns.countplot(x=response[real_features[7]],data=response)
g = sns.countplot(x=response[real_features[14]],data=response)


numeric = {real_features[2]:     {'No':0, 'Yes':1, "I don't know":2},
                real_features[3]: {'Yes':1, 'I am not sure':2, 'No':0},
                 real_features[4]:{'No':0, 'Yes':1, "I don't know":2},
                  real_features[5]:{'No':0, 'Yes':1, "I don't know":2},
                   real_features[6]:{"I don't know":2, 'Yes':1, 'No':0},
                    real_features[7]:{'Very easy':0 ,'Somewhat easy':1, 'Neither easy nor difficult':2,'Very difficult':-1,
 'Somewhat difficult':-2, "I don't know":2}, #### MODIFIED DIRECTLY
                real_features[8]:{'No':0, 'Maybe':2, 'Yes':1},
                real_features[9]:{'No':0, 'Maybe':2, 'Yes':1},
                 real_features[10]:{'No':0, 'Maybe':2, 'Yes':1},
                 real_features[11]:{'No':0, 'Maybe':2, 'Yes':1},
                 real_features[12]:{"I don't know":2, 'Yes':1, 'No':0},
                 real_features[13]:{'No':0, 'Yes':1},
                 real_features[14]:{'No, none did':0, 'Yes, they all did':1, "I don't know":2, 'Some did':3},
                 real_features[15]:{'N/A (not currently aware)':0, 'I was aware of some':1,
 'Yes, I was aware of all of them':1, 'No, I only became aware later':0},  ### MODIFIED DIRECTLY
                real_features[16]:{"I don't know":2, 'None did':0, 'Some did':3,'Yes, they all did':1},
                real_features[17]:{'None did':0, 'Some did':3, 'Yes, they all did':1},
                real_features[18]:{"I don't know":2, 'Yes, always':1, 'Sometimes':3, 'No':0},
                real_features[19]:{'Some of them':3, 'None of them':0, "I don't know":2, 'Yes, all of them':1},
                real_features[20]:{'None of them':0, 'Some of them':3, 'Yes, all of them':1},
                real_features[21]:{'Some of my previous employers':3, 'No, at none of my previous employers':0,
 'Yes, at all of my previous employers':1},
                real_features[22]:{'Some of my previous employers':3, "I don't know":2, 'No, at none of my previous employers':0,
 'Yes, at all of my previous employers':1},
                real_features[23]:{"I don't know":2, 'Some did':3, 'None did':0, 'Yes, they all did':1},
                real_features[24]:{'None of them':0, 'Some of them':3, 'Yes, all of them':1},
                real_features[25]:{'Maybe':2, 'Yes':1, 'No':0},
                real_features[26]:{'Maybe':2, 'Yes':1, 'No':0},
                real_features[27]:{'Maybe':2, "No, I don't think it would":0, 'Yes, I think it would':1,
 'No, it has not':0, 'Yes, it has':1},  ### MODIFIED DIRECTLY
                real_features[28]:{"No, I don't think they would":0, 'Maybe':2, 'Yes, they do':1,'Yes, I think they would':1, 'No, they do not':0},  ## MODIFIED DIRECTLY
                real_features[29]:{'Somewhat open':1, 'Not applicable to me (I do not have a mental illness)':4,
 'Very open':2, 'Not open at all':-2 ,'Neutral':0, 'Somewhat not open':-1}, ### MODIFIED DIRECTLY
                real_features[30]:{'No':0, 'Maybe/Not sure':2, 'Yes, I experienced':1, 'Yes, I observed':1},
                real_features[31]:{'No':0, 'Yes':1, "I don't know":2},
                real_features[32]:{'Yes':1, 'Maybe':2, 'No':0},
                real_features[33]:{'Yes':1, 'Maybe':2, 'No':0},
                real_features[34]:{'Yes':1, 'No':0},
                real_features[36]:{'Not applicable to me':4, 'Rarely':0, 'Sometimes':3, 'Never':0, 'Often':1},
                real_features[37]:{'Not applicable to me':4, 'Sometimes':3, 'Often':1, 'Rarely':0, 'Never':0},
                real_features[39]:{'Male':1, 'Female':0, 'Genderqueer/Other':2},
                real_features[41]:{'Sometimes':3, 'Never':0, 'Always':1}
          }
response.replace(numeric, inplace=True)
response.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
target=response['Have you ever sought treatment for a mental health issue from a mental health professional?']
response.drop(['Have you ever sought treatment for a mental health issue from a mental health professional?'],axis=1,inplace=True)
#target.unique()
#response.shape
X=response
X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.2, random_state=0)
clf = RandomForestClassifier(max_depth=14, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))
# NOT USED
# count=0
# # print(match,unmatch)
# k=response['Have you been diagnosed with a mental health condition by a medical professional?'].tolist()
# p=response['Have you ever sought treatment for a mental health issue from a mental health professional?'].tolist()

# print(k[0],p[0])
# for i,val in enumerate(k):
#     if(val==p[i]):
#         count+=1
# print(count)
#g = sns.countplot(x='How many employees does your company or organization have?',data=response)
# for i in _notans169:
#     print(response.iloc[i].values)
#     print('\n')
# match=0
# unmatch=0
# np.where(response['What country do you live in?']==response['What country do you work in?'], match+=1,unmatch+=1)
#count=0
# print(match,unmatch)
# k=response['What country do you live in?'].tolist()
# p=response['What country do you work in?'].tolist()

# #print(k)
# for i,val in enumerate(k):
#     if(val==p[i]):
#         count+=1
# print(count)
#1407 people works on same place where they live
#print(response[:]['What country do you work in?'],response[:]['What country do you live in?'])
# USEFUL
# count=0
# for i,col in enumerate(real_features):
#     idx=response.index[response[col].isnull()]
#     #response.drop(idx,inplace=True)
#     if(len(idx)==169):
#         print(col)
#         print(i,idx)
#         _notans169=idx
#         count+=1
# print(count,_notans169)
#idx=response.index[response['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].isnull()]
#response.drop(idx,inplace=True)
# USEFUL
# count=0
# tcount=0
# selfnotans=[]
# for index,col in enumerate(features):
#     idx=response.index[response[col].isnull()]
#     tcount+=1
#     #response.drop(idx,inplace=True)
#     if(len(idx)==287):
#         #print(features[index])
#         selfempl=idx
#         selfnotans.append(features[index])
#         #print('\n')
#         #print(index,idx)
#         count+=1
# #print(count,tcount)

# print((selfempl[:]))
# for i in selfempl:
#     print(response.loc[i,'Do you currently have a mental health disorder?'])