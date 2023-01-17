import pandas as pd

import numpy as np



def import_data():

    data_paths = {

    '2017': '../input/osmi-mental-health-in-tech-survey-2017/OSMI Mental Health in Tech Survey 2017.csv',

    '2018': '../input/osmi-mental-health-in-tech-survey-2018/OSMI Mental Health in Tech Survey 2018.csv'}

    mh17, mh18 = [pd.read_csv(url) for url in [data_paths["2017"], data_paths["2018"]]]

    #The following four column names have encoding issues in the 2017 dataset. Since both datasets have the exact same columns, we can just replace the strangely encoded column names from 2017 with the same column name in 2018:

    mh17.rename(columns={[mh17.columns[5]][0]:[mh18.columns[5]][0],

                     [mh17.columns[103]][0]:[mh18.columns[103]][0],

                     [mh17.columns[111]][0]:[mh18.columns[111]][0],

                     [mh17.columns[93]][0]:[mh18.columns[93]][0]}, inplace=True)

    #Combine the datasets

    return pd.concat([mh17, mh18], ignore_index=True).drop_duplicates()



data = import_data()

data.head()
data.drop(['#','Have you ever had a coworker discuss their or another coworker\'s mental health with you?', 

          'Describe the conversation your coworker had with you about their mental health (please do not use names).',

          'Do you know local or online resources to seek help for a mental health issue?',

          '<strong>If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?</strong>',

          'If you have revealed a mental health disorder to a client or business contact, how has this affected you or the relationship?',

          'If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?',

          'Did you ever have a previous coworker discuss their or another coworker\'s mental health with you?',

          'Describe the conversation your coworker had with you about their mental health (please do not use names)..1',

          'Has being identified as a person with a mental health issue affected your career?',

          'How has it affected your career?',

          'Would you be willing to talk to one of us more extensively about your experiences with mental health issues in the tech industry? (Note that all interview responses would be used <em>anonymously</em> and only with your permission.)',

          'What US state or territory do you <strong>live</strong> in?',

          'What US state or territory do you <strong>work</strong> in?',

          'Start Date (UTC)', 'Submit Date (UTC)', 'Network ID'],inplace=True,axis=1)



data.rename(columns={'<strong>Are you self-employed?</strong>':'self_employed',

                      'How many employees does your company or organization have?':'num_employees',

                      'Is your employer primarily a tech company/organization?':'tech_company',

                     'Is your primary role within your company related to tech/IT?':'tech_role',

                     'Does your employer provide mental health benefits as part of healthcare coverage?':'benefits',

                     'Do you know the options for mental health care available under your employer-provided health coverage?':'care_options',

                     'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?':'wellness_program',

                     'Does your employer offer resources to learn more about mental health disorders and options for seeking help?':'seek_help_resources',

                     'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?':'anonymity',

                     'If a mental health issue prompted you to request a medical leave from work, how easy or difficult would it be to ask for that leave?':'leave_difficulty',

                     'Would you feel more comfortable talking to your coworkers about your physical health or your mental health?':'phys_vs_mental',

                     'Would you feel comfortable discussing a mental health issue with your direct supervisor(s)?':'supervisor_willingness',

                     'Have you ever discussed your mental health with your employer?':'supervisor_experience',

                     'Describe the conversation you had with your employer about your mental health, including their reactions and what actions were taken to address your mental health issue/questions.':'supervisor_description',

                     'Would you feel comfortable discussing a mental health issue with your coworkers?':'coworkers_willingness',

                     'Have you ever discussed your mental health with coworkers?':'coworkers_experience',

                     'Describe the conversation with coworkers you had about your mental health including their reactions.':'coworkers_description',

                     'Overall, how much importance does your employer place on physical health?':'phys_health_importance',

                     'Overall, how much importance does your employer place on mental health?':'mental_health_importance',

                     'Do you have medical coverage (private insurance or state-provided) that includes treatment of mental health disorders?':'mental_health_coverage',

                     '<strong>If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?</strong>':'tell_others_experience',

                     'If you have revealed a mental health disorder to a coworker or employee, how has this impacted you or the relationship?':'tell_others_consequence',

                     'Do you believe your productivity is ever affected by a mental health issue?':'scope_productivity',

                     '<strong>Do you have previous employers?</strong>':'prev_employer',

                     'Was your employer primarily a tech company/organization?':'prev_tech_company',

                     '<strong>Have your previous employers provided mental health benefits?</strong>':'prev_benefits',

                     '<strong>Were you aware of the options for mental health care provided by your previous employers?</strong>':'prev_care_options',

                     'Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?':'prev_wellness_program',

                     'Did your previous employers provide resources to learn more about mental health disorders and how to seek help?':'prev_seek_help_resources',

                     'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?':'prev_anonymity',

                     'Would you have felt more comfortable talking to your previous employer about your physical health or your mental health?':'prev_phys_vs_mental',

                     'Would you have been willing to discuss your mental health with your direct supervisor(s)?':'prev_supervisor_willingness',

                     'Did you ever discuss your mental health with your previous employer?':'prev_supervisor_experience',

                     'Describe the conversation you had with your previous employer about your mental health, including their reactions and actions taken to address your mental health issue/questions.':'prev_supervisor_description',

                     '<strong>Would you have been willing to discuss your mental health with your coworkers at previous employers?</strong>':'prev_coworkers_willingness',

                     'Did you ever discuss your mental health with a previous coworker(s)?':'prev_coworkers_experience',

                     'Describe the conversation you had with your previous coworkers about your mental health including their reactions.':'prev_coworkers_description',

                     'Overall, how much importance did your previous employer place on physical health?':'prev_phys_health_importance',

                     'Overall, how much importance did your previous employer place on mental health?':'prev_mental_health_importance',

                     'Do you currently have a mental health disorder?':'mh_disorder',

                     'Have you ever been diagnosed with a mental health disorder?':'mh_diagnosis',

                     'Have you had a mental health disorder in the past?':'mh_past',

                     'Have you ever sought treatment for a mental health disorder from a mental health professional?':'mh_treatment',

                     'Do you have a family history of mental illness?':'mh_family_history',

                     'If you have a mental health disorder, how often do you feel that it interferes with your work <strong>when being treated effectively?</strong>':'work_interfere_treated',

                     'If you have a mental health disorder, how often do you feel that it interferes with your work <strong>when <em>NOT</em> being treated effectively (i.e., when you are experiencing symptoms)?</strong>':'work_interfere_not_treated',

                     'Have your observations of how another individual who discussed a mental health issue made you less likely to reveal a mental health issue yourself in your current workplace?':'experience_deterrence',

                     'How willing would you be to share with friends and family that you have a mental illness?':'friends_fam_willingness',

                     'Would you be willing to bring up a physical health issue with a potential employer in an interview?':'phys_interview',

                     'Why or why not?':'phys_interview_description',

                     'Would you bring up your mental health with a potential employer in an interview?':'mental_interview',

                     'Why or why not?.1':'mental_interview_description',

                     'Are you openly identified at work as a person with a mental health issue?':'do_people_know',

                     'If they knew you suffered from a mental health disorder, how do you think that team members/co-workers would react?':'team_reaction',

                     '<strong>Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?</strong>':'witness_unsupportive',

                     'Describe the circumstances of the badly handled or unsupportive response.':'unsupportive_description',

                     '<strong>Have you observed or experienced supportive or well handled response to a mental health issue in your current or previous workplace?</strong>':'witness_supportive',

                     'Describe the circumstances of the supportive or well handled response.':'supportive_description',

                     'Overall, how well do you think the tech industry supports employees with mental health issues?':'tech_ind_support',

                     'Briefly describe what you think the industry as a whole and/or employers could do to improve mental health support for employees.':'tech_improvements_description',

                     'If there is anything else you would like to tell us that has not been covered by the survey questions, please use this space to do so.':'comments',

                     'What is your age?':'age',

                     'What is your gender?':'gender',

                     'What country do you <strong>live</strong> in?':'country_live',

                     'What is your race?':'race', 'Other.3':'race_other',

                     'What country do you <strong>work</strong> in?':'country_work'}, inplace=True)



data = data[['age', 'gender', 'country_live', 'country_work', 

             'race', 'race_other'] + list(data.columns[0:100])]

data.head()
#This is a function that removes all irrelevant variables:



description_list = []



def remove_vars():

    #Removing all text boxes, whose column keys include the keyword "description" to easily retrieve all text data. Let's keep these column names stored in a list in case you want to do sentiment analysis later.

    for colName in data.columns:

        if "description" in colName:

            description_list.append(colName) 

            data.drop(colName, axis=1, inplace=True)

    data.drop("comments", axis=1, inplace=True)

    #I found that for the most part, those who were not working in a tech company WERE working in a tech role and vice versa. I decided then that this only proved that the dataset is focused on tech industry employees, which we already knew. 

    data.drop(["tech_company", "tech_role", "prev_tech_company"], axis=1, inplace=True)

    

remove_vars()

data.shape
#GENDER COLUMN

def fix_gender_column():

    data["gender"] = data["gender"].str.lower()

    data["gender"].replace(['f','female ', 'femalw', 'femail', 'female (cis)', 'female (cis) ','cis female ',

                            'my sex is female.', 'female (cisgender)', 'woman-identified', 'cis-female', 

                            'cis female', 'f, cisgender', 'female-ish', 'trans woman', 'i identify as female',

                            '*shrug emoji* (f)', 'cis woman', 'cisgendered woman', 'trans female', 'woman',

                            'cisgender female'], 'female', inplace=True)

    data["gender"].replace(['m', 'man', 'cis-male', 'mail', 'male/androgynous ','cis hetero male', 

                            'male (cis)','male (hey this is the tech industry you\'re talking about)',

                            'god king of the valajar', 'cis male', 'male ', 'male, cis', 'cis male ',

                            'male-ish','dude','ostensibly male','male, born with xy chromosoms','malel', 

                            'trans man','cisgender male', 'swm',], "male", inplace=True)

    data["gender"].replace(['gender non-binary/other','nonbinary','non-binary','non binary','uhhhhhhhhh fem genderqueer?',

                            'agender/genderfluid','sometimes','contextual','genderqueer demigirl','genderqueer/non-binary',

                            '\\-','transfeminine','agender','male (or female, or both)','female/gender non-binary.', 

                            'genderqueer','demiguy','she/her/they/them','other','nonbinary/femme','genderfluid', 'none',

                           'transgender', 'nb', 'gender non-conforming woman'], 

                           "gender non-binary/other", inplace=True)

    data["gender"].fillna(value="gender non-binary/other", inplace=True)

    

fix_gender_column()

#Should have three categories

data["gender"].unique()
#DUPLICATE DISORDERS COLUMNS

#There are three consecutive 12-column blocks of the dataset containing one hot encoded data where each column represents one mental health disorder the respondent could have checked off as having.



#Columns 40-52 are completely empty, and are titled the 12 disorder names.

#Columns 53-65 are titled the 12 disorder names followed by ".1"

#Columns 66-78 are titled the 12 disorder names followed by ".2" and for some reason contains different data than the previous 12 column block. 



#eg:

#Outcome that preserves responses that are in matching columns:

# Anxiety + Anxiety.1  == Unified Anxiety Column

#    T    |   F        ==         1

#    F    |   F        ==         0

#    F    |   T        ==         1



#Let's condense these sections into one



def fix_disorders_columns():

    #The PTSD duplicate columns vary in their capitalization. Synchonize them so that the rest of the function works on these columns:

    data.rename(columns={'Post-traumatic Stress Disorder':'Post-traumatic Stress Disorder.2'}, inplace=True)

    data.rename(columns={'Post-Traumatic Stress Disorder':'Post-traumatic Stress Disorder'}, inplace=True)

    #Handle the left-side of the duplicated data, turn to 0's and 1's

    for col in data.iloc[:,53:65]:

        for i in range(0,data.shape[0]):

            if data.at[i,col] == col[:-2]:

                data.at[i,col[:-2]] = 1

            else:

                data.at[i,col[:-2]] = 0

    #Add right-side data to left side as 0's and 1's

    for col in data.iloc[:,66:78]:

        for i in range(0,data.shape[0]):

            if data.at[i,col] == col[:-2]:

                data.at[i,col[:-2]] = 1

    #The "Other" column is tricky. There is not enough data 

    #in this column to one hot encode it, so just tally the 'other' in general:

    for i in range(0,data.shape[0]):

        if (pd.isnull(data['Other.1'][i])) and (pd.isnull(data['Other.2'][i])):

            data.at[i,"Other"] = 0

        else:

            data.at[i,"Other"] = 1

    #Drop the duplicated columns:

    drop_cols = data.iloc[:,53:79].columns.tolist()

    data.drop(drop_cols, inplace=True, axis=1)



#Should be (93-26 dupes) = 67 columns

fix_disorders_columns()

data.shape
#RACE COLUMN(S)

#"What is your Race?" was asked in two separate survey questions, one with multiple choice options and the other a free input-box to write in "other."

#Additionally, some of the Race data was repetative and I tried to classify appropriately the unique values while still respecting the identity of survey respondants. 



def fix_race_column():

    for i in range(0,data.shape[0]):

        if str(data.at[i,"race_other"]) != "nan":

            data.at[i,"race"] = data.at[i,"race_other"]

    data.drop("race_other", inplace=True, axis=1)

    data["race"] = data["race"].fillna("I prefer not to answer")

    data["race"].replace({'I am of the race of Adam, the first human.':'I prefer not to answer',

                          'My race is white, but my ethnicity is Latin American':'White'}, inplace=True)

    data["race"].replace(["Latinx","Latino","Latina","Hispanic",'Hispanic, White'], "Hispanic or Latino", inplace=True)

            

fix_race_column()

#Should be 66 columns

data.shape
#Now that I've cleaned the data up a bit, I want to look into some of the remaining NaN data. 

#The idea for an isna() heatmap came from this source: 

#https://www.kaggle.com/andradaolteanu/preprocess-visualise-model-mental-health-in-tech



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



sns.heatmap(data = data.isna());
#data['self_employed'].isna().sum() #returns zero, so everyone was asked this

#data['self_employed'][data['self_employed'] == 1].sum() #returns 169, so there are 169 self-employed respondents

#(function just to check, does not need to be run)

def check_cols_for_NaN():

    checkCols = ["benefits", "care_options", "wellness_program", "seek_help_resources", "anonymity", "leave_difficulty"]

    numNaNDict = {}

    for aCol in checkCols:

        nullCount = data[aCol][data['self_employed'] == 1]

        numNaNDict[aCol] = nullCount.isna().sum()

    return numNaNDict

#check_cols_for_NaN()

#Notice that each value in the dictionary is 169. We can just drop respondents that are self-employed then, and then drop the whole self_employed column now that we are certain we only have not-self-employed respondents. 

data.dropna(subset=["benefits"], inplace=True) 

data.reset_index(drop=True, inplace=True)

data.drop("self_employed", axis=1, inplace=True)



#Should be 1,004 rows x 65 cols

data.shape
#Some of the data columns contain less than 50% of response-rate. Drop those columns.

def drop_major_not_ans():

    major_not_ans=[]

    for col in data.columns:

        if(sum(pd.isnull(data[col])) > data.shape[0]/2):

            major_not_ans.append(col)

    data.drop(major_not_ans, axis=1, inplace=True)

    return major_not_ans

#Return which columns were dropped 

drop_major_not_ans()
# Dealing with other missing values

# Maybe this is a bit lazy but just fill them in with the average...

from sklearn.impute import SimpleImputer

# Impute nan with the most frequent value on every row

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imp.fit(data)

data = pd.DataFrame(data=imp.transform(data), columns = data.columns)
#Let's get a sense of the data types in each of our features 

#Expand the code below for a collection of unique inputs for each column.

for index,val in enumerate(data.columns):

    p=data[val].unique()

    print(index,val, p, '\n')