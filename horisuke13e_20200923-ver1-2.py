import numpy as np

import pandas as pd

pd.options.display.max_rows = 1000

pd.options.display.max_columns = 200

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns



import category_encoders as ce



from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import mean_squared_error



import lightgbm as lgb



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

        

DIR = '/kaggle/input/exam-for-students20200923'
# 計算に時間がかかる処理などは辞書形式でグローバル変数にキャッシュとして保存しておく

caches = {}
# 関数を実行した際にログを出力するデコレータ

def logger(func):

    def print_log(*args, **kwargs):

        print(f'{func.__name__}()')

        return func(*args, **kwargs)

    return print_log
@logger

def get_df_concat(cache_name=None):

    global caches

    

    # キャッシュがあればキャッシュを返す

    if cache_name in caches.keys(): 

        print(' - already done, loading cache.')

        return caches[cache_name].copy()

    

    # キャッシュがなければ普通に読み込み

    df_train = pd.read_csv(f'{DIR}/train.csv')

    df_test = pd.read_csv(f'{DIR}/test.csv')

    df_train['is_test'] = 0

    df_test['is_test'] = 1

    df_concat = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)



    # cache_nameが指定されている場合はキャッシュを残す

    if cache_name is not None: 

        caches[cache_name] = df_concat.copy()

        

    return df_concat
@logger

def baseline_fe_for_tree_model(df_concat, cache_name=None):

    global caches



    # キャッシュがあればキャッシュを返す

    if cache_name in caches.keys(): 

        print(' - already done, loading cache.')

        return caches[cache_name].copy()

    

    # Hobby

    dict_hobby = {'Yes': 1, 'No': 0}

    df_concat['Hobby'] = df_concat['Hobby'].apply(lambda x: dict_hobby[x]).astype(int)



    # OpenSource

    dict_opensource = {'Yes': 1, 'No': 0}

    df_concat['OpenSource'] = df_concat['OpenSource'].apply(lambda x: dict_opensource[x]).astype(int)



    # Student

    df_concat['Student'] = df_concat['Student'].fillna('MISSING')

    dict_student = {'Yes, full-time': 2, 'Yes, part-time': 1, 'No': 0, 'MISSING': -1}

    df_concat['Student'] = df_concat['Student'].apply(lambda x: dict_student[x]).astype(int)



    # Employment

    dict_employment = {

        'Employed full-time': 4, 

        'Employed part-time': 3,

        'Independent contractor, freelancer, or self-employed': 4,

        'Retired': 2, 

        'Not employed, but looking for work': 1, 

        'Not employed, and not looking for work': 0, 

        'MISSING': -1

    }

    df_concat['Employment'] = df_concat['Employment'].fillna('MISSING')

    df_concat['Employment_Employed'] = df_concat['Employment'].isin(['Employed full-time', 'Employed part-time']).astype(int)

    df_concat['Employment_SelfEmployed'] = (df_concat['Employment'] == 'Independent contractor, freelancer, or self-employed').astype(int)

    df_concat['Employment_Retired'] = (df_concat['Employment'] == 'Retired').astype(int)

    df_concat['Employment_NotEmployed'] = df_concat['Employment'].isin(['Not employed, but looking for work', 'Not employed, and not looking for work']).astype(int)

    df_concat['Employment'] = df_concat['Employment'].apply(lambda x: dict_employment[x]).astype(int)



    # FormalEducation

    dict_formaleducation = {

        'Professional degree (JD, MD, etc.)': 6,

        'Other doctoral degree (Ph.D, Ed.D., etc.)': 6, 

        'Master’s degree (MA, MS, M.Eng., MBA, etc.)': 5, 

        'Bachelor’s degree (BA, BS, B.Eng., etc.)': 4,

        'Associate degree': 3,

        'Some college/university study without earning a degree': 2,

        'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 1,

        'Primary/elementary school': 1,

        'I never completed any formal education': 0, 

        'MISSING': -1

    }

    df_concat['FormalEducation'] = df_concat['FormalEducation'].fillna('MISSING')

    df_concat['FormalEducation_WithDegree'] = df_concat['FormalEducation'].isin([

        'Professional degree (JD, MD, etc.)', 'Other doctoral degree (Ph.D, Ed.D., etc.)', 

        'Master’s degree (MA, MS, M.Eng., MBA, etc.)', 'Bachelor’s degree (BA, BS, B.Eng., etc.)',

        'Associate degree'

    ]).astype(int)

    df_concat['FormalEducation'] = df_concat['FormalEducation'].apply(lambda x: dict_formaleducation[x]).astype(int)



    # CompanySize

    dict_companysize = {

        '10,000 or more employees': 10000, '5,000 to 9,999 employees': 5000, '1,000 to 4,999 employees': 1000, 

        '500 to 999 employees': 500, '100 to 499 employees': 100, '20 to 99 employees': 20, 

        '10 to 19 employees': 10, 'Fewer than 10 employees': 1, 'MISSING': -1

    }

    df_concat['CompanySize'] = df_concat['CompanySize'].fillna('MISSING')

    df_concat['CompanySize'] = df_concat['CompanySize'].apply(lambda x: dict_companysize[x]).astype(int)



    # DevType

    df_concat['DevType'] = df_concat['DevType'].fillna('MISSING')

    val_list = [

        'Data scientist or machine learning specialist', 'Marketing or sales professional', 

        'Database administrator', 'Engineering manager', 'QA or test developer', 'Back-end developer', 

        'DevOps specialist', 'C-suite executive (CEO, CTO, etc.)', 'Data or business analyst', 

        'Game or graphics developer', 'Mobile developer', 'Product manager', 

        'Desktop or enterprise applications developer', 'Student', 'Front-end developer', 

        'Full-stack developer', 'Designer', 'Embedded applications or devices developer', 

        'System administrator', 'Educator or academic researcher', 

        'MISSING'

    ]

    arr_one_hot = [[0]*len(val_list) for i in range(len(df_concat))]

    for i, vals in enumerate(df_concat['DevType']):

        for v in vals.split(';'):

            arr_one_hot[i][val_list.index(v)] = 1

    df_vals = pd.DataFrame(columns=[f'DevType_{i}' if v != 'MISSING' else 'DevType_MISSING' for i, v in enumerate(val_list)], data=arr_one_hot)

    df_concat = pd.concat([df_concat, df_vals], axis=1).drop(columns=['DevType'])

    del df_vals



    # YearsCoding

    dict_yearscoding = {

        '30 or more years': 30, '27-29 years': 27,  '24-26 years': 24, '21-23 years': 21, 

        '18-20 years': 18, '15-17 years': 15, '12-14 years': 12, '9-11 years': 9, '6-8 years': 6, 

        '3-5 years': 3, '0-2 years': 1,

        'MISSING': -1

    }

    df_concat['YearsCoding'] = df_concat['YearsCoding'].fillna('MISSING')

    df_concat['YearsCoding'] = df_concat['YearsCoding'].apply(lambda x: dict_yearscoding[x]).astype(int)



    #YearsCodingProf

    dict_yearscodingprof = {

        '30 or more years': 30, '27-29 years': 27, '24-26 years': 24,'21-23 years': 21, 

        '18-20 years': 18, '15-17 years': 15, '12-14 years': 12, '9-11 years': 9, '6-8 years': 6, 

        '3-5 years': 3, '0-2 years': 1, 

        'MISSING': -1

    }

    df_concat['YearsCodingProf'] = df_concat['YearsCodingProf'].fillna('MISSING')

    df_concat['YearsCodingProf'] = df_concat['YearsCodingProf'].apply(lambda x: dict_yearscodingprof[x]).astype(int)



    # JobSatisfaction

    dict_JobSatisfaction = {

        'Extremely satisfied': 6, 'Moderately satisfied': 5, 'Slightly satisfied': 4,

        'Neither satisfied nor dissatisfied': 3,

        'Slightly dissatisfied': 2, 'Moderately dissatisfied': 1, 'Extremely dissatisfied': 0, 

        'MISSING': -1

    }

    df_concat['JobSatisfaction'] = df_concat['JobSatisfaction'].fillna('MISSING')

    df_concat['JobSatisfaction_Satisfied'] = df_concat['JobSatisfaction'].isin(['Extremely satisfied', 'Moderately satisfied', 'Slightly satisfied']).astype(int)

    df_concat['JobSatisfaction_DisSatisfied'] = df_concat['JobSatisfaction'].isin(['Extremely dissatisfied', 'Moderately dissatisfied', 'Slightly dissatisfied']).astype(int)

    df_concat['JobSatisfaction'] = df_concat['JobSatisfaction'].apply(lambda x: dict_JobSatisfaction[x]).astype(int)



    # CareerSatisfaction

    dict_CareerSatisfaction = {

        'Extremely satisfied': 6, 'Moderately satisfied': 5, 'Slightly satisfied': 4,

        'Neither satisfied nor dissatisfied': 3,

        'Slightly dissatisfied': 2, 'Moderately dissatisfied': 1, 'Extremely dissatisfied': 0, 

        'MISSING': -1

    }

    df_concat['CareerSatisfaction'] = df_concat['CareerSatisfaction'].fillna('MISSING')

    df_concat['CareerSatisfaction_Satisfied'] = df_concat['CareerSatisfaction'].isin(['Extremely satisfied', 'Moderately satisfied', 'Slightly satisfied']).astype(int)

    df_concat['CareerSatisfaction_DisSatisfied'] = df_concat['CareerSatisfaction'].isin(['Extremely dissatisfied', 'Moderately dissatisfied', 'Slightly dissatisfied']).astype(int)

    df_concat['CareerSatisfaction'] = df_concat['CareerSatisfaction'].apply(lambda x: dict_CareerSatisfaction[x]).astype(int)



    # HopeFiveYears

    df_concat['HopeFiveYears_Retirement'] = (df_concat['HopeFiveYears'] == 'Retirement').astype(int)



    # JobSearchStatus

    dict_JobSearchStatus = {

        'I am actively looking for a job': 2,

        'I’m not actively looking, but I am open to new opportunities': 1,

        'I am not interested in new job opportunities': 0, 

        'MISSING': -1

    }

    df_concat['JobSearchStatus'] = df_concat['JobSearchStatus'].fillna('MISSING')

    df_concat['JobSearchStatus'] = df_concat['JobSearchStatus'].apply(lambda x: dict_JobSearchStatus[x]).astype(int)



    # LastNewJob

    dict_LastNewJob = {

        'More than 4 years ago': 4, 'Between 2 and 4 years ago': 3, 

        'Between 1 and 2 years ago': 2, 'Less than a year ago': 1, "I've never had a job": 0,

        'MISSING': -1

    }

    df_concat['LastNewJob'] = df_concat['LastNewJob'].fillna('MISSING')

    df_concat['LastNewJob'] = df_concat['LastNewJob'].apply(lambda x: dict_LastNewJob[x]).astype(int)



    # CommunicationTools

    df_concat['CommunicationTools'] = df_concat['CommunicationTools'].fillna('MISSING')

    val_list = [

        'HipChat', 'Facebook', 'Slack', 'Jira', 'Other wiki tool (Github, Google Sites, proprietary software, etc.)', 

        'Office / productivity suite (Microsoft Office, Google Suite, etc.)', 'Google Hangouts/Chat', 'Confluence', 

        'Stack Overflow Enterprise', 'Trello', 'Other chat system (IRC, proprietary software, etc.)',

        'MISSING'

    ]

    arr_one_hot = [[0]*len(val_list) for i in range(len(df_concat))]

    for i, vals in enumerate(df_concat['CommunicationTools']):

        for v in vals.split(';'):

            arr_one_hot[i][val_list.index(v)] = 1

    df_vals = pd.DataFrame(columns=[f'CommunicationTools_{i}' if v != 'MISSING' else 'CommunicationTools_MISSING' for i, v in enumerate(val_list)], data=arr_one_hot)

    df_concat = pd.concat([df_concat, df_vals], axis=1).drop(columns=['CommunicationTools'])

    del df_vals



    # TimeFullyProductive

    dict_TimeFullyProductive = {

        'More than a year': 12, 'Nine months to a year': 9, 'Six to nine months': 6, 

        'Three to six months': 3, 'One to three months': 1, 'Less than a month': 0,

        'MISSING': -1

    }

    df_concat['TimeFullyProductive'] = df_concat['TimeFullyProductive'].fillna('MISSING')

    df_concat['TimeFullyProductive'] = df_concat['TimeFullyProductive'].apply(lambda x: dict_TimeFullyProductive[x]).astype(int)



    # AgreeDisagree

    dict_AgreeDisagree = {

        'Strongly agree': 4,'Agree': 3, 

        'Neither Agree nor Disagree': 2,

        'Disagree': 1, 'Strongly disagree': 0, 

        'MISSING': -1

    }

    df_concat['AgreeDisagree1'] = df_concat['AgreeDisagree1'].fillna('MISSING')

    df_concat['AgreeDisagree2'] = df_concat['AgreeDisagree2'].fillna('MISSING')

    df_concat['AgreeDisagree3'] = df_concat['AgreeDisagree3'].fillna('MISSING')

    df_concat['AgreeDisagree1'] = df_concat['AgreeDisagree1'].apply(lambda x: dict_AgreeDisagree[x]).astype(int)

    df_concat['AgreeDisagree2'] = df_concat['AgreeDisagree2'].apply(lambda x: dict_AgreeDisagree[x]).astype(int)

    df_concat['AgreeDisagree3'] = df_concat['AgreeDisagree3'].apply(lambda x: dict_AgreeDisagree[x]).astype(int)



    # FrameworkWorkedWith

    df_concat['FrameworkWorkedWith'] = df_concat['FrameworkWorkedWith'].fillna('MISSING')

    val_list = [

        'React', 'Xamarin', 'Node.js', 'Torch/PyTorch', 'Hadoop', 'Cordova', 

        'Django', '.NET Core', 'TensorFlow', 'Spark', 'Angular', 'Spring', 

        'MISSING'

    ]

    arr_one_hot = [[0]*len(val_list) for i in range(len(df_concat))]

    for i, vals in enumerate(df_concat['FrameworkWorkedWith']):

        for v in vals.split(';'):

            arr_one_hot[i][val_list.index(v)] = 1

    df_vals = pd.DataFrame(columns=[f'FrameworkWorkedWith_{i}' if v != 'MISSING' else 'FrameworkWorkedWith_MISSING' for i, v in enumerate(val_list)], data=arr_one_hot)

    df_concat = pd.concat([df_concat, df_vals], axis=1).drop(columns=['FrameworkWorkedWith'])

    del df_vals



    # NumberMonitors

    dict_NumberMonitors = {

        'More than 4': 5, '4': 4, '3': 3, '2': 2, '1': 1, 

        'MISSING': -1

    }

    df_concat['NumberMonitors'] = df_concat['NumberMonitors'].fillna('MISSING')

    df_concat['NumberMonitors'] = df_concat['NumberMonitors'].apply(lambda x: dict_NumberMonitors[x]).astype(int)



    # CheckInCode

    dict_CheckInCode = {

        'Multiple times per day': 5, 

        'Once a day': 4, 

        'A few times per week': 3, 

        'Weekly or a few times per month': 2,

        'Less than once per month': 1,

        'Never': 0, 

        'MISSING': -1

    }

    df_concat['CheckInCode'] = df_concat['CheckInCode'].fillna('MISSING')

    df_concat['CheckInCode'] = df_concat['CheckInCode'].apply(lambda x: dict_CheckInCode[x]).astype(int)



    # AdsAgreeDisagree 

    dict_AdsAgreeDisagree = {

        'Strongly agree': 4, 'Somewhat agree': 3,

        'Neither agree nor disagree': 2,

        'Somewhat disagree': 1, 'Strongly disagree': 0,

        'MISSING': -1

    }

    df_concat['AdsAgreeDisagree1'] = df_concat['AdsAgreeDisagree1'].fillna('MISSING')

    df_concat['AdsAgreeDisagree2'] = df_concat['AdsAgreeDisagree2'].fillna('MISSING')

    df_concat['AdsAgreeDisagree3'] = df_concat['AdsAgreeDisagree3'].fillna('MISSING')

    df_concat['AdsAgreeDisagree1'] = df_concat['AdsAgreeDisagree1'].apply(lambda x: dict_AdsAgreeDisagree[x]).astype(int)

    df_concat['AdsAgreeDisagree2'] = df_concat['AdsAgreeDisagree2'].apply(lambda x: dict_AdsAgreeDisagree[x]).astype(int)

    df_concat['AdsAgreeDisagree3'] = df_concat['AdsAgreeDisagree3'].apply(lambda x: dict_AdsAgreeDisagree[x]).astype(int)



    # AdsActions

    df_concat['AdsActions'] = df_concat['AdsActions'].fillna('MISSING')

    val_list = [

        'Saw an online advertisement and then researched it (without clicking on the ad)', 

        'Clicked on an online advertisement', 'Paid to access a website advertisement-free', 

        'Stopped going to a website because of their advertising', 

        'MISSING'

    ]

    arr_one_hot = [[0]*len(val_list) for i in range(len(df_concat))]

    for i, vals in enumerate(df_concat['AdsActions']):

        for v in vals.split(';'):

            arr_one_hot[i][val_list.index(v)] = 1

    df_vals = pd.DataFrame(columns=[f'AdsActions_{i}' if v != 'MISSING' else 'AdsActions_MISSING' for i, v in enumerate(val_list)], data=arr_one_hot)

    df_concat = pd.concat([df_concat, df_vals], axis=1).drop(columns=['AdsActions'])

    del df_vals



    # StackOverflowRecommend

    dict_StackOverflowRecommend = {

        '10 (Very Likely)': 10, '9': 9, '8': 8, '7': 7, '6': 6, 

        '5': 5, '4': 4, '3': 3, '2': 2, '1': 1, '0 (Not Likely)': 0, 

        'MISSING': -1

    }

    df_concat['StackOverflowRecommend'] = df_concat['StackOverflowRecommend'].fillna('MISSING')

    df_concat['StackOverflowRecommend'] = df_concat['StackOverflowRecommend'].apply(lambda x: dict_StackOverflowRecommend[x]).astype(int)



    # StackOverflowVisit

    dict_StackOverflowVisit = {

        'Multiple times per day': 5, 

        'Daily or almost daily': 4,

        'A few times per week': 3,

        'A few times per month or weekly': 2, 

        'Less than once per month or monthly': 1,

        'I have never visited Stack Overflow (before today)': 0, 

        'MISSING': -1

    }

    df_concat['StackOverflowVisit'] = df_concat['StackOverflowVisit'].fillna('MISSING')

    df_concat['StackOverflowVisit'] = df_concat['StackOverflowVisit'].apply(lambda x: dict_StackOverflowVisit[x]).astype(int)



    # StackOverflowParticipate 

    dict_StackOverflowParticipate = {

        'Multiple times per day': 5,

        'Daily or almost daily': 4,

        'A few times per week': 3, 

        'A few times per month or weekly': 2,

        'Less than once per month or monthly': 1,

        'I have never participated in Q&A on Stack Overflow': 0,

        'MISSING': -1

    }

    df_concat['StackOverflowParticipate'] = df_concat['StackOverflowParticipate'].fillna('MISSING')

    df_concat['StackOverflowParticipate'] = df_concat['StackOverflowParticipate'].apply(lambda x: dict_StackOverflowParticipate[x]).astype(int)



    # StackOverflowJobsRecommend

    dict_StackOverflowJobsRecommend = {

        '10 (Very Likely)': 10, '9': 9, '8': 8, '7': 7, '6': 6, 

        '5': 5, '4': 4, '3': 3, '2': 2, '1': 1, '0 (Not Likely)': 0, 

        'MISSING': -1

    }

    df_concat['StackOverflowJobsRecommend'] = df_concat['StackOverflowJobsRecommend'].fillna('MISSING')

    df_concat['StackOverflowJobsRecommend'] = df_concat['StackOverflowJobsRecommend'].apply(lambda x: dict_StackOverflowJobsRecommend[x]).astype(int)



    # HypotheticalTools

    dict_HypotheticalTools = {

        'Extremely interested': 4,

        'Very interested': 3,

        'Somewhat interested': 2,

        'A little bit interested': 1,

        'Not at all interested': 0,

        'MISSING': -1

    }

    for i in [1, 2, 3, 4, 5]:

        df_concat[f'HypotheticalTools{i}'] = df_concat[f'HypotheticalTools{i}'].fillna('MISSING')

        df_concat[f'HypotheticalTools{i}'] = df_concat[f'HypotheticalTools{i}'].apply(lambda x: dict_HypotheticalTools[x]).astype(int)



    # HoursComputer

    dict_HoursComputer = {

        'Over 12 hours': 12, '9 - 12 hours': 9, '5 - 8 hours': 5, 

        '1 - 4 hours': 1, 'Less than 1 hour': 0,

        'MISSING': -1

    }

    df_concat['HoursComputer'] = df_concat['HoursComputer'].fillna('MISSING')

    df_concat['HoursComputer'] = df_concat['HoursComputer'].apply(lambda x: dict_HoursComputer[x]).astype(int)



    # HoursOutside

    dict_HoursOutside = {

        'Over 4 hours': 240, '3 - 4 hours': 180, 

        '1 - 2 hours': 60, '30 - 59 minutes': 30, 

        'Less than 30 minutes': 30,

        'MISSING': -1

    }

    df_concat['HoursOutside'] = df_concat['HoursOutside'].fillna('MISSING')

    df_concat['HoursOutside'] = df_concat['HoursOutside'].apply(lambda x: dict_HoursOutside[x]).astype(int)



    # SkipMeals

    dict_SkipMeals = {

        'Daily or almost every day': 7, '3 - 4 times per week': 3, '1 - 2 times per week': 1, 'Never': 0,

        'MISSING': -1

    }

    df_concat['SkipMeals'] = df_concat['SkipMeals'].fillna('MISSING')

    df_concat['SkipMeals'] = df_concat['SkipMeals'].apply(lambda x: dict_SkipMeals[x]).astype(int)



    # ErgonomicDevices

    df_concat['ErgonomicDevices'] = df_concat['ErgonomicDevices'].fillna('MISSING')

    val_list = [

        'Standing desk', 'Fatigue-relieving floor mat', 'Wrist/hand supports or braces', 'Ergonomic keyboard or mouse', 

        'MISSING'

    ]

    arr_one_hot = [[0]*len(val_list) for i in range(len(df_concat))]

    for i, vals in enumerate(df_concat['ErgonomicDevices']):

        for v in vals.split(';'):

            arr_one_hot[i][val_list.index(v)] = 1

    df_vals = pd.DataFrame(columns=[f'ErgonomicDevices_{i}' if v != 'MISSING' else 'ErgonomicDevices_MISSING' for i, v in enumerate(val_list)], data=arr_one_hot)

    df_concat = pd.concat([df_concat, df_vals], axis=1).drop(columns=['ErgonomicDevices'])

    del df_vals



    # Exercise

    dict_Exercise = {

        'Daily or almost every day': 7, '3 - 4 times per week': 3, '1 - 2 times per week': 1, "I don't typically exercise": 0, 

        'MISSING': -1

    }

    df_concat['Exercise'] = df_concat['Exercise'].fillna('MISSING')

    df_concat['Exercise'] = df_concat['Exercise'].apply(lambda x: dict_Exercise[x]).astype(int)



    # Gender

    df_concat['Gender'] = df_concat['Gender'].fillna('MISSING')

    val_list = [

        'Non-binary, genderqueer, or gender non-conforming', 'Female', 'Transgender', 'Male', 

        'MISSING'

    ]

    arr_one_hot = [[0]*len(val_list) for i in range(len(df_concat))]

    for i, vals in enumerate(df_concat['Gender']):

        for v in vals.split(';'):

            arr_one_hot[i][val_list.index(v)] = 1

    df_vals = pd.DataFrame(columns=[f'Gender_{i}' if v != 'MISSING' else 'Gender_MISSING' for i, v in enumerate(val_list)], data=arr_one_hot)

    df_concat = pd.concat([df_concat, df_vals], axis=1).drop(columns=['Gender'])

    del df_vals



    # SexualOrientation

    df_concat['SexualOrientation'] = df_concat['SexualOrientation'].fillna('MISSING')

    val_list = [

        'Straight or heterosexual', 'Gay or Lesbian', 'Asexual', 'Bisexual or Queer', 

        'MISSING'

    ]

    arr_one_hot = [[0]*len(val_list) for i in range(len(df_concat))]

    for i, vals in enumerate(df_concat['SexualOrientation']):

        for v in vals.split(';'):

            arr_one_hot[i][val_list.index(v)] = 1

    df_vals = pd.DataFrame(columns=[f'SexualOrientation_{i}' if v != 'MISSING' else 'SexualOrientation_MISSING' for i, v in enumerate(val_list)], data=arr_one_hot)

    df_concat = pd.concat([df_concat, df_vals], axis=1).drop(columns=['SexualOrientation'])

    del df_vals



    # EducationParents

    dict_EducationParents = {

        'Professional degree (JD, MD, etc.)': 6,

        'Other doctoral degree (Ph.D, Ed.D., etc.)': 6, 

        'Master’s degree (MA, MS, M.Eng., MBA, etc.)': 5, 

        'Bachelor’s degree (BA, BS, B.Eng., etc.)': 4,

        'Associate degree': 3,

        'Some college/university study without earning a degree': 2,

        'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 1,

        'Primary/elementary school': 1,

        'They never completed any formal education': 0, 

        'MISSING': -1

    }

    df_concat['EducationParents'] = df_concat['EducationParents'].fillna('MISSING')

    df_concat['EducationParents_WithDegree'] = df_concat['EducationParents'].isin([

        'Professional degree (JD, MD, etc.)', 'Other doctoral degree (Ph.D, Ed.D., etc.)', 

        'Master’s degree (MA, MS, M.Eng., MBA, etc.)', 'Bachelor’s degree (BA, BS, B.Eng., etc.)',

        'Associate degree'

    ]).astype(int)

    df_concat['EducationParents'] = df_concat['EducationParents'].apply(lambda x: dict_EducationParents[x]).astype(int)



    # RaceEthnicity

    df_concat['RaceEthnicity'] = df_concat['RaceEthnicity'].fillna('MISSING')

    val_list = [

        'South Asian', 'East Asian', 'Middle Eastern', 'Native American, Pacific Islander, or Indigenous Australian',

        'Hispanic or Latino/Latina', 'Black or of African descent', 'White or of European descent', 

        'MISSING'

    ]

    arr_one_hot = [[0]*len(val_list) for i in range(len(df_concat))]

    for i, vals in enumerate(df_concat['RaceEthnicity']):

        for v in vals.split(';'):

            arr_one_hot[i][val_list.index(v)] = 1

    df_vals = pd.DataFrame(columns=[f'RaceEthnicity_{i}' if v != 'MISSING' else 'RaceEthnicity_MISSING' for i, v in enumerate(val_list)], data=arr_one_hot)

    df_concat = pd.concat([df_concat, df_vals], axis=1).drop(columns=['RaceEthnicity'])

    del df_vals



    # Age

    dict_Age = {

        '65 years or older': 65, '55 - 64 years old': 55, '45 - 54 years old': 45, 

        '35 - 44 years old': 35, '25 - 34 years old': 25, '18 - 24 years old': 18, 'Under 18 years old': 0, 

        'MISSING': -1

    }

    df_concat['Age'] = df_concat['Age'].fillna('MISSING')

    df_concat['Age'] = df_concat['Age'].apply(lambda x: dict_Age[x]).astype(int)



    # SurveyTooLong

    dict_SurveyTooLong = {

        'The survey was too long': 2,

        'The survey was an appropriate length': 1, 

        'The survey was too short': 0, 

        'MISSING': -1

    }

    df_concat['SurveyTooLong'] = df_concat['SurveyTooLong'].fillna('MISSING')

    df_concat['SurveyTooLong'] = df_concat['SurveyTooLong'].apply(lambda x: dict_SurveyTooLong[x]).astype(int)



    # SurveyEasy

    dict_SurveyEasy = {

        'Very easy': 4, 

        'Somewhat easy': 3, 

        'Neither easy nor difficult': 2,

        'Somewhat difficult': 1, 

        'Very difficult': 0,

        'MISSING': -1

    }

    df_concat['SurveyEasy'] = df_concat['SurveyEasy'].fillna('MISSING')

    df_concat['SurveyEasy'] = df_concat['SurveyEasy'].apply(lambda x: dict_SurveyEasy[x]).astype(int)

    

    # cache_nameが指定されている場合はキャッシュを残す

    if cache_name is not None: 

        caches[cache_name] = df_concat.copy()

        

    return df_concat
# Count Encoding

@logger

def count_encoding(df_concat, cols, drop_original=False):

    # Encoding

    encoder = ce.CountEncoder()

    df_encoded = encoder.fit_transform(df_concat[cols]).add_prefix('COUNT:')

    df_concat = pd.concat([df_concat, df_encoded], axis=1)

    

    # Drop Original Columns

    if drop_original:

        df_concat = df_concat.drop(columns=cols)

        

    return df_concat
# Target Encoding

@logger

def target_encoding(df_concat, cols, drop_original=False):

    df_train = df_concat[df_concat['is_test'] != 1]

    df_test = df_concat[df_concat['is_test'] == 1]

    

    # Encoding

    encoder = ce.TargetEncoder(cols=cols, handle_unknown='ignore')

    encoder.fit(df_train[cols], df_train[COL_TARGET])

    arr_encoded_train = encoder.transform(df_train[cols]).values

    arr_encoded_test = encoder.transform(df_test[cols]).values

    

    # Merge Results

    df_concat.loc[df_concat['is_test'] != 1, [f'TARGET:{col}' for col in cols]] = arr_encoded_train

    df_concat.loc[df_concat['is_test'] == 1, [f'TARGET:{col}' for col in cols]] = arr_encoded_test

    

    # Drop Original Columns

    if drop_original:

        df_concat = df_concat.drop(columns=cols)

    

    return df_concat
# Holdout

@logger 

def split_holdout(df_concat, rate=0.2):

    ix_train = df_concat[df_concat['is_test'] != 1].index.tolist()

    

    # Define Holdout Indices

    ix_holdout = np.random.choice(ix_train, int(len(ix_train)*rate), replace=False)

    ix_concat = [ix for ix in df_concat.index.tolist() if ix not in ix_holdout]

    

    # Split DataFrame

    df_holdout = df_concat.iloc[ix_holdout].copy().reset_index(drop=True)

    df_concat = df_concat.iloc[ix_concat].copy().reset_index(drop=True)

    

    return df_concat, df_holdout
ver = '1_2'

HOLDOUT = False



N_FOLDS = 5

params = {'objective': 'regression', 'boosting_type': 'gbdt', 'metric': 'rmse'}
### Import Data

df_concat = get_df_concat(cache_name='df_concat')

df_concat['n_nan'] = df_concat.isnull().sum(axis=1) - df_concat['is_test']

COL_TARGET = 'ConvertedSalary'

COL_ID = 'Respondent'





### Cleaning

df_concat[COL_TARGET] = df_concat[COL_TARGET].apply(np.log1p)

df_concat = baseline_fe_for_tree_model(df_concat, cache_name='Baseline_FE')





### Encoding

cols_unnecessary = [COL_ID, COL_TARGET, 'is_test'] + ['Country', 'Currency', 'CurrencySymbol', 'MilitaryUS']



cols_qualitative = [col for col, dtype in zip(df_concat.dtypes.index, df_concat.dtypes.values) if ((dtype == 'object') and (col not in cols_unnecessary))]

cols_quantitative = [col for col, dtype in zip(df_concat.dtypes.index, df_concat.dtypes.values) if ((dtype != 'object') and (col not in cols_unnecessary))]

df_concat[cols_qualitative] = df_concat[cols_qualitative].fillna('MISSING')

df_concat[cols_quantitative] = df_concat[cols_quantitative].fillna(-1)



df_concat = count_encoding(df_concat, cols=cols_qualitative)

df_concat = target_encoding(df_concat, cols=cols_qualitative, drop_original=True)





### Modeling



# Split Holdout Data

df_holdout = None

if HOLDOUT:

    df_concat, df_holdout = split_holdout(df_concat, rate=0.2)



# Cross Validation

df_train = df_concat[df_concat['is_test'] != 1]

df_test = df_concat[df_concat['is_test'] == 1]



y_train = df_train[COL_TARGET].values

X_train = df_train.drop(columns=cols_unnecessary).values

X_test = df_test.drop(columns=cols_unnecessary).values



cv_preds = [None]*N_FOLDS

cv_scores = [None]*N_FOLDS

skf = KFold(n_splits=N_FOLDS, random_state=71, shuffle=True)

for i, (ix_train, ix_valid) in enumerate(skf.split(X_train, y_train)):

    _X_train, _y_train = X_train[ix_train], y_train[ix_train] 

    _X_valid, _y_valid = X_train[ix_valid], y_train[ix_valid]

    

    print(f'---------- Fold {i+1}/{N_FOLDS} ----------')

    

    # Define Validation

    _set_train = lgb.Dataset(_X_train, label=_y_train)

    _set_valid = lgb.Dataset(_X_valid, label=_y_valid)

    

    # Train Model

    _clf = lgb.train(

        params, 

        train_set=_set_train,

        valid_sets=[_set_train, _set_valid], 

        verbose_eval=100, 

        early_stopping_rounds=1000

    )

 

    # Prediction

    _y_pred = _clf.predict(_X_valid) 

    _score = np.sqrt(mean_squared_error(_y_valid, _y_pred))

    cv_scores[i] = _score

    cv_preds[i] = _clf.predict(X_test)

    print(f' > RMSE: {_score:.5f}')



print(f'---------- Average of CV ----------')

print(f' > RMSLE: {np.mean(cv_scores):.5f}')



# Holdout Validation

if HOLDOUT:

    y_holdout = df_holdout[COL_TARGET].values

    X_holdout = df_holdout.drop(columns=cols_unnecessary).values

    print(f'---------- Holdout Validation ----------')

    set_train = lgb.Dataset(X_train, label=y_train)

    clf = lgb.train(

        params, 

        train_set=set_train,

        valid_sets=[set_train, set_train], 

        verbose_eval=100, 

        early_stopping_rounds=1000

    ) 

    y_pred = clf.predict(X_holdout)

    score = np.sqrt(mean_squared_error(y_holdout, y_pred))

    print(f' > RMSE: {score:.5f}')

    

# CV Ensemble

cv_preds = [[np.exp(p)-1 for p in pred] for pred in cv_preds]

y_pred = np.mean(cv_preds, axis=0)

y_pred = [int(p) for p in y_pred]



# Submission

now = dt.datetime.now().strftime('%Y%m%d')

path = f'{now}_submission_ver{ver}_lgbm.csv'

df_submission = pd.read_csv(f'{DIR}/sample_submission.csv')

df_submission[COL_TARGET] = y_pred

df_submission.to_csv(path, index=False)

print(f' > saved: {path}')
print(df_concat.columns.tolist())
print([col for col in df_concat.columns if 'AdsPriorities' in col])

cols_score = [

    'AssessJob1', 'AssessJob2', 'AssessJob3', 'AssessJob4', 'AssessJob5', 

    'AssessJob6', 'AssessJob7', 'AssessJob8', 'AssessJob9', 'AssessJob10',

    'AssessBenefits1', 'AssessBenefits2', 'AssessBenefits3', 'AssessBenefits4', 

    'AssessBenefits5', 'AssessBenefits6', 'AssessBenefits7', 'AssessBenefits8', 

    'AssessBenefits9', 'AssessBenefits10', 'AssessBenefits11',

    'JobContactPriorities1', 'JobContactPriorities2', 'JobContactPriorities3', 

    'JobContactPriorities4', 'JobContactPriorities5',

    'AdsPriorities1', 'AdsPriorities2', 'AdsPriorities3', 'AdsPriorities4', 

    'AdsPriorities5', 'AdsPriorities6', 'AdsPriorities7'

]
# EDA



# Country：使えない

# Currency：なんだこれ？テストだけ分布が違う

# CurrencySymbol：これもわからん

# MilitaryUS：これも調べる必要あり



def hist_train_vs_test(feature,bins,clip = False):

    

    train = df_concat[df_concat['is_test'] != 1]

    test = df_concat[df_concat['is_test'] == 1]

    

    if train[feature].nunique() > 1000:

        print('Toooooooo MANYYYyyyyyyy')

        return 

    

    plt.figure(figsize=(16, 8))

    if clip:

        th_train = np.percentile(train[feature], 99)

        th_test = np.percentile(test[feature], 99)

        plt.hist(x=[train[train[feature]<th_train][feature], test[test[feature]<th_test][feature]])

    else:

        plt.hist(x=[train[feature].dropna(), test[feature].dropna()])

    plt.legend(['train', 'test'])

    plt.show()

    

#for col in df_concat.columns:

#    print(col)

#    hist_train_vs_test(col, 10)