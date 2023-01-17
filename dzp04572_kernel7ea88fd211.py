# 卒業試験

import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

import gc

import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm_notebook as tqdm

import lightgbm as lgb

from lightgbm import LGBMClassifier
df_train = pd.read_csv('../input/exam-for-students20200129/train.csv')

df_test = pd.read_csv('../input/exam-for-students20200129/test.csv')

gc.collect()
pd.set_option('display.max_columns', 50)

# df_train
# 欠損補完

# まずはテキストを"#"にする

df_train = df_train.fillna({'Student':'#'})

df_train = df_train.fillna({'Employment':'#'})

df_train = df_train.fillna({'FormalEducation':'#'})

df_train = df_train.fillna({'UndergradMajor':'#'})

df_train = df_train.fillna({'DevType':'#'})

df_train = df_train.fillna({'JobSatisfaction':'#'})

df_train = df_train.fillna({'CareerSatisfaction':'#'})

df_train = df_train.fillna({'HopeFiveYears':'#'})

df_train = df_train.fillna({'JobSearchStatus':'#'})

df_train = df_train.fillna({'UpdateCV':'#'})

df_train = df_train.fillna({'SalaryType':'#'})

df_train = df_train.fillna({'CommunicationTools':'#'})

df_train = df_train.fillna({'TimeAfterBootcamp':'#'})

df_train = df_train.fillna({'AgreeDisagree1':'#'})

df_train = df_train.fillna({'AgreeDisagree2':'#'})

df_train = df_train.fillna({'AgreeDisagree3':'#'})

df_train = df_train.fillna({'FrameworkWorkedWith':'#'})

df_train = df_train.fillna({'OperatingSystem':'#'})

df_train = df_train.fillna({'CheckInCode':'#'})

df_train = df_train.fillna({'AdsAgreeDisagree1':'#'})

df_train = df_train.fillna({'AdsAgreeDisagree2':'#'})

df_train = df_train.fillna({'AdsAgreeDisagree3':'#'})

df_train = df_train.fillna({'AdsActions':'#'})

df_train = df_train.fillna({'AIDangerous':'#'})

df_train = df_train.fillna({'AIInteresting':'#'})

df_train = df_train.fillna({'AIResponsible':'#'})

df_train = df_train.fillna({'AIFuture':'#'})

df_train = df_train.fillna({'EthicsResponsible':'#'})

df_train = df_train.fillna({'StackOverflowVisit':'#'})

df_train = df_train.fillna({'StackOverflowParticipate':'#'})

df_train = df_train.fillna({'HypotheticalTools1':'#'})

df_train = df_train.fillna({'HypotheticalTools2':'#'})

df_train = df_train.fillna({'HypotheticalTools3':'#'})

df_train = df_train.fillna({'HypotheticalTools4':'#'})

df_train = df_train.fillna({'HypotheticalTools5':'#'})

df_train = df_train.fillna({'ErgonomicDevices':'#'})

df_train = df_train.fillna({'Gender':'#'})

df_train = df_train.fillna({'SexualOrientation':'#'})

df_train = df_train.fillna({'EducationParents':'#'})

df_train = df_train.fillna({'RaceEthnicity':'#'})

df_train = df_train.fillna({'Dependents':'#'})

df_train = df_train.fillna({'MilitaryUS':'#'})

df_train = df_train.fillna({'SurveyTooLong':'#'})

df_train = df_train.fillna({'SurveyEasy':'#'})



df_test = df_test.fillna({'Student':'#'})

df_test = df_test.fillna({'Employment':'#'})

df_test = df_test.fillna({'FormalEducation':'#'})

df_test = df_test.fillna({'UndergradMajor':'#'})

df_test = df_test.fillna({'DevType':'#'})

df_test = df_test.fillna({'JobSatisfaction':'#'})

df_test = df_test.fillna({'CareerSatisfaction':'#'})

df_test = df_test.fillna({'HopeFiveYears':'#'})

df_test = df_test.fillna({'JobSearchStatus':'#'})

df_test = df_test.fillna({'UpdateCV':'#'})

df_test = df_test.fillna({'SalaryType':'#'})

df_test = df_test.fillna({'CommunicationTools':'#'})

df_test = df_test.fillna({'TimeAfterBootcamp':'#'})

df_test = df_test.fillna({'AgreeDisagree1':'#'})

df_test = df_test.fillna({'AgreeDisagree2':'#'})

df_test = df_test.fillna({'AgreeDisagree3':'#'})

df_test = df_test.fillna({'FrameworkWorkedWith':'#'})

df_test = df_test.fillna({'OperatingSystem':'#'})

df_test = df_test.fillna({'CheckInCode':'#'})

df_test = df_test.fillna({'AdsAgreeDisagree1':'#'})

df_test = df_test.fillna({'AdsAgreeDisagree2':'#'})

df_test = df_test.fillna({'AdsAgreeDisagree3':'#'})

df_test = df_test.fillna({'AdsActions':'#'})

df_test = df_test.fillna({'AIDangerous':'#'})

df_test = df_test.fillna({'AIInteresting':'#'})

df_test = df_test.fillna({'AIResponsible':'#'})

df_test = df_test.fillna({'AIFuture':'#'})

df_test = df_test.fillna({'EthicsResponsible':'#'})

df_test = df_test.fillna({'StackOverflowVisit':'#'})

df_test = df_test.fillna({'StackOverflowParticipate':'#'})

df_test = df_test.fillna({'HypotheticalTools1':'#'})

df_test = df_test.fillna({'HypotheticalTools2':'#'})

df_test = df_test.fillna({'HypotheticalTools3':'#'})

df_test = df_test.fillna({'HypotheticalTools4':'#'})

df_test = df_test.fillna({'HypotheticalTools5':'#'})

df_test = df_test.fillna({'ErgonomicDevices':'#'})

df_test = df_test.fillna({'Gender':'#'})

df_test = df_test.fillna({'SexualOrientation':'#'})

df_test = df_test.fillna({'EducationParents':'#'})

df_test = df_test.fillna({'RaceEthnicity':'#'})

df_test = df_test.fillna({'Dependents':'#'})

df_test = df_test.fillna({'MilitaryUS':'#'})

df_test = df_test.fillna({'SurveyTooLong':'#'})

df_test = df_test.fillna({'SurveyEasy':'#'})
# 欠損補完

# 次に#以外の固定文字で埋める

df_train = df_train.fillna({'AdBlocker':'I\'m not sure/I don\'t know'})

df_train = df_train.fillna({'AdBlockerDisable':'I\'m not sure/I can\'t remember'})

df_train = df_train.fillna({'EthicsChoice':'Depends on what it is'})

df_train = df_train.fillna({'EthicsReport':'Depends on what it is'})

df_train = df_train.fillna({'EthicalImplications':'Unsure / I don\'t know'})

df_train = df_train.fillna({'StackOverflowHasAccount':'I\'m not sure / I can\'t remember'})

df_train = df_train.fillna({'StackOverflowJobs':'No, I #'})

df_train = df_train.fillna({'StackOverflowDevStory':'No, I #'})

df_train = df_train.fillna({'StackOverflowConsiderMember':'I\'m not sure'})

df_train = df_train.fillna({'WakeTime':'I do not have a set schedule'})

df_train = df_train.fillna({'SkipMeals':'Never'})

df_train = df_train.fillna({'Exercise':'I don\'t typically exercise'})



df_test = df_test.fillna({'AdBlocker':'I\'m not sure/I don\'t know'})

df_test = df_test.fillna({'AdBlockerDisable':'I\'m not sure/I can\'t remember'})

df_test = df_test.fillna({'EthicsChoice':'Depends on what it is'})

df_test = df_test.fillna({'EthicsReport':'Depends on what it is'})

df_test = df_test.fillna({'EthicalImplications':'Unsure / I don\'t know'})

df_test = df_test.fillna({'StackOverflowHasAccount':'I\'m not sure / I can\'t remember'})

df_test = df_test.fillna({'StackOverflowJobs':'No, I #'})

df_test = df_test.fillna({'StackOverflowDevStory':'No, I #'})

df_test = df_test.fillna({'StackOverflowConsiderMember':'I\'m not sure'})

df_test = df_test.fillna({'WakeTime':'I do not have a set schedule'})

df_test = df_test.fillna({'SkipMeals':'Never'})

df_test = df_test.fillna({'Exercise':'I don\'t typically exercise'})
# 欠損補完

# 次はmean補完

df_train = df_train.fillna({'AssessJob1':df_train['AssessJob1'].mean()})

df_train = df_train.fillna({'AssessJob2':df_train['AssessJob2'].mean()})

df_train = df_train.fillna({'AssessJob3':df_train['AssessJob3'].mean()})

df_train = df_train.fillna({'AssessJob4':df_train['AssessJob4'].mean()})

df_train = df_train.fillna({'AssessJob5':df_train['AssessJob5'].mean()})

df_train = df_train.fillna({'AssessJob6':df_train['AssessJob6'].mean()})

df_train = df_train.fillna({'AssessJob7':df_train['AssessJob7'].mean()})

df_train = df_train.fillna({'AssessJob8':df_train['AssessJob8'].mean()})

df_train = df_train.fillna({'AssessJob9':df_train['AssessJob9'].mean()})

df_train = df_train.fillna({'AssessJob10':df_train['AssessJob10'].mean()})

df_train = df_train.fillna({'AssessBenefits1':df_train['AssessBenefits1'].mean()})

df_train = df_train.fillna({'AssessBenefits2':df_train['AssessBenefits2'].mean()})

df_train = df_train.fillna({'AssessBenefits3':df_train['AssessBenefits3'].mean()})

df_train = df_train.fillna({'AssessBenefits4':df_train['AssessBenefits4'].mean()})

df_train = df_train.fillna({'AssessBenefits5':df_train['AssessBenefits5'].mean()})

df_train = df_train.fillna({'AssessBenefits6':df_train['AssessBenefits6'].mean()})

df_train = df_train.fillna({'AssessBenefits7':df_train['AssessBenefits7'].mean()})

df_train = df_train.fillna({'AssessBenefits8':df_train['AssessBenefits8'].mean()})

df_train = df_train.fillna({'AssessBenefits9':df_train['AssessBenefits9'].mean()})

df_train = df_train.fillna({'AssessBenefits10':df_train['AssessBenefits10'].mean()})

df_train = df_train.fillna({'AssessBenefits11':df_train['AssessBenefits11'].mean()})

df_train = df_train.fillna({'JobContactPriorities1':df_train['JobContactPriorities1'].mean()})

df_train = df_train.fillna({'JobContactPriorities2':df_train['JobContactPriorities2'].mean()})

df_train = df_train.fillna({'JobContactPriorities3':df_train['JobContactPriorities3'].mean()})

df_train = df_train.fillna({'JobContactPriorities4':df_train['JobContactPriorities4'].mean()})

df_train = df_train.fillna({'JobContactPriorities5':df_train['JobContactPriorities5'].mean()})

df_train = df_train.fillna({'JobEmailPriorities1':df_train['JobEmailPriorities1'].mean()})

df_train = df_train.fillna({'JobEmailPriorities2':df_train['JobEmailPriorities2'].mean()})

df_train = df_train.fillna({'JobEmailPriorities3':df_train['JobEmailPriorities3'].mean()})

df_train = df_train.fillna({'JobEmailPriorities4':df_train['JobEmailPriorities4'].mean()})

df_train = df_train.fillna({'JobEmailPriorities5':df_train['JobEmailPriorities5'].mean()})

df_train = df_train.fillna({'JobEmailPriorities6':df_train['JobEmailPriorities6'].mean()})

df_train = df_train.fillna({'JobEmailPriorities7':df_train['JobEmailPriorities7'].mean()})

df_train = df_train.fillna({'AdsPriorities1':df_train['AdsPriorities1'].mean()})

df_train = df_train.fillna({'AdsPriorities2':df_train['AdsPriorities2'].mean()})

df_train = df_train.fillna({'AdsPriorities3':df_train['AdsPriorities3'].mean()})

df_train = df_train.fillna({'AdsPriorities4':df_train['AdsPriorities4'].mean()})

df_train = df_train.fillna({'AdsPriorities5':df_train['AdsPriorities5'].mean()})

df_train = df_train.fillna({'AdsPriorities6':df_train['AdsPriorities6'].mean()})

df_train = df_train.fillna({'AdsPriorities7':df_train['AdsPriorities7'].mean()})



df_test = df_test.fillna({'AssessJob1':df_train['AssessJob1'].mean()})

df_test = df_test.fillna({'AssessJob2':df_train['AssessJob2'].mean()})

df_test = df_test.fillna({'AssessJob3':df_train['AssessJob3'].mean()})

df_test = df_test.fillna({'AssessJob4':df_train['AssessJob4'].mean()})

df_test = df_test.fillna({'AssessJob5':df_train['AssessJob5'].mean()})

df_test = df_test.fillna({'AssessJob6':df_train['AssessJob6'].mean()})

df_test = df_test.fillna({'AssessJob7':df_train['AssessJob7'].mean()})

df_test = df_test.fillna({'AssessJob8':df_train['AssessJob8'].mean()})

df_test = df_test.fillna({'AssessJob9':df_train['AssessJob9'].mean()})

df_test = df_test.fillna({'AssessJob10':df_train['AssessJob10'].mean()})

df_test = df_test.fillna({'AssessBenefits1':df_train['AssessBenefits1'].mean()})

df_test = df_test.fillna({'AssessBenefits2':df_train['AssessBenefits2'].mean()})

df_test = df_test.fillna({'AssessBenefits3':df_train['AssessBenefits3'].mean()})

df_test = df_test.fillna({'AssessBenefits4':df_train['AssessBenefits4'].mean()})

df_test = df_test.fillna({'AssessBenefits5':df_train['AssessBenefits5'].mean()})

df_test = df_test.fillna({'AssessBenefits6':df_train['AssessBenefits6'].mean()})

df_test = df_test.fillna({'AssessBenefits7':df_train['AssessBenefits7'].mean()})

df_test = df_test.fillna({'AssessBenefits8':df_train['AssessBenefits8'].mean()})

df_test = df_test.fillna({'AssessBenefits9':df_train['AssessBenefits9'].mean()})

df_test = df_test.fillna({'AssessBenefits10':df_train['AssessBenefits10'].mean()})

df_test = df_test.fillna({'AssessBenefits11':df_train['AssessBenefits11'].mean()})

df_test = df_test.fillna({'JobContactPriorities1':df_train['JobContactPriorities1'].mean()})

df_test = df_test.fillna({'JobContactPriorities2':df_train['JobContactPriorities2'].mean()})

df_test = df_test.fillna({'JobContactPriorities3':df_train['JobContactPriorities3'].mean()})

df_test = df_test.fillna({'JobContactPriorities4':df_train['JobContactPriorities4'].mean()})

df_test = df_test.fillna({'JobContactPriorities5':df_train['JobContactPriorities5'].mean()})

df_test = df_test.fillna({'JobEmailPriorities1':df_train['JobEmailPriorities1'].mean()})

df_test = df_test.fillna({'JobEmailPriorities2':df_train['JobEmailPriorities2'].mean()})

df_test = df_test.fillna({'JobEmailPriorities3':df_train['JobEmailPriorities3'].mean()})

df_test = df_test.fillna({'JobEmailPriorities4':df_train['JobEmailPriorities4'].mean()})

df_test = df_test.fillna({'JobEmailPriorities5':df_train['JobEmailPriorities5'].mean()})

df_test = df_test.fillna({'JobEmailPriorities6':df_train['JobEmailPriorities6'].mean()})

df_test = df_test.fillna({'JobEmailPriorities7':df_train['JobEmailPriorities7'].mean()})

df_test = df_test.fillna({'AdsPriorities1':df_train['AdsPriorities1'].mean()})

df_test = df_test.fillna({'AdsPriorities2':df_train['AdsPriorities2'].mean()})

df_test = df_test.fillna({'AdsPriorities3':df_train['AdsPriorities3'].mean()})

df_test = df_test.fillna({'AdsPriorities4':df_train['AdsPriorities4'].mean()})

df_test = df_test.fillna({'AdsPriorities5':df_train['AdsPriorities5'].mean()})

df_test = df_test.fillna({'AdsPriorities6':df_train['AdsPriorities6'].mean()})

df_test = df_test.fillna({'AdsPriorities7':df_train['AdsPriorities7'].mean()})
# 欠損補完

# CompanySize変換

df_train.loc[df_train['CompanySize']=='Fewer than 10 employees', 'CompanySize'] = '1'

df_train.loc[df_train['CompanySize']=='10 to 19 employees', 'CompanySize'] = '10'

df_train.loc[df_train['CompanySize']=='20 to 99 employees', 'CompanySize'] = '20'

df_train.loc[df_train['CompanySize']=='100 to 499 employees', 'CompanySize'] = '100'

df_train.loc[df_train['CompanySize']=='500 to 999 employees', 'CompanySize'] = '500'

df_train.loc[df_train['CompanySize']=='1,000 to 4,999 employees', 'CompanySize'] = '1000'

df_train.loc[df_train['CompanySize']=='5,000 to 9,999 employees', 'CompanySize'] = '5000'

df_train.loc[df_train['CompanySize']=='10,000 or more employees', 'CompanySize'] = '10000'

# df_train_CompanySize = df_train[df_train['CompanySize'].isnull()==False]

# df_train_CompanySize['CompanySize_int'] = df_train_CompanySize['CompanySize'].astype(int)

# print(df_train_CompanySize['CompanySize_int'].mean())

df_train = df_train.fillna({'CompanySize':'1811'})

df_train



df_test.loc[df_test['CompanySize']=='Fewer than 10 employees', 'CompanySize'] = '1'

df_test.loc[df_test['CompanySize']=='10 to 19 employees', 'CompanySize'] = '10'

df_test.loc[df_test['CompanySize']=='20 to 99 employees', 'CompanySize'] = '20'

df_test.loc[df_test['CompanySize']=='100 to 499 employees', 'CompanySize'] = '100'

df_test.loc[df_test['CompanySize']=='500 to 999 employees', 'CompanySize'] = '500'

df_test.loc[df_test['CompanySize']=='1,000 to 4,999 employees', 'CompanySize'] = '1000'

df_test.loc[df_test['CompanySize']=='5,000 to 9,999 employees', 'CompanySize'] = '5000'

df_test.loc[df_test['CompanySize']=='10,000 or more employees', 'CompanySize'] = '10000'

df_test = df_test.fillna({'CompanySize':'1811'})
# 欠損補完

# CompanySize変換

df_train.loc[df_train['YearsCoding']=='0-2 years', 'YearsCoding'] = '0'

df_train.loc[df_train['YearsCoding']=='3-5 years', 'YearsCoding'] = '3'

df_train.loc[df_train['YearsCoding']=='6-8 years', 'YearsCoding'] = '6'

df_train.loc[df_train['YearsCoding']=='9-11 years', 'YearsCoding'] = '9'

df_train.loc[df_train['YearsCoding']=='12-14 years', 'YearsCoding'] = '12'

df_train.loc[df_train['YearsCoding']=='15-17 years', 'YearsCoding'] = '15'

df_train.loc[df_train['YearsCoding']=='18-20 years', 'YearsCoding'] = '18'

df_train.loc[df_train['YearsCoding']=='21-23 years', 'YearsCoding'] = '21'

df_train.loc[df_train['YearsCoding']=='24-26 years', 'YearsCoding'] = '24'

df_train.loc[df_train['YearsCoding']=='27-29 years', 'YearsCoding'] = '27'

df_train.loc[df_train['YearsCoding']=='30 or more years', 'YearsCoding'] = '30'

# df_train_YearsCoding = df_train[df_train['YearsCoding'].isnull()==False]

# df_train_YearsCoding['YearsCoding_int'] = df_train_YearsCoding['YearsCoding'].astype(int)

# print(df_train_YearsCoding['YearsCoding_int'].mean())

df_train = df_train.fillna({'YearsCoding':'10'})

df_train



df_test.loc[df_test['YearsCoding']=='0-2 years', 'YearsCoding'] = '0'

df_test.loc[df_test['YearsCoding']=='3-5 years', 'YearsCoding'] = '3'

df_test.loc[df_test['YearsCoding']=='6-8 years', 'YearsCoding'] = '6'

df_test.loc[df_test['YearsCoding']=='9-11 years', 'YearsCoding'] = '9'

df_test.loc[df_test['YearsCoding']=='12-14 years', 'YearsCoding'] = '12'

df_test.loc[df_test['YearsCoding']=='15-17 years', 'YearsCoding'] = '15'

df_test.loc[df_test['YearsCoding']=='18-20 years', 'YearsCoding'] = '18'

df_test.loc[df_test['YearsCoding']=='21-23 years', 'YearsCoding'] = '21'

df_test.loc[df_test['YearsCoding']=='24-26 years', 'YearsCoding'] = '24'

df_test.loc[df_test['YearsCoding']=='27-29 years', 'YearsCoding'] = '27'

df_test.loc[df_test['YearsCoding']=='30 or more years', 'YearsCoding'] = '30'

df_test = df_test.fillna({'YearsCoding':'10'})
# 欠損補完

# YearsCodingProf変換

df_train.loc[df_train['YearsCodingProf']=='0-2 years', 'YearsCodingProf'] = '0'

df_train.loc[df_train['YearsCodingProf']=='3-5 years', 'YearsCodingProf'] = '3'

df_train.loc[df_train['YearsCodingProf']=='6-8 years', 'YearsCodingProf'] = '6'

df_train.loc[df_train['YearsCodingProf']=='9-11 years', 'YearsCodingProf'] = '9'

df_train.loc[df_train['YearsCodingProf']=='12-14 years', 'YearsCodingProf'] = '12'

df_train.loc[df_train['YearsCodingProf']=='15-17 years', 'YearsCodingProf'] = '15'

df_train.loc[df_train['YearsCodingProf']=='18-20 years', 'YearsCodingProf'] = '18'

df_train.loc[df_train['YearsCodingProf']=='21-23 years', 'YearsCodingProf'] = '21'

df_train.loc[df_train['YearsCodingProf']=='24-26 years', 'YearsCodingProf'] = '24'

df_train.loc[df_train['YearsCodingProf']=='27-29 years', 'YearsCodingProf'] = '27'

df_train.loc[df_train['YearsCodingProf']=='30 or more years', 'YearsCodingProf'] = '30'

# df_train_YearsCodingProf = df_train[df_train['YearsCodingProf'].isnull()==False]

# df_train_YearsCodingProf['YearsCodingProf_int'] = df_train_YearsCodingProf['YearsCodingProf'].astype(int)

# print(df_train_YearsCodingProf['YearsCodingProf_int'].mean())

df_train = df_train.fillna({'YearsCoding':'6'})

df_train



df_test.loc[df_test['YearsCodingProf']=='0-2 years', 'YearsCodingProf'] = '0'

df_test.loc[df_test['YearsCodingProf']=='3-5 years', 'YearsCodingProf'] = '3'

df_test.loc[df_test['YearsCodingProf']=='6-8 years', 'YearsCodingProf'] = '6'

df_test.loc[df_test['YearsCodingProf']=='9-11 years', 'YearsCodingProf'] = '9'

df_test.loc[df_test['YearsCodingProf']=='12-14 years', 'YearsCodingProf'] = '12'

df_test.loc[df_test['YearsCodingProf']=='15-17 years', 'YearsCodingProf'] = '15'

df_test.loc[df_test['YearsCodingProf']=='18-20 years', 'YearsCodingProf'] = '18'

df_test.loc[df_test['YearsCodingProf']=='21-23 years', 'YearsCodingProf'] = '21'

df_test.loc[df_test['YearsCodingProf']=='24-26 years', 'YearsCodingProf'] = '24'

df_test.loc[df_test['YearsCodingProf']=='27-29 years', 'YearsCodingProf'] = '27'

df_test.loc[df_test['YearsCodingProf']=='30 or more years', 'YearsCodingProf'] = '30'

df_test = df_test.fillna({'YearsCoding':'6'})
# 欠損補完

# LastNewJob変換

df_train.loc[df_train['LastNewJob']=='Less than a year ago', 'LastNewJob'] = '1'

df_train.loc[df_train['LastNewJob']=='Between 1 and 2 years ago', 'LastNewJob'] = '2'

df_train.loc[df_train['LastNewJob']=='Between 2 and 4 years ago', 'LastNewJob'] = '3'

df_train.loc[df_train['LastNewJob']=='More than 4 years ago', 'LastNewJob'] = '4'

df_train.loc[df_train['LastNewJob']=='I\'ve never had a job', 'LastNewJob'] = '0'

# df_train_LastNewJob = df_train[df_train['LastNewJob'].isnull()==False]

# df_train_LastNewJob['LastNewJob_int'] = df_train_LastNewJob['LastNewJob'].astype(int)

# print(df_train_LastNewJob['LastNewJob_int'].mean())

df_train = df_train.fillna({'LastNewJob':'2'})



df_test.loc[df_test['LastNewJob']=='Less than a year ago', 'LastNewJob'] = '1'

df_test.loc[df_test['LastNewJob']=='Between 1 and 2 years ago', 'LastNewJob'] = '2'

df_test.loc[df_test['LastNewJob']=='Between 2 and 4 years ago', 'LastNewJob'] = '3'

df_test.loc[df_test['LastNewJob']=='More than 4 years ago', 'LastNewJob'] = '4'

df_test.loc[df_test['LastNewJob']=='I\'ve never had a job', 'LastNewJob'] = '0'

df_test = df_test.fillna({'LastNewJob':'2'})
# 欠損補完

# TimeFullyProductive変換

df_train.loc[df_train['TimeFullyProductive']=='Less than a month', 'TimeFullyProductive'] = '1'

df_train.loc[df_train['TimeFullyProductive']=='One to three months', 'TimeFullyProductive'] = '3'

df_train.loc[df_train['TimeFullyProductive']=='Three to six months', 'TimeFullyProductive'] = '6'

df_train.loc[df_train['TimeFullyProductive']=='Six to nine months', 'TimeFullyProductive'] = '9'

df_train.loc[df_train['TimeFullyProductive']=='Nine months to a year', 'TimeFullyProductive'] = '12'

df_train.loc[df_train['TimeFullyProductive']=='More than a year', 'TimeFullyProductive'] = '24'

# df_train_TimeFullyProductive = df_train[df_train['TimeFullyProductive'].isnull()==False]

# df_train_TimeFullyProductive['TimeFullyProductive_int'] = df_train_TimeFullyProductive['TimeFullyProductive'].astype(int)

# print(df_train_TimeFullyProductive['TimeFullyProductive_int'].mean())

df_train = df_train.fillna({'TimeFullyProductive':'3'})



df_test.loc[df_test['TimeFullyProductive']=='Less than a month', 'TimeFullyProductive'] = '1'

df_test.loc[df_test['TimeFullyProductive']=='One to three months', 'TimeFullyProductive'] = '3'

df_test.loc[df_test['TimeFullyProductive']=='Three to six months', 'TimeFullyProductive'] = '6'

df_test.loc[df_test['TimeFullyProductive']=='Six to nine months', 'TimeFullyProductive'] = '9'

df_test.loc[df_test['TimeFullyProductive']=='Nine months to a year', 'TimeFullyProductive'] = '12'

df_test.loc[df_test['TimeFullyProductive']=='More than a year', 'TimeFullyProductive'] = '24'

df_test = df_test.fillna({'TimeFullyProductive':'3'})
# 欠損補完

# NumberMonitors変換

df_train.loc[df_train['NumberMonitors']=='More than 4', 'NumberMonitors'] = '5'

# df_train_NumberMonitors = df_train[df_train['NumberMonitors'].isnull()==False]

# df_train_NumberMonitors['NumberMonitors_int'] = df_train_NumberMonitors['NumberMonitors'].astype(int)

# print(df_train_NumberMonitors['NumberMonitors_int'].mean())

df_train = df_train.fillna({'TimeFullyProductive':'2'})



df_test.loc[df_test['NumberMonitors']=='More than 4', 'NumberMonitors'] = '5'

df_test = df_test.fillna({'TimeFullyProductive':'2'})
# 欠損補完

# StackOverflowRecommend変換

df_train.loc[df_train['StackOverflowRecommend']=='10 (Very Likely)', 'StackOverflowRecommend'] = '10'

df_train.loc[df_train['StackOverflowRecommend']=='0 (Not Likely)', 'StackOverflowRecommend'] = '0'

# df_train_NumberMonitors = df_train[df_train['NumberMonitors'].isnull()==False]

# df_train_NumberMonitors['NumberMonitors_int'] = df_train_NumberMonitors['NumberMonitors'].astype(int)

# print(df_train_NumberMonitors['NumberMonitors_int'].mean())

df_train = df_train.fillna({'StackOverflowRecommend':'0'})



df_test.loc[df_test['StackOverflowRecommend']=='10 (Very Likely)', 'StackOverflowRecommend'] = '10'

df_test.loc[df_test['StackOverflowRecommend']=='0 (Not Likely)', 'StackOverflowRecommend'] = '0'

df_test = df_test.fillna({'StackOverflowRecommend':'0'})
# 欠損補完

# StackOverflowJobsRecommend変換

df_train.loc[df_train['StackOverflowJobsRecommend']=='10 (Very Likely)', 'StackOverflowJobsRecommend'] = '10'

df_train.loc[df_train['StackOverflowJobsRecommend']=='0 (Not Likely)', 'StackOverflowJobsRecommend'] = '0'

# df_train_NumberMonitors = df_train[df_train['NumberMonitors'].isnull()==False]

# df_train_NumberMonitors['NumberMonitors_int'] = df_train_NumberMonitors['NumberMonitors'].astype(int)

# print(df_train_NumberMonitors['NumberMonitors_int'].mean())

df_train = df_train.fillna({'StackOverflowJobsRecommend':'0'})



df_test.loc[df_test['StackOverflowJobsRecommend']=='10 (Very Likely)', 'StackOverflowJobsRecommend'] = '10'

df_test.loc[df_test['StackOverflowJobsRecommend']=='0 (Not Likely)', 'StackOverflowJobsRecommend'] = '0'

df_test = df_test.fillna({'StackOverflowJobsRecommend':'0'})
# 欠損補完

# HoursComputer変換

df_train.loc[df_train['HoursComputer']=='Less than 1 hour', 'HoursComputer'] = '0.5'

df_train.loc[df_train['HoursComputer']=='1 - 4 hours', 'HoursComputer'] = '1'

df_train.loc[df_train['HoursComputer']=='5 - 8 hours', 'HoursComputer'] = '5'

df_train.loc[df_train['HoursComputer']=='9 - 12 hours', 'HoursComputer'] = '9'

df_train.loc[df_train['HoursComputer']=='Over 12 hours', 'HoursComputer'] = '12'

# df_train_HoursComputer = df_train[df_train['HoursComputer'].isnull()==False]

# df_train_HoursComputer['HoursComputer_int'] = df_train_HoursComputer['HoursComputer'].astype(float)

# print(df_train_HoursComputer['HoursComputer_int'].mean())

df_train = df_train.fillna({'HoursComputer':'8'})



df_test.loc[df_test['HoursComputer']=='Less than 1 hour', 'HoursComputer'] = '0.5'

df_test.loc[df_test['HoursComputer']=='1 - 4 hours', 'HoursComputer'] = '1'

df_test.loc[df_test['HoursComputer']=='5 - 8 hours', 'HoursComputer'] = '5'

df_test.loc[df_test['HoursComputer']=='9 - 12 hours', 'HoursComputer'] = '9'

df_test.loc[df_test['HoursComputer']=='Over 12 hours', 'HoursComputer'] = '12'

df_test = df_test.fillna({'HoursComputer':'8'})
# 欠損補完

# HoursOutside変換

df_train.loc[df_train['HoursOutside']=='1 - 2 hours', 'HoursOutside'] = '120'

df_train.loc[df_train['HoursOutside']=='3 - 4 hours', 'HoursOutside'] = '240'

df_train.loc[df_train['HoursOutside']=='30 - 59 minutes', 'HoursOutside'] = '60'

df_train.loc[df_train['HoursOutside']=='Less than 30 minutes', 'HoursOutside'] = '30'

df_train.loc[df_train['HoursOutside']=='Over 4 hours', 'HoursOutside'] = '480'

# df_train_HoursOutside = df_train[df_train['HoursOutside'].isnull()==False]

# df_train_HoursOutside['HoursOutside_int'] = df_train_HoursOutside['HoursOutside'].astype(float)

# print(df_train_HoursOutside['HoursOutside_int'].mean())

df_train = df_train.fillna({'HoursOutside':'99'})



df_test.loc[df_test['HoursOutside']=='1 - 2 hours', 'HoursOutside'] = '120'

df_test.loc[df_test['HoursOutside']=='3 - 4 hours', 'HoursOutside'] = '240'

df_test.loc[df_test['HoursOutside']=='30 - 59 minutes', 'HoursOutside'] = '60'

df_test.loc[df_test['HoursOutside']=='Less than 30 minutes', 'HoursOutside'] = '30'

df_test.loc[df_test['HoursOutside']=='Over 4 hours', 'HoursOutside'] = '480'

df_test = df_test.fillna({'HoursOutside':'99'})

# 欠損補完

# Age変換

df_train.loc[df_train['Age']=='18 - 24 years old', 'Age'] = '18'

df_train.loc[df_train['Age']=='25 - 34 years old', 'Age'] = '25'

df_train.loc[df_train['Age']=='35 - 44 years old', 'Age'] = '35'

df_train.loc[df_train['Age']=='45 - 54 years old', 'Age'] = '45'

df_train.loc[df_train['Age']=='55 - 64 years old', 'Age'] = '55'

df_train.loc[df_train['Age']=='65 years or older', 'Age'] = '65'

df_train.loc[df_train['Age']=='Under 18 years old', 'Age'] = '15'

# df_train_Age = df_train[df_train['Age'].isnull()==False]

# df_train_Age['Age_int'] = df_train_Age['Age'].astype(float)

# print(df_train_Age['Age_int'].mean())

df_train = df_train.fillna({'Age':'27'})



df_test.loc[df_test['Age']=='18 - 24 years old', 'Age'] = '18'

df_test.loc[df_test['Age']=='25 - 34 years old', 'Age'] = '25'

df_test.loc[df_test['Age']=='35 - 44 years old', 'Age'] = '35'

df_test.loc[df_test['Age']=='45 - 54 years old', 'Age'] = '45'

df_test.loc[df_test['Age']=='55 - 64 years old', 'Age'] = '55'

df_test.loc[df_test['Age']=='65 years or older', 'Age'] = '65'

df_test.loc[df_test['Age']=='Under 18 years old', 'Age'] = '15'

df_test = df_test.fillna({'Age':'27'})
# ここからまだですー
cats = []

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        cats.append(col)

      

        print(col, df_train[col].nunique())
# oe = OrdinalEncoder()

# dic = ['emp_length', 'title', 'zip_code', 'earliest_cr_line']



# oe_train = pd.DataFrame(oe.fit_transform(df_train[dic]), columns=dic)

# oe_test = pd.DataFrame(oe.transform(df_test[dic]), columns=dic)



# df_train['emp_length'] = oe_train['emp_length'].values

# df_train['zip_code'] = oe_train['zip_code'].values

# df_train['title'] = oe_train['title'].values

# df_train['earliest_cr_line'] = oe_train['earliest_cr_line'].values

# df_test['emp_length'] = oe_test['emp_length'].values

# df_test['zip_code'] = oe_test['zip_code'].values

# df_test['title'] = oe_test['title'].values

# df_test['earliest_cr_line'] = oe_test['earliest_cr_line'].values
# df_train['loan_amnt'] = df_train['loan_amnt'].apply(np.log1p)

# df_train['annual_inc'] = df_train['annual_inc'].apply(np.log1p)

# df_train['revol_bal'] = df_train['revol_bal'].apply(np.log1p)

# df_train['tot_cur_bal'] = df_train['tot_cur_bal'].apply(np.log1p)

# df_test['loan_amnt'] = df_test['loan_amnt'].apply(np.log1p)

# df_test['annual_inc'] = df_test['annual_inc'].apply(np.log1p)

# df_test['revol_bal'] = df_test['revol_bal'].apply(np.log1p)

# df_test['tot_cur_bal'] = df_test['tot_cur_bal'].apply(np.log1p)
# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()

# dic2 = ['loan_amnt', 'annual_inc', 'revol_bal', 'tot_cur_bal']

# sc_train = pd.DataFrame(sc.fit_transform(df_train[dic2]), columns=dic2)

# sc_test = pd.DataFrame(sc.transform(df_test[dic2]), columns=dic2)
# df_train = df_train.reset_index(drop=True)

# df_train['loan_amnt'] = sc_train['loan_amnt'].values

# df_train['revol_bal'] = sc_train['revol_bal'].values

# df_train['tot_cur_bal'] = sc_train['tot_cur_bal'].values

# df_test = df_test.reset_index(drop=True)

# df_test['loan_amnt'] = sc_test['loan_amnt'].values

# df_test['revol_bal'] = sc_test['revol_bal'].values

# df_test['tot_cur_bal'] = sc_test['tot_cur_bal'].values
# y_train = df_train.loan_condition

# X_train = df_train.drop(['loan_condition'], axis=1)

# X_test = df_test
# col = 'grade'

# target = 'loan_condition'

# X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

# summary = X_temp.groupby([col])[target].mean()

# enc_test = X_test[col].map(summary)

# enc_test2 = pd.DataFrame(enc_test)

# enc_test2.rename(columns={'grade':'enc_grade'}, inplace=True)



# X_trainのカテゴリ変数をoofでエンコーディングする

# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



# enc_train = Series(np.zeros(len(X_train)), index=df_train.index)



# for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

#     X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#     X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



#     summary = X_train_.groupby([col])[target].mean()

#     enc_train.iloc[val_ix] = X_val[col].map(summary)

# enc_train2 = pd.DataFrame(enc_train)

# enc_train2.rename(columns=lambda s:'enc_grade', inplace=True)
# col = 'sub_grade'

# target = 'loan_condition'

# X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

# summary = X_temp.groupby([col])[target].mean()

# enc_test = X_test[col].map(summary)

# enc_test3 = pd.DataFrame(enc_test)

venc_test3.rename(columns={'sub_grade':'enc_sub_grade'}, inplace=True)



# X_trainのカテゴリ変数をoofでエンコーディングする

# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



# enc_train = Series(np.zeros(len(X_train)), index=df_train.index)



# for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

#     X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#     X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



#     summary = X_train_.groupby([col])[target].mean()

#     enc_train.iloc[val_ix] = X_val[col].map(summary)

# enc_train3 = pd.DataFrame(enc_train)

# enc_train3.rename(columns=lambda s:'enc_sub_grade', inplace=True)
# col = 'home_ownership'

# target = 'loan_condition'

# X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

# summary = X_temp.groupby([col])[target].mean()

# enc_test = X_test[col].map(summary)

# enc_test4 = pd.DataFrame(enc_test)

# enc_test4.rename(columns={'home_ownership':'enc_home_ownership'}, inplace=True)



# X_trainのカテゴリ変数をoofでエンコーディングする

# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



# enc_train = Series(np.zeros(len(X_train)), index=df_train.index)



# for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

#     X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#     X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



#     summary = X_train_.groupby([col])[target].mean()

#     enc_train.iloc[val_ix] = X_val[col].map(summary)

# enc_train4 = pd.DataFrame(enc_train)

# enc_train4.rename(columns=lambda s:'enc_home_ownership', inplace=True)
# TXT_train = X_train['emp_title']

# TXT_test = X_test['emp_title']

# TXT_train.fillna('#', inplace=True)

# TXT_test.fillna('#', inplace=True)
# tfidf = TfidfVectorizer(max_features=1000, use_idf=True)

# TXT_train = tfidf.fit_transform(TXT_train)

# TXT_test = tfidf.transform(TXT_test)
# coo_train = TXT_train.tocoo(copy=False)

# coo_test = TXT_test.tocoo(copy=False)

# df_txt_train=pd.DataFrame({'index': coo_train.row, 'col': coo_train.col, 'data': coo_train.data}

#                  )[['index', 'col', 'data']].sort_values(['index', 'col']

#                  ).reset_index(drop=True)

# df_txt_test=pd.DataFrame({'index': coo_test.row, 'col': coo_test.col, 'data': coo_test.data}

#                  )[['index', 'col', 'data']].sort_values(['index', 'col']

#                  ).reset_index(drop=True)

# df_txt_train=df_txt_train.groupby(['index'], as_index=False)['col','data'].min()

# df_txt_test=df_txt_test.groupby(['index'], as_index=False)['col','data'].min()

# X_train['index'] = X_train.index

# X_test['index'] = X_test.index

# X_train = pd.merge(X_train, df_txt_train, on = 'index', how ='outer')

# X_test = pd.merge(X_test, df_txt_test, on = 'index', how ='outer')

# X_train = X_train.drop(['index'],axis=1)

# X_test = X_test.drop(['index'],axis=1)

# X_train.fillna(-9999,inplace=True)

# X_test.fillna(-9999,inplace=True)
# cats = []

# for col in X_train.columns:

#     if X_train[col].dtype == 'object':

#         if col != 'issue_d':

#             cats.append(col)

#             print(col, X_train[col].nunique())

# del cats[3]

# del cats[2]

# del cats[1]

# del cats[0]

# cats
# groups = X_train['issue_d'].values
# ohe = OneHotEncoder(cols=cats, return_df=False)

# ohe_train = pd.DataFrame(ohe.fit_transform(X_train[cats]))

# ohe_test = pd.DataFrame(ohe.transform(X_test[cats]))
# X_train2 = pd.concat([ohe_train, X_train], axis=1)

# X_test2 = pd.concat([ohe_test, X_test], axis=1)

# X_train3 = pd.concat([enc_train4, X_train2], axis=1)

# X_test3 = pd.concat([enc_test4, X_test2], axis=1)

# X_train4 = pd.concat([enc_train3, X_train3], axis=1)

# X_test4 = pd.concat([enc_test3, X_test3], axis=1)

# X_train5 = pd.concat([enc_train2, X_train4], axis=1)

# X_test5 = pd.concat([enc_test2, X_test4], axis=1)
# del X_train5['grade']

# del X_train5['sub_grade']

# del X_train5['home_ownership']

# del X_train5['emp_title']

# del X_train5['issue_d']

# del X_train5['purpose']

# del X_train5['addr_state']

# del X_train5['initial_list_status']

# del X_train5['application_type']

# del X_train5['tot_coll_amt_nan']

# del X_test5['grade']

# del X_test5['sub_grade']

# del X_test5['home_ownership']

# del X_test5['emp_title']

# del X_test5['issue_d']

# del X_test5['purpose']

# del X_test5['addr_state']

# del X_test5['initial_list_status']

# del X_test5['application_type']

# del X_test5['tot_coll_amt_nan']
# X_train0 = X_train5.set_index(df_train_label)

# X_test0 = X_test5.set_index(df_test_label)
# X_train = X_train0

# X_test = X_test0
# X_train['sub_grade_x_loan_amnt'] = X_train['enc_sub_grade'].values * X_train['loan_amnt'].values

# X_test['sub_grade_x_loan_amnt'] = X_test['enc_sub_grade'].values * X_test['loan_amnt'].values
# print(X_train.isnull().any())
# from hyperopt import fmin, tpe, hp, rand, Trials

# from sklearn.model_selection import StratifiedKFold

# from sklearn.metrics import roc_auc_score



# from lightgbm import LGBMClassifier
# def objective(space):

#     gkf = GroupKFold(n_splits=3)

#     scores = []

#     for i, (train_ix, test_ix) in enumerate(tqdm(gkf.split(X_train, y_train, groups))):   

#         X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]

#         X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]

#         clf = LGBMClassifier(n_estimators=100, **space) 

#         clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

#         y_pred = clf.predict_proba(X_val)[:,1]

#         score = roc_auc_score(y_val, y_pred)

#         scores.append(score)       

#     scores = np.array(scores)

#     print(scores.mean())

  

#     return -scores.mean()
# space ={

#         'max_depth': hp.choice('max_depth', np.arange(10, 30, dtype=int)),

#         'subsample': hp.uniform ('subsample', 0.8, 1),

#         'learning_rate' : hp.quniform('learning_rate', 0.05, 0.1, 0.025),

#         'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05)

# }
# trials = Trials()



# best = fmin(fn=objective,

#               space=space, 

#               algo=tpe.suggest,

#               max_evals=10, 

#               trials=trials, 

#               rstate=np.random.RandomState(71) 

#              )
# LGBMClassifier(**best)
# trials.best_trial['result']
# trials.best_trial
# %%time

# gkf = GroupKFold(n_splits=3)

# scores = []

# for i, (train_ix, test_ix) in enumerate(tqdm(gkf.split(X_train, y_train, groups))):   

#     X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]

#     X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]

#     print('Train Groups', np.unique(groups_train_))

#     print('Val Groups', np.unique(groups_val))

#     clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

#                             importance_type='split', learning_rate=0.05, max_depth=-1,

#                             min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#                             n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

#                             random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#                             subsample=0.935, subsample_for_bin=200000, subsample_freq=0)

#     clf.fit(X_train_, y_train_)

#     y_pred = clf.predict_proba(X_val)[:,1]

#     score = roc_auc_score(y_val, y_pred)

#     scores.append(score)

#     print('CV Score of Fold_%d is %f' % (i, score))

#     print('\n')
# %%time

# gkf = GroupKFold(n_splits=3)

# scores = []

# for i, (train_ix, test_ix) in enumerate(tqdm(gkf.split(X_train, y_train, groups))):   

#     X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]

#     X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]

#     print('Train Groups', np.unique(groups_train_))

#     print('Val Groups', np.unique(groups_val))

#     clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

#                             importance_type='split', learning_rate=0.05, max_depth=-1,

#                             min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#                             n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

#                             random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#                             subsample=0.935, subsample_for_bin=200000, subsample_freq=0)

#     clf.fit(X_train_, y_train_)

#     y_pred = clf.predict_proba(X_val)[:,1]

#     score = roc_auc_score(y_val, y_pred)

#     scores.append(score)

#     print('CV Score of Fold_%d is %f' % (i, score))

#     print('\n')
# print(np.mean(scores))

# print(scores)
# %%time

# clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

#                         importance_type='split', learning_rate=0.05, max_depth=-1,

#                         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#                         n_estimators=1000, n_jobs=-1, num_leaves=15, objective=None,

#                         random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#                         subsample=0.935, subsample_for_bin=200000, subsample_freq=0)

# clf.fit(X_train_, y_train_)
# DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance'])
# fig, ax = plt.subplots(figsize=(14, 14))

# lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
# y_pred = clf.predict_proba(X_test)[:,1]

# y_pred
# submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)

# submission.loan_condition = y_pred

# submission.to_csv('submission.csv')