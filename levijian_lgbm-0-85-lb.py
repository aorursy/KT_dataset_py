# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data= pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
train_data = train_data[(train_data.stroke_in_2018 == '0') | (train_data.stroke_in_2018 == '1')]

train_data.stroke_in_2018.value_counts()
def process_sex(gender):

    gender = gender.upper()

    if gender[0] == 'F':

        return 'Female'

    elif gender[0] == 'M':

        return 'Male'

    return 'Other'



def order_sex_and_age(s):

    if not s:

        return s

    s = str(s)

    s = s.strip()

    l = s.split(",")

    l = list(map(lambda x : x.strip(), l))

    if l[0][0].isnumeric():

        l[1] = process_sex(l[1])

        return ",".join(l[::-1])

    l[0] = process_sex(l[0])

    return ",".join(l)



# train data process

train_data['sex and age'] = train_data['sex and age'].apply(order_sex_and_age)

sex_age = train_data["sex and age"].str.split(",", n = 1, expand = True) 

train_data['sex'] = sex_age[0]

train_data['age'] = sex_age[1]



# test data process

test_data['sex and age'] = test_data['sex and age'].apply(order_sex_and_age)

sex_age = test_data["sex and age"].str.split(",", n = 1, expand = True) 

test_data['sex'] = sex_age[0]

test_data['age'] = sex_age[1]
convert_list = ['high_BP', 'heart_condition_detected_2017', 'married', 'BMI', 'age', 'stroke_in_2018']

for col in convert_list:

    train_data[col] = train_data[col].convert_objects(convert_numeric=True)

    

convert_list = ['high_BP', 'heart_condition_detected_2017', 'married', 'BMI', 'age']

for col in convert_list:

    test_data[col] = test_data[col].convert_objects(convert_numeric=True)
train_data['age'] = train_data['age'].fillna(-1).astype(np.int8)

test_data['age'] = test_data['age'].fillna(-1).astype(np.int8)
gender_encoding = pd.get_dummies(train_data.sex, prefix='Gender')

train_data = train_data.join(gender_encoding)



gender_encoding = pd.get_dummies(test_data.sex, prefix='Gender')

test_data = test_data.join(gender_encoding)
def job_and_location(ss):

    if ss!=ss:

        return "other?other"

    s=ss.lower().split('?')

    if len(s)<2:

        return ""

    # job?location

    # check if c... or r...

    if s[0]:

        if s[0][0]=='c'or s[0][0]=='r':

            tmp=s[0]

            s[0]=s[1]

            s[1]=tmp

    locat=""

    job=""

    if s[1] and s[1]==s[1]:

        if s[1][0]=='c':

            locat="city"

        elif s[1][0]=='r':

            locat="remote"

        else:

            locat="other"

    else:

        locat="other"

    # u... or pr... or pa... or g.... or b.....

    if s[0]:

        if s[0][0]=='p' and s[0][1]=='r':

            job="private_sector"

        elif s[0][0]=='p' and s[0][1]=='a':

            job="parental_leave"

        elif s[0][0]=='u':

            job="unemployed"

        elif s[0][0]=='g':

            job="government"

        elif s[0][0]=='b':

            job="business_owner"

        else:

            job="other"

    else:

        job="other"

    return job+'?'+locat 



train_data['job_status and living_area'] = train_data['job_status and living_area'].apply(job_and_location)

job_status_living_area = train_data['job_status and living_area'].str.split("?", n=1, expand=True) 

train_data['job_status'] = job_status_living_area[0]

train_data['living_area'] = job_status_living_area[1]



job_encoding = pd.get_dummies(train_data.job_status, prefix='job_status')

living_encoding = pd.get_dummies(train_data.living_area, prefix='living')

train_data = train_data.join(job_encoding)

train_data = train_data.join(living_encoding)



test_data['job_status and living_area'] = test_data['job_status and living_area'].apply(job_and_location)

job_status_living_area = test_data['job_status and living_area'].str.split("?", n=1, expand=True) 

test_data['job_status'] = job_status_living_area[0]

test_data['living_area'] = job_status_living_area[1]



job_encoding = pd.get_dummies(test_data.job_status, prefix='job_status')

living_encoding = pd.get_dummies(test_data.living_area, prefix='living')

test_data = test_data.join(job_encoding)

test_data = test_data.join(living_encoding)
def smoker(ss):

    if ss!=ss:

        return "other"

    ret=""

    ss=ss.lower()

    if ss:

        if ss[0]=='a':

            ret="active_smoker"

        elif ss[0]=='n':

            ret="non_smoker"

        elif ss[0]=='q':

            ret="quit"

        else:

            ret="other"

    else:

        ret="other"

    return ret



train_data['smoker_status'] = train_data['smoker_status'].apply(smoker)

smoker_encoding = pd.get_dummies(train_data.smoker_status, prefix='smoker')

train_data = train_data.join(smoker_encoding)



test_data['smoker_status'] = test_data['smoker_status'].apply(smoker)

smoker_encoding = pd.get_dummies(test_data.smoker_status, prefix='smoker')

test_data = test_data.join(smoker_encoding)
total_data_set = pd.concat([train_data, test_data])



train_data.high_BP.fillna(-1, inplace=True)

train_data.heart_condition_detected_2017.fillna(-1, inplace=True)

train_data.average_blood_sugar.fillna(-1, inplace=True)

train_data.BMI.fillna(total_data_set.BMI.mean(), inplace=True)



test_data.high_BP.fillna(-1, inplace=True)

test_data.heart_condition_detected_2017.fillna(-1, inplace=True)

test_data.average_blood_sugar.fillna(-1, inplace=True)

test_data.BMI.fillna(total_data_set.BMI.mean(), inplace=True)
train_data.Gender_Other.fillna(0, inplace=True)

train_data.job_status_business_owner.fillna(0, inplace=True)

train_data.job_status_parental_leave.fillna(0, inplace=True)

train_data.job_status_unemployed.fillna(0, inplace=True)

train_data.living_other.fillna(0, inplace=True)

train_data.smoker_other.fillna(0, inplace=True)
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,auc,roc_auc_score,precision_score,recall_score,roc_curve
drop_column = ['id', 'sex and age', 

               'job_status and living_area', 

               'smoker_status', 'TreatmentA', 

               'TreatmentB', 'TreatmentC', 

               'TreatmentD', 'sex', 'job_status', 

               'living_area']

X = train_data.drop(drop_column, axis=1)

y  = train_data.stroke_in_2018

X = X.drop(['stroke_in_2018'], axis=1)





test_drop_column = ['id', 'sex and age', 

               'job_status and living_area', 

               'smoker_status', 'TreatmentA', 

               'TreatmentB', 'TreatmentC', 

               'TreatmentD', 'sex', 'job_status', 

               'living_area']

test_X = test_data.drop(drop_column, axis=1)
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)
oofPreds = np.zeros(X.shape[0])

subPreds = np.zeros(test_X.shape[0])
for n_fold, (trainXId, validXId) in enumerate(folds.split(X, y)):

    # Create TrainXY and ValidationXY set based on fold-indexes

    trainX, trainY = X.iloc[trainXId], y.iloc[trainXId]

    validX, validY = X.iloc[validXId], y.iloc[validXId]



    print('== Fold: ' + str(n_fold))



    # LightGBM parameters

    lgbm = LGBMClassifier(

        objective = 'binary',

        boosting_type = 'gbdt',

        n_estimators = 2500,

        learning_rate = 0.05, 

        num_leaves = 250,

        min_data_in_leaf = 125, 

        bagging_fraction = 0.901,

        max_depth = 13, 

        reg_alpha = 2.5,

        reg_lambda = 2.5,

        min_split_gain = 0.0001,

        min_child_weight = 25,

        feature_fraction = 0.5, 

        silent = -1,

        verbose = -1,

        #n_jobs is set to -1 instead of 4 otherwise the kernell will time out

        n_jobs = -1) 



    lgbm.fit(trainX, trainY, 

        eval_set=[(trainX, trainY), (validX, validY)], 

        eval_metric = 'auc', 

        verbose = 250, 

        early_stopping_rounds = 100)



    oofPreds[validXId] = lgbm.predict_proba(validX, num_iteration = lgbm.best_iteration_)[:, 1]

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(validY, oofPreds[validXId])))

    subPreds += lgbm.predict_proba(test_X, num_iteration = lgbm.best_iteration_)[:, 1] / folds.n_splits
test_prediction = pd.Series(subPreds, name="stroke_in_2018")

results = pd.concat([test_data.id,test_prediction],axis=1)

results.to_csv("lgbm_submission.csv", index=False)