# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

from tqdm import tqdm

from collections import Counter

import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
print('training data shape',train.shape)

print('test data shape',test.shape)



unique_id_train = len(train['installation_id'].unique())

unique_id_test = len(test['installation_id'].unique())



print('number of unique installation_id in training data' ,unique_id_train)

print('number of unique installation_id in test data' ,unique_id_test)
train.head()
test.head()
# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline

# Some of the installation_id from training data do not attempt for Assessment even for a single time.

# First we will get rid of those ids as we will not able to predict the class.

required_id = train[train['type'] == "Assessment"]['installation_id'].drop_duplicates()

train = pd.merge(train, required_id , on="installation_id", how="inner")



unique_id_train = len(train['installation_id'].unique())



print('training data shape',train.shape)

print('number of unique installation_id in training data',unique_id_train)
# In test data also we have some installation id that attempt for Assessment but corresponding event_code is not 4100 or 4110(Bird Measurer)

test[(test['installation_id']=='017c5718') & (test['type']=='Assessment')]
# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline

# In the reference blog they have used two separate function for featurization. I slightly modified and compressed it in only one function.

def get_features(installation_id , dataframe_for_an_id , test_flag=False):

    '''

    

    This function will calculate features for train and test data. 

    It will create 4 columns for four unique world(including None) and

    will create 44 columns for 44 unique title and

    will create 4 columns for four unique type and

    will create 42 columns for 42 unique event_code and

    will create 6 more columns for 'total_duration','total_action','correct_count','incorrect_count','accuracy','accuracy_group'

               ---

        total  100 columns 

    

    except total_duration, accuracy and accuracy_group all other features is number of counts of those feature in a game_session

    if test_flag is True then return last entry of list 

    '''

    # temp_dict initialized with keys (100 columns) and value = 0

    features = []

    features.extend(list(set(train['world'].unique()).union(set(test['world'].unique())))) # all unique worlds in train and test data

    features.extend(list(set(train['title'].unique()).union(set(test['title'].unique())))) # all unique title in train and test data

    features.extend(list(set(train['type'].unique()).union(set(test['type'].unique())))) # all unique type in train and test data

    features.extend(list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))) # all unique event_code in train and test data 

    features.extend(['total_duration','total_action','correct_count','incorrect_count','accuracy','accuracy_group'])

    temp_dict = dict.fromkeys(features,0)

    list_of_features = []

    

    

    def get_all_attempt(sample_df):

        '''

        This fuction will return the dataframe which is used to calculate accuracy_group

        '''

        if sample_df['title'].iloc[0] != 'Bird Measurer (Assessment)':

            all_attempt = sample_df.query('event_code == 4100')

        elif sample_df['title'].iloc[0] == 'Bird Measurer (Assessment)':

            all_attempt = sample_df.query('event_code == 4110')

        return all_attempt

    

    for i, sample_df in dataframe_for_an_id.groupby(by = ['game_session']):

        # sample_df is groupby object 

        # In sample_df 'type','title' and 'world' will not change so first entry of those column is piced

        temp_dict['installation_id'] = installation_id

        temp_type = sample_df['type'].iloc[0]

        temp_title = sample_df['title'].iloc[0]

        temp_world = sample_df['world'].iloc[0]

        temp_event_code = Counter(sample_df['event_code'].values)



        session_size = len(sample_df)



        temp_dict[temp_type]+=session_size

        temp_dict[temp_title]+=session_size    # corresponding type , title and world is incremented by session size

        temp_dict[temp_world]+=session_size                 



        for code, code_count in temp_event_code.items():    # corresponding event_code is incremented

            temp_dict[code]+= code_count                  

            

        duration_in_sec = float(sample_df['game_time'].iloc[-1])/1000   # total_duration is duration of game_session in seconds

        temp_dict['total_duration'] += duration_in_sec

        

        action_count_in_game_session = session_size     # total number of action performed in game_session

        temp_dict['total_action'] += action_count_in_game_session  

        

        isAssessment = temp_type == 'Assessment'

        isBirdMeasureAssessment = isAssessment and temp_title == 'Bird Measurer (Assessment)'

        isAssessment_with_code4110 = isBirdMeasureAssessment and 4110 in list(sample_df['event_code'])

        isNonBirdMeasureAssessment = isAssessment and temp_title != 'Bird Measurer (Assessment)'

        isAssessment_with_code4100 = isNonBirdMeasureAssessment and 4100 in list(sample_df['event_code'])

        

        criterion_to_accuracy_group = isAssessment_with_code4110 or isAssessment_with_code4100 

        

        

        if test_flag and isAssessment and (criterion_to_accuracy_group == False):

            temp_dict['accuracy'] = 0           # there are lots of installation_id in test data that attempt for

            temp_dict['accuracy_group'] = 0     # Assessment but not with event_code 4100 or 4110

            list_of_features.append(temp_dict)  # So I assumed those id belongs to class 0

            

            

        if criterion_to_accuracy_group == False:

            continue

        

        # below section is only performed when criterion_to_accuracy_group is True

        

        all_attempt = get_all_attempt(sample_df)

        correct_count = all_attempt['event_data'].str.contains('true').sum()     

        incorrect_count = all_attempt['event_data'].str.contains('false').sum()

        temp_dict['correct_count'] = correct_count  

        temp_dict['incorrect_count'] = incorrect_count



        if correct_count == 0 and incorrect_count == 0:

            temp_dict['accuracy'] = 0

        else:

            temp_dict['accuracy'] = correct_count/(correct_count + incorrect_count)



        if temp_dict['accuracy']==1:

            temp_dict['accuracy_group']=3

        elif temp_dict['accuracy']==0.5:

            temp_dict['accuracy_group']=2

        elif temp_dict['accuracy']==0:

            temp_dict['accuracy_group']=0

        else :

            temp_dict['accuracy_group']=1



        list_of_features.append(temp_dict)

        temp_dict = dict.fromkeys(features,0)

        

        

    if test_flag:                    # If given data is from test data then return only the last entry of the list

        return list_of_features[-1]

    

    return list_of_features
# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline

# below is testing code to check whether get_features function works properly for training data

sample_df = train[train.installation_id == "0006a69f"]

list_of_feature = get_features("0006a69f",sample_df)

list_of_feature

temp_df = pd.DataFrame(list_of_feature)

temp_df
# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline

# below is testing code to check whether get_features function works properly for test data

sample_df = train[train.installation_id == "0006a69f"]

list_of_feature = get_features('0006a69f',sample_df,True)

list_of_feature

temp_df = pd.DataFrame(list_of_feature,index=[0])

temp_df
from tqdm.notebook import tqdm
'''

# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline

final_training_data_list = []

training_groupby_id = train.groupby(by=['installation_id']) 



for installation_id , df_with_unique_id in tqdm(training_groupby_id):

    final_training_data_list.extend(get_features(installation_id,df_with_unique_id))



final_training_data = pd.DataFrame(final_training_data_list)

'''
final_training_data = pd.read_csv('/kaggle/input/final-training-data/final_training_data.csv')
print('featurized training data shape :',final_training_data.shape)

final_training_data.head()
'''

# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline

final_test_data_list = []

test_groupby_id = test.groupby(by=['installation_id'])

for installation_id , df_with_unique_id in tqdm(test_groupby_id):

    final_test_data_list.append(get_features(installation_id,df_with_unique_id,True))

final_test_data = pd.DataFrame(final_test_data_list)

'''
final_test_data = pd.read_csv('/kaggle/input/final-test-data/final_test_data.csv')
print('featurized test data shape :',final_test_data.shape)

final_test_data.head()
import xgboost as xgb

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import time

from sklearn.metrics import make_scorer
# reference : https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps



def calculate_QWK(actual_label,predicted_label):

    '''

    this function will calculate quadratic weighted kappa given actual 

    and predicted label array.

    '''

    N = 4 # unique labels

    hist_actual_label = np.zeros(N)

    hist_predicted_label = np.zeros(N)

    w = np.zeros((N,N))

    numerator = 0       # w and O

    denominator = 0     # w and E

    

    conf_mat = confusion_matrix(actual_label,predicted_label)



    for i in actual_label:               # this part will calculate histogram for actual and predicted label

        hist_actual_label[i]+=1

    for j in predicted_label:

        hist_predicted_label[j]+=1



    E = np.outer(hist_actual_label, hist_predicted_label)  # E is N-by-N matrix which is outer product of 

                                                           # histogram of actual and predicted label    

    for i in range(N):                   # w is N-by-N matrix which is calculated by the given expression

        for j in range(N):

            w[i][j] = (i-j)**2/((N-1)**2)



    E = E/E.sum()

    O = conf_mat/conf_mat.sum()  # normalize confusion matrix and E



    for i in range(N):

        for j in range(N):                # this section calculates numerator and denominator 

            numerator+=w[i][j]*O[i][j]

            denominator+=w[i][j]*E[i][j]



    kappa = 1-numerator/denominator

    

    return kappa
## testing code for function calculate_QWK()

actual_label_temp = np.array([0,3,2,3,1,0,2,1,2,1,0])

predicted_label_temp = np.array([0,3,2,3,1,0,2,1,2,1,0])

print('QWK when actual_label and predicted_label are same is :',calculate_QWK(actual_label_temp,predicted_label_temp))



actual_label_temp = np.array([0,3,0,3,2,0,3,1,2,3,0])

predicted_label_temp = np.array([0,3,2,3,1,0,2,1,2,1,0])

print('QWK when actual_label and predicted_label are different is :',calculate_QWK(actual_label_temp,predicted_label_temp))
X = final_training_data.copy()

X_test = final_test_data.copy()

y = X['accuracy_group'].values

y_test = X_test['accuracy_group'].values
## if we include features 'correct_count','incorect_count' and 'accuracy' to train a model then

## it will become a trvial task like if-else condition to predict the label that we dont want. 

## we calculated 'correct_count','incorect_count' and 'accuracy' to get the label of training  and test data but

## we want our model to predict the label without those feature thats why we will remove those feature.



X = X.drop(['correct_count','incorrect_count','accuracy','accuracy_group','installation_id'], axis=1)

X_test = X_test.drop(['correct_count','incorrect_count','accuracy','accuracy_group','installation_id'],axis =1)



X_train, X_cv, y_train, y_cv = train_test_split(X, y,stratify=y,test_size=0.2)

X_train = X_train.values

X_cv = X_cv.values

X_test = X_test.values



print('size of training data and labels :',X_train.shape,y_train.shape)

print('size of cv data and labels :',X_cv.shape,y_cv.shape)

print('size of test data and labels :',X_test.shape,y_test.shape)
## train a very simple XGBClassifier base model with default parameter

start = time.time()

model = XGBClassifier()

model.fit(X_train,y_train)



actual_label = y_test

predicted_label = model.predict(X_test)



print('Quadratic weighted kappa with simple base model :',calculate_QWK(actual_label,predicted_label))

print('time: ',time.time() - start)
# reference : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

# reference : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer

'''

start = time.time()

params = {'max_depth':[3,4,5,6,7],

          'min_child_weight':[0.01,0.1,1,10],

          'n_estimators':[50,100,200,500]}



QWK_scorer = make_scorer(calculate_QWK, greater_is_better=True)



model  = xgb.XGBClassifier(booster='gbtree')

grid = RandomizedSearchCV(model, param_distributions=params, scoring = QWK_scorer, \

                    n_jobs=-1,cv=5,return_train_score=True) 

                                                

grid.fit(X_train,y_train) 



model = grid.best_estimator_

model.fit(X_train,y_train)



print('time taken to train the model in sec:',time.time() - start)

'''
actual_label = y_train

predicted_label = model.predict(X_train)

print('Quadratic weighed kappa for training data is :',calculate_QWK(actual_label,predicted_label))



actual_label = y_cv

predicted_label = model.predict(X_cv)

print('Quadratic weighed kappa for cross validation data is :',calculate_QWK(actual_label,predicted_label))



actual_label = y_test

predicted_label = model.predict(X_test)

print('Quadratic weighed kappa for test data is :',calculate_QWK(actual_label,predicted_label))
my_submission = pd.DataFrame(data = final_test_data['installation_id'],columns=['installation_id'])

my_submission['accuracy_group'] = predicted_label
k = 0

for i in range(len(sample_submission)):

    if sample_submission['installation_id'][i]==my_submission['installation_id'][i]:

        k+=1

    else:

        print(sample_submission['installation_id'][i])

        print(my_submission['installation_id'][i])

print(k)

type(sample_submission['accuracy_group'][0])==type(my_submission['accuracy_group'][0])
comp_test_df = final_test_data.reset_index()

comp_test_df = comp_test_df[['installation_id']]

comp_test_df['accuracy_group'] = predicted_label

sample_submission.drop('accuracy_group', inplace = True, axis = 1)

sample_submission = sample_submission.merge(comp_test_df, on = 'installation_id')

sample_submission.to_csv('submission.csv', index = False)

print('done !')