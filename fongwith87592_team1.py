import pandas as pd

import numpy as np

import json

from tqdm import tqdm

from dateutil.relativedelta import relativedelta

import gc

import os

import warnings

import datetime

warnings.filterwarnings('ignore')



#Models

from sklearn import metrics

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
train_data = pd.read_csv("../input/openedulearningdataanalysis/hack_training.csv")

question_data = pd.read_csv("../input/question-data/hack_question02.csv")



# Get course id from training data.

train_course_id = set(train_data.course_id.values)

train_uid = set(train_data.user_id.values)
def get_alldata_by_course(course_id):

    for i in range(4, 8):

        print("Processing the 201" + str(i) + " log file......")



        if i <= 5:

            log_path = "../input/hack-2014-2015-log/hack_log_201" + str(i) + "/hack_log_201" + str(i)

        else:

            log_path = "../input/hack-2016-2017-log/hack_log_201" + str(i) + "/hack_log_201" + str(i)

            

        log_list = os.listdir(log_path)

        data_df = pd.DataFrame()

        dic = {}

        for log_name in tqdm(log_list):

            with open(log_path + "/" + log_name) as file:

                count = 0

                try:

                    for line in file:

                        try:

                            j = json.loads(line)

                            if j['context']['course_id'] in course_id:

                                dic[count] = j

                        except:

                            pass

                        count += 1

                except:

                    pass



        df = pd.DataFrame.from_dict(dic, orient = 'index')

        data_df = pd.concat([data_df, df], axis = 0, ignore_index = True)        



        del dic

        del df

        gc.collect()



    return data_df



print("Get training DataFrame......")

train_df = get_alldata_by_course(train_course_id)
def clean_data(data):

    user_data_dic = {}



    for i in tqdm(range(len(data))):

        content = []

        try:

            content.append(data['context'].iloc[i]['course_id'])

        except:

            content.append(np.nan)

        try:

            content.append(data.iloc[i]['session'])

        except:

            content.append(np.nan)

        try:

            content.append(data['context'].iloc[i]['user_id'])

        except:

            content.append(np.nan)

        try:

            content.append(data.iloc[i]['event_type'])

        except:

            content.append(np.nan)

        try:

            content.append(eval(data.iloc[i]['event'])['id'])

        except:

            content.append(np.nan)

        try:

            content.append(eval(data.iloc[i]['event'])['currentTime'])

        except:

            content.append(np.nan)

        try:

            content.append(data.iloc[i]['time'])

        except:

            content.append(np.nan)

        user_data_dic[i] = content



    clean_data = pd.DataFrame.from_dict(user_data_dic, orient = 'index', columns=['course_id', 'session', 

                                                                            'user_id', 'event_type', 

                                                                            'id', 'currentTime',

                                                                            'time'])



    clean_data['date']=clean_data['time'].str.split("T").str[0]

    clean_data['time1']=clean_data['time'].str.split("T").str[1]

    clean_data['time1']=clean_data['time1'].str.split(".").str[0]

    return clean_data



clean_train_df = clean_data(train_df)
def get_uid_data(data, UID):

    num=1

    fdic={}

    u_dic={}

    c_dic={}

    dc_dic={}

    vc_dic={}

    at_dic={}

    sc_dic={}

    se_dic={}

    pl_dic={}

    sevi_dic={}

    spva_dic={}

    for uid in tqdm(UID):

        uid_course = data[data['user_id']==uid]['course_id'].value_counts().index



        for course in uid_course:

            df = data[(data['user_id']==uid) & (data['course_id']==course)]

            #user_id

            u_dic[num]=uid

            #course_id

            c_dic[num]=course

            #date_count

            dc_dic[num]=len(df['date'].value_counts())

            #video_count

            vc_dic[num]=len(df['id'].value_counts())

            #影片觀看次數

            a=0

            session = []

            for len_session in range(len(df['session'].value_counts())):

                if df['session'] not in session:

                    session.append(df['session'])

                    a+=len(df['id'].value_counts())

            sevi_dic[num] = a

            #pause_count

            pl_dic[num]=len(df[df['event_type']=='pause_video'])

            #speed_change_count

            sc_dic[num]=len(df[df['event_type']=='speed_change_video'])



            #all_time

            total=0

            p=[]

            n=df.index

            for a in n:

                if df['event_type'][a]=='play_video':

                    p.append(a)

                elif df['event_type'][a]=='pause_video' and p!=[]:

                    if df['session'][a] == df['session'][p[-1]]:

                        total += df['currentTime'][a] - df['currentTime'][p[-1]]

                    p=[]

            at_dic[num]=total                  

            num+=1



    fdic['user_id']=u_dic

    fdic['course_id']=c_dic

    fdic['datecount']=dc_dic

    fdic['videocount']=vc_dic

    fdic['totaltime']=at_dic

    fdic['pause_video_count']=pl_dic

    fdic['speed_change_count']=sc_dic

    train_df=pd.DataFrame.from_dict(fdic)

    

    c_id = train_df.course_id.value_counts().index

    courseperid, ddm = {}, []

    num = 0

    for each in c_id:

        courseperid[each] = train_df[train_df['course_id']==each]['videocount'].max()

    for a in range(1,len(train_df)+1):

        ddm.append(train_df['videocount'][a]/courseperid[train_df['course_id'][a]])

    train_df['ddm'] = ddm

    train_df.ddm = train_df.ddm.fillna(0)

    return train_df



train_uid_df = get_uid_data(clean_train_df, train_uid)

train = train_uid_df.merge(train_data, on=('user_id','course_id'), left_index=True, right_index=True)

train.target = train.target.map({"T":1, "F":0})
#KNN

X_train, X_test, y_train, y_test = train_test_split(train.drop('target', 1), train.target, test_size=0.3, random_state=1000)

knn = KNeighborsClassifier()

X_test['course_id'] = LabelEncoder().fit_transform(X_test['course_id'])

X_train['course_id'] = LabelEncoder().fit_transform(X_train['course_id'])

knn.fit(X_train, y_train)



#預測準確率

y_predict = knn.predict(X_test)

print('Misclassified samples: %d' % (y_test != y_predict).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_predict))