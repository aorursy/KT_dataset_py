import pandas as pd

from time import time

import datetime

import numpy as np

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px

from plotly.offline import  init_notebook_mode

import random

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

from catboost import CatBoostClassifier

from sklearn.metrics import confusion_matrix

import colorlover as cl

from tqdm import tqdm_notebook as tqdm

sns.set(rc={'figure.figsize':(11.7,8.27)})

init_notebook_mode(connected=True)
train_labels_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")

train_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")

specs_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")

test_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")

submission_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")
print ("In train dataset we have total of " + str(train_df['installation_id'].nunique()) + " unique Installation ID")

print ("In test dataset we have total of " + str(test_df['installation_id'].nunique()) + " unique Installation ID")
train_df.head()
specs_df.head()
train_labels_df.head()
temp_df = train_labels_df.accuracy_group.value_counts(normalize = True) *100

temp_df = temp_df.round(2)

text = [str(x) + "%" for x in temp_df.values]

fig = go.Figure(data = go.Bar(x = temp_df.index,y = temp_df.values, text = text,textposition='auto'))

fig.update_traces(marker_color='#D95219', marker_line_color='#D95219',marker_line_width=1.5, opacity=0.6)

fig.update_layout(title={'text': "Percentage of accuracy group",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'} )

fig.show()
temp_df = train_df['world'].value_counts(normalize = True) * 100

temp_df = temp_df.round(2)

text = [str(x) + "%" for x in temp_df.values]

fig = go.Figure(data = go.Bar(x = temp_df.values,y = temp_df.index, text = text,textposition='auto',orientation='h'))

fig.update_traces(marker_color='#611F8D', marker_line_color='#611F8D',marker_line_width=1.5, opacity=0.6)

fig.update_layout(title={'text': "Percentage of World",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'} )

fig.show(title = "Percentage of world")
temp_df = train_df.groupby('world')['type'].value_counts(normalize = True).reset_index(name="percentage")

temp_df['percentage'] = temp_df['percentage'] *100

temp_df = temp_df.round(2)

data = []

type_ = temp_df['type'].unique()

colors = [x.replace(")","").replace("rgb(","") for x in cl.scales['4']['qual']['Paired']]

count = 0

for i in type_:



    text = [str(x) + "%" for x in temp_df[temp_df['type'] == i]['percentage'].values]

    data.append(go.Bar(name = i, x =temp_df[temp_df['type'] == i]['world'].values,text = text,textposition='auto',

                      y =  temp_df[temp_df['type'] == i]['percentage'].values,marker=dict(

        color='rgba(' + colors[count] + ',0.6)',

        line=dict(color='rgba(' + colors[count] + ',1.0)', width=1)

    )))

    count = count + 1

fig = go.Figure(data=data)

fig.update_layout(barmode='stack')

fig.update_layout(title={'text': "Percentage of media types in each world",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'} )

fig.show()

# lets see the fav title

temp_df = train_df['title'].str.replace("\(Activity\)","").replace("\(Assessment\)","")

text = ' ' .join(val for val in temp_df)

wordcloud = WordCloud(width=1600, height=800, stopwords = {'None','etc','and','other'}).generate(text)

plt.figure(figsize=(20,10), facecolor='k')

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
train_df['timestamp'] = pd.to_datetime(train_df.timestamp)

train_df['date'] = train_df['timestamp'].dt.date

train_df['month'] = train_df['timestamp'].dt.month_name()

train_df['weekday_name'] = train_df['timestamp'].dt.weekday_name

train_df['hour'] = train_df['timestamp'].dt.hour

train_df['minute'] = train_df['timestamp'].dt.minute
date_df = train_df.groupby("date")['event_id'].count()

month_df = train_df.groupby("month")['event_id'].count().reset_index(name="count")

month_df['month'] = pd.Categorical(month_df['month'],categories=['December','November','October','September','August','July','June','May','April','March','February','January'],ordered=True)

month_df = month_df.sort_values('month',ascending=False)



weekday_df = train_df.groupby("weekday_name")['event_id'].count().reset_index(name="count")

weekday_df['weekday'] = pd.Categorical(weekday_df['weekday_name'],categories=['Saturday','Friday','Thursday','Wednesday','Tuesday','Monday','Sunday'],ordered=True)

weekday_df = weekday_df.sort_values('weekday',ascending=False)



hour_df = train_df.groupby("hour")['event_id'].count()

minute_df = train_df.groupby("minute")['event_id'].count()

fig = make_subplots(rows = 5,cols = 1)



installation_df = train_df.groupby("date")['installation_id'].count()

fig.append_trace(go.Scatter(x = minute_df.index, y = minute_df.values, mode = "lines", name = "Minute"),row=1,col=1)

fig.append_trace(go.Scatter(x = hour_df.index, y = hour_df.values, mode = "markers", name = "Hour"),row=2,col=1)

fig.append_trace(go.Scatter(x = weekday_df['weekday'], y = weekday_df['count'], mode = "lines+markers", name = "Week Day"),row=3,col=1)

fig.append_trace(go.Scatter(x = date_df.index, y = date_df.values, mode = "lines+markers", name = "Date"),row=4,col=1)

fig.append_trace(go.Scatter(x = month_df['month'], y = month_df['count'], mode = "lines", name = "Month"),row=5,col=1)









fig.update_layout(height=1000)

fig.show()
temp_df = train_labels_df.groupby('title')['accuracy_group'].value_counts(normalize=True).reset_index(name="percentage")

temp_df['percentage'] = temp_df['percentage']*100

temp_df = temp_df.round(2)

temp_df['title'] = temp_df['title'].str.replace("\(Assessment\)","")

colors = [x.replace(")","").replace("rgb(","") for x in cl.scales['4']['qual']['Dark2']]

data = []

for i in range(4):

    text = [str(x) + "%" for x in temp_df[temp_df['accuracy_group'] == i]['percentage'].values]

    data.append(go.Bar(name = i, x = temp_df[temp_df['accuracy_group'] == i]['title'].values,

                       text = text,textposition='auto',

                      y = temp_df[temp_df['accuracy_group'] == i]['percentage'].values,marker=dict(

        color='rgba(' + colors[i] + ',0.6)',

        line=dict(color='rgba(' + colors[i] + ',1.0)', width=1)

    )))

fig = go.Figure(data=data)

fig.update_layout(barmode='stack', title={'text': "Percentage of accuracy group for different type of Assessment",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})

fig.show()
temp_df =  train_labels_df['title'].value_counts()

data = go.Bar(x = temp_df.index,y = temp_df.values,text = temp_df.values,  textposition='auto')

fig = go.Figure(data = data)

fig.update_traces(marker_color='#C5197D', marker_line_color='#8E0052',marker_line_width=1.5, opacity=0.6)

fig.update_layout(barmode='stack', title={'text': "Different typess of Assessment",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})

fig.show()
temp_df = train_df[train_df.installation_id=="0001e90f"]

temp_df
print("Out of 1357 rows we have " + str(temp_df.event_id.nunique()) + " unique event ID and " + str(temp_df.game_session.nunique()) + " unique game session")
temp_df[temp_df.game_session == "0848ef14a8dc6892"]
specs_df
train_df['game_time_log'] = train_df['game_time'].apply(np.log1p)

train_df = train_df.head(1000000)

# fig = px.box(train_df, y="game_time_log",x = "type",color='month',title={'text': "Distribution of game_time by type based on month",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},

#              color_discrete_sequence=cl.scales['3']['qual']['Dark2'])

# fig.show()

ax = sns.catplot(x="type", y="game_time_log", data=train_df,col="month",kind="box", aspect=.7)
# fig = px.box(train_df, y="game_time_log",x = "type",color='weekday_name',title={'text': "Distribution of game_time by type based on weekday",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},

#              color_discrete_sequence=cl.scales['3']['qual']['Dark2'])

# fig.show()

ax = sns.catplot(x="type", y="game_time_log", data=train_df,col="weekday_name",kind="box", aspect=.7)
# fig = px.box(train_df, y="game_time_log",x = "type",color='world',title={'text': "Distribution of game_time by type based on world",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},

#              color_discrete_sequence=cl.scales['3']['qual']['Dark2'])

# fig.show()

plt.figure(figsize=(16, 6))



ax = sns.catplot(x="type", y="game_time_log", data=train_df,col="world",kind="strip", aspect=.7)
# fig = px.strip(train_df, y="game_time_log",x = "world",title={'text': "Distribution of game_time by world",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},

#              color_discrete_sequence=cl.scales['3']['qual']['Dark2'])

# fig.show()



ax = sns.catplot(x="world", y="game_time_log", data=train_df,kind="strip", aspect=.7)
incorrect = train_labels_df.groupby(['title','accuracy_group'])['num_incorrect'].value_counts().reset_index(name="count")

correct = train_labels_df.groupby(['title','accuracy_group'])['num_correct'].value_counts().reset_index(name="count")
px.scatter(incorrect[incorrect['title'] == "Bird Measurer (Assessment)"], x="accuracy_group", y="count",color = "num_incorrect",size = "count",hover_name="accuracy_group",title="Bird Measurer incorrect answers")
px.scatter(correct[correct['title'] == "Bird Measurer (Assessment)"], x="accuracy_group", y="count",color = "num_correct",size = "count",hover_name="accuracy_group",title="Bird Measurer correct answers")
px.scatter(incorrect[incorrect['title'] == "Mushroom Sorter (Assessment)"], x="accuracy_group", y="count",color = "num_incorrect",size = "count",hover_name="accuracy_group",title="Mushroom Sorter incorrect answers")
px.scatter(correct[correct['title'] == "Mushroom Sorter (Assessment)"], x="accuracy_group", y="count",color = "num_correct",size = "count",hover_name="accuracy_group",title="Mushroom Sorter correct answers")
px.scatter(incorrect[incorrect['title'] == "Cauldron Filler (Assessment)"], x="accuracy_group", y="count",color = "num_incorrect",size = "count",hover_name="accuracy_group",title="Cauldron Filler incorrect answers")
px.scatter(correct[correct['title'] == "Cauldron Filler (Assessment)"], x="accuracy_group", y="count",color = "num_correct",size = "count",hover_name="accuracy_group",title="Cauldron Filler correct answers")
px.scatter(incorrect[incorrect['title'] == "Chest Sorter (Assessment)"], x="accuracy_group", y="count",color = "num_incorrect",size = "count",hover_name="accuracy_group",title="Chest Sorter incorrect answers")
px.scatter(correct[correct['title'] == "Chest Sorter (Assessment)"], x="accuracy_group", y="count",color = "num_correct",size = "count",hover_name="accuracy_group",title="Chest Sorter correct answers")
px.scatter(incorrect[incorrect['title'] == "Cart Balancer (Assessment)"], x="accuracy_group", y="count",color = "num_incorrect",size = "count",hover_name="accuracy_group",title="Cart Balancer incorrect answers")
px.scatter(correct[correct['title'] == "Cart Balancer (Assessment)"], x="accuracy_group", y="count",color = "num_correct",size = "count",hover_name="accuracy_group",title="Cart Balancer correct answers")
def qwk(act,pred,n=4,hist_range=(0,3)):

    

    O = confusion_matrix(act,pred)

    O = np.divide(O,np.sum(O))

    

    W = np.zeros((n,n))

    for i in range(n):

        for j in range(n):

            W[i][j] = ((i-j)**2)/((n-1)**2)

            

    act_hist = np.histogram(act,bins=n,range=hist_range)[0]

    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]

    

    E = np.outer(act_hist,prd_hist)

    E = np.divide(E,np.sum(E))

    

    num = np.sum(np.multiply(W,O))

    den = np.sum(np.multiply(W,E))

        

    return 1-np.divide(num,den)
list_of_user_activities = list(set(train_df['title'].unique()).union(set(test_df['title'].unique())))

activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))



train_df['title'] = train_df['title'].map(activities_map)

test_df['title'] = test_df['title'].map(activities_map)

train_labels_df['title'] = train_labels_df['title'].map(activities_map)



win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

win_code[activities_map['Bird Measurer (Assessment)']] = 4110



train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
def get_data(user_sample, test_set=False):

    last_activity = 0

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy=0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0 

    accumulated_actions = 0

    counter = 0

    durations = []

    for i, session in user_sample.groupby('game_session', sort=False):

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        if test_set == True:

            second_condition = True

        else:

            if len(session)>1:

                second_condition = True

            else:

                second_condition= False

            

        if (session_type == 'Assessment') & (second_condition):

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            features = user_activities_count.copy()

            features['session_title'] = session['title'].iloc[0] 

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            if durations == []:

                features['duration_mean'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            if accuracy == 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1



            features.update(accuracy_groups)

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            features['accumulated_actions'] = accumulated_actions

            accumulated_accuracy_group += features['accuracy_group']

            accuracy_groups[features['accuracy_group']] += 1

            if test_set == True:

                all_assessments.append(features)

            else:

                if true_attempts+false_attempts > 0:

                    all_assessments.append(features)

                

            counter += 1



        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activity = session_type



    if test_set:

        return all_assessments[-1] 

    return all_assessments
compiled_data = []

installation_id = train_df['installation_id'].nunique()

for i, (ins_id, user_sample) in tqdm(enumerate(train_df.groupby('installation_id', sort=False)), total=installation_id):

    compiled_data += get_data(user_sample)
new_train = pd.DataFrame(compiled_data)

del compiled_data

new_train.shape
new_train.head()
all_features = [x for x in new_train.columns if x not in ['accuracy_group']]

cat_features = ['session_title']

X, y = new_train[all_features], new_train['accuracy_group']

del train_df
clf = CatBoostClassifier(loss_function='MultiClass',task_type="CPU",learning_rate=0.05,iterations=3000,od_type="Iter",early_stopping_rounds=500,random_seed=21)

clf.fit(X, y, verbose=500, cat_features=cat_features)

del X, y
new_test = []

for ins_id, user_sample in tqdm(test_df.groupby('installation_id', sort=False), total=1000):

    a = get_data(user_sample, test_set=True)

    new_test.append(a)

    

X_test = pd.DataFrame(new_test)

del test_df
preds = clf.predict(X_test)

del X_test
submission_df['accuracy_group'] = np.round(preds).astype('int')

submission_df.to_csv('submission.csv', index=None)

submission_df.head()