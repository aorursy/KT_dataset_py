import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('../input/poe_stats.csv')

df.head()
df['ladder'].unique()
df_ladder = df.groupby('ladder')

df_ladder.size()
def create_order_by_division(division):

    divisionclass = df_ladder.get_group(division).groupby('class')

    ordered_data = pd.DataFrame(0, index=np.arange(len(divisionclass.size())), columns=['class','number'])

    ordered_data['class'] = divisionclass.size().index

    for i in range(len(ordered_data)):

        ordered_data.iat[i,1] =  int(divisionclass.size()[i])

    ordered_data = ordered_data.sort_values(by = 'number', ascending=False)

    return ordered_data
create_order_by_division('Harbinger').head()
create_order_by_division('SSF Harbinger HC').head()
create_order_by_division('Hardcore Harbinger').head()
create_order_by_division('SSF Harbinger').head()
sns.set_style('darkgrid', {'axis.facecolor':'black'})

f, axes = plt.subplots(2, 2, figsize=(30,45), sharex=True)

DivisionList = df['ladder'].unique()

times = 0

for i in range(2):

    for j in range(2):

        plt.sca(axes[i, j])

        plot_data = create_order_by_division(DivisionList[times])

        plot_data = plot_data.reset_index()

        x = plot_data['number']

        y = len(plot_data.index) - plot_data.index

        labels = plot_data['class']

        plt.scatter(x, y, color='g', label = 'Number of Class')

        plt.xticks(size = 22)

        plt.yticks(y, labels, fontsize=18)

        plt.title('Class by Ordered in '+ DivisionList[times] +' Division', fontsize = 30, color = 'Red')

        plt.legend(loc=2, fontsize =20)

        times=times+1

plt.show()
plt.rcParams.update({'font.size': 16})

df['stream'] = np.zeros(len(df))

df.loc[df['twitch']== 'null', 'stream'] = 'who_not_stream'

df.loc[df['twitch']!= 'null', 'stream'] = 'who_stream'

df['stream'] = df['stream'].astype('category')

sns.set_style('darkgrid', {'axis.facecolor':'black'})

f, axes = plt.subplots(6, 2, figsize=(25,60))

vis1= sns.distplot(df[df['twitch']!= 'null']['challenges'], bins=35, ax=axes[0,0])

axes[0,0].set_title('who_stream', fontsize=16,color = 'red')

axes[0,0].set_ylim(0,0.2)

vis2= sns.distplot(df[df['twitch']== 'null']['challenges'], bins=35,  ax=axes[0,1], label = 'who_not_stream')

axes[0,1].set_title('who_not_stream', fontsize=16,color = 'red')

axes[0,1].set_ylim(0,0.2)

vis3= sns.distplot(df[df['twitch']!= 'null']['rank'], bins=35,  ax=axes[1,0], label = 'who_stream')

axes[1,0].set_title('who_stream', fontsize=16,color = 'red')

axes[1,0].set_ylim(0,0.0001)

vis4= sns.distplot(df[df['twitch']== 'null']['rank'], bins=35,  ax=axes[1,1], label = 'who_not_stream')

axes[1,1].set_title('who_not_stream', fontsize=16,color = 'red')

axes[1,1].set_ylim(0,0.0001)

vis5= sns.distplot(df[((df.twitch != 'null') & ((df.ladder == 'SSF Harbinger HC') | (df.ladder == 'Hardcore Harbinger')))]['dead'], bins=35,  ax=axes[2,0], label = 'who_stream')

axes[2,0].set_title('who_stream', fontsize=16,color = 'red')

axes[2,0].set_ylim(0,25)

vis6= sns.distplot(df[((df.twitch == 'null') & ((df.ladder == 'SSF Harbinger HC') | (df.ladder == 'Hardcore Harbinger')))]['dead'], bins=35,  ax=axes[2,1], label = 'who_not_stream')

axes[2,1].set_title('who_not_stream', fontsize=16,color = 'red')

axes[2,1].set_ylim(0,25)

z = sns.violinplot(data=df, x='stream', y ='challenges',  ax=axes[3,0])

z3 = sns.boxplot(data=df, x='stream', y ='challenges',  ax=axes[3,1])

z4 = sns.violinplot(data=df[((df.ladder == 'SSF Harbinger HC') | (df.ladder == 'Hardcore Harbinger'))], x='stream', y ='level',  ax=axes[4,0])

z5 = sns.boxplot(data=df[((df.ladder == 'SSF Harbinger HC') | (df.ladder == 'Hardcore Harbinger'))], x='stream', y ='level',  ax=axes[4,1])

z6 = sns.violinplot(data=df, x='stream', y ='rank',  ax=axes[5,0])

z7 = sns.boxplot(data=df, x='stream', y ='rank',  ax=axes[5,1])
from sklearn.model_selection import train_test_split

import xgboost as xgb

import warnings

warnings.filterwarnings('ignore')

#data process

df['is_in_top_30'] = np.zeros(len(df))

df.loc[df['rank'] <= 30, 'is_in_top_30'] = True

df.loc[df['rank'] >30,  'is_in_top_30'] = False

labelencoder_y= LabelEncoder()

df['is_in_top_30'] = labelencoder_y.fit_transform(df['is_in_top_30'])

X = df[['dead', 'online', 'level', 'class', 'challenges', 'ladder', 'stream']]

y = df['is_in_top_30']

# Encoding the categorical data

labelencoder_X_1 = LabelEncoder()

X['dead'] = labelencoder_X_1.fit_transform(X['dead'])

labelencoder_X_2 = LabelEncoder()

X['online'] = labelencoder_X_2.fit_transform(X['online'])

labelencoder_X_3 = LabelEncoder()

X['class'] = labelencoder_X_3.fit_transform(X['class'])

labelencoder_X_5 = LabelEncoder()

X['ladder'] = labelencoder_X_5.fit_transform(X['ladder'])

labelencoder_X_6 = LabelEncoder()

X['stream'] = labelencoder_X_6.fit_transform(X['stream'])

# Splitting the dataset into the Training set and Validation set

Xt, Xv, yt, yv = train_test_split(X, y, test_size = 0.20, random_state = 0)

dt = xgb.DMatrix(Xt.as_matrix(),label=yt.as_matrix())

dv = xgb.DMatrix(Xv.as_matrix(),label=yv.as_matrix())
#Build the model

params = {

    "eta": 0.2,

    "max_depth": 4,

    "objective": "binary:logistic",

    "silent": 1,

    "base_score": np.mean(yt),

    'n_estimators': 1000,

    "eval_metric": "logloss"

}

model = xgb.train(params, dt, 5000, [(dt, "train"),(dv, "valid")], verbose_eval=500)
from sklearn.metrics import confusion_matrix

from termcolor import colored

#Prediction on validation set

y_pred = model.predict(dv)



# Making the Confusion Matrix

cm = confusion_matrix(yv, (y_pred>0.5))

print(colored('The Confusion Matrix is: ', 'red'),'\n', cm)

# Calculate the accuracy on test set

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

print(colored('The Accuracy on Test Set is: ', 'blue'), colored(predict_accuracy_on_test_set, 'blue'))
'dead', 'online', 'class', 'challenges', 'ladder', 'stream'

# Input the data you want to predict

print("please input the folowing information: Is dead?")

input_1 = input("'dead':")

print("please input the folowing information: Online?")

input_2 = input("'online':")

print("please input the folowing information: level?")

input_3 = input("'level:'")

print("please input the folowing informatione: Which class?")

input_4 = input("class:")

print("please input the folowing information: Challenges completed?")

input_5 = input("challenges:")

print("please input the folowing information: Which division?")

input_6 = input("ladder:")

print("please input the folowing information: Is stream?")

input_7 = input("stream:")

#  Encoding categorical data

input_1 = labelencoder_X_1.transform(np.array([[input_1]]))

input_2 = labelencoder_X_2.transform(np.array([[input_2]]))

input_4 = labelencoder_X_3.transform(np.array([[input_4]]))

input_6 = labelencoder_X_5.transform(np.array([[input_6]]))

input_7 = labelencoder_X_6.transform(np.array([[input_7]]))

# Make prediction

new_prediction = model.predict(xgb.DMatrix([[int(input_1), int(input_2), int(input_3), int(input_4), int(input_5), int(input_6), int(input_7)]]))

if(new_prediction > 0.5):

    print(colored("You Should be in Top 30!", 'red'))

else:

    print(colored("You Should not be in Top 30!", 'green'))
df_ladder = df.groupby('ladder')

Mean_challenges = df_ladder['challenges'].mean()

Mean_challenges
plt.rcParams.update({'font.size': 32})

plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')

vis = sns.violinplot(data=df, x='ladder', y ='challenges', fontsize = 35, hue = 'stream', palette="muted", split=True)
died = df[df['dead']==True]

plt.rcParams.update({'font.size': 20})

plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')

ax = sns.boxplot(data=died, y='class', x ='level')