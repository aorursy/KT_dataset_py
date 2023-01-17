# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier, plot_importance
import category_encoders as ce

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/riiid-test-answer-prediction/train.csv",
                    nrows=10**5,
                    dtype={'row_id': 'int64', 
                           'timestamp': 'int64', 
                           'user_id': 'int32', 
                           'content_id': 'int16', 
                           'content_type_id': 'int8',
                           'task_container_id': 'int16', 
                           'user_answer': 'int8', 
                           'answered_correctly': 'int8', 
                           'prior_question_elapsed_time': 'float32', 
                           'prior_question_had_explanation': 'boolean'}
                      )
print("train shape: ",train.shape)
lectures = pd.read_csv("/kaggle/input/riiid-test-answer-prediction/lectures.csv")
print("lectures shape: ",lectures.shape)
questions = pd.read_csv("/kaggle/input/riiid-test-answer-prediction/questions.csv")
print("questions shape: ",questions.shape)
# partMapping 1;4 = Listening, 5-7; Reading
# Listening : 0
# Reading : 1
def TOEICSection(part):
    if part >= 1 and part <= 4:
        return "Listening"
    elif part >= 5 and part <= 7:
        return "Reading"
    else:
        return "Missing"
questionsdf = pd.concat([questions, lectures.rename({"lecture_id":"question_id"}, axis = 1)], axis = 0).reset_index(drop = True)
questionsdf.tags= questionsdf.tags.fillna(questionsdf.tag)

questionsdf.type_of = questionsdf.type_of.fillna("question")
questionsdf["content_type_id"] = questionsdf["type_of"] != 'question'
# questionsdf["content_type_id"] = questionsdf["type_of"].apply(lambda x : 1 if x != 'question' else 0)
questionsdf = questionsdf.drop("tag", axis = 1)
questionsdf = questionsdf.fillna(-1)

questionsdf = questionsdf.rename({"question_id": "content_id"}, axis=1)

questionsdf.tags = questionsdf.tags.apply(lambda x: [int(x)] if type(x) != str else list(map(int, x.split())))
questionsdf['reading_section'] = questionsdf['part'].apply(lambda x: TOEICSection(x))
questionsdf["tags"] = questionsdf["tags"].apply(lambda x: sorted(x))
questionsdf["tag_len"] = questionsdf["tags"].apply(lambda x: len(x))
questionsdf[['tags1','tags2','tags3','tags4','tags5','tags6']] = pd.DataFrame(questionsdf["tags"] .tolist())
questionsdf["tags1"] = questionsdf["tags1"].fillna(-1)
questionsdf["tags2"] = questionsdf["tags2"].fillna(-1)
questionsdf["tags3"] = questionsdf["tags3"].fillna(-1)
questionsdf["tags4"] = questionsdf["tags4"].fillna(-1)
questionsdf["tags5"] = questionsdf["tags5"].fillna(-1)
questionsdf["tags6"] = questionsdf["tags6"].fillna(-1)

questionsdf["tags1"] = questionsdf["tags1"].astype(int)
questionsdf["tags2"] = questionsdf["tags2"].astype(int)
questionsdf["tags3"] = questionsdf["tags3"].astype(int)
questionsdf["tags4"] = questionsdf["tags4"].astype(int)
questionsdf["tags5"] = questionsdf["tags5"].astype(int)
questionsdf["tags6"] = questionsdf["tags6"].astype(int)
question_train_data = train.loc[train.content_type_id == 0,]
lectore_train_data = train.loc[train.content_type_id == 1,]
que_df = questionsdf.loc[questionsdf.content_type_id == False,]
lec_df = questionsdf.loc[questionsdf.content_type_id == True,]
print(question_train_data.shape)
print(lectore_train_data.shape)
print(que_df.shape)
print(lec_df.shape)
question_train_data = question_train_data.merge(que_df, how = "inner", on = ["content_id", "content_type_id"])
lectore_train_data = lectore_train_data.merge(lec_df, how = "inner", on = ["content_id", "content_type_id"])
train_df = pd.concat([question_train_data,lectore_train_data])
remove_col = ["row_id","correct_answer"]
train_df = train_df.drop(remove_col, axis = 1)
train_df["bundle_id"] = train_df["bundle_id"].astype(int)
train_que_df = train_df.loc[train_df["content_type_id"] == 0,]
figure, ax = plt.subplots(figsize = (15,10))
sns.heatmap(train_que_df.corr(), annot = True, cmap="YlGnBu")
figure, ax = plt.subplots(figsize = (15,10))
plt.subplot(3,2,1)
sns.distplot(train_que_df["user_id"])
plt.subplot(3,2,2)
sns.distplot(train_que_df["content_id"])
plt.subplot(3,2,3)
sns.distplot(train_que_df["task_container_id"])
plt.subplot(3,2,4)
sns.distplot(train_que_df["prior_question_elapsed_time"])
figure, ax = plt.subplots(figsize = (10,8))
sns.scatterplot(x = "user_id", y = "prior_question_elapsed_time", hue = "user_answer", size = "user_answer", data = train_que_df)
figure, ax = plt.subplots(figsize = (10,8))
sns.scatterplot(x = "user_id", y = "task_container_id", hue = "user_answer", size = "user_answer", data = train_que_df)
## Mean time users took to solve the questions
user_mean = train_que_df.groupby('user_id')["timestamp"].mean()
user_df = pd.DataFrame(user_mean)
user_df["user_id"] = user_df.index
figure, ax = plt.subplots(figsize = (10,8))
sns.scatterplot(x = "user_id",y ="timestamp", data = user_df)
correct_ans = train_que_df['answered_correctly'].value_counts().reset_index()
correct_ans.columns = ['answered_correctly','per']
correct_ans['per'] /= len(train_que_df)

colours =  ('blue','violet')
explode = (0.1, 0.1)
def func(pct, allvalues): 
    absolute = int(pct / 100.*np.sum(allvalues)) 
    return "{:.1f}%".format(pct)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,22))
wedges, texts, autotexts = ax1.pie(correct_ans['per'],  
                                  autopct = lambda pct: func(pct, correct_ans['per']), 
                                  explode = explode,  
                                  labels = correct_ans['answered_correctly'], 
                                  shadow = True, 
                                  colors = colours, 
                                  startangle = 90, 
                                
                                  textprops = dict(color ="white"))

correct_ans = train_que_df['user_answer'].value_counts().reset_index()
correct_ans.columns = ['user_answer','per']
correct_ans['per'] /= len(train_que_df)

colors1 = ('blue','violet','brown','black')

explode1 = (0.1,0.1,0.1,0.1)
wedges, texts, autotexts = ax2.pie(correct_ans['per'],  
                                  autopct = lambda pct: func(pct, correct_ans['per']), 
                                  explode = explode1,  
                                  labels = correct_ans['user_answer'], 
                                  shadow = True, 
                                  colors = colors1, 
                                  startangle = 90, 
                                  textprops = dict(color ="white")) 
mini_df= train_que_df.copy()


mini_df = mini_df.sort_values(by=['timestamp'])
mini_df = mini_df.drop_duplicates('timestamp')
mini_df["timestamp"] = mini_df["timestamp"] / 1000000

plt.figure(figsize=(20,10))
sns.set_style('dark')
plt.subplot(3, 1, 1)
mid_df = mini_df.head(100)
sns.pointplot(x = "timestamp", 
              y = "answered_correctly",              
              data = mid_df, 
              linestyle='--',
              color='violet',
              hue = 'prior_question_had_explanation',
              markers='x')

plt.subplot(3, 1, 2)
mid_df = mini_df[50000:51100]
sns.pointplot(x = "timestamp", 
              y = "answered_correctly",              
              data = mid_df, 
              linestyle='--',
              color='violet',
              hue = 'prior_question_had_explanation',
              markers='x')

plt.subplot(3, 1, 3)
mid_df = mini_df.tail(100)
sns.pointplot(x = "timestamp", 
              y = "answered_correctly",              
              data = mid_df, 
              linestyle='--',
              color='violet',
              hue = 'prior_question_had_explanation',
              markers='x')
plt.figure(figsize = (10,8))
sns.countplot('user_answer', hue = 'prior_question_had_explanation', data = train_que_df)
plt.figure(figsize = (10,8))
sns.scatterplot(x = "task_container_id", y = "prior_question_elapsed_time", hue = "user_id", data = train_que_df , size='user_id' ,alpha=0.7)
plt.figure(figsize = (10,8))
sns.scatterplot(x = "task_container_id", y = "prior_question_elapsed_time", hue = "answered_correctly", data = train_que_df , size='answered_correctly' ,alpha=0.7)
plt.figure(figsize = (10,8))
sns.barplot(x = "part", y = "bundle_id", hue = 'answered_correctly', data = train_que_df)
plt.figure(figsize = (10,8))
sns.barplot(x = "part", y = "user_id", hue = 'answered_correctly', data = train_que_df)
training = train_que_df.copy()
training.shape
training.dropna(inplace=True)
training.reset_index(drop=True, inplace=True)
def max_value(inputlist):
    return max([sublist[-1] for sublist in inputlist])

def generateSequencedlist(inputList):
    blankLs = [0] * maxLength
    for l in inputList:
        blankLs[l-1] = 1
    return blankLs
maxLength = max_value(training.tags.tolist())
stags = pd.DataFrame(training.tags.apply(lambda x : generateSequencedlist(x)).tolist())
colDict = {}
for c in stags.columns:
    colDict[c] = "tag_{}".format(c)
stags.rename(columns=colDict, inplace = True)
# training = pd.concat([training, stags], axis = 1)
removeCol = ["bundle_id","tags","content_type_id"]
training = training.drop(removeCol, axis = 1)
# apply StandardScaler
scaler = StandardScaler() 
standard_df = scaler.fit_transform(training[['timestamp',"prior_question_elapsed_time"]]) 
standard_df = pd.DataFrame(standard_df, columns =['timestamp',"prior_question_elapsed_time"]) 
training["timestamp"] = standard_df["timestamp"]
training["prior_question_elapsed_time"] = standard_df["prior_question_elapsed_time"]
# "type_of", "reading_section"
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder() 
training["type_of"] = label_encoder.fit_transform(training['type_of']) 
training["reading_section"] = label_encoder.fit_transform(training['reading_section']) 
training["prior_question_had_explanation"] = label_encoder.fit_transform(training['prior_question_had_explanation']) 
grouped_by_user_df = training.groupby('user_id')
user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count', 'sum']}).copy()
user_answers_df.columns = ['mean_answered_correctly_user', 'questions_answered', 'sum_answered_correctly_user']


grouped_by_content_df = training.groupby('content_id')
content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count'] }).copy()
content_answers_df.columns = ['mean_answered_correctly_content', 'question_asked']

training = training.merge(user_answers_df, on = "user_id")
training = training.merge(content_answers_df, on = "content_id")
# Apply encoding for user_id,content_id,task_container_id,bundle_id,part
y = training["answered_correctly"]
X = training.drop(["answered_correctly"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = XGBClassifier(
    tree_method="hist",
    learning_rate=0.1,
    gamma=0.2,
    n_estimators=200,
    max_depth=8,
    min_child_weight=40,
    subsample=0.87,
    colsample_bytree=0.95,
    reg_alpha=0.04,
    reg_lambda=0.073,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27
)
# cross_val_score(model, X_train, y_train.values.ravel(), cv=5, scoring="roc_auc")
model.fit(X_train, y_train.values.ravel())
roc_auc_score(y_train.values, model.predict_proba(X_train)[:,1])
roc_auc_score(y_test.values, model.predict_proba(X_test)[:,1])
fig, ax = plt.subplots(figsize=(10,8))
plot_importance(model, ax=ax)
plt.show()
