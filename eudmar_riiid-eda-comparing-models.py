import numpy as np

import pandas as pd



# Plot

import matplotlib.pyplot as plt

from matplotlib import pyplot

import seaborn as sns



# Training and test data

from sklearn.model_selection import train_test_split



# AUC score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



# Model

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier



# Submission

import riiideducation



import warnings

warnings.filterwarnings("ignore")
df_train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', 

                       nrows=10**6,

                       dtype={'row_id': 'int64', 

                              'timestamp': 'int64', 

                              'user_id': 'int32',

                              'content_id': 'int16',

                              'content_type_id': 'int8',

                              'task_container_id': 'int16',

                              'user_answer': 'int8',

                              'answered_correctly': 'int8',

                              'prior_question_elapsed_time': 'float32',

                              'prior_question_had_explanation': 'boolean'})
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

    return summary
resumetable(df_train)
plt.figure(figsize=(15, 5))



plt.suptitle('Time between this interaction and first event', fontsize = 18)

plt.hist(df_train['timestamp'], bins = 50, color = "skyblue")

plt.ylabel('Count', fontsize = 15)

plt.xlabel('timestamp', fontsize = 15)



plt.show()
plt.figure(figsize=(15, 5))



p = sns.distplot(df_train['user_id'])

p.set_title("Code for the user", fontsize=18)

p.set_xlabel("user_id", fontsize = 15)

p.set_ylabel("Probability", fontsize = 15)



plt.show()
plt.figure(figsize=(15, 5))



p = sns.distplot(df_train['content_id'])

p.set_title("The user interaction", fontsize = 18)

p.set_xlabel("content_id", fontsize = 15)

p.set_ylabel("Probability", fontsize = 15)



plt.show()
plt.figure(figsize=(15, 5))



p3 = sns.distplot(df_train['task_container_id'])

p3.set_title("Code for the batch of questions or lectures", fontsize = 18)

p3.set_xlabel("task_container_id", fontsize = 15)

p3.set_ylabel("Probability", fontsize = 15)



plt.show()
plt.figure(figsize=(15, 5))



p3 = sns.distplot(df_train['prior_question_elapsed_time'].dropna())

p3.set_title("How long it took a user to answer their previous question bundle", fontsize = 18)

p3.set_xlabel("prior_question_elapsed_time", fontsize = 15)

p3.set_ylabel("Probability", fontsize = 15)



plt.show()
plt.figure(figsize=(12, 5))



freq = len(df_train)



g = sns.countplot(df_train['content_type_id'])

g.set_title("", fontsize = 18)

g.set_xlabel("content_type_id", fontsize = 15)

g.set_ylabel("Count", fontsize = 15)



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x() + p.get_width() / 2., height + 3,

          '{:1.2f}%'.format(height / freq * 100),

          ha = "center", fontsize = 18)
plt.figure(figsize=(15, 5))



freq = len(df_train)



g = sns.countplot(df_train['user_answer'])

g.set_title("The user's answer to the question", fontsize = 18)

g.set_xlabel("user_answer", fontsize = 15)

g.set_ylabel("Count", fontsize = 15)



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x() + p.get_width() / 2., height + 3,

          '{:1.2f}%'.format(height / freq * 100),

          ha = "center", fontsize = 18)
plt.figure(figsize=(15, 5))



freq = len(df_train)



g = sns.countplot(df_train['answered_correctly'])

g.set_title("If the user responded correctly", fontsize = 18)

g.set_xlabel("answered_correctly", fontsize = 15)

g.set_ylabel("Count", fontsize = 15)



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x() + p.get_width() / 2., height + 3,

          '{:1.2f}%'.format(height / freq * 100),

          ha = "center", fontsize = 18)
plt.figure(figsize=(12, 5))



freq = len(df_train)



g = sns.countplot(df_train['prior_question_had_explanation'])

g.set_title("Whether or not the user saw an explanation and the correct response (s) \n after answering the previous question bundle",

            fontsize = 18)

g.set_xlabel("prior_question_had_explanation", fontsize = 15)

g.set_ylabel("Count", fontsize = 15)



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x() + p.get_width() / 2., height + 3,

          '{:1.2f}%'.format(height / freq * 100),

          ha = "center", fontsize = 18)
plt.figure(figsize=(12, 5))

g = sns.scatterplot(data = df_train, x = "timestamp", y = "prior_question_elapsed_time", hue = "prior_question_had_explanation", 

                style = "prior_question_had_explanation")

g.set_xlabel("timestamp", fontsize = 15)

g.set_ylabel("prior_question_elapsed_time", fontsize = 15)



plt.show()
train = df_train[df_train['answered_correctly']!=-1]
plt.figure(figsize=(15, 5))



sns.relplot(

    data= train, x = "timestamp", y = "prior_question_elapsed_time",

    col = "prior_question_had_explanation", hue = "answered_correctly", style = "answered_correctly",

    kind="scatter"

);
used_data_types_dict = {

    'row_id': 'int64',

    'timestamp': 'int64',

    'user_id': 'int32',

    'content_id': 'int16',

    'answered_correctly': 'int8',

    'prior_question_elapsed_time': 'float16'

}



train_df = pd.read_csv(

    '/kaggle/input/riiid-test-answer-prediction/train.csv',

    usecols = used_data_types_dict.keys(),

    dtype=used_data_types_dict, 

    index_col = 0

)
features_df = train_df.iloc[:int(9 /10 * len(train_df))]

train_df = train_df.iloc[int(9 /10 * len(train_df)):]
train_questions_only_df = features_df[features_df['answered_correctly']!=-1]

grouped_by_user_df = train_questions_only_df.groupby('user_id')

user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count', 'std']}).copy()

user_answers_df.columns = ['mean_user_accuracy', 'questions_answered', 'std_user_accuracy']
grouped_by_content_df = train_questions_only_df.groupby('content_id')

content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count', 'std'] }).copy()

content_answers_df.columns = ['mean_accuracy', 'question_asked', 'std_accuracy']
import gc



del features_df

del grouped_by_user_df

del grouped_by_content_df



gc.collect()
features = [

    'timestamp',

    'mean_user_accuracy', 

    'questions_answered',

    'std_user_accuracy',

    'mean_accuracy', 

    'question_asked',

    'std_accuracy',

    'prior_question_elapsed_time'

]

target = 'answered_correctly'
train_df = train_df[train_df[target] != -1]
train_df = train_df.merge(user_answers_df, how='left', on='user_id')

train_df = train_df.merge(content_answers_df, how='left', on='content_id')

train_df
train_df = train_df[features + [target]]
train_df = train_df.replace([np.inf, -np.inf], np.nan)

train_df = train_df.fillna(0)

train_df
# Function to reduce the df size

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
# Reducing memory

train_df = reduce_mem_usage(train_df)
# Training and test data

train_df, test_df = train_test_split(train_df, random_state = 123, test_size = 0.3)
# Creating the model

model_LR = LogisticRegression()



# Training the model

model_LR.fit(train_df[features], train_df[target])
ns_probs = [0 for _ in range(len(train_df[target]))]
# predict probabilities

LR_probs = model_LR.predict_proba(train_df[features])



# keep probabilities for the positive outcome only

LR_probs = LR_probs[:, 1]



# calculate scores

ns_auc = roc_auc_score(train_df[target], ns_probs)

LR_auc = roc_auc_score(train_df[target], LR_probs)



# result print

print('Logistic: ROC AUC = %.3f' % (LR_auc * 100))
# calculate roc curves

ns_fpr, ns_tpr, _ = roc_curve(train_df[target], ns_probs)

LR_fpr, LR_tpr, _ = roc_curve(train_df[target], LR_probs)



# figure size

plt.rcParams["figure.figsize"] = (9, 5)



# plot the roc curve for the model

pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

pyplot.plot(LR_fpr, LR_tpr, marker='.', label='Logistic')



# axis labels

pyplot.xlabel('False Positive Rate', fontsize = 15)

pyplot.ylabel('True Positive Rate', fontsize = 15)



# show the legend

pyplot.legend(fontsize = 15)



# show the plot

pyplot.show()
# Creating the model

model_XGB = XGBClassifier()



# Training the model

model_XGB.fit(train_df[features], train_df[target])
# predict probabilities

XGB_probs = model_XGB.predict_proba(train_df[features])



# keep probabilities for the positive outcome only

XGB_probs = XGB_probs[:, 1]



# calculate scores

ns_auc = roc_auc_score(train_df[target], ns_probs)

XGB_auc = roc_auc_score(train_df[target], XGB_probs)



# result print

print('XGBoost: ROC AUC = %.3f' % (XGB_auc * 100))
# calculate roc curves

ns_fpr, ns_tpr, _ = roc_curve(train_df[target], ns_probs)

XGB_fpr, XGB_tpr, _ = roc_curve(train_df[target], XGB_probs)



# figure size

plt.rcParams["figure.figsize"] = (9, 5)



# plot the roc curve for the model

pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

pyplot.plot(LR_fpr, LR_tpr, marker='.', label='Logistic')

pyplot.plot(XGB_fpr, XGB_tpr, marker='.', label='XGBoost')



# axis labels

pyplot.xlabel('False Positive Rate', fontsize = 15)

pyplot.ylabel('True Positive Rate', fontsize = 15)



# show the legend

pyplot.legend(fontsize = 15)



# show the plot

pyplot.show()
env = riiideducation.make_env()



iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    test_df = test_df.merge(user_answers_df, how = 'left', on = 'user_id')

    test_df = test_df.merge(content_answers_df, how = 'left', on = 'content_id')

    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value = False).astype(bool)

    test_df.fillna(value = -1, inplace = True)

    

    test_df['answered_correctly'] = model_XGB.predict_proba(test_df[features])[:,1]

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])