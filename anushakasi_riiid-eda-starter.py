import os

import pandas as pd



# neural nets

import tensorflow as tf

import tensorflow.keras.models as M

import tensorflow.keras.layers as L



import riiideducation
INPUT_DIR = '/kaggle/input/riiid-test-answer-prediction/'

TRAIN_FILE = os.path.join(INPUT_DIR,'train.csv')

TEST_FILE = os.path.join(INPUT_DIR,'test.csv')

QUES_FILE = os.path.join(INPUT_DIR,'questions.csv')

LEC_FILE = os.path.join(INPUT_DIR,'lectures.csv')
tr = pd.read_csv(TRAIN_FILE, low_memory=False, nrows=10**7)

tr.head()
tr.info()
dir(tr)
print('Num rows, cols:',tr.shape)

print('Num rows:',len(tr))

print('Num cols:',len(tr.columns))

print('Num elements:',tr.size)
print('COLUMNS:\n----------------------------------')



for column in tr.columns:

    print(column)
for column in tr.columns:

    print('\n\n\nThis column is ',column)

    print(tr[column].shape)

    print(dir(tr[column]))
for column in tr.columns:

    print('\n\n\nThis column is ',column)

    print('Num elements: ',tr[column].shape,tr[column].values.shape,tr[column].size)

    print('Num non null elements:',tr[column].count())

    print('Num null elements:',tr[column].isna().sum())

    print('Num unique elements:',tr[column].unique().shape)
print('User answers:',tr['user_answer'].unique())

print('Answered correctly:',tr['answered_correctly'].unique())

print('Prior ques had explanation',tr['prior_question_had_explanation'].unique())
qu = pd.read_csv(QUES_FILE)

qu.head()
print('Num rows',len(qu))

print('Num cols',len(qu.columns))

print('Shape:',qu.shape)

print('Num elements:',qu.size)
print('Num unique questions:',qu['question_id'].unique().shape)

print('Num unique bundles:',qu['bundle_id'].unique().shape)

print('Num unique correct answers:',qu['correct_answer'].unique().shape, 'Unique correct answers:',qu['correct_answer'].unique())

print('Num unique parts:',qu['part'].unique().shape, 'Unique parts:',qu['part'].unique())

print('Num unique tags:',qu['tags'].unique().shape)
le = pd.read_csv(LEC_FILE)
print('Num rows',len(le))

print('Num cols',len(le.columns))

print('Shape:',le.shape)

print('Num elements:',le.size)

print(le.columns)
print('Num unique lec ids:',le['lecture_id'].unique().shape)

print('Num unique tags:',le['tag'].unique().shape)

print('Num unique parts:',le['part'].unique().shape, 'Unique parts:',le['part'].unique())

print('Num unique type of:',le['type_of'].unique().shape, 'Unique type of:', le['type_of'].unique())
# piv1 = tr.loc[tr.answered_correctly!=-1].groupby("content_id")["answered_correctly"].mean().reset_index()

print(tr[:10].loc[tr.answered_correctly!=-1])
# print(dir(tr.loc[tr.answered_correctly!=-1].groupby("content_id")))

# print(type(tr.loc[tr.answered_correctly!=-1].groupby("content_id")))

# print(tr.loc[tr.answered_correctly!=-1].groupby("content_id")['answered_correctly'].sum())

print(tr.loc[tr.answered_correctly!=-1].groupby("content_id")['answered_correctly'].sum().reset_index())
print(tr.loc[tr.answered_correctly!=-1].groupby("task_container_id")['answered_correctly'].sum().reset_index())
print(tr.loc[tr.answered_correctly!=-1].groupby("user_id")['answered_correctly'].sum().reset_index().sort_values(by=["answered_correctly"], ascending=False))
print(tr.loc[tr.answered_correctly!=-1].groupby(["task_container_id","user_id"])['answered_correctly'].max().reset_index())
%%time

piv1 = tr.loc[tr.answered_correctly!=-1].groupby("content_id")["answered_correctly"].mean().reset_index()

piv1.columns = ["content_id", "content_emb"]

piv2 = tr.loc[tr.answered_correctly!=-1].groupby("task_container_id")["answered_correctly"].mean().reset_index()

piv2.columns = ["task_container_id", "task_container_emb"]

piv3 = tr.loc[tr.answered_correctly!=-1].groupby("user_id")["answered_correctly"].mean().reset_index()

piv3.columns = ["user_id", "user_emb"]
TIME_MEAN = tr.prior_question_elapsed_time.median()

TIME_MIN = tr.prior_question_elapsed_time.min()

TIME_MAX = tr.prior_question_elapsed_time.max()

print(TIME_MEAN,TIME_MAX, TIME_MIN)

map_prior = {True:1, False:0}
def preprocess(df):

#     print('before merging:\n',df[:10])

    df = df.merge(piv1, how="left", on="content_id")

#     print('merged piv1:\n',df[:10])

    df["content_emb"] = df["content_emb"].fillna(0.5)

    df = df.merge(piv2, how="left", on="task_container_id")

    df["task_container_emb"] = df["task_container_emb"].fillna(0.5)

    df = df.merge(piv3, how="left", on="user_id")

    df["user_emb"] = df["user_emb"].fillna(0.5)

    df["prior_question_elapsed_time"] = df["prior_question_elapsed_time"].fillna(TIME_MEAN)

    df["duration"] = (df["prior_question_elapsed_time"] - TIME_MIN) / (TIME_MAX - TIME_MIN)

    df["prior_answer"] = df["prior_question_had_explanation"].map(map_prior)

    df["prior_answer"] = df["prior_answer"].fillna(0.5)

    return df
%%time

tr_preprocessed = preprocess(tr)
# print(tr_preprocessed.content_type_id==0)

# print(tr_preprocessed.loc[tr_preprocessed.content_type_id==0])

print(tr_preprocessed.loc[tr_preprocessed.content_type_id==1][:1])
print(tr_preprocessed.loc[tr_preprocessed.content_type_id==0].count())

print(tr_preprocessed.loc[tr_preprocessed.content_type_id==1].count())
FE = ["content_emb", "task_container_emb", "user_emb", "duration", "prior_answer"]

TARGET = "answered_correctly"
x = tr_preprocessed.loc[tr_preprocessed.answered_correctly!=-1, FE].values

y = tr_preprocessed.loc[tr_preprocessed.answered_correctly!=-1, TARGET].values
print(x.shape)

print(y.shape)
print(y[10:100])
def make_ann(n_in):

    inp = L.Input(shape=(n_in,), name="inp")

    d1 = L.Dense(100, activation="relu", name="d1")(inp)

    d2 = L.Dense(100, activation="relu", name="d2")(d1)

    preds = L.Dense(1, activation="sigmoid", name="preds")(d2)

    

    model = M.Model(inp, preds, name="ANN")

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
net = make_ann(x.shape[1])

print(net.summary())

net.fit(x, y, validation_split=0.2, batch_size=30_000, epochs=5)
print("just to see")

print(x[0], y[0])


env = riiideducation.make_env()

iter_test = env.iter_test()

it = 0



for test_df, sample_prediction_df in iter_test:

    print(it)

    it += 1

    if it % 100 == 0:

       print(it)

    test_df = preprocess(test_df)

    x_te = test_df[FE].values

    test_df['answered_correctly'] = net.predict(x_te, batch_size=50_000, verbose=0)[:, 0]

    print(env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']]))
print('hi')
print(dir(env))
# print(env.features)

print(len(list(iter_test)))

print(list(iter_test))
test_df.head()


iter_test = env.iter_test()

(test_df, sample_prediction_df) = next(iter_test)

test_df
