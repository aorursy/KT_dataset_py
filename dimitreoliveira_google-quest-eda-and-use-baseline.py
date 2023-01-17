import gc

import random

import warnings

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

from nltk import FreqDist

from nltk.corpus import stopwords

from ml_stratifiers import MultilabelStratifiedKFold

from sklearn.model_selection import train_test_split

from tensorflow.keras import Model, optimizers

from tensorflow.keras.layers import Lambda, Input, Dense, Dropout, Concatenate, BatchNormalization, Activation

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from googleqa_utilityscript import *



SEED = 0

seed_everything(SEED)

warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)

plt.rcParams.update({'font.size': 16})
train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

test = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')



train['set'] = 'train'

test['set'] = 'test'

complete_set = train.append(test)



print('Train samples: %s' % len(train))

print('Test samples: %s' % len(test))

display(train.head())
samp_id = 9

print('Question Title: %s \n' % train['question_title'].values[samp_id])

print('Question Body: %s \n' % train['question_body'].values[samp_id])

print('Answer: %s' % train['answer'].values[samp_id])
question_target_cols = ['question_asker_intent_understanding','question_body_critical', 'question_conversational', 

                        'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer',

                        'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent', 

                        'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice',

                        'question_type_compare', 'question_type_consequence', 'question_type_definition', 

                        'question_type_entity', 'question_type_instructions', 'question_type_procedure',

                        'question_type_reason_explanation', 'question_type_spelling', 'question_well_written']

answer_target_cols = ['answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance',

                      'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure', 

                      'answer_type_reason_explanation', 'answer_well_written']

target_cols = question_target_cols + answer_target_cols



print('Question labels')

display(train.iloc[[samp_id]][question_target_cols])

print('Answer labels')

display(train.iloc[[samp_id]][answer_target_cols])
train_users = set(train['question_user_page'].unique())

test_users = set(test['question_user_page'].unique())



print('Unique users in train set: %s' % len(train_users))

print('Unique users in test set: %s' % len(test_users))

print('Users in both sets: %s' % len(train_users & test_users))

print('What users are in both sets? %s' % list(train_users & test_users))
train_users = set(train['answer_user_page'].unique())

test_users = set(test['answer_user_page'].unique())



print('Unique users in train set: %s' % len(train_users))

print('Unique users in test set: %s' % len(test_users))

print('Users in both sets: %s' % len(train_users & test_users))
question_gp = complete_set[['qa_id', 'question_user_name', 'question_user_page']].groupby(['question_user_name', 'question_user_page'], as_index=False).count()

question_gp.columns = ['question_user_name', 'question_user_page', 'count']

display(question_gp.sort_values('count', ascending=False).head())



train_question_gp = train[['qa_id', 'question_user_page']].groupby('question_user_page', as_index=False).count()

test_question_gp = test[['qa_id', 'question_user_page']].groupby('question_user_page', as_index=False).count()

train_question_gp.columns = ['question_user_page', 'Question count']

test_question_gp.columns = ['question_user_page', 'Question count']



sns.set(style="darkgrid")

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

sns.countplot(x="Question count", data=train_question_gp, palette="Set3", ax=ax1).set_title("Train")

sns.countplot(x="Question count", data=test_question_gp, palette="Set3", ax=ax2).set_title("Test")

plt.show()
answer_gp = complete_set[['qa_id', 'answer_user_name', 'answer_user_page']].groupby(['answer_user_name', 'answer_user_page'], as_index=False).count()

answer_gp.columns = ['answer_user_name', 'answer_user_page', 'count']

display(answer_gp.sort_values('count', ascending=False).head())



train_answer_gp = train[['qa_id', 'answer_user_page']].groupby('answer_user_page', as_index=False).count()

test_answer_gp = test[['qa_id', 'answer_user_page']].groupby('answer_user_page', as_index=False).count()

train_answer_gp.columns = ['answer_user_page', 'Answer count']

test_answer_gp.columns = ['answer_user_page', 'Answer count']



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

sns.countplot(x="Answer count", data=train_answer_gp, palette="Set3", ax=ax1).set_title("Train")

sns.countplot(x="Answer count", data=test_answer_gp, palette="Set3", ax=ax2).set_title("Test")

plt.show()
question_title_gp = complete_set[['qa_id', 'question_title']].groupby('question_title', as_index=False).count()

question_title_gp.columns = ['question_title', 'count']

display(question_title_gp.sort_values('count', ascending=False).head())



train_question_title_gp = train[['qa_id', 'question_title']].groupby('question_title', as_index=False).count()

test_question_title_gp = test[['qa_id', 'question_title']].groupby('question_title', as_index=False).count()

train_question_title_gp.columns = ['question_title', 'Question title count']

test_question_title_gp.columns = ['question_title', 'Question title count']



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

sns.countplot(x="Question title count", data=train_question_title_gp, palette="Set3", ax=ax1).set_title("Train")

sns.countplot(x="Question title count", data=test_question_title_gp, palette="Set3", ax=ax2).set_title("Test")

plt.show()
question_body_gp = complete_set[['qa_id', 'question_body']].groupby('question_body', as_index=False).count()

question_body_gp.columns = ['question_body', 'count']

display(question_body_gp.sort_values('count', ascending=False).head())



train_question_body_gp = train[['qa_id', 'question_body']].groupby('question_body', as_index=False).count()

test_question_body_gp = test[['qa_id', 'question_body']].groupby('question_body', as_index=False).count()

train_question_body_gp.columns = ['question_body', 'Question body count']

test_question_body_gp.columns = ['question_body', 'Question body count']



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

sns.countplot(x="Question body count", data=train_question_body_gp, palette="Set3", ax=ax1).set_title("Train")

sns.countplot(x="Question body count", data=test_question_body_gp, palette="Set3", ax=ax2).set_title("Test")

plt.show()
complete_set['question_title_len'] = complete_set['question_title'].apply(lambda x : len(x))

complete_set['question_body_len'] = complete_set['question_body'].apply(lambda x : len(x))

complete_set['answer_len'] = complete_set['answer'].apply(lambda x : len(x))

complete_set['question_title_wordCnt'] = complete_set['question_title'].apply(lambda x : len(x.split(' ')))

complete_set['question_body_wordCnt'] = complete_set['question_body'].apply(lambda x : len(x.split(' ')))

complete_set['answer_wordCnt'] = complete_set['answer'].apply(lambda x : len(x.split(' ')))



f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7), sharex=True)

sns.distplot(complete_set[complete_set['set'] == 'train']['question_title_len'], ax=ax1).set_title("Train")

sns.distplot(complete_set[complete_set['set'] == 'test']['question_title_len'], ax=ax2).set_title("Test")

plt.show()



f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7), sharex=True)

sns.distplot(complete_set[complete_set['set'] == 'train']['question_title_wordCnt'], ax=ax1).set_title("Train")

sns.distplot(complete_set[complete_set['set'] == 'test']['question_title_wordCnt'], ax=ax2).set_title("Test")

plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7), sharex=True)

sns.distplot(complete_set[complete_set['set'] == 'train']['question_body_len'], ax=ax1).set_title("Train")

sns.distplot(complete_set[complete_set['set'] == 'test']['question_body_len'], ax=ax2).set_title("Test")

plt.show()



f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7), sharex=True)

sns.distplot(complete_set[complete_set['set'] == 'train']['question_body_wordCnt'], ax=ax1).set_title("Train")

sns.distplot(complete_set[complete_set['set'] == 'test']['question_body_wordCnt'], ax=ax2).set_title("Test")

plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7), sharex=True)

sns.distplot(complete_set[complete_set['set'] == 'train']['answer_len'], ax=ax1).set_title("Train")

sns.distplot(complete_set[complete_set['set'] == 'test']['answer_len'], ax=ax2).set_title("Test")

plt.show()



f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 7), sharex=True)

sns.distplot(complete_set[complete_set['set'] == 'train']['answer_wordCnt'], ax=ax1).set_title("Train")

sns.distplot(complete_set[complete_set['set'] == 'test']['answer_wordCnt'], ax=ax2).set_title("Test")

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 7), sharex=True)

sns.countplot(complete_set[complete_set['set'] == 'train']['category'], ax=ax1).set_title("Train")

sns.countplot(complete_set[complete_set['set'] == 'test']['category'], ax=ax2).set_title("Test")

plt.show()
complete_set['host_first'] = complete_set['host'].apply(lambda x : x.split('.')[0])

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), sharex=True)

sns.countplot(y=complete_set[complete_set['set'] == 'train']['host_first'], ax=ax1, palette="muted").set_title("Train")

sns.countplot(y=complete_set[complete_set['set'] == 'test']['host_first'], ax=ax2, palette="muted").set_title("Test")

plt.show()
f = plt.subplots(figsize=(24, 7))

for col in question_target_cols[:5]:

    sns.distplot(train[col], label=col, rug=True, hist=False)

plt.show()



f = plt.subplots(figsize=(24, 7))

for col in question_target_cols[5:10]:

    sns.distplot(train[col], label=col, rug=True, hist=False)

plt.show()



f = plt.subplots(figsize=(24, 7))

for col in question_target_cols[10:15]:

    sns.distplot(train[col], label=col, rug=True, hist=False)

plt.show()



f = plt.subplots(figsize=(24, 7))

for col in question_target_cols[15:]:

    sns.distplot(train[col], label=col, rug=True, hist=False)

plt.show()
f = plt.subplots(figsize=(24, 7))

for col in answer_target_cols[:5]:

    sns.distplot(train[col], label=col, rug=True, hist=False)

plt.show()



f = plt.subplots(figsize=(24, 7))

for col in answer_target_cols[5:]:

    sns.distplot(train[col], label=col, rug=True, hist=False)

plt.show()
eng_stopwords = stopwords.words('english')



complete_set['question_title'] = complete_set['question_title'].str.replace('[^a-z ]','')

complete_set['question_body'] = complete_set['question_body'].str.replace('[^a-z ]','')

complete_set['answer'] = complete_set['answer'].str.replace('[^a-z ]','')

complete_set['question_title'] = complete_set['question_title'].apply(lambda x: x.lower())

complete_set['question_body'] = complete_set['question_body'].apply(lambda x: x.lower())

complete_set['answer'] = complete_set['answer'].apply(lambda x: x.lower())



freq_dist = FreqDist([word for comment in complete_set['question_title'] for word in comment.split() if word not in eng_stopwords])

plt.figure(figsize=(20, 6))

plt.title('Word frequency on question title').set_fontsize(20)

freq_dist.plot(60, marker='.', markersize=10)

plt.show()



freq_dist = FreqDist([word for comment in complete_set['question_body'] for word in comment.split() if word not in eng_stopwords])

plt.figure(figsize=(20, 6))

plt.title('Word frequency on question body').set_fontsize(20)

freq_dist.plot(60, marker='.', markersize=10)

plt.show()



freq_dist = FreqDist([word for comment in complete_set['answer'] for word in comment.split() if word not in eng_stopwords])

plt.figure(figsize=(20, 6))

plt.title('Word frequency on answer').set_fontsize(20)

freq_dist.plot(60, marker='.', markersize=10)

plt.show()
gc.collect()
EPOCHS = 12

BATCH_SIZE = 32

LEARNING_RATE = 3e-4

EMBEDDDING_SIZE = 512

N_CLASS = len(target_cols)

ES_PATIENCE = 3

RLROP_PATIENCE = 2

DECAY_DROP = 0.3

module_url = "../input/universalsentenceencodermodels/universal-sentence-encoder-models/use-qa"



es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)

rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
use_embed = hub.load(module_url)



def USEEmbedding(x):

    return use_embed(tf.squeeze(tf.cast(x, tf.string)))



def model_fn():

    input_title = Input(shape=(1,), dtype=tf.string, name='input_title')

    embedding_title = Lambda(USEEmbedding, output_shape=(EMBEDDDING_SIZE,))(input_title)



    input_body = Input(shape=(1,), dtype=tf.string, name='input_body')

    embedding_body = Lambda(USEEmbedding, output_shape=(EMBEDDDING_SIZE,))(input_body)



    input_answer = Input(shape=(1,), dtype=tf.string, name='input_answer')

    embedding_answer = Lambda(USEEmbedding, output_shape=(EMBEDDDING_SIZE,))(input_answer)



    x = Concatenate()([embedding_title, embedding_body, embedding_answer])

    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu')(x)

    x = Dropout(0.5)(x)

    output = Dense(N_CLASS, activation='sigmoid', name='output')(x)

    model = Model(inputs=[input_title, input_body, input_answer], outputs=[output])



    optimizer = optimizers.Adam(LEARNING_RATE)

    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    

    return model
feature_cols = ['question_title', 'question_body', 'answer']

Y_train = train[target_cols]



NUM_FOLDS = 3

train_rho_kfolds = []

valid_rho_kfolds = []

model_path_list = []

kf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, random_state=SEED)



for ind, (tr, val) in enumerate(kf.split(train[feature_cols], Y_train)):

    print('FOLD', ind+1)

    X_tr = train[feature_cols].loc[tr]

    y_tr = Y_train.loc[tr].values

    X_vl = train[feature_cols].loc[val]

    y_vl = Y_train.loc[val].values



    X_tr = [X_tr[col] for col in feature_cols]

    X_vl = [X_vl[col] for col in feature_cols]

    

    

    model = model_fn()

    spearmanCallback = SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl))

    callback_list = [es, rlrop, spearmanCallback]

    history = model.fit(X_tr, y_tr, 

                        validation_data=(X_vl, y_vl), 

                        batch_size=BATCH_SIZE, 

                        callbacks=callback_list, 

                        epochs=EPOCHS, 

                        verbose=2).history

    

    preds_train = model.predict(X_tr)

    preds_val = model.predict(X_vl)



    rho_train = [spearmanr(y_tr[:, ind], preds_train[:, ind] + np.random.normal(0, 1e-7, preds_train.shape[0])).correlation for ind in range(preds_train.shape[1])]

    rho_val = [spearmanr(y_vl[:, ind], preds_val[:, ind] + np.random.normal(0, 1e-7, preds_val.shape[0])).correlation for ind in range(preds_val.shape[1])]



    train_rho_kfolds.append(rho_train)

    valid_rho_kfolds.append(rho_val)

    print('Train spearman-rho: %.3f' % np.mean(rho_train))

    print('Validation spearman-rho: %.3f' % np.mean(rho_val))

    

    model_path = '../working/use_baseline_fold_%d.h5' % (ind+1)

    model.save_weights(model_path)

    model_path_list.append(model_path)

    print('Saved model at: %s' % model_path)
sns.set(style="whitegrid")

for key in spearmanCallback.history.keys():

    history[key] = spearmanCallback.history[key]



plot_metrics(history, metric_list=['loss', 'spearman'])
print('Train')

print('Averaged spearman-rho: %.3f' % np.mean(train_rho_kfolds))

print('Averaged spearman-rho (nanmean): %.3f' % np.nanmean(train_rho_kfolds))

print('Averaged spearman-rho avg(regular and nanmean): %.3f +/- %.3f'% (np.mean(train_rho_kfolds), np.std(np.mean(train_rho_kfolds))))

print('\nValidation')

print('Averaged spearman-rho: %.3f' % np.mean(valid_rho_kfolds))

print('Averaged spearman-rho (nanmean): %.3f' % np.nanmean(valid_rho_kfolds))

print('Averaged spearman-rho avg(regular and nanmean): %.3f +/- %.3f'% (np.mean(valid_rho_kfolds), np.std(np.mean(valid_rho_kfolds))))



print('\nEach label :')

spearman_avg_per_label = np.mean(valid_rho_kfolds, axis=0)

spearman_std_per_label = np.std(valid_rho_kfolds, axis=0)

for ii in range(len(target_cols)):

    print('%d - %.3f +/- %.3f - %s' % (ii+1,spearman_avg_per_label[ii],spearman_std_per_label[ii],

                                       target_cols[ii] ))
# Test features

X_test_title = test['question_title']

X_test_body = test['question_body']

X_test_answer = test['answer']



X_test = [X_test_title, X_test_body, X_test_answer]

Y_test = np.zeros((len(test), len(target_cols)))



for model_path in model_path_list:

    model = model_fn()

    model.load_weights(model_path)

    Y_test += model.predict(X_test) / NUM_FOLDS
submission = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')

submission[target_cols] = Y_test

submission.to_csv("submission.csv", index=False)

display(submission.head())