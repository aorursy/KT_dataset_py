import datetime as dt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras import layers

from keras import Input

from keras.models import Model



import pickle



import matplotlib.pyplot as plt
import os

input_dir = "../input"

base_data_dir = os.path.join(input_dir,'data-science-for-good-careervillage')

fe_ts_dir = os.path.join(input_dir,'cv-feature-engineering-text-scores')

ag_data_dir = os.path.join(input_dir,'cv-data-augmentation-network-predictors-2')
tags = pd.read_csv(os.path.join(base_data_dir, 'tags.csv'))

print(tags.shape)

print(tags.head(3))
tag_users = pd.read_csv(os.path.join(base_data_dir, 'tag_users.csv'))

tag_users.sample(3)
# Count users per tag

tags_by_users_counts = tag_users.groupby('tag_users_tag_id')['tag_users_user_id'].count().reset_index()

tags_by_users_counts = tags_by_users_counts.rename(columns={'tag_users_user_id': 'users_count'})

print(tags_by_users_counts.shape)

tags_by_users_counts.head(3)
tags = tags.merge(tags_by_users_counts, 

                  left_on='tags_tag_id', right_on='tag_users_tag_id', how='left')

print(tags.shape)

print(tags.head(3))
tag_questions = pd.read_csv(os.path.join(base_data_dir, 'tag_questions.csv'))

print(tag_questions.shape)

print(tag_questions.sample(3))
# Count questions per tag

tags_by_questions_counts = tag_questions.groupby('tag_questions_tag_id')['tag_questions_question_id'].count().reset_index()

tags_by_questions_counts = tags_by_questions_counts.rename(columns={'tag_questions_question_id': 'questions_count'})

print(tags_by_questions_counts.shape)

tags_by_questions_counts.head(3)
tags = tags.merge(tags_by_questions_counts, 

                  left_on='tags_tag_id', right_on='tag_questions_tag_id', how='left')

print(tags.shape)

print(tags.head(3))
print(tags.shape)

tags = tags.dropna(how='any', subset=['users_count', 'questions_count'])

print(tags.shape)
print(tags.shape)

tags = tags[(tags['users_count'] >= 5) | (tags['questions_count'] >= 5)]

print(tags.shape)
tags = tags.reset_index()

del tags['index']

tags = tags.reset_index().set_index('tags_tag_id')

print(tags.head(5))

print(tags.tail(5))
print(tag_users.shape)

tag_users = tag_users[tag_users['tag_users_tag_id'].isin(tags['tag_users_tag_id'])]

print(tag_users.shape)
print(tag_questions.shape)

tag_questions = tag_questions[tag_questions['tag_questions_tag_id'].isin(tags['tag_users_tag_id'])]

print(tag_questions.shape)
tags = tags[['index', 'tags_tag_name']]

print(tags.head(5))

print(tags.tail(5))
examples = pd.read_parquet(os.path.join(ag_data_dir,'positive_negative_examples.parquet.gzip'))

examples = examples.sort_values(

    by=['questions_id', 'questions_date_added', 'answer_user_id', 'emails_date_sent'])

examples.sample(3)
print(examples.groupby(['questions_id', 'answer_user_id'])['questions_date_added'].count().sort_values(

    ascending=False).head(3))

print(examples[(examples['questions_id']=='9a42d4109ee141c0838fd966efcb5026') & 

               (examples['answer_user_id']=='6adc2bf866bd428892821b044eb8f0fe')].drop_duplicates())
val_period_start = dt.datetime(2018, 7, 1)
train_examples = examples[examples['questions_date_added'] < val_period_start]

print(train_examples.shape)

print(train_examples[train_examples['matched']==1].shape[0])
val_examples = examples[examples['questions_date_added'] >= val_period_start]

print(val_examples.shape)

print(val_examples[val_examples['matched']==1].shape[0])
def get_user_tags_vector(user_id, tags):

    user_tags = tag_users[tag_users['tag_users_user_id']==user_id]['tag_users_tag_id']

    user_tags_vector = np.zeros(tags.shape[0])

    for tag_id in user_tags:

        user_tags_vector[tags.loc[tag_id]['index']] = 1

    return user_tags_vector
def get_question_tags_vector(question_id, tags):

    question_tags = tag_questions[tag_questions['tag_questions_question_id']==question_id]['tag_questions_tag_id']

    question_tags_vector = np.zeros(tags.shape[0])

    for tag_id in question_tags:

        question_tags_vector[tags.loc[tag_id]['index']] = 1

    return question_tags_vector
def sample_question_professionals(question_id, examples, unmatched_matched_ratio):

    question_instances = examples[examples['questions_id']==question_id]

    

    matched_professionals = question_instances[question_instances['matched']==1]['answer_user_id'].values

    matched_professionals_targets = np.repeat(1, len(matched_professionals))

    

    unmatched_professionals = question_instances[question_instances['matched']==0]['answer_user_id'].values

    sampled_unmatched_professionals = np.random.choice(

        unmatched_professionals, size=unmatched_matched_ratio * len(matched_professionals), replace=False)

    sampled_unmatched_professionals_targets = np.repeat(0, len(sampled_unmatched_professionals))



    return (np.concatenate((matched_professionals, sampled_unmatched_professionals), axis=0), 

            np.concatenate((matched_professionals_targets, sampled_unmatched_professionals_targets), axis=0))
def sample_question_tags_vectors(question_id, examples, tags, unmatched_matched_ratio):

    

    sampled_question_professionals, matched_targets = sample_question_professionals(

        question_id, examples, unmatched_matched_ratio)



    question_tags_vector = get_question_tags_vector(question_id, tags) 

    question_tags_vectors = np.broadcast_to(question_tags_vector, 

                                            (len(sampled_question_professionals), len(question_tags_vector)))



    professionals_tags_vectors = []

    for professional_id in sampled_question_professionals:

        professionals_tags_vectors.append(get_user_tags_vector(professional_id, tags))

    professionals_tags_vectors = np.vstack(professionals_tags_vectors)

    

    return question_tags_vectors, professionals_tags_vectors, matched_targets
def generator(examples, tags, num_questions, unmatched_matched_ratio):

    

    question_statistics = examples.groupby('questions_id').agg({'matched': 'sum', 'answer_user_id': 'count'})

    question_statistics = question_statistics[

        ((question_statistics['matched']>0) &

         (question_statistics['answer_user_id']>((unmatched_matched_ratio + 1)*question_statistics['matched'])))]

    question_ids = question_statistics.index.values



    while True:



        cum_question_tags_vectors = []

        cum_professionals_tags_vectors = []

        cum_matched_targets = []

        

        for question_id in np.random.choice(question_ids, size=num_questions, replace=False):

            question_tags_vectors, professionals_tags_vectors, matched_targets = sample_question_tags_vectors(

                question_id, examples, tags, unmatched_matched_ratio)

            cum_question_tags_vectors.append(question_tags_vectors)

            cum_professionals_tags_vectors.append(professionals_tags_vectors)

            cum_matched_targets.append(matched_targets)



        cum_question_tags_vectors = np.concatenate(cum_question_tags_vectors, axis=0)

        cum_professionals_tags_vectors = np.concatenate(cum_professionals_tags_vectors, axis=0)

        cum_matched_targets = np.concatenate(cum_matched_targets, axis=0)

    

        yield [cum_question_tags_vectors, cum_professionals_tags_vectors], cum_matched_targets
 # the number of questions randomly drawn for each training round

training_num_questions = 16

# the ratio of the sample size of unmatched instances over matched ones for each training round

training_unmatched_matched_ratio = 1

train_gen = generator(train_examples, tags, training_num_questions, training_unmatched_matched_ratio)



 # the number of questions randomly drawn for each validation round

val_num_questions = 16

 # the ratio of the sample size of unmatched instances over matched ones for each validation round

val_unmatched_matched_ratio = 1

val_gen = generator(val_examples, tags, val_num_questions, val_unmatched_matched_ratio)
latent_dimensions = 64



# Both questions and professionals share the same tag embedding layer

embeddings = layers.Dense(latent_dimensions)



question_tags_input = Input(shape=(tags.shape[0],), name='question_tags')

question_tags_output = embeddings(question_tags_input)



professional_tags_input = Input(shape=(tags.shape[0],), name='professional_tags')

professional_tags_output = embeddings(professional_tags_input)



# There are many way to combine the embeddins of question tags and professional tags

# One simple option is used here just for demonstration

tags_multiplied = layers.multiply([question_tags_output, professional_tags_output])



# Merge with other features

merged_with_others = layers.concatenate([question_tags_output, professional_tags_output, tags_multiplied], axis=-1)



# Another feed forward layer

reduced_l1 = layers.Dense(16, activation='relu')(merged_with_others)



# The binary predictions of matched (1) versus unmatched (0) are modeled by a sigmoid function

predictions = layers.Dense(1, activation='sigmoid')(reduced_l1)



model = Model([question_tags_input, professional_tags_input], predictions)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# # Test Run

# history = model.fit_generator(train_gen, steps_per_epoch=5, epochs=5, 

#                               validation_data=val_gen, validation_steps=5)



# Full Run

history = model.fit_generator(train_gen, steps_per_epoch=50, epochs=75, 

                              validation_data=val_gen, validation_steps=50)
model.save('tags_embeddings.h5')
loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
acc = history.history['acc']

val_acc = history.history['val_acc']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, acc, 'bo', label='Training Accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()
with open('trainHistory_tags_embeddings', 'wb') as out_file:

    pickle.dump(history.history, out_file)