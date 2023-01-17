import pandas as pd

import tensorflow as tf



from tensorflow import keras

layers = keras.layers



# This code was tested with TensorFlow v1.7

print("You have TensorFlow version", tf.__version__)
# Convert the data to a Pandas data frame

comments = pd.read_csv('../input/metacritic_game_user_comments.csv')

# Shuffle with a fixed random seed

# This will help us to have the same training and test set every time

comments = comments.sample(frac=1, random_state=387)

comments = comments[pd.notnull(comments['Comment'])]

comments.drop(['Unnamed: 0','Username'], axis=1, inplace=True)



# Drop comments with less than 200 characters

# Modify this parameter to obtain different results

comments = comments[comments['Comment'].str.len() > 200]

# Print the first 5 rows

print(len(comments))

comments.head()
# Split data into train and test

train_size = int(len(comments) * .8)

print ("Train size: %d" % train_size)

print ("Test size: %d" % (len(comments) - train_size))



# Train features

comments_train = comments['Comment'][:train_size]

# Train labels

labels_train = comments['Userscore'][:train_size]

# Test features

comments_test = comments['Comment'][train_size:]

# Test labels

labels_test = comments['Userscore'][train_size:]
# Create a tokenizer to preprocess our text descriptions

vocab_size = 12000 # This is a hyperparameter, experiment with different values for your dataset

tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)

tokenize.fit_on_texts(comments_train) # only fit on train
# Define our wide model with the functional API

bow_inputs = layers.Input(shape=(vocab_size,))

inter = layers.Dense(256, activation='relu')(bow_inputs)

inter = layers.Dropout(0.3)(inter)

predictions = layers.Dense(1, activation='linear')(inter)

wide_model = keras.Model(inputs=bow_inputs, outputs=predictions)

wide_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

print(wide_model.summary())
max_seq_length = 200



# Define our deep model with the Functional API

deep_inputs = layers.Input(shape=(max_seq_length,))

embedding = layers.Embedding(vocab_size, 16, input_length=max_seq_length)(deep_inputs)

embedding = layers.Flatten()(embedding)

embedding = layers.Dense(64, activation='relu')(embedding)

embedding = layers.Dropout(0.3)(embedding)

embed_out = layers.Dense(1, activation='linear')(embedding)

deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)

deep_model.compile(loss='mse',

                   optimizer='adam',

                   metrics=['accuracy'])

print(deep_model.summary())
# Combine wide and deep into one model

merged_out = layers.concatenate([wide_model.output, deep_model.output])

merged_out = layers.Dense(64, activation='relu')(merged_out)

merged_out = layers.Dropout(0.3)(merged_out)

merged_out = layers.Dense(1)(merged_out)

combined_model = keras.Model([wide_model.input, deep_model.input], merged_out)

combined_model.compile(loss='mse',

                       optimizer='adam',

                       metrics=['accuracy'])

print(combined_model.summary())
def process_comments(comments, tokenize, max_seq_length):

    # Create the Bag Of Words and the embed version of only this

    # batch of examples. 

    # This is to avoid using all the memory at the same time

    bow = tokenize.texts_to_matrix(comments)

    embed = tokenize.texts_to_sequences(comments)

    embed = keras.preprocessing.sequence.pad_sequences(

        embed, maxlen=max_seq_length, padding="post"

    )

    return [bow, embed]

  

# Create the generator for fit and evaluate

def generator(comments_list, labels_list, batch_size, tokenize, max_seq_length):

    batch_number = 0

    data_set_len = len(comments_list)

    batches_per_epoch = int(data_set_len/batch_size)



    while True:

        initial = (batch_number*batch_size) % data_set_len

        final = initial + batch_size

        comments_to_send = comments_list[initial:final]



        x = process_comments(comments_to_send, tokenize, max_seq_length) 

        y = labels_list[initial:final]



        batch_number = (batch_number+1) % batches_per_epoch

        yield x, y
def on_epoch_end(epoch, logs, print_preditions=0):

    # Generate predictions

    predictions = combined_model.predict_generator(

        generator(comments_test, labels_test, 128, tokenize, max_seq_length),

        steps=int(len(comments_test)/128)

    )



    # Compare predictions with actual values for the first few items in our test dataset

    diff = 0

    printed = 0

    for i in range(len(predictions)):

        val = predictions[i]

        if print_preditions and printed < print_preditions:

            print(comments_test.iloc[i])

            print('Predicted: ', val[0], 'Actual: ', labels_test.iloc[i], '\n')

            printed += 1

        diff += abs(val[0] - labels_test.iloc[i])



    # Compare the average difference between actual price and the model's predicted price

    print('\nEpoch: %d. Average prediction difference: %0.4f\n' %

            (epoch+1, diff/len(predictions)))

    



print_callback = keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
# Run training

# It is a fairly deep network, it will take around 2 minutes per epoch

combined_model.fit_generator(

    generator(comments_train, labels_train, 128, tokenize, max_seq_length),

    steps_per_epoch=int(len(comments_train)/128),

    epochs=7,

    validation_data=generator(comments_test, labels_test, 128, tokenize, max_seq_length),

    validation_steps=int(len(comments_test)/128),

    callbacks=[print_callback]

)
# We manually call on_epoch_end with the trained model, but this time with 

# print_preditions=20

# It will print 20 examples of the training set, with its predicted and actual 

# value

on_epoch_end(7, {}, print_preditions=20)
# Let's try some user review of World War Z for Playstatin 4

# https://www.metacritic.com/game/playstation-4/world-war-z/user-reviews

test_comments = [

    "This game is fun. No microtransactions i can find. Buy it on sale unless you have friends then its a no brainer. It looks good and runs well on ultra 1060 60fps. Its not deep or big or original or creative but its solid which few zombie games are. They have kept it small and as a result its incompetent and capable. The single player works with decent ai which is a first for me. Your ai team will have your back though youll have to do the heavy lifting. There are no bosses which is a real missed opportunity. The progression system is very average but theres plenty to do. Sound and music is average. Gameplay is good. Lastability is what ever you make it but id like to see more free content within a month. The game can be scary and thrilling. The sequel could be great. If you love zombies its a no brainer (lol) and maybe a good fix while waiting on dayz gone.",

    "It's one of the worst games I've played in my life, the AI is extremely stupid, the enemies are completely idiots, there is no challenge in the game, the targets are very bad, there is no sense in anything that is ago, many missions end in a very foolish way. The bots are useless, they do not know how to heal themselves, they do not defend or help. It is a copy of very poor quality of l4d2 in the third person, l4d2, a game of the year 2009 has better graphics, artificial intelligence, objectives, history and a great etc. compared to wwz, if the game was on Steam it would be a complete failure, OTWD was bad, but WWZ is not much better.",

    "I would have given this game an 8-9, but because of the clear review bombing below I am upping the score to a 10. This game is super fun. It is mindless zombie action best played with friends. I agree with the guy below - there is now way this is worth less than a 7. The game is beyond insane in certain areas. It isn't Left4Dead because the action was nowhere near as intense in that game. Streamers are loving it and the one guy who gave a critic review was clearly doing it as clickbait. I would say a solid 8 is much more accurate. I will say it is much more fun online and Sony PSN was down yesterday so I haven't played as much as I would like.",

    "The game has a fun progression system, but the constant buggyness holds it back. I have ran into a bug daily since release, and it has really killed my enjoyment for the game. If you are hoping to play with friends, it's a gamble whether or not you can invite them. Some people can, others can't. There are no check points, so if you bug out (which happens often) you have to restart with 0 progress gained. The past days since release there has been constant connection errors online. And when you do join, half of the people I have seen are either hackers or extremely toxic. (I was doing a higher difficulty with a low level, and someone just kept shooting me every time I respawned claiming we were going to lose anyways.) And even if you do play offline, despite what people say, the bots are garbage. But they're probably designed that way so they don't nick things important to you. They don't pick up any new items, and when you set up defenses you have to do it all by yourself. Not to mention any mission where you have to haul multiple of the objective back to a point you have to do that solo too. With all that negativity, what does this game offer over left 4 dead? The progression system lets you upgrade guns you use frequently, and you get experience for classes which have special abilities (some let you hit more than one target with a melee ability, others let you rez people from a distance, etc.) and get different special equipment, like grenades or stun guns. This alone is what got me into the game, but since I'm usually playing solo and my team comp is just AI with aimbots with no abilities I never get to see classes shine. Sound design and enemy, design compared to L4D is also lackluster. Bombs feel like they're smothered by wet paper towels, along with grenade launchers. You go to different locations but the Bull (charger) special infected always has POLICE on it. One enemy spawns more zombies but it usually is too far away to shoot, and happens during defenses which is where you're already shooting a large amount of zombies so you can't kill it and get easily overrun. If the devs ever get around to fixing the numerous bugs and getting online stable, it'll already be out on steam and they'll be doing the same **** over again. Wait for left 4 dead 3, if it even exists."

]

# The scores are 

# 7

# 0

# 10

# 3



combined_model.predict(process_comments(test_comments, tokenize, max_seq_length))