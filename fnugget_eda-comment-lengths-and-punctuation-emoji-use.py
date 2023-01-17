import pandas

training_data = pandas.read_csv('../input/train.csv')

civil_comments = training_data[

    training_data['target'] < 0.5

]

toxic_comments = training_data[

    training_data['target'] >= 0.5

]



toxicity_types = [

    'severe_toxicity',

    'obscene',

    'threat',

    'insult',

    'identity_attack',

    'sexual_explicit']



identity_types = [

    # Gender

    'male', 'female', 

    'transgender', 'other_gender', # Not targets

    # Sexuality

    'homosexual_gay_or_lesbian',

    'heterosexual', 'bisexual', 'other_sexual_orientation', # Not targets

    # Religion

    'christian', 'jewish', 'muslim',

    'hindu', 'buddhist', 'atheist', 'other_religion', # Not targets

    # Race/Ethnicity

    'black', 'white',

    'asian', 'latino', 'other_race_or_ethnicity', # Not targets

    # Disability/Illness

    'psychiatric_or_mental_illness',

    'physical_disability', 'other_disability', 'intellectual_or_learning_disability' # Not targets

]



target_identity_types = [

      # Gender

    'male', 'female',

    # Sexuality

    'homosexual_gay_or_lesbian',

    # Religion

    'christian', 'jewish', 'muslim',

    # Race/Ethnicity

    'black', 'white',

    # Disability/Illness

    'psychiatric_or_mental_illness',

]
training_data[

    training_data['id'].isin(

        [59856, 59859, 59861])

]
positions = []

heights = []

labels = []

colors = []

for i, toxicity_type in enumerate(toxicity_types):

    positions.extend([2*i, 2*i+1])

    heights.extend([

        civil_comments[toxicity_type].mean(),

        toxic_comments[toxicity_type].mean()

    ])

    labels.extend([

        toxicity_type + ' (civil)', 

        toxicity_type + ' (toxic)'

    ])

    colors.extend(['C0', 'C1'])

plt.xticks(positions, labels, rotation='vertical')

plt.ylim(0)

plt.bar(positions, heights, color=colors)

plt.show()
positions = []

heights = []

labels = []

colors = []

for i, identity_type in enumerate(target_identity_types):

    positions.extend([2*i, 2*i+1])

    heights.extend([

        civil_comments.dropna()[identity_type].mean(),

        toxic_comments.dropna()[identity_type].mean()

    ])

    labels.extend([

        identity_type + ' (civil)', 

        identity_type + ' (toxic)'

    ])

    colors.extend(['C0', 'C1'])

plt.xticks(positions, labels, rotation='vertical')

plt.ylim(0, 0.4)

plt.bar(positions, heights, color=colors)

plt.show()
positions = []

heights = []

labels = []

colors = []



for i, identity_type in enumerate(set(identity_types)-set(target_identity_types)):

    positions.extend([2*i, 2*i+1])

    heights.extend([

        civil_comments.dropna()[identity_type].mean(),

        toxic_comments.dropna()[identity_type].mean()

    ])

    labels.extend([

        identity_type + ' (civil)', 

        identity_type + ' (toxic)'

    ])

    colors.extend(['C0', 'C1'])

plt.xticks(positions, labels, rotation='vertical')

plt.ylim(0, 0.4)

plt.bar(positions, heights, color=colors)

plt.figure(figsize=(200, 400))

plt.show()
print('Civil Comments Character Length:',

      civil_comments['comment_text'].apply(lambda c: len(c)).mean())

print('Toxic Comments Character Length:',

      toxic_comments['comment_text'].apply(lambda c: len(c)).mean())
print('Civil Comments Word Count:',

      civil_comments['comment_text'].apply(lambda c: len(c.split(' '))).mean())

print('Toxic Comments Word Count:',

      toxic_comments['comment_text'].apply(lambda c: len(c.split(' '))).mean())
import numpy as np

print('Civil Comments Word Length:',

      civil_comments['comment_text'].apply(lambda c: np.mean([

          len(word) for word in c.split(' ')])).mean())

print('Toxic Comments Word Length:',

      toxic_comments['comment_text'].apply(lambda c: np.mean([

          len(word) for word in c.split(' ')])).mean())
civil_word_counts = civil_comments['comment_text'].apply(lambda c: len(c.split(' ')))

toxic_word_counts = toxic_comments['comment_text'].apply(lambda c: len(c.split(' ')))



punctuation_marks = [

    '.', ',', ';', ':', '?', '!', '*', 

    '(', ')', '[', ']', '{', '}'

]



for punctuation in punctuation_marks:

    print(f'Civil Comments Number Punctuation ({punctuation}):',

          (civil_comments['comment_text'].apply(

              lambda c: c.count(punctuation))/

              civil_word_counts).mean()

        )

    print(f'Toxic Comments Number Punctuation ({punctuation}):',

          (toxic_comments['comment_text'].apply(

              lambda c: c.count(punctuation))/

              toxic_word_counts).mean()

        )
def add_noses(emojis):

    noses = ['^', '<', 'o']

    emojis_with_noses = []

    for emoji in emojis:

        for nose in noses:

            emojis_with_noses.append(

                emoji[:1] + nose + emoji[1:])

    return emojis_with_noses



positive_emojis = (

    add_noses([':)', '=)', ';)']) +

    ['<3'] + [':D', '=D', ':3'] +

    [':P', '=P'] +

    ['xD', 'XD'] +

    ['^.^', '^^']

)

negative_emojis = (

    add_noses([':(', '=(', ':/', ':|', '=/', '=|']) +

    [":'(", "='("] +

    ["-.-", 'O.o'])



emojis = positive_emojis + negative_emojis
def contains_emoji(text, emojis):

    return any(emoji in text for emoji in emojis)





print('Civil Comments Emoji Frequency (all):',

      civil_comments['comment_text'].apply(

          lambda c: int(contains_emoji(c, emojis))).mean()

)

print('Toxic Comments Emoji Frequency (all):',

      toxic_comments['comment_text'].apply(

          lambda c: int(contains_emoji(c, emojis))).mean()

)



print('Civil Comments Emoji Frequency (positive):',

      civil_comments['comment_text'].apply(

          lambda c: int(contains_emoji(c, positive_emojis))).mean()

)

print('Toxic Comments Emoji Frequency (positive):',

      toxic_comments['comment_text'].apply(

          lambda c: int(contains_emoji(c, positive_emojis))).mean()

)



print('Civil Comments Emoji Frequency (negative):',

      civil_comments['comment_text'].apply(

          lambda c: int(contains_emoji(c, negative_emojis))).mean()

)

print('Toxic Comments Emoji Frequency (negative):',

      toxic_comments['comment_text'].apply(

          lambda c: int(contains_emoji(c, negative_emojis))).mean()

)
civil_emojis = []

toxic_emojis = []

ambigious_emojis = []



for emoji in emojis:

    toxic_use_count = toxic_comments['comment_text'].apply(

        lambda c: int(contains_emoji(c, [emoji]))).sum()

    toxic_use_frequency = toxic_use_count/len(toxic_comments)

    

    civil_use_count = civil_comments['comment_text'].apply(

        lambda c: int(contains_emoji(c, [emoji]))).sum()

    civil_use_frequency = civil_use_count/len(civil_comments)



    if toxic_use_frequency + civil_use_frequency == 0:

        continue



    if toxic_use_frequency >= civil_use_frequency:

        confidence = (toxic_use_frequency*100/

            (civil_use_frequency*100 + toxic_use_frequency*100))

        toxic_emojis.append(

            (emoji, confidence, toxic_use_count + civil_use_count)

        )

    else:

        confidence = (civil_use_frequency*100/

            (civil_use_frequency*100 + toxic_use_frequency*100))

        civil_emojis.append(

            (emoji, confidence, toxic_use_count + civil_use_count)

        )

print('Civil emojis', civil_emojis)

print('Toxic emojis', toxic_emojis)