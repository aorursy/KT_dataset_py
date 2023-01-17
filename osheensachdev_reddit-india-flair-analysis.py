import pandas as pd

import numpy as np



def get_data(train_file, test_file = None):

    if test_file == None:

        frame = pd.read_csv(train_file)

        print(frame.head(5))

        data = frame.values

        np.random.shuffle(data)

        return data

    else:

        train_frame = pd.read_csv(train_file)

        test_frame = pd.read_csv(test_file, error_bad_lines=False, quoting = 2)



        train_data = train_frame.values

        test_data = test_frame.values

        np.random.shuffle(train_data)

        np.random.shuffle(test_data)



        return train_data, test_data





data = get_data('../input/Subreddit_India.csv')



flairwise = {'[R]eddiquette' : [], 'Sports' : [], 'AskIndia' : [], 'Photography' : [], 'Policy/Economy' : [], 'Science/Technology' : [], 'Politics' : [], 'Food' : [], 'Business/Finance' : []}

for row in data:

	flairwise[row[1]].append(row)

for flair in flairwise:

    print(flair, len(flairwise[flair]))
print(flairwise['Sports'][0])    
upvotes = {}

comments = {}

for flair in flairwise:

    score = 0

    comment_count = 0

    for submission in flairwise[flair]:

        score += submission[5]

        comment_count += submission[7]

        if  str(submission[4]) == 'nan' or submission[4] == '[removed]': 

            submission[4] = ''

    upvotes[flair] = score

    comments[flair] = comment_count

print(upvotes)

print(comments)

import matplotlib.pyplot as plt

flair_name = {'[R]eddiquette' : 'Redd', 'Sports': 'Sp', 'AskIndia' :'AI', 'Photography' : 'Ph', 'Policy/Economy' : 'PE', 'Science/Technology':'ST', 'Politics': 'Pol', 'Food': 'Food', 'Business/Finance': 'BF'}

plt.bar(flair_name.values(), upvotes.values())

plt.savefig('upvoteAnalysis.jpg')
plt.bar(flair_name.values(), comments.values())

plt.savefig('commentAnalysis.jpg')

from wordcloud import WordCloud



x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2

mask = 255 * mask.astype(int)



for flair in flairwise:

    wc = WordCloud(max_font_size=40, max_words=200, background_color='white', random_state=1337, mask=mask).generate(' '.join((row[3] + ' ' + row[4]) for row in flairwise[flair]))

    plt.figure(figsize=(10,10))

    plt.imshow(wc, interpolation='bilinear')

    plt.axis("off")

    plt.title(flair, fontsize=20)

    plt.show()

    plt.savefig(flair_name[flair] + '.jpg')
