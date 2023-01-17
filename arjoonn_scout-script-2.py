import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from collections import Counter

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.feature_extraction.text import CountVectorizer

%pylab inline
df = pd.read_csv('../input/primary_debates_cleaned.csv')

df.info()
df.head()
df.URL.str.split('/').str[2].unique()
# All from the same site, so I'll drop that. I'm interested in far more interesting things

df = df.drop('URL', axis=1)
sns.countplot('Party', data=df)

plt.title('Which party spoke the most.')
sns.countplot('Date', data=df, hue='Party')

plt.title('On which dates were the most words exchanged?')
df['candidate'] = (df.Speaker == 'Sanders') | (df.Speaker == 'Clinton') | (df.Speaker == 'Trump')

sns.countplot('Speaker', data=df.loc[df.candidate])

plt.title('Which speaker spoke the most?')
# since the end is near :3, we'll concentrate only on the two notorious ones.

df['candidate'] = (df.Speaker == 'Clinton') | (df.Speaker == 'Trump')

df['candidate'] = df.candidate > 0

df = df.loc[df.candidate]
df['textlen'] = df.Text.str.len()

sns.barplot('Speaker', 'textlen', data=df)

plt.title('Who has longer sentences?')
def repeated_fraction(sentence):

    "What fraction of the sentence is repeated?"

    seen, seen_count, sentences = [], 0, sentence.split(' ')

    for word in sentences:

        if word in seen:

            seen_count += 1

        else:

            seen.append(word)

    return seen_count / float(len(sentences))

df['seen_frac'] = df.Text.apply(repeated_fraction)

sns.barplot('Speaker', 'seen_frac', data=df)

plt.title('Who repeats words in sentences?')
# But we remember Trump speaking the most right? Actually, who had more air time?

df['speak_time'] = df.Text.str.len()

sns.barplot('Speaker', 'speak_time', data=df, estimator=np.sum)

plt.title('Who uttered more words overall?')
# Trump is more of a "I'll interrupt you to death guy I think"

# I'll do more of this tomorrow.