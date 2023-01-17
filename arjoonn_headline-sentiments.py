import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import textblob as tb

from tqdm import tqdm



import seaborn as sns

%pylab inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# There's some issue with pandas read_csv. We'll do it the old fashioned way.

table = []

with open('../input/abcnews-date-text.csv', 'r') as fl:

    for line in fl.readlines():

        date, headline = line.split(',', 1)

        table.append((date, headline.strip()))

df = pd.DataFrame(table[1:], columns=['date', 'headline'])

df['date'] = pd.to_datetime(df['date'])



def sent(x):

    t = tb.TextBlob(x)

    return t.sentiment.polarity, t.sentiment.subjectivity

df.head()
tqdm.pandas(leave=False, mininterval=25)

vals = df.headline.progress_apply(sent)
df['polarity'] = vals.str[0]

df['subj'] = vals.str[1]

df.sort_values('date', inplace=True)

df['times'] = df['date'].astype(int)
def plot_data(df):

    mean_pol = list(dict(df.groupby('date')['polarity'].mean()).items())

    mean_pol.sort(key=lambda x: x[0])



    plt.subplots(figsize=(15, 10))

    plt.subplot(2, 2, 1)

    plt.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])

    plt.title('Mean polarity over time')



    plt.subplot(2, 2, 2)

    mean_pol = list(dict(df.groupby('date')['subj'].mean()).items())

    mean_pol.sort(key=lambda x: x[0])

    plt.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])

    plt.title('Mean subjectivity over time')



    plt.subplot(2, 2, 3)

    mean_pol = list(dict(df.groupby('date')['polarity'].std()).items())

    mean_pol.sort(key=lambda x: x[0])

    plt.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])

    plt.title('Std Dev of polarity over time')



    plt.subplot(2, 2, 4)

    mean_pol = list(dict(df.groupby('date')['subj'].std()).items())

    mean_pol.sort(key=lambda x: x[0])

    plt.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])

    plt.title('Std dev of subjectivity over time')

plot_data(df)