# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Pip additional libraries 

!pip install adjustText
import pandas as pd

import numpy as np



import pickle



from collections import Counter

from sklearn.feature_extraction import text 

from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob



from matplotlib import pyplot as plt

from adjustText import adjust_text

import math

from tqdm import tqdm

import seaborn as sns
def split_text(text, n=10):

    """

        Splits text into n partitions

    """

    

    # Get partition ranges

    length = len(text)

    partition_size = math.floor(length/n)

    partition = np.arange(0, length, partition_size)

    

    # split text

    text_partition = []

    for split in range(n):

        text_partition.append(text[partition[split]:partition[split]+partition_size])

    return text_partition
def build_presidents_polarity_transcripts(corpus, n=10):

    """

        Returns the polarity of the corpus for each president's text

    """

    

    polarity_transcripts = {}

    for president in corpus.index:

        transcript = corpus.loc[president].transcripts

        partitioned_text = split_text(transcript, n)

        polarity_text = list(map(pol, partitioned_text))

        polarity_transcripts[president] = polarity_text

    return polarity_transcripts
def build_presidents_subjectivity_transcripts(corpus, n=10):

    """

        Returns the subjectivity of the corpus for each president's text

    """

    

    polarity_transcripts = {}

    for president in corpus.index:

        transcript = corpus.loc[president].transcripts

        partitioned_text = split_text(transcript, n)

        polarity_text = list(map(sub, partitioned_text))

        polarity_transcripts[president] = polarity_text

    return polarity_transcripts
def build_party_polarity_transcripts(corpus, n=10):

    """

        Returns the polarity of the corpus for each party's text

    """

    

    polarity_transcripts = {}

    for party in corpus.index:

        transcript = corpus.loc[party].transcripts

        partitioned_text = split_text(transcript, n)

        polarity_text = list(map(pol, partitioned_text))

        polarity_transcripts[party] = polarity_text

    return polarity_transcripts
def build_party_subjectivity_transcripts(corpus, n=10):

    """

        Returns the subjectivity of the corpus for each party's text

    """

    

    polarity_transcripts = {}

    for party in corpus.index:

        transcript = corpus.loc[party].transcripts

        partitioned_text = split_text(transcript, n)

        polarity_text = list(map(sub, partitioned_text))

        polarity_transcripts[party] = polarity_text

    return polarity_transcripts



def build_party_subjectivity_transcripts(corpus, n=10):

    """

        Returns the subjectivity of the corpus for each party's text

    """

    

    polarity_transcripts = {}

    for party in corpus.index:

        transcript = corpus.loc[party].transcripts

        partitioned_text = split_text(transcript, n)

        polarity_text = list(map(sub, partitioned_text))

        polarity_transcripts[party] = polarity_text

    return polarity_transcripts
def build_party_corpus(corpus):

    """

        Returns the corpus of the political parties and their respective transcripts

    """

    

    # Build corpus

    party_texts = []

    parties = list(set(corpus.Party.values))

    for party in parties:

        text = ""

        for row in corpus[corpus.Party == party].itertuples(index=False):

            text += row[1]

        party_texts.append(text)

    party_corpus =  pd.DataFrame({'Party': parties, 'transcripts': party_texts}).set_index('Party')

    

    # Include only non-null parties

    party_order = []

    parties = ['Democratic', 'Republican', 'Democratic-Republican', 'Whig', 'Federalist', 'Unaffiliated']

    for party in parties:

        if party in set(corpus.Party.values):

            party_order.append(party)

    party_corpus = party_corpus.reindex(index = party_order)

    return party_corpus
def graph_political_era_polarity(corpus, title, N=10):

    

    # Build party corpus

    party_corpus = build_party_corpus(corpus)

    

    # Number of partitions

    N = 10

    polarity_transcripts = build_party_polarity_transcripts(party_corpus, N)

    

    # Get number of rows necessary and if last subplot needs to be removed

    delete_last = False

    row_count = len(polarity_transcripts)

    if row_count % 2 == 1:

        delete_last = True

    while row_count % 2 != 0:

        row_count += 1

    row_count //= 2

        

    # Plot

    fig, ax = plt.subplots(row_count, 2, figsize=(12,2.5 + (2.5 * row_count)))

    fig.suptitle(title, fontsize=16)

    

    if row_count == 1:

        for i, party in enumerate(party_corpus.index):

            

            # Label for partition size

            length = len(party_corpus.loc[party].transcripts)

            partition_size = math.floor(length/N)

            x_label = 'Partitions ({} characters per partition)'.format(partition_size)



            # Plot subplot

            sns.lineplot(x=range(0, N), y=polarity_transcripts[party], ax=ax[i], color=parties_color[party])

            ax[i].set(xlabel=x_label, ylabel='<-- Negative ----------- Positive -->')

            ax[i].set_title(party)        

    else:

        i = -1

        for k, party in enumerate(party_corpus.index):

            if k % 2 == 0:

                i += 1

            j = k % 2

            

            # Label for partition size

            length = len(party_corpus.loc[party].transcripts)

            partition_size = math.floor(length/N)

            x_label = 'Partitions ({} characters per partition)'.format(partition_size)

            

            # Plot subplot

            sns.lineplot(x=range(0, N), y=polarity_transcripts[party], ax=ax[i, j], color=parties_color[party])

            ax[i, j].set(xlabel=x_label, ylabel='<-- Negative ----------- Positive -->')

            ax[i, j].set_title(party)



    # Delete unused subplot

    if delete_last:

        fig.delaxes(ax[row_count - 1, 1])

            

    fig.tight_layout(pad=3)

    fig.subplots_adjust(top=0.88)
def graph_political_era_subjectivity(corpus, title, N = 10):

    

    # Build party corpus

    party_corpus = build_party_corpus(corpus)

    

    # Number of partitions

    N = 10

    subjectivity_transcripts = build_party_subjectivity_transcripts(party_corpus, N)

    



    # Get number of rows necessary and if last subplot needs to be removed

    delete_last = False

    row_count = len(subjectivity_transcripts)

    if row_count % 2 == 1:

        delete_last = True

    while row_count % 2 != 0:

        row_count += 1

    row_count //= 2

        

    # Plot

    fig, ax = plt.subplots(row_count, 2, figsize=(12,2.5 + (2.5 * row_count)))

    fig.suptitle(title, fontsize=16)

    

    if row_count == 1:

        for i, party in enumerate(party_corpus.index):

            

            # Label for partition size

            length = len(party_corpus.loc[party].transcripts)

            partition_size = math.floor(length/N)

            x_label = 'Partitions ({} characters per partition)'.format(partition_size)



            # Plot subplot

            sns.lineplot(x=range(0, N), y=subjectivity_transcripts[party], ax=ax[i], color=parties_color[party])

            ax[i].set(xlabel=x_label, ylabel='<-- Facts ----------- Opinions -->')

            ax[i].set_title(party)   

    else:

        i = -1

        for k, party in enumerate(party_corpus.index):

            if k % 2 == 0:

                i += 1

            j = k % 2

            

            # Label for partition size

            length = len(party_corpus.loc[party].transcripts)

            partition_size = math.floor(length/N)

            x_label = 'Partitions (Size: {} characters per partition)'.format(partition_size)

            

            # Plot subplot

            sns.lineplot(x=range(0, N), y=subjectivity_transcripts[party], ax=ax[i, j], color=parties_color[party])

            ax[i, j].set(xlabel=x_label, ylabel='<-- Facts ----------- Opinions -->')

            ax[i, j].set_title(party)



    # Delete unused subplot

    if delete_last:

        fig.delaxes(ax[row_count - 1, 1])

            

    # Add padding between subplots and title

    fig.tight_layout(pad=3)

    fig.subplots_adjust(top=0.88)
def graph_political_era_comparison(corpus, title, N = 10):

    

    # Build party corpus

    party_corpus = build_party_corpus(corpus)

    

    # Number of partitions

    N = 10

    polarity_transcripts = build_party_polarity_transcripts(party_corpus, N)

    subjectivity_transcripts = build_party_subjectivity_transcripts(party_corpus, N)

    

    # Number of parties

    party_count = len(party_corpus)

        

    # Plot

    fig, ax = plt.subplots(party_count + 1, 2, figsize=(12, 10 + (2.5 * party_count)))

    fig.suptitle(title, fontsize=16)

    

    # Individual Plots

    for i, party in enumerate(party_corpus.index):

        # Label for partition size

        length = len(party_corpus.loc[party].transcripts)

        partition_size = math.floor(length/N)

        x_label = 'Partitions ({} characters per partition)'.format(partition_size)

        

        # Plot polarity subplot

        sns.lineplot(x=range(0, N), y=polarity_transcripts[party], ax=ax[i, 0], color=parties_color[party])

        ax[i, 0].set(xlabel=x_label, ylabel='<-- Negative ------ Positive -->')

        ax[i, 0].set_title(party + " (polarity)")  

        

        # Plot subjectivity subplot

        sns.lineplot(x=range(0, N), y=subjectivity_transcripts[party], ax=ax[i, 1], color=parties_color[party])

        ax[i, 1].set(xlabel=x_label, ylabel='<-- Facts ------ Opinions -->')

        ax[i, 1].set_title(party + " (subjectivity)") 

        

    # Comparison Plots

    for i, party in enumerate(party_corpus.index):

        # Label for partition size

        length = len(party_corpus.loc[party].transcripts)

        partition_size = math.floor(length/N)

        x_label = 'Partitions ({} characters per partition)'.format(partition_size)

        

        # Plot polarity subplot

        sns.lineplot(x=range(0, N), y=polarity_transcripts[party], ax=ax[party_count, 0], color=parties_color[party], label=party)

        ax[party_count, 0].set(xlabel=x_label, ylabel='<-- Negative ------ Positive -->')

        ax[party_count, 0].set_title("Polarity Comparison")  

        

        # Plot subjectivity subplot

        sns.lineplot(x=range(0, N), y=subjectivity_transcripts[party], ax=ax[party_count, 1], color=parties_color[party], label=party)

        ax[party_count, 1].set(xlabel=x_label, ylabel='<-- Facts ------ Opinions -->')

        ax[party_count, 1].set_title("Subjectivity Comparison") 

            

    # Add padding between subplots and title

    fig.tight_layout(pad=3)

    fig.subplots_adjust(top=0.88)
# Load corpuses

corpus = pd.read_csv('/kaggle/input/united-states-presidential-speeches/corpus.csv')

first_party_corpus = pd.read_csv('/kaggle/input/united-states-presidential-speeches/first_party_corpus.csv')

second_party_corpus = pd.read_csv('/kaggle/input/united-states-presidential-speeches/second_party_corpus.csv')

third_party_corpus = pd.read_csv('/kaggle/input/united-states-presidential-speeches/third_party_corpus.csv')

fourth_party_corpus = pd.read_csv('/kaggle/input/united-states-presidential-speeches/fourth_party_corpus.csv')

fifth_party_corpus = pd.read_csv('/kaggle/input/united-states-presidential-speeches/fifth_party_corpus.csv')

sixth_party_corpus = pd.read_csv('/kaggle/input/united-states-presidential-speeches/sixth_party_corpus.csv')
# Fix indicies

corpus = corpus.rename(columns={'Unnamed: 0':'President'}).set_index('President')

first_party_corpus = first_party_corpus.rename(columns={'Unnamed: 0':'President'}).set_index('President')

second_party_corpus = second_party_corpus.rename(columns={'Unnamed: 0':'President'}).set_index('President')

third_party_corpus = third_party_corpus.rename(columns={'Unnamed: 0':'President'}).set_index('President')

fourth_party_corpus = fourth_party_corpus.rename(columns={'Unnamed: 0':'President'}).set_index('President')

fifth_party_corpus = fifth_party_corpus.rename(columns={'Unnamed: 0':'President'}).set_index('President')

sixth_party_corpus = sixth_party_corpus.rename(columns={'Unnamed: 0':'President'}).set_index('President')
# Find polarity and subjectivity of texts for each president

pol = lambda x: TextBlob(x).sentiment.polarity

sub = lambda x: TextBlob(x).sentiment.subjectivity



corpus['pol'] = corpus.transcripts.apply(pol)

corpus['subj'] = corpus.transcripts.apply(sub)



corpus
# Plot

fig, ax = plt.subplots(figsize=(20,12))

sns.scatterplot(x=corpus.pol.values, y=corpus.subj.values)



texts = [ax.text(corpus.pol.values[line], corpus.subj.values[line], corpus.index[line], horizontalalignment='center', size='large', color='black', weight='semibold') for line in range(0,corpus.shape[0])]

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))



plt.title('Sentiment Analysis', fontsize=20)

plt.xlabel('<-- Negative ------------------------ Positive -->', fontsize=15)

plt.ylabel('<-- Facts ------------------------ Opinions -->', fontsize=15)



plt.show()
# Party coloring

parties_color = {

            'Democratic': 'blue',

            'Democratic-Republican': 'green',

            'Federalist': 'orange',

            'Republican': 'red',

            'Unaffiliated': 'black',

            'Whig': 'purple'

          }



# Plot speeches per year

fig, ax = plt.subplots(figsize=(20,12))



texts = []



# Democratic

party_corpus = corpus[corpus['Party'] == 'Democratic']

sns.scatterplot(x=party_corpus.pol.values, y=party_corpus.subj.values, label='Democratic')

texts += [ax.text(party_corpus.pol.values[line], party_corpus.subj.values[line], party_corpus.index[line], horizontalalignment='center', size='large', color=parties_color['Democratic'], weight='semibold') for line in range(0,party_corpus.shape[0])]

#adjust_text(texts, arrowprops=dict(arrowstyle='->', color='blue'))



# Democratic-Republican

party_corpus = corpus[corpus['Party'] == 'Democratic-Republican']

sns.scatterplot(x=party_corpus.pol.values, y=party_corpus.subj.values, label='Democratic-Republican')

texts += [ax.text(party_corpus.pol.values[line], party_corpus.subj.values[line], party_corpus.index[line], horizontalalignment='center', size='large', color=parties_color['Democratic-Republican'], weight='semibold') for line in range(0,party_corpus.shape[0])]

#adjust_text(texts, arrowprops=dict(arrowstyle='->', color='green'))



# Federalist

party_corpus = corpus[corpus['Party'] == 'Federalist']

sns.scatterplot(x=party_corpus.pol.values, y=party_corpus.subj.values, label='Federalist')

texts += [ax.text(party_corpus.pol.values[line], party_corpus.subj.values[line], party_corpus.index[line], horizontalalignment='center', size='large', color=parties_color['Federalist'], weight='semibold') for line in range(0,party_corpus.shape[0])]

#adjust_text(texts, arrowprops=dict(arrowstyle='->', color='orange'))



# Republican

party_corpus = corpus[corpus['Party'] == 'Republican']

sns.scatterplot(x=party_corpus.pol.values, y=party_corpus.subj.values, label='Republican')

texts += [ax.text(party_corpus.pol.values[line], party_corpus.subj.values[line], party_corpus.index[line], horizontalalignment='center', size='large', color=parties_color['Republican'], weight='semibold') for line in range(0,party_corpus.shape[0])]

#adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))



# Unaffiliated

party_corpus = corpus[corpus['Party'] == 'Unaffiliated']

sns.scatterplot(x=party_corpus.pol.values, y=party_corpus.subj.values, label='Unaffiliated')

texts += [ax.text(party_corpus.pol.values[line], party_corpus.subj.values[line], party_corpus.index[line], horizontalalignment='center', size='large', color=parties_color['Unaffiliated'], weight='semibold') for line in range(0,party_corpus.shape[0])]

#adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))



# Whig

party_corpus = corpus[corpus['Party'] == 'Whig']

sns.scatterplot(x=party_corpus.pol.values, y=party_corpus.subj.values, label='Whig')

texts += [ax.text(party_corpus.pol.values[line], party_corpus.subj.values[line], party_corpus.index[line], horizontalalignment='center', size='large', color=parties_color['Whig'], weight='semibold') for line in range(0,party_corpus.shape[0])]



adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))

plt.title('Sentiment Analysis', fontsize=20)

plt.xlabel('<-- Negative ------------------------ Positive -->', fontsize=15)

plt.ylabel('<-- Facts ------------------------ Opinions -->', fontsize=15)



plt.show()
# Number of partitions

N = 10

polarity_transcripts = build_presidents_polarity_transcripts(corpus, N)

subjectivity_transcripts = build_presidents_subjectivity_transcripts(corpus, N)
# Plots

fig, ax = plt.subplots(11, 4, figsize=(15,40))



i = -1

for k, president in enumerate(corpus.index):

    if k % 4 == 0:

        i += 1

    j = k % 4

    sns.lineplot(x=range(0, N), y=polarity_transcripts[president], ax=ax[i, j], color=parties_color[corpus.loc[president].Party])

    ax[i, j].set(xlabel=president, ylabel='<-- Negative ------------------------ Positive -->')



fig.tight_layout()
# Build party corpus

party_corpus = build_party_corpus(corpus)

party_corpus = party_corpus.reset_index()



# Plot polarity and subjectivity

graph_political_era_polarity(party_corpus, 'Polarity of Political Parties')

graph_political_era_subjectivity(party_corpus, 'Subjectivity of Political Parties')
# Find polarity and subjectivity of texts for each political era



# First party system

first_party_corpus['pol'] = first_party_corpus.transcripts.apply(pol)

first_party_corpus['subj'] = first_party_corpus.transcripts.apply(sub)



# Second party system

second_party_corpus['pol'] = second_party_corpus.transcripts.apply(pol)

second_party_corpus['subj'] = second_party_corpus.transcripts.apply(sub)



# Third party system

third_party_corpus['pol'] = third_party_corpus.transcripts.apply(pol)

third_party_corpus['subj'] = third_party_corpus.transcripts.apply(sub)



# Fourth party system

fourth_party_corpus['pol'] = fourth_party_corpus.transcripts.apply(pol)

fourth_party_corpus['subj'] = fourth_party_corpus.transcripts.apply(sub)



# Fifth party system

fifth_party_corpus['pol'] = fifth_party_corpus.transcripts.apply(pol)

fifth_party_corpus['subj'] = fifth_party_corpus.transcripts.apply(sub)



# Sixth party system

sixth_party_corpus['pol'] = sixth_party_corpus.transcripts.apply(pol)

sixth_party_corpus['subj'] = sixth_party_corpus.transcripts.apply(sub)
party_system_corpuses = [

                            first_party_corpus, 

                            second_party_corpus,

                            third_party_corpus,

                            fourth_party_corpus,

                            fifth_party_corpus,

                            sixth_party_corpus

                        ]



party_system_titles = [

                            'First Party System (1792–1824)',

                            'Second Party System (1828–1854)',

                            'Third Party System (1854–1895)',

                            'Fourth Party System (1896–1932)',

                            'Fifth Party System (1932–1964)',

                            'Sixth Party System (1964–present)'

                        ]



graph_political_era_tuples = list(zip(party_system_corpuses, party_system_titles))
for corpus, title in graph_political_era_tuples:

    graph_political_era_polarity(corpus, title)
for corpus, title in graph_political_era_tuples:

    graph_political_era_subjectivity(corpus, title)
# Modify titles for comparisons

party_system_titles_comparison = [party + ' Comparison' for party in party_system_titles]

graph_political_era_comparison_tuples = list(zip(party_system_corpuses, party_system_titles_comparison))



for corpus, title in graph_political_era_comparison_tuples:

    graph_political_era_comparison(corpus, title)