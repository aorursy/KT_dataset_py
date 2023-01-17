import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_variants_df = pd.read_csv("../input/training_variants")

test_variants_df = pd.read_csv("../input/test_variants")

train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

print("Train and Test variants shape : ",train_variants_df.shape, test_variants_df.shape)

print("Train and Test text shape : ",train_text_df.shape, test_text_df.shape)
train_variants_df.head()
train_text_df.head()
counter = 0

with open("../input/training_text") as infile:

    while True:

        counter += 1

        line = infile.readline()

        print(line)

        if counter==2:

            break
plt.figure(figsize=(12,8))

sns.countplot(x="Class", data=train_variants_df)

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('Class Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of Classes", fontsize=15)

plt.show()
train_genes = train_variants_df.groupby('Gene')['Gene'].count()



for i in [2, 5, 10, 20, 50, 100, 300]:

    print('Genes that appear less than {} times: {}%'.format(i, round((train_genes < i).mean() * 100, 2)))



plt.figure(figsize=(12, 8))

plt.hist(train_genes.values, bins=50, log=True)

plt.xlabel('Number of times Gene appeared', fontsize=12)

plt.ylabel('log of Count', fontsize=12)

plt.show()
train_variation = train_variants_df.groupby('Variation')['Variation'].count()



for i in [2,3,5,50,100]:

    print('Genes that appear less than {} times: {}%'.format(i, round((train_variation < i).mean() * 100, 2)))



plt.figure(figsize=(12, 8))

plt.hist(train_variation.values, bins=50, log=True, color='green')

plt.xlabel('Number of times Variation appeared', fontsize=12)

plt.ylabel('log of Count', fontsize=12)

plt.show()
train_text_df["Text_num_words"] = train_text_df["Text"].apply(lambda x: len(str(x).split()) )

train_text_df["Text_num_chars"] = train_text_df["Text"].apply(lambda x: len(str(x)) )
plt.figure(figsize=(12, 8))

sns.distplot(train_text_df.Text_num_words.values, bins=50, kde=False, color='red')

plt.xlabel('Number of words in text', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.title("Frequency of number of words", fontsize=15)

plt.show()
plt.figure(figsize=(12, 8))

sns.distplot(train_text_df.Text_num_chars.values, bins=50, kde=False, color='brown')

plt.xlabel('Number of characters in text', fontsize=12)

plt.ylabel('log of Count', fontsize=12)

plt.title("Frequency of Number of characters", fontsize=15)

plt.show()
train_df = pd.merge(train_variants_df, train_text_df, on='ID')



plt.figure(figsize=(12,8))

sns.boxplot(x='Class', y='Text_num_words', data=train_df)

plt.xlabel('Class', fontsize=12)

plt.ylabel('Text - Number of words', fontsize=12)

plt.show()