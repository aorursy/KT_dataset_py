import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.context('seaborn')

import seaborn as sns

sns.set_style("darkgrid")

sns.set_palette('Set2')



# Skip the first row in order to not take questions, only answers

df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv', skiprows=[1])



# Get array of questions

questions = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')

questions = questions.loc[0]



for i in range(1, 34):

    print('{}- {}'.format(i,questions[i]), end='\n\n')
# Return labels and counts for unique choice questions

def getDataFromUniqueAnswers(df, question):

    return df['Q{}'.format(question)].value_counts().index, df['Q{}'.format(question)].value_counts().values



# Return labels and counts for multiple choices questions

def getDataFromMultipleAnswers(df, question, numOfChoices):

    answers = [[df['Q{}_Part_{}'.format(question, i)].dropna().unique()[0],

            df['Q{}_Part_{}'.format(question, i)].count()]

            for i in range(1, numOfChoices + 1)]

    answers.sort(key = lambda x: x[1], reverse = True)

    return [answer[0] for answer in answers], [answer[1] for answer in answers]



# Return array of strings with the label and the pourcentage

def getNamePlusPourcentage(labels, counts):

    return ['{} ({:.2f}%)'.format(label, count / sum(y) * 100) for label, count in zip(labels, counts)]



x, y = getDataFromUniqueAnswers(df, 5)

fig, ax = plt.subplots(figsize=(8, 8))

ax.pie(y, labels=x, autopct='%1.f%%')

ax.set_title(questions[5], fontdict={'fontweight': 'bold'})

plt.show()
dfDataScientists = df[df['Q5'] == 'Data Scientist']



x, y = getDataFromMultipleAnswers(dfDataScientists, 9, 8)

fig, ax = plt.subplots(figsize=(7, 7))

ax.set_title(questions[9] + '\n', fontdict={'fontweight': 'bold'}, loc='right')

ax = sns.barplot(y, x, orient='h')
x, y = getDataFromMultipleAnswers(dfDataScientists, 18, 12)

fig, ax = plt.subplots(figsize=(15, 7))

ax.set_title(questions[18] + '\n', fontdict={'fontweight': 'bold'})

ax = sns.barplot(x, y, orient='v')
x, y = getDataFromUniqueAnswers(dfDataScientists, 19)

fig, ax = plt.subplots(figsize=(7, 7))

ax.pie(y)

ax.set_title(questions[19] + '\n', fontdict={'fontweight': 'bold'})

plt.legend(getNamePlusPourcentage(x, y), loc="upper left")

plt.show()
x, y = getDataFromMultipleAnswers(dfDataScientists, 16, 12)

fig, ax = plt.subplots(figsize=(12, 7))

ax.set_title(questions[16] + '\n', fontdict={'fontweight': 'bold'}, loc='right')

ax = sns.barplot(y, x, orient='h')
x, y = getDataFromUniqueAnswers(dfDataScientists, 14)

fig, ax = plt.subplots(figsize=(10, 7))

ax.set_title(questions[14] + '\n', fontdict={'fontweight': 'bold'}, loc='right')

ax = sns.barplot(y, x, orient='h')
x, y = getDataFromMultipleAnswers(dfDataScientists, 24, 12)

fig, ax = plt.subplots(figsize=(8, 8))

ax.pie(y, labels=x, autopct='%1.f%%')

ax.set_title(questions[24], fontdict={'fontweight': 'bold'})

plt.show()
x, y = getDataFromMultipleAnswers(dfDataScientists, 12, 12)

fig, ax = plt.subplots(figsize=(10, 7))

ax.set_title(questions[12] + '\n', fontdict={'fontweight': 'bold'}, loc='right')

ax = sns.barplot(y, x, orient='h')
x, y = getDataFromMultipleAnswers(dfDataScientists, 13, 12)

fig, ax = plt.subplots(figsize=(8, 8))

ax.pie(y, labels=x, autopct='%1.f%%')

ax.set_title(questions[13], fontdict={'fontweight': 'bold'})

plt.show()
dfFrenchDataScientists = dfDataScientists.query('Q3 == "France"')

x, y = dfFrenchDataScientists['Q10'].value_counts().index, dfFrenchDataScientists['Q10'].value_counts().values



# Create a new array in order to sort the axis by compensation

axis = list(map(lambda index: (int(index.split('-')[0].replace(',', '').replace('$', '').replace('>', '')), index), x))

axis = [(index, count) for index, count in zip(axis, y)]

axis.sort()



# Recreate x and y sorted with only compensations above $20,000

x = [row[0][1] for row in axis if row[0][0]>=20000]

y = [row[1] for row in axis if row[0][0]>=20000]



fig, ax = plt.subplots(figsize=(20, 7))

ax.set_title(questions[10] + '\n', fontdict={'fontweight': 'bold'})

plt.xticks(rotation=45)

ax = sns.barplot(x, y)
# Group by compensations and company sizes

dfGrouped = dfDataScientists.groupby(['Q10', 'Q6']).size().unstack()

dfGrouped.loc[x[0]]



# Create new df with salary sorted and above $20,000

newDf = pd.DataFrame()

for index, label in enumerate(x):

    newDf = newDf.append(dfGrouped.loc[label])



# Sort the columns

cols = ['0-49 employees', '50-249 employees', '250-999 employees', '1000-9,999 employees', '> 10,000 employees']

newDf = newDf[cols]



# Create a new column with the total for each compensation

newDf['sum'] = newDf.sum(axis=1)

 

# Divide by the total in order to get the part of every company sizes for each compensation

newDf = newDf.loc[:,cols].div(newDf['sum'], axis=0)



fig, ax = plt.subplots(figsize=(16, 8))

ax.set_title(questions[10] + '\n', fontdict={'fontweight': 'bold'})

newDf.plot(kind='bar', stacked=True, ax=ax)

plt.xticks(rotation=45)

plt.show()