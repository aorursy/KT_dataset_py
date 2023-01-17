from IPython.display import Image

Image("../input/jobbulletinresults/machine.png")
from IPython.display import Image

Image("../input/jobbulletinresults/Bias.png")
from IPython.display import Image

Image("../input/jobbulletinresults/Text2CSV.png")
'''

Script:  Stats.py

Purpose: Provide new insights about Job Bulletins and Career Paths

'''

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



# dnlp:  get statistics from JBR_NLP.py

dnlp = pd.read_csv('../input/jobbulletinresults/JBR_Output/fileStats.csv')

pd.options.display.max_columns=len(dnlp)



# dbias and dbw:  get statistics from biasVerifier.py

dbias = pd.read_csv('../input/jobbulletinresults/JBR_Output/BiasedWords.csv')

dbw = dbias[['FILE_NAME','WORD']]

# dbw2:  dataframe containing filename and the number of biased words found in it

dbw2 = pd.DataFrame({'TOT_BIAS_WORDS':dbw.groupby(['FILE_NAME'])['WORD'].count()}).reset_index()



# dstats: dataframe with statistics from word analysis in JBR_NLP and bias analysis in biasVerifier

dstats = pd.merge(dnlp,dbw2, how="left", on='FILE_NAME')

dstats["TOT_BIAS_WORDS"] = dstats["TOT_BIAS_WORDS"].fillna(0.0)

dstats["TOT_BIAS_WORDS"] = dstats["TOT_BIAS_WORDS"].astype('int64')
# djbstats:  merge statistics with Job Bulletin data

jb = pd.read_csv("../input/jobbulletinresults/JBR_Output/JobBulletin.csv")

djbstats = pd.merge(jb,dstats,how="left",on='FILE_NAME')

# df and df2: slice the dataframe to gather only numeric and categories useful for analysis

df = djbstats[['DRIVER_LICENSE_REQ', 'DRIV_LIC_TYPE', 'EDUCATION_YEARS','EXPERIENCE_LENGTH', 'FULL_TIME_PART_TIME', 'HIGH_SALARY', 'LOW_SALARY','SCHOOL_TYPE', 'TOT_DIF_WORDS', 'TOT_LONG_WORDS','TOT_WORDS', 'TOT_BIAS_WORDS']]

df = df.fillna(0)

df2 = pd.get_dummies(df)
# CHARTS and REPORTS



dnlp['TOT_WORDS'].plot(figsize=(10,5), kind='hist', color='black', title='Frequency of words in Job Bulletins')

plt.show()    



print("\nLongest Job Bulletin: ")

longest = dnlp['TOT_WORDS'].max()

longs = dnlp['FILE_NAME'][dnlp['TOT_WORDS'] == longest]

for long in longs: print(longest, " words in ", long)



print("\nShortest Job Bulletin:")

shortest = dnlp['TOT_WORDS'].min()

shorts = dnlp['FILE_NAME'][dnlp['TOT_WORDS'] == shortest]

for short in shorts: print(shortest, " words in ", short)



print("\nAverage Job Bulletin length:")

avgJB = dnlp['TOT_WORDS'].mean()

print(int(round(avgJB,0)), " words")
dnlp['TOT_DIF_WORDS'].plot(figsize=(10,5), kind='hist', color='navy', title='Frequency of different words in Job Bulletins')

plt.show()    

print("\nJob Bulletin with most different words: ")

longest = dnlp['TOT_DIF_WORDS'].max()

longs = dnlp['FILE_NAME'][dnlp['TOT_DIF_WORDS'] == longest]

for long in longs: print(longest, " words in ", long)



print("\nJob Bulletin with the least number of different words:")

shortest = dnlp['TOT_DIF_WORDS'].min()

shorts = dnlp['FILE_NAME'][dnlp['TOT_DIF_WORDS'] == shortest]

for short in shorts: print(shortest, " words in ", short)



print("\nAverage Job Bulletin length based on different words:")

avgJB = dnlp['TOT_DIF_WORDS'].mean()

print(int(round(avgJB,0)), " words")
dnlp['TOT_LONG_WORDS'].plot(figsize=(10,5), kind='hist', color='lightblue', title='Frequency of complex words found in Job Bulletins')

plt.show()    

print("\nJob Bulletin with the most complex words: ")

longest = dnlp['TOT_LONG_WORDS'].max()

longs = dnlp['FILE_NAME'][dnlp['TOT_LONG_WORDS'] == longest]

for long in longs: print(longest, " words in ", long)



print("\nJob Bulletin with the least number of complex words:")

shortest = dnlp['TOT_LONG_WORDS'].min()

shorts = dnlp['FILE_NAME'][dnlp['TOT_LONG_WORDS'] == shortest]

for short in shorts: print(shortest, " words in ", short)



print("\nAverage Job Bulletin complexity:")

avgJB = dnlp['TOT_LONG_WORDS'].mean()

print(int(round(avgJB,0)), " words")
print("Job Bulletins ranked from longest to shortest:\n")

dtot = dnlp[['FILE_NAME', 'TOT_WORDS']]

print(dtot.sort_values('TOT_WORDS', ascending=False))
print("Job Bulletins ranked from longest to shortest based on number of different words used:\n")

ddif = dnlp[['FILE_NAME', 'TOT_DIF_WORDS']]

print(ddif.sort_values('TOT_DIF_WORDS', ascending=False))
print("Job Bulletin ranked from most complex words used to least:\n")

dlong = dnlp[['FILE_NAME', 'TOT_LONG_WORDS']]

print(dlong.sort_values('TOT_LONG_WORDS', ascending=False))
# Feature correlation

# Features are columns of data.  Feature correlation shows which columns affect each other.  

# A heatmap will highlight correlated data so we can investigate it 



print("\nFeatures (interesting data that can be analyzed): ")

cols = df2.columns

for col in cols:  print(col)



#plt.matshow(df2.corr())

#plt.show()



def seabornCM(corr):

    import seaborn as sns

    sns.heatmap(corr, cmap="Blues")



seabornCM(df2.corr())

plt.show()

print("\nQUESTIONS POSED FROM ANALYZING FEATURES")



df2.plot(figsize=(10,5), x="LOW_SALARY", y="TOT_BIAS_WORDS", kind="scatter", title="Do jobs with higher salaries have more biased words?")

plt.show()



print("\nDo jobs with higher salaries have more biased words?")

print("Yes, jobs with higher salaries tend to have more biased words.")
print("\nI'm a Public Relations Specialist, what is the next step in my career path?")

nextPromotions = jb['NEXT_PROMOTION'][jb['JOB_CLASS_TITLE'] == 'PUBLIC RELATIONS SPECIALIST']

for nextPromotion in nextPromotions: print(nextPromotion)

careerPathImg = "../input/jobbulletinresults/"+jb['DAG_FILE'][jb['JOB_CLASS_TITLE'] == 'PUBLIC RELATIONS SPECIALIST']



img = mpimg.imread(careerPathImg.item())

imgplot = plt.imshow(img)

plt.show()
df2["EXPERIENCE_LENGTH"] = df2["EXPERIENCE_LENGTH"].fillna(0.0)

df2.plot(figsize=(10,5), x="EXPERIENCE_LENGTH", y="TOT_BIAS_WORDS", kind="scatter",  color="grey", title="Do jobs requiring more experience have more biased words?")

plt.show()



print("\nDo jobs requiring more experience have more biased words?")

print("Inconclusive because jobs paying $100k to 150k tend to be more scientific so these Job Bulletins have more complex words.  I plan to investigate only the gender-based words as a future project")

df2["EDUCATION_YEARS"] = df2["EDUCATION_YEARS"].fillna(0.0)

df2.plot(figsize=(10,5), x="EDUCATION_YEARS", y="TOT_BIAS_WORDS", kind="scatter",  color="teal", title="Do jobs requiring more years of education have more biased words?")

plt.show()



print("\nDo jobs requiring more years of education have more biased words?")

print("Inconclusive for non-native speakers bias because Job Bulletins for jobs requiring more education often include words related to a specific field of study, such as oceanography...  If a professional is skilled in that field, they should know these words even if they are not a native speaker.")
df2["EXPERIENCE_LENGTH"] = df2["EXPERIENCE_LENGTH"].fillna(0.0)

df2.plot(figsize=(10,5), x="EXPERIENCE_LENGTH", y="LOW_SALARY", kind="scatter", color="green", title="Do jobs requiring more experience pay a higher salary?")

plt.show()



print("\nDo jobs requiring more experience pay a higher salary?")

print("Inconclusive because EXPERIENCE_LENGTH needs to be double-checked.  It appears that some Job Bulletins only request the last 2-3 years of experience rather than a complete history while others ask for the history")

word_in_file = dbw.groupby(['FILE_NAME'])['WORD'].count()

print("\nSTATISTICS FOR BIASED WORDS USED IN EACH JOB BULLETIN:")

print("\nMost biased words used in one Job Bulletin: ", word_in_file.max())

print("Least biased words used in one Job Bulletin: ", word_in_file.min())

print("Average biased words per Job Bulletin: ", int(round(word_in_file.mean(),0)))



print("\nThe number of times a biased word was used in each file:\n")

print(word_in_file.sort_values(ascending=False))

print("\nThe number of times each biased word was used in each file:")

print(dbw.groupby(['FILE_NAME','WORD'])['WORD'].count())

dstats['TOT_BIAS_WORDS'].plot(figsize=(10,5), kind='hist', color='teal', title='Frequency of potentially biased words found in Job Bulletins')

plt.show()   
img = mpimg.imread("../input/jobbulletinresults/results.png")

imgplot = plt.imshow(img)

plt.show()
locateBias = dbias[['FILE_NAME','WORD_LOCATION','SENTENCE']]

locateBias = locateBias.drop_duplicates()

print("Number of times a biased word was found after dropping duplicates: ", len(locateBias))
print("Job Bulletins containing 'open competitive candidates'")

res = locateBias[locateBias["SENTENCE"].str.contains("open competitive candidates", na=False)]

print(res)



# "Competition attracts male candidates.  Could a gender-neutral term be used instead?

# Perhaps "open-hire candidates" or "new-hire candidates"?  

# If "open competitive candidates" is the required term, then the City can place this word on the non-bias list

# and it will not appear in future analysis

img = mpimg.imread("../input/jobbulletinresults/male2.png")

imgplot = plt.imshow(img)

plt.show()
print('Job Bulletins containing "COMPETITIVE BASIS"\n')

res = locateBias[locateBias["SENTENCE"].str.contains(">>> COMPETITIVE <<<  BASIS", na=False)]

print(res)



#Competitive basis" is a term that refers to the examination process.  Could a gender-neutral term be used such as "skills basis"

# If not, the City can add this term to the non-bias list 
print('Job Bulletins containing "determines"\n')



res = locateBias[locateBias["SENTENCE"].str.contains("determines", na=False)]

print(res)



# "determines" attracts male candidates.  Could a gender-neutral word be used in these sentences?

# For example, "determines procedures and methods" could be replaced with "finds procedures and methods"

print('Job Bulletins containing "competencies"\n')

res = locateBias[locateBias["SENTENCE"].str.contains("competencies", na=False)]

print(res)



# "competencies" is a complex word rarely used in daily speech.

# "competencies" could be replaced with "skills" because "skills" is a more commonly used word

print("Other words were found that are biased towards women and non-native speakers")
img = mpimg.imread('../input/jobbulletinresults/female.png')

imgplot = plt.imshow(img)

plt.show()



print('Job Bulletins containing the most used word biased towards women:  "responsibilities"\n')

res = locateBias[locateBias["SENTENCE"].str.contains("responsibilities", na=False)]

print(res)
img = mpimg.imread('../input/jobbulletinresults/female2.png')

imgplot = plt.imshow(img)

plt.show()
img = mpimg.imread('../input/jobbulletinresults/nonnative.png')

imgplot = plt.imshow(img)

plt.show()
img = mpimg.imread('../input/jobbulletinresults/nonnative-2.png')

imgplot = plt.imshow(img)

plt.show()



print('Job Bulletins containing the most used word biased towards non-native speakers: "examination"\n')

res = locateBias[locateBias["SENTENCE"].str.contains("examination", na=False)]

print(res)



# Can "examination" be replaced with "exam" or "test"?