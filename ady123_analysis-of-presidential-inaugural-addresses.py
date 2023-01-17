#Import essential libraries

!pip install textstat

import textstat

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
#Getting the data

speeches = pd.read_csv("/kaggle/input/presidentialaddress/inaug_speeches.csv", encoding = "latin1")

speeches.head()
#Create lists to store information

years, numChars, numWords, numDifficult, daleScores = [], [], [], [], []



for i in range(len(speeches)):

  years.append(int(speeches["Date"][i][-4::]))

  numChars.append(len(speeches["text"][i]))

  numWords.append(len(speeches["text"][i].split(" ")))

  numDifficult.append(textstat.difficult_words(speeches["text"][i]))

  daleScores.append(textstat.dale_chall_readability_score(speeches["text"][i]))
#Function to plot a scatterplot of any variable against the year

def plot(yList, yLabel):

  plt.rcParams["figure.figsize"] = (14, 8)

  plt.xticks(range(len(years)), years, rotation = 90)

  plt.xlabel("Year that Inaugural Address was given")

  plt.ylabel(yLabel)



  x = [i for i in range(len(speeches))]

  plt.scatter(x, yList)



  coeff = np.polyfit([i for i in range(len(speeches))], yList, 1)

  regLine = [((coeff[0] * val) + coeff[1]) for val in x]

  plt.plot(x, regLine)



  plt.show()



  print("\nCorrelation Coefficient: " + str(np.corrcoef(x, yList, 1)[0][1]))





#Function that generates the five-number summary for any list

def fiveNum(inputList):

  quartiles = np.percentile(inputList, [0, 25, 50, 75, 100])

  print("Minimum: " + str(quartiles[0]))

  print("1st Quartile: " + str(quartiles[1]))

  print("Median: " + str(quartiles[2]))

  print("3rd Quartile: " + str(quartiles[3]))

  print("Maximum: " + str(quartiles[4]))
#Plot the number of characters against the year

plot(numChars, "Number of characters in speech")
#Plot the number of space-separated words against the year

plot(numWords, "Number of space-separated words in speech")
#Plot the number of difficult words against the year

plot(numDifficult, "Number of difficult words in speech")
#Plot the Dale-Chall readability scores against the year

plot(daleScores, "Dale–Chall readability score")
#Plot word cloud of all inaugural addresses

combinedText = ""

for i in range(len(speeches)):

  combinedText += " " + speeches["text"][i]



wordcloud = WordCloud(stopwords = set(STOPWORDS), max_font_size = 50, max_words = 70, background_color = "white").generate(combinedText)



plt.rcParams["figure.figsize"] = (16, 8)

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
#Ten longest inaugural addresses

long = []

long.extend(numWords)

long.sort(reverse = True)



for i in range(10):

  index = numWords.index(long[i])

  print(str(i + 1) + ") " + list(speeches["Name"])[index] + ": " + str(long[i]) + " words" + " (" + str(years[index]) + ")")
#Ten shortest inaugural addresses

short = []

short.extend(numWords)

short.sort()



for i in range(10):

  index = numWords.index(short[i])

  print(str(i + 1) + ") " + list(speeches["Name"])[index] + ": " + str(short[i]) + " words" + " (" + str(years[index]) + ")")
#Five-number summaries of all lists

print("Five-number summary of number of characters: ")

fiveNum(numChars)

print("")



print("Five-number summary of number of words: ")

fiveNum(numWords)

print("")



print("Five-number summary of number of difficult words: ")

fiveNum(numDifficult)

print("")



print("Five-number summary of Dale-Chall scores: ")

fiveNum(daleScores)