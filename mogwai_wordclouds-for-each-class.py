%matplotlib inline

import pandas as pd

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt
STOPWORDS.add('et')

STOPWORDS.add('al')

STOPWORDS.add('â')

STOPWORDS.add('Figure')

STOPWORDS.add('figure')

STOPWORDS.add('fig')

STOPWORDS.add('Supplementary')
textDF = pd.read_csv('../input/training_text',sep="\|\|",header=None,names=['ID','Text'])

textDF = textDF[textDF.ID != 'ID,Text']

variantsDF = pd.read_csv('../input/training_variants').drop(['Gene','Variation'],1)

textDF.head()

variantsDF.head()

fullDF = textDF

fullDF['Class'] = variantsDF['Class']

fullDF.head()
class1DF = fullDF[fullDF.Class == 1]

class2DF = fullDF[fullDF.Class == 2]

class3DF = fullDF[fullDF.Class == 3]

class4DF = fullDF[fullDF.Class == 4]

class5DF = fullDF[fullDF.Class == 5]

class6DF = fullDF[fullDF.Class == 6]

class7DF = fullDF[fullDF.Class == 7]

class8DF = fullDF[fullDF.Class == 8]

class9DF = fullDF[fullDF.Class == 9]
class1 = class1DF['Text'].tolist()

string1 = ''

for i in range(len(class1)):

    string1 += class1[i]



class2 = class2DF['Text'].tolist()

string2 = ''

for i in range(len(class2)):

    string2 += class2[i]



class3 = class3DF['Text'].tolist()

string3 = ''

for i in range(len(class3)):

    string3 += class3[i]



class4 = class4DF['Text'].tolist()

string4 = ''

for i in range(len(class4)):

    string4 += class4[i]



class5 = class5DF['Text'].tolist()

string5 = ''

for i in range(len(class5)):

    string5 += class5[i]



class6 = class6DF['Text'].tolist()

string6 = ''

for i in range(len(class6)):

    string6 += class6[i]



class7 = class7DF['Text'].tolist()

string7 = ''

for i in range(len(class7)):

    string7 += class7[i]



class8 = class8DF['Text'].tolist()

string8 = ''

for i in range(len(class8)):

    string8 += class8[i]



class9 = class9DF['Text'].tolist()

string9 = ''

for i in range(len(class9)):

    string9 += class9[i]
wordcloud1 = WordCloud(   stopwords=STOPWORDS,

                          background_color='white',



                       max_words=25

                         ).generate(string1)



wordcloud2 = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',



                        max_words=25

                         ).generate(string2)



wordcloud3 = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',



                        max_words=25

                         ).generate(string3)



wordcloud4 = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',



                        max_words=25

                         ).generate(string4)



wordcloud5 = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',



                        max_words=25

                         ).generate(string5)



wordcloud6 = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',



                        max_words=25

                         ).generate(string6)



wordcloud7 = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',



                        max_words=25

                         ).generate(string7)



wordcloud8 = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',



                        max_words=25

                         ).generate(string8)



wordcloud9 = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',



                        max_words=25

                         ).generate(string9)
print("Class 1")

plt.imshow(wordcloud1)

plt.axis('off')

plt.show()



print("Class 2")

plt.imshow(wordcloud2)

plt.axis('off')

plt.show()



print("Class 3")

plt.imshow(wordcloud3)

plt.axis('off')

plt.show()



print("Class 4")

plt.imshow(wordcloud4)

plt.axis('off')

plt.show()



print("Class 5")

plt.imshow(wordcloud5)

plt.axis('off')

plt.show()



print("Class 6")

plt.imshow(wordcloud6)

plt.axis('off')

plt.show()



print("Class 7")

plt.imshow(wordcloud7)

plt.axis('off')

plt.show()



print("Class 8")

plt.imshow(wordcloud8)

plt.axis('off')

plt.show()



print("Class 9")

plt.imshow(wordcloud9)

plt.axis('off')

plt.show()