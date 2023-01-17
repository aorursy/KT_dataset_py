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
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import re
import pandas as pd

# nltk.download()
paragraph = """Mr. President, Last year I spoke at one of. As the marches in Chicago for comprehensive immigration reform. 
I looked out across the faces of the crowd. I saw mothers and fathers, citizens and noncitizens, people of Polish and Mexican descent, working Americans, and children. 
And what I know is that these are people we should embrace, not fear. We can and should be able to see ourselves in them.
I do not say that to diminish the complexity of the task. I say it because I believe that attitude must guide our discourse. We can and should be able to fix our broken immigration system and do so in a way that's reflective of American values and ideals and the tradition that we have of accepting immigrants to our shores.
Now, I think that the bill that has come to the floor is a fine first step, but I strongly believe it requires some changes. And I'm working with others to improve it.
In approaching immigration reform, I believe that we must enact tough, practical reforms that ensure and promote the legal and orderly entry of immigrants into our country. Just as important, we must respect the humanity of the carpenters and bricklayers who help build America; the humanity of garment workers and farmworkers who come to America to join their families; the humanity of students like my father who came to America in search of a dream. We are a nation of immigrants, and we must respect that shared history as this debate moves forward.
To fix the system in a way that does not require us to revisit the same problem in twenty years, I continue to believe that we need stronger enforcement on the border and at the workplace. And that means a workable mandatory system that employers must use to verify the legality of their workers.
But for reform to work, we also must respond to what pulls people to America and what pushes them out of their home countries. Where we can reunite families, we should. Where we can bring in more foreign-born workers with the skills our economy needs, we should. And these goals are not mutually exclusive. We should not say that Spanish speaking or working class immigrants are only good enough to be temporary workers and cannot earn the right to be a part of the American family.
with regard to the most pressing part of the immigration challenge -- the 12 million undocumented immigrants living in the U.S. -- we must create an earned path to citizenship. Now, no one condones unauthorized entry into the United States. And by supporting an earned path to citizenship, I am not saying that illegal entry should go unpunished. The path to permanent residence and eventual citizenship must be tough enough to make it clear that an unauthorized entry was wrong and will be punished.
But these immigrants are our neighbors. They go to our churches, and their kids go to our schools. They provide the hard labor that supports many of the industries in our country. We should bring them out of hiding, make them pay the appropriate fines for their mistakes, and then help them become tax paying, law-abiding, productive members of society. And so I'm heartened by the agreement that we have in this bill to put all 12 million or so undocumented immigrants on a path to earned citizenship. And I applaud those who worked on this compromise to achieve that pathway.
But there are other parts of the compromise deal before us that cause me serious concern.
I want to just address those now, Mr. President.
In order to stem the demand for illegal workers, we need a mandatory employment verification system that is actually mandatory. It needs to allow employers to check with the Department of Homeland Security to see that their employees are legally eligible to work in the United States. This is something that I worked on last year when we crafted comprehensive reform and passed it out of the Senate.
Unfortunately, this this year's version of the employment eligibility verification system would give the Department of Homeland Security too much power to force the screening of everyone working in America without appropriate safeguards. I will be working with others to offer an amendment to make this provision closer to what we proposed last year.
As for the guestworker program in the bill, it proposes to create a new 400,000 person annual temporary worker program that could grow to 600,000 without Congressional approval. And it expands the existing seasonal guestworker program from 66,000 up to 100,000 in the first year, and 200,000 after that.
At the end of their temporary status, almost all of these workers would have to go home. That means at the end of the first three years, we would have at least 1.2 million of these new guestworkers in the country with only 30,000 of them having any real hope of being able to stay. And it is my strong belief, Mr. President, that we are setting ourselves up for failure and creating a new undocumented immigrant population.
As we've learned with misguided immigration policies in the past, it is naive to think that people who do not have a way to stay legally will just abide by the system and leave. They will not. This new group of second-class workers will replace the current group of undocumented immigrants, placing downward pressure on American wages and working conditions. And where their -- when their time is up, they will go into the shadows where our current system exploits the undocumented today.
I will support amendments aimed at fixing the temporary worker program that Senator Bingaman and others have offered. And if we're going to have a new temporary worker program, those workers should have the opportunity to stay if they prove themselves capable and willing to participate in this country.
I have to say though, Mr. President, that the most disturbing aspect of this bill is the point system for future immigrants. As currently drafted, it does not reflect how much Americans value the family ties that bind people to their brothers and sisters, or to their parents.
As I understand it, a similar point system is used in Australia and Canada and is intended to attract immigrants who can help produce more goods. But we need to consider more than economics; we also need to consider our nation's unique history and values and what family-based preferences are designed to accomplish. As currently structured, the point system gives no preference to an immigrant with a brother or sister or even a parent who is a U.S. citizen unless the immigrant meets some minimum and arbitrary threshold on education and skills.
That's wrong and fails to recognize the fundamental morality of uniting Americans with their family members. It also places a person's job skills over his character and work ethic. How many of our forefathers would have measured up under this point system? How many would have been turned back at Ellis Island?
I've cosponsored an amendment with the president [Presiding Officer Senator Menendez] to remove that arbitrary minimum threshold of points before family starts to count and to bump up the points for family ties.
And at the appropriate time, I will be offering another amendment with the president [Presiding Officer Senator Menendez] to sunset the points system in the bill. The proposed point system constitutes, at minimum, a radical experiment in social engineering and a departure from our tradition of having family and employers invite immigrants to come. If we are going to allow this to go forward, then Congress should at the very least revisit the point system in five years to give us time to examine the concept in depth and determine whether its intended or unintended consequences are worth the cost of continuing the experiment, or, whether we should return to the existing system that allows immigrants to be sponsored through family and employers.
In closing, Mr. President, we must construct a final product that has broad bipartisan support and will work. I agree with Senator Brownback that the time is now to fix our broken immigration system. If we do not fix it this year, I fear that divisions over the issue will only deepen and the challenge will only grow.
But I also believe, Mr. President, that we have to get it right. And I think that it is critical that as we embark on this enormous venture to update our immigration system, that it is fully reflective of the powerful tradition of immigration in this country and fully reflective of our values and our ideals."""
# Sentence tokenize to convert the paragraph in to Sentences
sentences = nltk.sent_tokenize(paragraph)
len(sentences)
sentences
# Word tokenize to convert sentences in words
# Then Stemming those words which are not Stopwords.
# stopwords.words('english')

# stemmer = PorterStemmer()
# for i in range(len(sentences)):
#     words = nltk.word_tokenize(sentences[i])
#     words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
#     sentences[i] = ' '.join(words)

# sentences
# Apply lemmitization
def lemma(sentences):
    lemmatizer = WordNetLemmatizer()
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
        sentences[i] = ' '.join(words)
    return sentences
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]',' ',sentences[i])
    review = review.lower()
    review = review.split()
    review = lemma(review)
    review = ' '.join(review)
    corpus.append(review)

# corpus
# type(review)
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
X
cv = TfidfVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
X