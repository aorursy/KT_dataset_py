#dependencies
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from numpy import array
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import LabelBinarizer
def load_data(filename,encoding):
    data = pd.read_csv(filename,encoding=encoding)
    return data

def create_train_test_sets(data,split):
    np.random.seed(0)
    mask = np.random.rand(len(data)) < split
    train_data = data[mask]
    test_data = data[~mask]
    return train_data,test_data

def clean_and_get_tokens(doc):
    tokens = doc.split()
    table = str.maketrans('','',punctuation)  #removes punctuations using 
    tokens = [w.translate(table) for w in tokens] #dictionary of punctuations
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word)>2]
    return tokens
data = load_data('../input/news-classification-dataset-bbc-news/BBC_news.csv','latin1')
words = set()
vocab = {}


token = data['texts'][0].split()
table = str.maketrans('','',punctuation)
tokens = [w.translate(table) for w in token] 
#print(tokens)
tokens = [word for word in tokens if word.isalpha()]
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
tokens = [word for word in tokens if len(word)>2]
#print(tokens)

documents = data['texts']
for doc in documents:
    tokens = clean_and_get_tokens(doc)
    for token in tokens:
        if token in vocab:
            vocab[token] += 1
        else:
            vocab[token] = 1

for word in vocab:
    if vocab[word] > 5:
        words.add(word)



train_data,test_data = create_train_test_sets(data,0.8)

train_documents = []
for doc in train_data['texts']:
    tokens = doc.split()
    final_tokens = []
    #final_string = ''
    for token in tokens:
        if token in words:
            final_tokens.append(token)
    final_string = ' '.join(final_tokens)
    train_documents.append(final_string)

test_documents = []
for doc in test_data['texts']:
    tokens = doc.split()
    final_tokens = []
    #final_string = ''
    for token in tokens:
        if token in words:
            final_tokens.append(token)
    final_string = ' '.join(final_tokens)
    test_documents.append(final_string)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_documents)
encoded_docs = tokenizer.texts_to_sequences(train_documents)

max_length = max(([len(s.split()) for s in train_documents]))
labels = train_data['CAT']
train_labels = labels
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
ytrain = keras.utils.to_categorical(labels, num_classes=5)
encoded_docs = tokenizer.texts_to_sequences(test_documents)
labels = test_data['CAT']
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
ytest = keras.utils.to_categorical(labels, num_classes=5)
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length = max_length))
model.add(Conv1D(filters=16, kernel_size=16, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2, validation_data = (Xtest,ytest))
random_text = ["Yukos unit buyer faces loan claim  The owners of embattled Russian oil giant Yukos are to ask the buyer of its former production unit to pay back a $900m (Ã‚Â£479m) loan.  State-owned Rosneft bought the Yugansk unit for $9.3bn in a sale forced by Russia to part settle a $27.5bn tax claim against Yukos. Yukos owner Menatep Group says it will ask Rosneft to repay a loan that Yugansk had secured on its assets. Rosneft already faces a similar $540m repayment demand from foreign banks. Legal experts said Rosneft's purchase of Yugansk would include such obligations. The pledged assets are with Rosneft, so it will have to pay real money to the creditors to avoid seizure of Yugansk assets, said Moscow-based US lawyer Jamie Firestone, who is not connected to the case. Menatep Group's managing director Tim Osborne told the Reuters news agency: If they default, we will fight them where the rule of law exists under the international arbitration clauses of the credit.  Rosneft officials were unavailable for comment. But the company has said it intends to take action against Menatep to recover some of the tax claims and debts owed by Yugansk. Yukos had filed for bankruptcy protection in a US court in an attempt to prevent the forced sale of its main production arm. The sale went ahead in December and Yugansk was sold to a little-known shell company which in turn was bought by Rosneft. Yukos claims its downfall was punishment for the political ambitions of its founder Mikhail Khodorkovsky and has vowed to sue any participant in the sale. "]
encoded_text = tokenizer.texts_to_sequences(random_text)
test_text = pad_sequences(encoded_text, maxlen = max_length, padding= 'post')
model.predict(test_text)
random_text = ["Retirement age could be scrapped  The myth that ageing is a barrier to contributing to society needs to be exploded, the work and pensions minister has said.  This was why the government was considering scrapping the retirement age entirely, Alan Johnson said. It was also committed to stamping out age discrimination and would outlaw it, he told a conference on ageing. All three parties have been wooing older voters with both the Tories and Lib Dems pledging higher pensions.  Mr Johnson told Age Concern's Age Agenda in London the government was 'seriously considering' introducing pensions based on residency rather than national insurance contributions. This idea has been adopted by the Lib Dems as policy, while the Tories have pledged to boost pensions by restoring the link between earnings and pensions. Mr Johnson's speech comes after he last week unveiled plans to find a consensus on how to reform the country's pension system. This would be based on a series of principles including tackling pensioner poverty and fairer pensions for women, he said. Speaking at the London conference he said: Generalised stereotypes of people past state pension age as dependant, incapable and vulnerable are a particularly pernicious form of age discrimination.  The government wanted to tackle this by moving to a culture where retirement ages were increasingly consigned to the past. We're sweeping them away entirely for people under 65, and we're giving those above that age a right to request to work past 65 which their employers will have to engage with seriously. And the review in 2011, which will look at whether it is time to sweep retirement ages away entirely, is to be tied to evidence ... showing that retirement ages are increasingly outmoded. Mr Johnson said his department had a long-term aspiration of moving towards an 80% employment rate. This would involve an extra one million older people joining the work force, he said. "]
encoded_text = tokenizer.texts_to_sequences(random_text)
test_text = pad_sequences(encoded_text, maxlen = max_length, padding= 'post')
model.predict(test_text)
random_text = ["Unilever has pledged to drop fossil fuels from its cleaning products by 2030 to reduce carbon emissions. The consumer goods giant said it would invest €1bn (£890m, $1.2bn) in the effort. Unilever said it would replace petrochemicals with ingredients made from plants, and marine sources like algae. The company's best-selling cleaning brands include Omo, Cif, Sunlight and Domestos. Unilever said the chemicals used in its cleaning and laundry products make up 46% of its overall carbon footprint. Replacing them with more sustainable ingredients will reduce that footprint by up to 20%."]
encoded_text = tokenizer.texts_to_sequences(random_text)
test_text = pad_sequences(encoded_text, maxlen = max_length, padding= 'post')
model.predict(test_text)
random_text = ["He is the most charismatic figure in technology with some amazing achievements to his name, from making electric cars desirable to developing rockets that can return to earth and be reused. But dare to suggest that anything Elon Musk does is not groundbreaking or visionary and you can expect a backlash from the great man and his army of passionate fans. That is what happened when a British academic criticised Musk's demo on Friday of his Neuralink project - and the retaliation he faced was largely my fault. Neuralink is a hugely ambitious plan to link the human brain to a computer. It might eventually allow people with conditions such as Parkinson's disease to control their physical movements or manipulate machines via the power of thought. There are plenty of scientists already at work in this field. But Musk has far greater ambitions than most, talking of developing superhuman cognition - enhancing the human brain in part to combat the threat he sees from artificial intelligence. Friday night's demo involved a pig called Gertrude fitted with what the tech tycoon described as a Fitbit in your skull. A tiny device recorded the animals neural activity and sent it wirelessly to a screen. A series of beeps happened every time her snout was touched, indicating activity in the part of her brain seeking out food. 'I think this is incredibly profound', commented Musk. Some neuroscience experts were not quite as impressed"]
encoded_text = tokenizer.texts_to_sequences(random_text)
test_text = pad_sequences(encoded_text, maxlen = max_length, padding= 'post')
model.predict(test_text)

random_text = ["Banks are free to restructure loans but they cannot penalise honest borrowers by charging interest on deferred EMI payments under the moratorium scheme during the Covid-19 pandemic, a petitioner opposing the move said in the Supreme Court on Wednesday. A bench headed by Justice Ashok Bhushan, which commenced final hearing on a batch of pleas raising the issue of interest on instalments deferred under the scheme during the moratorium period, was told that paying interest on interest is a double whammy for borrowers."]
encoded_text = tokenizer.texts_to_sequences(random_text)
test_text = pad_sequences(encoded_text, maxlen = max_length, padding= 'post')
model.predict(test_text)
random_text = ["It is a three-season partnership which will see the association run through till the 2022 season. CRED, which allows users to pay credit card bills through its mobile applications, aims to make financial decisions smooth and rewarding for its members. Mr Brijesh Patel, Chairman, IPL, said: We are very pleased to have CRED on board as the 'Official Partner' of the Indian Premier League 2020 to 2022. IPL is one of the most innovative sporting leagues of the world and we are delighted to have a brand as unique and innovative as CRED partner us. I am sure more people across the country will take notice of them as we embark on this exciting journey.Kunal Shah, Founder and CEO, CRED, said: “We are extremely pleased to be associated with IPL, without a question among the most high-profile events on the world’s sporting calendar. CRED is aimed at giving millions of people access to the good life through improved credit standing, trusted community, and special experiences. IPL represents the pinnacle of consumer experiences, powered by a global community of cricketers, fans and enthusiasts. We look forward to participating in this festival of sporting excellence, which celebrates peak performance for the individual, the team and the community. Through this synergy, we want to celebrate and recognize millions of Indians who hold the same values that IPL and CRED cherish."]
encoded_text = tokenizer.texts_to_sequences(random_text)
test_text = pad_sequences(encoded_text, maxlen = max_length, padding= 'post')
model.predict(test_text)
random_text = ["Ishant Sharma has been an unsung hero for India, featuring in many historic moments for the team. As he turns 32, here's looking back at those moments and other contributions made by the persistent fast bowler."]
encoded_text = tokenizer.texts_to_sequences(random_text)
test_text = pad_sequences(encoded_text, maxlen = max_length, padding= 'post')
model.predict(test_text)
random_text = ["PUBG Mobile has announced that the New Erangel is coming this September with lots of changes and improvement. But, before the official release, the developers are kickstarting a community event called the Dawn of a New Era where they want players to show their favourite memories over the past two years on Erangel. There will be three categories, which will include Favourite Chicken Dinner memory (Screenshot of Clip), Favourite screenshot on Erangel and Favourite Erangel video clip, to showcase past experiences on Erangel. Once the submissions are over, three winners will be chosen for each category by the community team. The first-place winners will get $100 UC and have their clips or images showcased on the official PUBG Mobile Twitter handle. The second and third place winners will receive an award of $50 UC and $25 UC, respectively. Here's how you can submit your clippings for the PUBG Mobile Dawn of a New Era event. How to participate for PUBG Mobile Dawn of a New Era event Interested participants need to post their clippings on Twitter, Facebook, or Instagram with the hashtag #PUBGMErangel. The post should be visible to public view.The judging will be based on - How unique the moment/memory is, how well it relates to the event and how it captures the feeling we all know and love of playing on Erangel.You can submit multiple entries for all of the categories, but if one of your submissions is selected as a winner, all others will be disqualified. One person cannot win multiple times for multiple categories"]
encoded_text = tokenizer.texts_to_sequences(random_text)
test_text = pad_sequences(encoded_text, maxlen = max_length, padding= 'post')
model.predict(test_text)
random_text = ["The galaxy, called AUDFs01 was discovered by a team of astronomers led by IUCAA's Kanak Saha, announced Union Minister of State Department of Space Jitendra Singh.In a landmark achievement in Space missions, a team of scientists from Pune's Inter-University Centre for Astronomy & Astrophysics (IUCAA) have discovered one of the farthest galaxies in the universe, officials said here on Tuesday.The galaxy, called AUDFs01 was discovered by a team of astronomers led by IUCAA's Kanak Saha, announced Union Minister of State Department of Space Jitendra Singh.The galaxy was discovered by India's first Multi-Wavelength Space Observatory 'AstroSat', which detected extreme-UV light from a galaxy located at an astronomical distance of 9.30 billion light-years from Earth.This is the second achievement for Maharashtra in the past fortnight and comes after two students of IIT-Bombay discovered an SUV-sized asteroid that zoomed past just 2,950 km above Earth's surface on August 16. The students are Kunal Deshmukh from Pune and Kritti Sharma from Panchkula, Haryana.The importance and uniqueness of this original discovery can be made out from the fact that it has been reported in the leading international journal 'Nature Astronomy' published from Britain. India's AstroSat/UVIT was able to achieve this unique feat because the background noise in the UVIT detector is much less than one on the Hubble Space Telescope of US-based NASA. Singh lauded India's Space Scientists for once again proving to the world that India's capability in Space technology has risen to a distinguished level from where our scientists are now offering cues and giving leads to the Space scientists in other parts of the world.IUCAA Director Somak Ray Chaudhury said: This discovery is a very important clue to how the dark ages of the Universe ended and there was light in the Universe. We need to know when this started, but it has been very hard to find the earliest sources of light.A noteworthy aspect is that AstroSat, which made this discovery, was launched by the Indian Space Research Organization (ISRO) on September 28, 2015, during the first term of the Modi government.It was developed by a team led by Shyam Tandon, Ex Emeritus Professor, IUCAA with the full support of ISRO. According to Tandon, the excellent spatial resolution and high sensitivity is a tribute to the hard work of the UVIT core team of scientists for over a decade."]
encoded_text = tokenizer.texts_to_sequences(random_text)
test_text = pad_sequences(encoded_text, maxlen = max_length, padding= 'post')
model.predict(test_text)
random_text = [" Despite job uncertainty, work from home necessitated due to Covid-19 pandemic may have an upside for working professionals as well. An average working professional is saving Rs 5,000 per month in saved expenses on food, clothing and almost 2 hours in commuting time, found a survey conducted by co-working space provider Awfis. The survey also found that as many as 74% of this workforce is ready to work remotely, either from cafes or their homes, as this saves their time and money. 80% feel that their jobs can be performed from a remote location instead of the office. The survey found that as many as 20% are saving Rs 5,000-10,000 every month as they are working from home while 19% are saving over Rs 10,000.'In a highly price and cost-sensitive country like India, the savings done by working from home is a major factor in the acceptance of this change. On average, an employee saves Rs 5,520 per month which was earlier spent on food, commute clothing, etc. This comes to approximately 17% of an average Indian’s salary', the report said. The survey was conducted across seven Indian metro cities during June and July with a sample size of 1,000. Another reason why working from home or remotely is popular is due to the amount of time saved by professionals in commuting to and from office. As many as 60% respondents said they spend more than an hour commuting to and from work which they are saving. This leads to 1.5 hours of saved time per employee which translates into 44 additional working days in a year.'This means that for a company with 100 employees, 18 FTE (full-time equivalent) days are added without any additional cost,' said Amit Ramani, founder and CEO of Awfis."]
encoded_text = tokenizer.texts_to_sequences(random_text)
test_text = pad_sequences(encoded_text, maxlen = max_length, padding= 'post')
model.predict(test_text)
random_text = ["Banks are free to restructure loans but they cannot penalise honest borrowers by charging interest on deferred EMI payments under the moratorium scheme during the Covid-19 pandemic, a petitioner opposing the move said in the Supreme Court on Wednesday. A bench headed by Justice Ashok Bhushan, which commenced final hearing on a batch of pleas raising the issue of interest on instalments deferred under the scheme during the moratorium period, was told that paying interest on interest is a double whammy for borrowers."]
encoded_text = tokenizer.texts_to_sequences(random_text)
test_text = pad_sequences(encoded_text, maxlen = max_length, padding= 'post')
model.predict(test_text)
ypred = model.predict(Xtest)
pred_labels = []
for probs in ypred:
    label = np.argmax(probs, axis=-1)
    pred_labels.append(int(label))
actual_labels = list(labels)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(actual_labels, pred_labels)
import matplotlib.pyplot as plt
import itertools
cmap = plt.cm.Blues
title = "Confusion Matrix"
classes = 5
normalize = False
tick_marks = np.arange(classes)
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(5)
#plt.xticks(tick_marks, classes, rotation=45)
#plt.yticks(tick_marks, classes)

fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()