
from google.colab import drive
drive.mount('/content/gdrive/', force_remount=True)
#Importing tensorflow
%tensorflow_version 2.x
import tensorflow as tf

#Checking GPU availability. This is not necessary if you are not working with GPUs
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
#Importing the necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy import array, asarray, zeros
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Embedding, LSTM
from tensorflow.keras.optimizers import Adam, Adadelta
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
#Install the NLTK data
nltk.download("stopwords")
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
gs_uri = r'/content/gdrive/My Drive/NLPg/IMDB Dataset.csv'

#load the data into a DataFrame
movie_reviews = pd.read_csv(gs_uri)

#Rows that have missing data are removed
movie_reviews.fillna(value="", inplace=True)
movie_reviews.dropna(inplace=True)

#Display the size of the dataframe (50,000 rows = No. reviews and 2 columns = [review, sentiment])
movie_reviews.shape
def clean_text(text):
  #Convert to lowercase
  text = text.lower()

  # Remove html tags
  text = re.compile(r'<[^>]+>').sub('', text)

  # Remove punctuations and numbers, emails, websites
  text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
  text = re.sub("[^a-zA-Z]", " ", text)

  # Single character removal
  text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

  # Removing multiple spaces
  text = re.sub(r'\s+', ' ', text)
  
  #Removing stopwords
  text_token = word_tokenize(text)

  return [word for word in text_token if word not in stop_words]

x = []
sentences = list(movie_reviews['review'])

for sen in sentences:
  x.append(clean_text(sen))
# 1 and 0 are assigned for positive and negative sentiment respectively
labels = movie_reviews['sentiment']
labels = np.array(list(map(lambda x: 1 if x=="positive" else 0, labels)))

#It keeps the words of the review that are within the first 5000 most frequent words throughout the corpus
tokenizer = Tokenizer(num_words=5000)

#Update the vocabulary
tokenizer.fit_on_texts(x)

# Transform the text of each review into a sequence of integers
reviews = tokenizer.texts_to_sequences(x)


import matplotlib.pyplot as plt
%matplotlib inline
reviews_len = [len(x) for x in reviews]
pd.Series(reviews_len).hist(bins=50)
plt.show()
pd.Series(reviews_len).describe(percentiles=[0.1, 0.5])
#remove very short reviews (those below the 10th percentile)
reviews = [ reviews[i] for i, len_rev in enumerate(reviews_len) if len_rev > 37 ]
labels = [ labels[i] for i, len_rev in enumerate(reviews_len) if len_rev > 37 ]
import matplotlib.pyplot as plt
%matplotlib inline
reviews_len = [len(x) for x in reviews]
pd.Series(reviews_len).hist(bins=50)
plt.show()
pd.Series(reviews_len).describe(percentiles=[0.1, 0.5])
labels = np.asarray(labels)
reviews_train, reviews_test, labels_train, labels_test = train_test_split(reviews, labels, test_size=0.20, random_state=31)
# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 104 #the value is chosen considering that it is in the average of the number of words in the reviews

#Set reviews to a fixed size of 104
reviews_train = pad_sequences(reviews_train, padding='post', maxlen=maxlen)
reviews_test = pad_sequences(reviews_test, padding='post', maxlen=maxlen)
embeddings_dictionary = dict()
glove_file = open(r'/content/gdrive/My Drive/NLPg/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
  records = line.split()
  word = records[0]
  vector_dimensions = asarray(records[1:], dtype='float32')
  embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
 embedding_vector = embeddings_dictionary.get(word)
 if embedding_vector is not None:
  embedding_matrix[index] = embedding_vector
#initializing a sequential model
model = Sequential()

#Embedding Layer, using pre-trained embedding matrix
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

#creating an LSTM layer with 128 neurons
model.add(LSTM(128))

#Output Layer
model.add(Dense(1, activation='sigmoid'))

#Try the model
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(reviews_train, labels_train, batch_size=128, epochs=8, verbose=1, validation_split=0.2)
score = model.evaluate(reviews_test, labels_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
def prediction(text, model):
  #Cleaning and tokenizing of data
  text_clean = clean_text(text)
  instance = tokenizer.texts_to_sequences(text_clean)

  flat_list = []
  for sublist in instance:
      for item in sublist:
          flat_list.append(item)

  flat_list = [flat_list]

  instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

  value = model.predict(instance)[0][0]
  sentiment = ''

  if round(value) > 0.5:
    sentiment = 'positive'
  else:
    sentiment = 'negative'


  return [value, sentiment]
#The reviews used here were taken from https://www.imdb.com/title/tt4520988/reviews?ref_=tt_ql_3
#POSITIVE REVIEWS
#Reviews with 10 star ratings
text10a = "This. This was the sequel I was HOPING for. I was in tears of every variety for the majority of the film. The animation was flawless, the songs were abundant and wonderful (nothing quite as earwormy as Let It Go, though) and the laughter was non-stop. The character growth was also a treat. I didn't expect that, especially from a character like Olaf. All the negative reviews I see complain about it either being too dark or all over the place. It was neither. Sure, it had some darkness to it, but so does life, and like most moments in life, their is a shining bright light waiting on the other side, even if it's not the ending I would have expected. And to those saying it was all over the place? It wasn't, if you bothered to pay attention. But most of you are adults who write off every animated feature as children's fare and don't really care to pay attention. And that is because you've lost the child inside. You've forgotten how to feel joy and excitement and embrace it like you did as a child. You confuse your contentedness with your children's happiness with actual joy and live vicariously through them than embracing what's inside of you and living your own happiness yourself. And for that, I pity you all."
text10b = "This was the first Just want to say I disagree with you Gunsjoe- the plot was definitely not simplistic. In fact it was a bit complicated if you ask me. I thought the movie and the story overall was excellent, as was the music. It even includes an 80s style ballad from Kristoff. My young girls loved it as did my wife and I."
text10c = "We cried, we sang, we were engaged. The story is warm and rounded: the search for the truth from the past by our two beloved sisters. The songs are rampant, earthy and satisfying, and the colours and laughter enthral. Ignore the jaded critics, this is a wonderful film. A little more age in the audience was evident, but we have all grown up a few years since the explosive first film. Let's celebrate how wonderful a fairytale can be, and right the wrongs of the past generation!"

#Reviews with 9 star ratings
text9a = "The songs are great. Especially Kristoff's song, an 80's jam as Jonathan Groff would say it. Even the animation looks like an 80's music video which I find funny. Olaf's comedic skits are even funnier. For Into the Unknown, not as catchy as Let it Go but time will tell. The animation, the best. Elsa got even beautifuller. If you saw the water animation you would see *much* more of that. The story is bigger, questions are answered, but there are some points that look silly. The plot is predictable but personally I enjoyed it."
text9b = "I'm not sure why some people say it sucked. I went with my nephew and I thought the story was good and kept me intrigued throughout and had so many emotions, great visuals and was as good if not better than the first. I would def. Recommend this movie if you liked the 1st one! The characters from Frozen are awesome!"
text9c = "Love the songs and can defo imagine a stage production of this film. The story is more grown up and parts will go over the heads of younger ones. My little boy who is 5 got a bit restless but managed to stick with it. Really want to go back and watch again without the distraction of lots of fidgety children LOL."

#Reviews with 8 star ratings
text8a = "Disney Sequels are an interesting conundrum. Almost every single Disney Movie sequel was produced by a different film studio (DisneyToon Studios) who gave us some terrible direct to video Disney Sequels such as Return of Jafar, Little Mermaid 2, and Fox and the Hound II. As of now only 4 Disney films have been given sequels made by Walt Disney Animation (Rescuers Down Under, Winnie the Pooh, Ralph Breaks the Internet, and Frozen II) Each Sequel is without a doubt better than the direct to video films, but as of right now This is the best of that bunch. Frozen II has a script and story that is much better than the first and actually gives heart and depth to a lot of our characters. Kristen Bell and the Wickedly Talented Idina Menzel give astounding vocal performances, and Josh Gad even made me cry. Though only time will tell how the songs fair to the public, I believe Frozen II is a sequel that could've been much worse, (and probably should have been) but it goes above and beyond and I don't think anyone will be Letting Go anytime soon."
text8b = "Frozen 2 is a great sequel that explores mature themes but is brought down slightly by a few predictable narrative revelations. Kristen Bell and Idina Menzel both give incredible performances and Josh Gad is also really good. The animation is incredible and the film is visually stunning throughout. It's well paced and the music by Christophe Beck is fantastic."
text8c = "I enjoyed watching Frozen 2. I dad a really good time. No, it's not as good as the first part but it's a very good movie. There were very good comedy scenes, some thrilling moments and almost everything that was in Frozen. My fellow theatre viewers also enjoyed it as there were big laughs and some ooooh moments. The animation is really really good. Overall, Frozen 2 is a fun adventure and I think no one will dislike it as these characters are hard to hate."

#Reviews with 7 star ratings
text7a = "It's not hard to imagine how big this movie will be. I, as well as many others, am a huge fan of the first film and the short films that followed it. Frozen II has a different feel to it. While the first film was brimming with quirky moments, the follow up feels a little more mature in its plot and themes. This story heavily centers around the importance of change, even if that means letting go. Throughout the film, I found myself dazzled with the quality and beauty of the visuals that the animation team laid out. The story, for me, felt a little flat, however. I think the writing team missed the mark when trying to achieve the same quirkiness from the original. With the more dramatic path the writers chose to take, also came more epic musical numbers and effects. Sure there were still plenty of humorous moments, but I personally felt myself longing for more. I also find irony in the fact that this film centers around change and the overall feel of the film felt like a major shift from the first. I do think this film will still appeal to many others and I, myself, am still very excited to watch it a second time. Overall, I'd say this film found success in what it was trying to achieve and the message it was trying to send to its viewers, but couldn't quite live up to the whimsicality of the original."
text7b = "it has amazing and catchy songs and an overall improved story that proves that, at Walt Disney Animation, Some Things Never Change. However, the complicated execution of the story will bring audiences into the unknown, forcing fans to say that the next right thing to do is tell the story in a less complicated way."
text7c = "This movie was way different than the first which has both good and bad. The comedy in this movie was great, had my dying at some parts. The cinematography in this was magical, some very creative cinematography, for example, Kristoff's music video. The part where this movie struggled was the story. The main problem is the lack of rules set. This movie has a lot of mystical powers, but not enough laws and rules in this universe were established to make us have real fear, because we didn't understand the consequences. The climax was not very climactic. The physicalness was, but there was not enough internal conflicts or challenges to make the climax something special."

#Reviews with 6 star ratings
text6a = "Frozen II felt unsatisfying. There was lots of build up to a very short climax that felt way too easy and simple. I kept expecting another adventure which never came. The film feels short and unfinished. That said, the animation is absolutely gorgeous and I did find the film quite funny. If you go into this with low expectations I think you'll enjoy yourself."
text6b = "This is a beautiful movie, and Elsa's journey is a powerful one to watch. But outside of this it's narrative is completely unfocused and none of the character moments are given the time they deserve on screen. It seems that it's trying to be more adult with more lore and a darker tone, but also still gives Olaf an unnecessary about of screen time doing things that the little kids would laugh at. The pacing also feels quite rushed, even by kids movie standards. You could have cut 20 minutes out of this movie and it would have been just as good."
text6c = "Sterling animation and a solid soundtrack are the strongest points in this sequel's favor. For the most part, sequels don't measure up to the original and this is no exception. This film surprised me in how easily it resolved things. Unfortunately, it also put me to sleep at one or two points; that is unusual. What it lacked in storytelling, it made up for with tremendous visuals. Recommended mostly to viewers with young children."

#NEGATIVE REVIEWS
#Reviews with 4 star ratings
text4a = "From the visual effect. Like usual notthing can beat Disney. But from the story line. It has lost its magic. The first movie is far stronger to enchant you. The first movie will be in your head for years. And this movie will be in your head for two weeks. Or maybe you will forget it only in a week. Maybe it is only for me. But this movie keeps giving you music nonstop. Some are really good. Some are just not that good. Or you will think why that scene needs a singing scene."
text4b = "I wasn't a massive fan of the first film but I must admit it had everything a Disney film needs, catchy songs, great animation and characters which are somewhat relatable. Frozen 2 was disappointing. The villain in this was so stupid and at times I felt like wanting to fall asleep. The running joke of Kristof not finding the right time to propose to Anna is stupid and at first got a few chuckles but after it was just repetitive and annoying. The music was alright but not as good as the first one. The animation was good as always but of course you have to remember it is a Disney film. I wasn't a big fan of the story this time around but otherwise 6/10"
text4c = "I set my expectations too high for Frozen 2. The movie has Elsa and Anna and Olaf which was all that was needed for my little girl to love the movie and beg to see it in the theaters a second time but this just isn't a good movie. I like the songs but the plot is really lacking and so much feels forced. It was really disappointing since I have been looking forward to this movie for at 2 years now and I really wanted to like the movie."

#Reviews with 3 star ratings
text3a = "The worst script writing in an animation I have ever seen, with awful lines such as let me see what I can see"
text3b = "Too bad It took them this long to write a new movie and came up with an ok story, but had no clue on how to execute it. The songs are somewhat ok but they more or less pause the story during the songs and made them theatrical and dull, so if you took all the songs out of the movie you would end up with a 40 min story. That was so disappointing compared to the first. I Found myself only laughing in scenes featuring Olaf. Why was that? Because nothing else in this movie is fun. There's no fun like in the first. There's no action like in the first. So I was very disappointed with this one. Too bad since I think they could have done a lot more with the story."
text3c = "I seriously do not understand the ratings this disappointment is getting. No outstanding songs and a convoluted plot no child, (Not to mention a lot of adults) could follow. Sure Olof is cute but he's pretty much the highlight of this movie. My wife's a kindergarten/ preschool teacher and she was disappointed too. Don't waste your money."

#Reviews with 2 star ratings
text2a = "Everything is nonsense and completely disaster The first movie I can give a 7-8. But this new one is so not like that. The music is so bad, so bad compared to let it go. The plot is very plain, and the special effects are not good. Ughh I want my time back so I can write report instead of going to this annoying film. I hate everything. I can't be in the movie cause it is too bad. And the theatre no one clapped or laughed at all. I felt some people leaving in the middle. The songs are rubbish, the lyrics are empty and the worst."
text2b = "What do you know. Disney decides to unnecessarily preach to us about reparations, while dangling it's liberal propaganda in our faces and then distracting us with witty humor. Leave your agendas out of the movies. It is getting so ridiculous."
text2c = "Had to watch it because of my 4 year old. Was soo boring. Songs, something which we all look forward to in a disney offering was a let down and was below average. Maybe they should take a lesson from the musical team of Moana, which was more than awesome. No doubt there is magical graphics but not enough to keep me glued to the seat. I was just waiting for the movie to end. Easily the most disappointing movie i watched this year."

#Reviews with 1 star ratings
text1a = "I loved the first one. Saw it three times in theaters. Immediately bought the soundtrack afterward. The first one had me on my toes the whole time. The story progressed smoothly. The songs had purpose. The theme of let it go was relatable. This sequel had none of that. They stuff a contrived story down your throat - having to explain everything at each step because nothing made sense. The songs are totally forgettable and filler. There is no meaningful conflict and the climax is non-existent."
text1b = "One star for animation only. The story is so dull and the songs don't serve any purpose to move the story. Anna and Olaf are even annoying this time whilst Elsa is still the same melancholic and timid girl she was in the first. Overall, not worth my money and time! Still, one man's trash is another person's treasure; so, the kids probably love them."
text1c = "Watched with me and my family and I can say movie was sooo crappy with songs every 10 mins. I believe there were more then 10 songs and most of it were irritating and boring. The plot was soo thin with no purpose. It is a very bad sequal in my honest brutal opinion. Do watch it if you want to sleep."

texts_positives = [
         text10a, text10b, text10c,
         text9a, text9b, text9c,
         text8a, text8b, text8c,
         text7a, text7b, text7c,
         text6a, text6b, text6c,
]

texts_negatives = [
         text4a, text4b, text4c,
         text3a, text3b, text3c,
         text2a, text2b, text2c,
         text1a, text1b, text1c, 
]

print("Predicting Reviews Positives\n")
for i in texts_positives:
  print(prediction(i, model))

print("\n\nPredicting Reviews Negatives\n")
for i in texts_negatives:
  print(prediction(i, model))