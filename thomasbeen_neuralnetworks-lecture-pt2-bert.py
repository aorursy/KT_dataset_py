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
# install tesorflow bert package
!pip install bert-for-tf2

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert

#Loding pretrained bert layer
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)


# Loading tokenizer from the bert layer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocab_file, do_lower_case)

print("done!")
text = 'Encoding will be clear with this example'
# tokenize
tokens_list = tokenizer.tokenize(text)
print('Text after tokenization')
print(tokens_list)

# initilize dimension
max_len =34
text = tokens_list[:max_len-2]
input_sequence = ["[CLS]"] + text + ["[SEP]"]
print("After adding  flasges -[CLS] and [SEP]: ")
print(input_sequence)


tokens = tokenizer.convert_tokens_to_ids(input_sequence )
print("tokens to id ")
print(tokens)

pad_len = max_len -len(input_sequence)
tokens += [0] * pad_len
print("tokens: ")
print(tokens)

print(pad_len)
pad_masks = [1] * len(input_sequence) + [0] * pad_len
print("Pad Masking: ")
print(pad_masks)

segment_ids = [0] * max_len
print("Segment Ids: ")
print(segment_ids)
import numpy as np
def bert_encode(texts, tokenizer, max_len=520):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

MAX_LEN = 20

# encode train set 
train_input = bert_encode([
    'love is more powerful than hate',
    'hope is more powerful than fear',
    'light is more powerful than darkness',
    'this is our moment',
    'this is our mission',
    'we cant take four more years of donald trump', 
    'i have always believed you can define america in one word possibilities', 
    'together, we can and will rebuild our economy', 
    'donald trump has failed to protect the american people and that is unforgivable',
    'donald trump continues to cozy up to russia',
    'i am grateful for your support and honored to be the nominee',
    'character is on the ballot',
    'compassion is on the ballot',
    'decency science democracy',
    'this campaign is about winning the heart and the soul of america', 
    'our president has failed in his most basic duty to this nation',
    'you wont hear me racebaiting',
    'you wont hear me dividing', 
    'i will ensure the rights of lgbtq people', 
    'we have to heal this nation', 
    'i am running as a proud democrat',
    'i am going to be an american president',
    'we are a diverse country',
    'improving the criminal justice system',
    'putting black americans in a position to gain generational wealth',
    'this moment demands moral leadership',
    'we can choose a different path one of hope unity and light', 
    'lets overcome this era of darkness together',
    'to lead america you have to understand america', 
    'the cruelty of this president truly knows no bounds', 
    'immigrants are not your political props',
    'president trump has failed our nation', 
    'leadership requires foresight',
    'its time we bring integrity back to the white house',
    'we have to beat donald trump',
    'times of crisis often bring out our best as americans',
    'donald trump put our nation on the sidelines',
    'trump still does not have a plan to get this virus under control',
    'we can choose four more years of fear division and hate', 
    'you will not pay a penny more in taxes under my administration',
    'the importance of wearing masks and keeping a safe social distance', 
    'president obama and i left donald trump a booming economy',
    'donald trump can lie about the economy all he wants',
    'we must stand up to hate and intolerance', 
    'its time we ensured all people are treated equally as well',
    'the only thing this president has done alone is fail america',
    'no lies fear-mongering or malarkey',
    'we run this campaign led by the values that guide my life',
    'we must not become a country at war with ourselves',
    'we are facing so many crises under donald trump',
    'the simple truth is donald trump failed to protect america',
    'donald trump looks at this violence and sees a political lifeline',
    'he failed to protect america so now hes trying to scare america',
    'fear never builds the future but hope does',
    'together we are going to build back better than before',
    'donald trump is determined to instill fear and divide us',
    'president trump has refused to stand up to russia',
    'this is our moment to root out systemic racism',
    'be a patriot and wear a mask',
    'it is time we reward work and not just wealth',
    'we need to restore honor and decency to the white house',
    'unlike donald trump i will listen to the experts',
    'donald trump is incapable of providing the leadership this moment requires',
    'we have got to flip the senate folks',
    'nature knows',
    'if president trump has his way in the us supreme court its unconscionable',
    'There is only one way to end this horror vote',
    'i made a promise to his family that i will not let him become just another hashtag',
    'donald trump wants to destroy obamacare',
    'vote like your health care depends on it', 
    'i will be a president for all americans',
    'the right to vote is the most sacred american right there is',
    'i will always tell you the truth',
    'i will listen to the experts',
    'you deserve a president who tells you the truth', 
    'how many more people have to suffer because of president trumps lies', 
    'do your part and wear a mask', 
    'i know we can beat trump and build this nation back better', 
    'weapons of war have no place in our communities',
    'science knows',
    'i believe that every american deserves a fair shot to get ahead',
    'science will win',
    'donald trump wants to give his rich friends another tax cut',
    'we need a president who cares about more than the wealthy',
    'you lost your freedom because president trump did not act',
    'unlike president trump i will be a president for all americans',
    'honor and decency are on the ballot this november', 
    'i will actually listen to advice and expertise not attack him for telling the truth', 
    'we all know president trump has a tendency to stray from the truth', 
    'donald trump the greatest failure of presidential leadership in our nations history', 
    'donald trump has been trying to throw out obamacare for years', 
    'we are going to make donald trump a one-term president',
    'i am ready to fight for you and for our nation', 
    'we need a president who will unite our country', 
    'we honor the hardworking men and women who drive our economy', 
    'the longer donald trump is president the more reckless he gets', 
    'i know americans are not looking for a handout',
    'let me be clear lgbtq rights are human rights',
    'hope over fear',
    'truth over lies',
    'there is a nasty rumor out there',
    'massive red wave coming',
    'biden lied',
    'polls numbers are looking very strong',
    'big crowds great enthusiasm',
    'thank you libertarians',
    'corrupt politician',
    'he will never let you down',
    'vote trump',
    'sleepy joe biden had a very bad showing last night',
    'the radical left',
    'negative biden news',
    'the debate was rigged',
    'the trump campaign was not treated fairly',
    'the radical left will destroy our country',
    'joe biden and the democrat socialists will kill your jobs',
    'the debate was rigged',
    'joe biden and the democrat socialists will  dismantle your police departments'
    'there is a nasty rumor out there',
    'massive red wave coming',
    'biden lied',
    'polls numbers are looking very strong',
    'big crowds great enthusiasm',
    'thank you libertarians',
    'corrupt politician',
    'he will never let you down',
    'vote trump',
    'sleepy joe biden had a very bad showing last night',
    'the radical left',
    'negative biden news',
    'the debate was rigged',
    'the trump campaign was not treated fairly',
    'the radical left will destroy our country',
    'joe biden and the democrat socialists will kill your jobs',
    'the debate was rigged',
    'joe biden and the democrat socialists will  dismantle your police departments',
    'if biden wins china wins',
    'when we win you win north carolina wins and america wins',
    'if biden wins china will own the united states',
    'nobody has ever done as much for iowa as i have done for iowa',
    'think of where we would be now without fake and fraudulent stories',
    'He has always been a corrupt politician',
    'joe biden must immediately release all emails',
    'there is nothing worse than a corrupt politician',
    'this is your chance to make america great again',
    'fight hard republicans',
    'they have been taking advantage of the system for years',
    'one of the most important issues for pennsylvania is the survival of your fracking industry', 
    'joe biden has repeatedly pledged to abolish fracking',
    'with me you are going to frack',
    'see you in court',
    'fight hard republicans',
    'fake news is devastated',
    'protecting people with preexisting conditions', 
    'he has done nothing on healthcare cost or otherwise or virtually anything else',
    'guy is a total loser',
    'early voting is underway in the great states of georgia and texas',
    'find out where to early vote by clicking below',
    'volunteer to be a trump election poll watcher',
    'rigged election',
    'the radical left is trying hard to undermine the christopher columbus legacy',
    'california hired a pure sleepy joe democrat firm to count and harvest votes',
    'no way republicans get a fair shake',
    'vote trump and watch the greatest comeback of them all',
    'biden disrespects voters with courtpacking dodge',
    'people want the truth',
    'vote trump',
    'save your 2nd amendment',
    'vote for trump what the hell do you have to lose',
    'nancy pelosi could not care less about the american people',
    'taxes too high',
    'crime too high',
    'lockdowns too severe',
    'i have gone through years of a fake illegal witchhunt',
    'it was a hoax',
    'massive corruption surrounding sleepy joe biden',
    'protect people with preexisting conditions',
    'make america great again',
    'fight hard republicans',
    'we are taking back our country',
    'totally negative china virus reports',
    'joe has never been a nice or kind guy',
    'fake news',
    'joe biden spoke fondly of this racist today',
    'almost nobody showed up to the sleepy joe biden rally',
    'the reporting and polls are a media con job fake news',
    'we have far more support and enthusiasm than even in 2016',
    'biden is coughing and hacking and playing fingers with his mask all over the place',
    'journalism has reached the all time low in history',
    'joe has never been a nice or kind guy',
    'there joe goes again',
    'we are winning',
    'economy is starting to boom',
    'i keep reading fake news stories that my campaign is running low on money',
    'not true and if it were so i would put up money myself',
    'the fact is that we have much more money than we had four years ago',
    'china virus',
    'i keep reading fake news stories',
    'crooked hillary',
    'leftwing radicals',
    'radical left justices',
    
    


    
], tokenizer, max_len=MAX_LEN)
train_labels = np.array([
    [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
    [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
    [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
    [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
    [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
    [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
    [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
    [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
    [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
    [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
   
])

print("number of test samples:", len(train_input[0]), "labels:", len(train_labels))

# first define input for token, mask and segment id  
from tensorflow.keras.layers import  Input
input_word_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
input_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_mask")
segment_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="segment_ids")

#  output  
from tensorflow.keras.layers import Dense
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])  
clf_output = sequence_output[:, 0, :]
out = Dense(2, activation='softmax')(clf_output)

# intilize model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
model.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# train
train_history = model.fit(
    train_input, train_labels,
    validation_split=0.05,
    epochs=4,
    batch_size=1
)

model.save('model.h5')
print("done and saved!")
test_input = bert_encode(['donald trump is a bigot', 'this is fake news'], tokenizer, max_len= MAX_LEN )
test_pred = model.predict(test_input)
preds = np.around(test_pred, 3)
preds