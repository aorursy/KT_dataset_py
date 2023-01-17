# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# df_train=pd.read_csv('../input/train.csv').sample(100000,random_state=0)

df_train=pd.read_csv('../input/train.csv', nrows=100000)

df_train['label']=np.where(df_train.target>=0.5,1,0)

df_train['label']=df_train['label'].astype('int8')
# df_train=pd.read_csv('../input/data-augmentation/train.csv')

df_test=pd.read_csv('../input/test.csv')
print(df_test.shape,df_train.shape)
contraction_mapping = {

    "Trump's" : 'trump is',"'cause": 'because',',cause': 'because',';cause': 'because',"ain't": 'am not','ain,t': 'am not',

    'ain;t': 'am not','ain´t': 'am not','ain’t': 'am not',"aren't": 'are not',

    'aren,t': 'are not','aren;t': 'are not','aren´t': 'are not','aren’t': 'are not',"can't": 'cannot',"can't've": 'cannot have','can,t': 'cannot','can,t,ve': 'cannot have',

    'can;t': 'cannot','can;t;ve': 'cannot have',

    'can´t': 'cannot','can´t´ve': 'cannot have','can’t': 'cannot','can’t’ve': 'cannot have',

    "could've": 'could have','could,ve': 'could have','could;ve': 'could have',"couldn't": 'could not',"couldn't've": 'could not have','couldn,t': 'could not','couldn,t,ve': 'could not have','couldn;t': 'could not',

    'couldn;t;ve': 'could not have','couldn´t': 'could not',

    'couldn´t´ve': 'could not have','couldn’t': 'could not','couldn’t’ve': 'could not have','could´ve': 'could have',

    'could’ve': 'could have',"didn't": 'did not','didn,t': 'did not','didn;t': 'did not','didn´t': 'did not',

    'didn’t': 'did not',"doesn't": 'does not','doesn,t': 'does not','doesn;t': 'does not','doesn´t': 'does not','doesn’t': 'does not',"don't": 'do not','don,t': 'do not','don;t': 'do not','don´t': 'do not','don’t': 'do not',

    "hadn't": 'had not',"hadn't've": 'had not have','hadn,t': 'had not','hadn,t,ve': 'had not have','hadn;t': 'had not',

    'hadn;t;ve': 'had not have','hadn´t': 'had not','hadn´t´ve': 'had not have','hadn’t': 'had not','hadn’t’ve': 'had not have',"hasn't": 'has not','hasn,t': 'has not','hasn;t': 'has not','hasn´t': 'has not','hasn’t': 'has not',

    "haven't": 'have not','haven,t': 'have not','haven;t': 'have not','haven´t': 'have not','haven’t': 'have not',"he'd": 'he would',

    "he'd've": 'he would have',"he'll": 'he will',

    "he's": 'he is','he,d': 'he would','he,d,ve': 'he would have','he,ll': 'he will','he,s': 'he is','he;d': 'he would',

    'he;d;ve': 'he would have','he;ll': 'he will','he;s': 'he is','he´d': 'he would','he´d´ve': 'he would have','he´ll': 'he will',

    'he´s': 'he is','he’d': 'he would','he’d’ve': 'he would have','he’ll': 'he will','he’s': 'he is',"how'd": 'how did',"how'll": 'how will',

    "how's": 'how is','how,d': 'how did','how,ll': 'how will','how,s': 'how is','how;d': 'how did','how;ll': 'how will',

    'how;s': 'how is','how´d': 'how did','how´ll': 'how will','how´s': 'how is','how’d': 'how did','how’ll': 'how will',

    'how’s': 'how is',"i'd": 'i would',"i'll": 'i will',"i'm": 'i am',"i've": 'i have','i,d': 'i would','i,ll': 'i will',

    'i,m': 'i am','i,ve': 'i have','i;d': 'i would','i;ll': 'i will','i;m': 'i am','i;ve': 'i have',"isn't": 'is not','isn,t': 'is not','isn;t': 'is not','isn´t': 'is not','isn’t': 'is not',"it'd": 'it would',"it'll": 'it will',"It's":'it is',

    "it's": 'it is','it,d': 'it would','it,ll': 'it will','it,s': 'it is','it;d': 'it would','it;ll': 'it will','it;s': 'it is','it´d': 'it would','it´ll': 'it will','it´s': 'it is',

    'it’d': 'it would','it’ll': 'it will','it’s': 'it is',

    'i´d': 'i would','i´ll': 'i will','i´m': 'i am','i´ve': 'i have','i’d': 'i would','i’ll': 'i will','i’m': 'i am',

    'i’ve': 'i have',"let's": 'let us','let,s': 'let us','let;s': 'let us','let´s': 'let us',

    'let’s': 'let us',"ma'am": 'madam','ma,am': 'madam','ma;am': 'madam',"mayn't": 'may not','mayn,t': 'may not','mayn;t': 'may not',

    'mayn´t': 'may not','mayn’t': 'may not','ma´am': 'madam','ma’am': 'madam',"might've": 'might have','might,ve': 'might have','might;ve': 'might have',"mightn't": 'might not','mightn,t': 'might not','mightn;t': 'might not','mightn´t': 'might not',

    'mightn’t': 'might not','might´ve': 'might have','might’ve': 'might have',"must've": 'must have','must,ve': 'must have','must;ve': 'must have',

    "mustn't": 'must not','mustn,t': 'must not','mustn;t': 'must not','mustn´t': 'must not','mustn’t': 'must not','must´ve': 'must have',

    'must’ve': 'must have',"needn't": 'need not','needn,t': 'need not','needn;t': 'need not','needn´t': 'need not','needn’t': 'need not',"oughtn't": 'ought not','oughtn,t': 'ought not','oughtn;t': 'ought not','oughtn´t': 'ought not','oughtn’t': 'ought not',"sha'n't": 'shall not','sha,n,t': 'shall not','sha;n;t': 'shall not',"shan't": 'shall not',

    'shan,t': 'shall not','shan;t': 'shall not','shan´t': 'shall not','shan’t': 'shall not','sha´n´t': 'shall not','sha’n’t': 'shall not',

    "she'd": 'she would',"she'll": 'she will',"she's": 'she is','she,d': 'she would','she,ll': 'she will',

    'she,s': 'she is','she;d': 'she would','she;ll': 'she will','she;s': 'she is','she´d': 'she would','she´ll': 'she will',

    'she´s': 'she is','she’d': 'she would','she’ll': 'she will','she’s': 'she is',"should've": 'should have','should,ve': 'should have','should;ve': 'should have',

    "shouldn't": 'should not','shouldn,t': 'should not','shouldn;t': 'should not','shouldn´t': 'should not','shouldn’t': 'should not','should´ve': 'should have',

    'should’ve': 'should have',"that'd": 'that would',"that's": 'that is','that,d': 'that would','that,s': 'that is','that;d': 'that would',

    'that;s': 'that is','that´d': 'that would','that´s': 'that is','that’d': 'that would','that’s': 'that is',"there'd": 'there had',

    "there's": 'there is','there,d': 'there had','there,s': 'there is','there;d': 'there had','there;s': 'there is',

    'there´d': 'there had','there´s': 'there is','there’d': 'there had','there’s': 'there is',"they'd": 'they would',"they'll": 'they will',"they're": 'they are',"they've": 'they have',

    'they,d': 'they would','they,ll': 'they will','they,re': 'they are','they,ve': 'they have','they;d': 'they would','they;ll': 'they will','they;re': 'they are',

    'they;ve': 'they have','they´d': 'they would','they´ll': 'they will','they´re': 'they are','they´ve': 'they have','they’d': 'they would','they’ll': 'they will',

    'they’re': 'they are','they’ve': 'they have',"wasn't": 'was not','wasn,t': 'was not','wasn;t': 'was not','wasn´t': 'was not',

    'wasn’t': 'was not',"we'd": 'we would',"we'll": 'we will',"we're": 'we are',"we've": 'we have','we,d': 'we would','we,ll': 'we will',

    'we,re': 'we are','we,ve': 'we have','we;d': 'we would','we;ll': 'we will','we;re': 'we are','we;ve': 'we have',

    "weren't": 'were not','weren,t': 'were not','weren;t': 'were not','weren´t': 'were not','weren’t': 'were not','we´d': 'we would','we´ll': 'we will',

    'we´re': 'we are','we´ve': 'we have','we’d': 'we would','we’ll': 'we will','we’re': 'we are','we’ve': 'we have',"what'll": 'what will',"what're": 'what are',"what's": 'what is',

    "what've": 'what have','what,ll': 'what will','what,re': 'what are','what,s': 'what is','what,ve': 'what have','what;ll': 'what will','what;re': 'what are',

    'what;s': 'what is','what;ve': 'what have','what´ll': 'what will',

    'what´re': 'what are','what´s': 'what is','what´ve': 'what have','what’ll': 'what will','what’re': 'what are','what’s': 'what is','what’ve': 'what have',"where'd": 'where did',"where's": 'where is','where,d': 'where did','where,s': 'where is','where;d': 'where did',

    'where;s': 'where is','where´d': 'where did','where´s': 'where is','where’d': 'where did','where’s': 'where is',

    "who'll": 'who will',"who's": 'who is','who,ll': 'who will','who,s': 'who is','who;ll': 'who will','who;s': 'who is',

    'who´ll': 'who will','who´s': 'who is','who’ll': 'who will','who’s': 'who is',"won't": 'will not','won,t': 'will not','won;t': 'will not',

    'won´t': 'will not','won’t': 'will not',"wouldn't": 'would not','wouldn,t': 'would not','wouldn;t': 'would not','wouldn´t': 'would not',

    'wouldn’t': 'would not',"you'd": 'you would',"you'll": 'you will',"you're": 'you are','you,d': 'you would','you,ll': 'you will',

    'you,re': 'you are','you;d': 'you would','you;ll': 'you will',

    'you;re': 'you are','you´d': 'you would','you´ll': 'you will','you´re': 'you are','you’d': 'you would','you’ll': 'you will','you’re': 'you are',

    '´cause': 'because','’cause': 'because',"you've": "you have","could'nt": 'could not',

    "havn't": 'have not',"here’s": "here is",'i""m': 'i am',"i'am": 'i am',"i'l": "i will","i'v": 'i have',"wan't": 'want',"was'nt": "was not","who'd": "who would",

    "who're": "who are","who've": "who have","why'd": "why would","would've": "would have","y'all": "you all","y'know": "you know","you.i": "you i","your'e": "you are","arn't": "are not","agains't": "against","c'mon": "common","doens't": "does not",'don""t': "do not","dosen't": "does not",

    "dosn't": "does not","shoudn't": "should not","that'll": "that will","there'll": "there will","there're": "there are",

    "this'll": "this all","u're": "you are", "ya'll": "you all","you'r": "you are","you’ve": "you have","d'int": "did not","did'nt": "did not","din't": "did not","dont't": "do not","gov't": "government",

    "i'ma": "i am","is'nt": "is not","‘I":'I',

    'ᴀɴᴅ':'and','ᴛʜᴇ':'the','ʜᴏᴍᴇ':'home','ᴜᴘ':'up','ʙʏ':'by','ᴀᴛ':'at','…and':'and','civilbeat':'civil beat',\

    'TrumpCare':'Trump care','Trumpcare':'Trump care', 'OBAMAcare':'Obama care','ᴄʜᴇᴄᴋ':'check','ғᴏʀ':'for','ᴛʜɪs':'this','ᴄᴏᴍᴘᴜᴛᴇʀ':'computer',\

    'ᴍᴏɴᴛʜ':'month','ᴡᴏʀᴋɪɴɢ':'working','ᴊᴏʙ':'job','ғʀᴏᴍ':'from','Sᴛᴀʀᴛ':'start','gubmit':'submit','CO₂':'carbon dioxide','ғɪʀsᴛ':'first',\

    'ᴇɴᴅ':'end','ᴄᴀɴ':'can','ʜᴀᴠᴇ':'have','ᴛᴏ':'to','ʟɪɴᴋ':'link','ᴏғ':'of','ʜᴏᴜʀʟʏ':'hourly','ᴡᴇᴇᴋ':'week','ᴇɴᴅ':'end','ᴇxᴛʀᴀ':'extra',\

    'Gʀᴇᴀᴛ':'great','sᴛᴜᴅᴇɴᴛs':'student','sᴛᴀʏ':'stay','ᴍᴏᴍs':'mother','ᴏʀ':'or','ᴀɴʏᴏɴᴇ':'anyone','ɴᴇᴇᴅɪɴɢ':'needing','ᴀɴ':'an','ɪɴᴄᴏᴍᴇ':'income',\

    'ʀᴇʟɪᴀʙʟᴇ':'reliable','ғɪʀsᴛ':'first','ʏᴏᴜʀ':'your','sɪɢɴɪɴɢ':'signing','ʙᴏᴛᴛᴏᴍ':'bottom','ғᴏʟʟᴏᴡɪɴɢ':'following','Mᴀᴋᴇ':'make',\

    'ᴄᴏɴɴᴇᴄᴛɪᴏɴ':'connection','ɪɴᴛᴇʀɴᴇᴛ':'internet','financialpost':'financial post', 'ʜaᴠᴇ':' have ', 'ᴄaɴ':' can ', 'Maᴋᴇ':' make ', 'ʀᴇʟɪaʙʟᴇ':' reliable ', 'ɴᴇᴇᴅ':' need ','ᴏɴʟʏ':' only ', 'ᴇxᴛʀa':' extra ', 'aɴ':' an ', 'aɴʏᴏɴᴇ':' anyone ', 'sᴛaʏ':' stay ', 'Sᴛaʀᴛ':' start', 'SHOPO':'shop',

    }



contraction_mapping=dict((k.lower(), v.lower()) for k,v in contraction_mapping.items())



# df_train['treated_comment']=df_train['comment_text']

def lowercase(sen):

    y=" ".join(x.lower() for x in sen.split())

    return y





def clean_contractions(text, mapping):   # 总觉得哪里不对....有待细化

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text





# df_train['treated_comment'].head()



import re

def clean_text(x):

    pattern = r'[^a-zA-z0-9\s]'

    text = re.sub(pattern, '', x) # re.sub is time-comsuming

    return text





from nltk.corpus import stopwords

stop=stopwords.words('english')

def stop_remove(sen):

    y=" ".join(x for x in sen.split() if x not in stop)

    return y



def number_remove(x):

    y=re.sub('\d','',x)

    return y



               

def preprocess(x):

    x=lowercase(x)

    x=clean_contractions(x,contraction_mapping)

    x=clean_text(x)

    x=stop_remove(x)

    x=number_remove(x)

    return x
%%time

df_train['comment_text']=df_train['comment_text'].apply(preprocess)



# df_train['label'] = np.where(df_train['target'] >= .5, 1, 0)

# df_train['label']=df_train['label'].astype('int8')



df_test['comment_text']=df_test['comment_text'].apply(preprocess)
# df_train=df_train.sample(100000,random_state=1)
import random

from random import shuffle

random.seed(1)



#stop words list

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 

			'ours', 'ourselves', 'you', 'your', 'yours', 

			'yourself', 'yourselves', 'he', 'him', 'his', 

			'himself', 'she', 'her', 'hers', 'herself', 

			'it', 'its', 'itself', 'they', 'them', 'their', 

			'theirs', 'themselves', 'what', 'which', 'who', 

			'whom', 'this', 'that', 'these', 'those', 'am', 

			'is', 'are', 'was', 'were', 'be', 'been', 'being', 

			'have', 'has', 'had', 'having', 'do', 'does', 'did',

			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',

			'because', 'as', 'until', 'while', 'of', 'at', 

			'by', 'for', 'with', 'about', 'against', 'between',

			'into', 'through', 'during', 'before', 'after', 

			'above', 'below', 'to', 'from', 'up', 'down', 'in',

			'out', 'on', 'off', 'over', 'under', 'again', 

			'further', 'then', 'once', 'here', 'there', 'when', 

			'where', 'why', 'how', 'all', 'any', 'both', 'each', 

			'few', 'more', 'most', 'other', 'some', 'such', 'no', 

			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 

			'very', 's', 't', 'can', 'will', 'just', 'don', 

			'should', 'now', '']



#cleaning up text

import re

def get_only_chars(line):



    clean_line = ""



    line = line.replace("’", "")

    line = line.replace("'", "")

    line = line.replace("-", " ") #replace hyphens with spaces

    line = line.replace("\t", " ")

    line = line.replace("\n", " ")

    line = line.lower()



    for char in line:

        if char in 'qwertyuiopasdfghjklzxcvbnm ':

            clean_line += char

        else:

            clean_line += ' '



    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces

    if clean_line[0] == ' ':

        clean_line = clean_line[1:]

    return clean_line



########################################################################

# Synonym replacement

# Replace n words in the sentence with synonyms from wordnet

########################################################################



#for the first time you use wordnet

#import nltk

#nltk.download('wordnet')

from nltk.corpus import wordnet 



def synonym_replacement(words, n):

	new_words = words.copy()

	random_word_list = list(set([word for word in words if word not in stop_words]))

	random.shuffle(random_word_list)

	num_replaced = 0

	for random_word in random_word_list:

		synonyms = get_synonyms(random_word)

		if len(synonyms) >= 1:

			synonym = random.choice(list(synonyms))

			new_words = [synonym if word == random_word else word for word in new_words]

			#print("replaced", random_word, "with", synonym)

			num_replaced += 1

		if num_replaced >= n: #only replace up to n words

			break



	#this is stupid but we need it, trust me

	sentence = ' '.join(new_words)

	new_words = sentence.split(' ')



	return new_words



def get_synonyms(word):

	synonyms = set()

	for syn in wordnet.synsets(word): 

		for l in syn.lemmas(): 

			synonym = l.name().replace("_", " ").replace("-", " ").lower()

			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])

			synonyms.add(synonym) 

	if word in synonyms:

		synonyms.remove(word)

	return list(synonyms)



########################################################################

# Random deletion

# Randomly delete words from the sentence with probability p

########################################################################



def random_deletion(words, p):



	#obviously, if there's only one word, don't delete it

	if len(words) == 1:

		return words



	#randomly delete words with probability p

	new_words = []

	for word in words:

		r = random.uniform(0, 1)

		if r > p:

			new_words.append(word)



	#if you end up deleting all words, just return a random word

	if len(new_words) == 0:

		rand_int = random.randint(0, len(words)-1)

		return [words[rand_int]]



	return new_words



########################################################################

# Random swap

# Randomly swap two words in the sentence n times

########################################################################



def random_swap(words, n):

	new_words = words.copy()

	for _ in range(n):

		new_words = swap_word(new_words)

	return new_words



def swap_word(new_words):

	random_idx_1 = random.randint(0, len(new_words)-1)

	random_idx_2 = random_idx_1

	counter = 0

	while random_idx_2 == random_idx_1:

		random_idx_2 = random.randint(0, len(new_words)-1)

		counter += 1

		if counter > 3:

			return new_words

	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 

	return new_words



########################################################################

# Random insertion

# Randomly insert n words into the sentence

########################################################################



def random_insertion(words, n):

	new_words = words.copy()

	for _ in range(n):

		add_word(new_words)

	return new_words



def add_word(new_words):

	synonyms = []

	counter = 0

	while len(synonyms) < 1:

		random_word = new_words[random.randint(0, len(new_words)-1)]

		synonyms = get_synonyms(random_word)

		counter += 1

		if counter >= 10:

			return

	random_synonym = synonyms[0]

	random_idx = random.randint(0, len(new_words)-1)

	new_words.insert(random_idx, random_synonym)



########################################################################

# main data augmentation function

########################################################################



def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):

	

	sentence = get_only_chars(sentence)

	words = sentence.split(' ')

	words = [word for word in words if word is not '']

	num_words = len(words)

	

	augmented_sentences = []

	num_new_per_technique = int(num_aug/4)+1

	n_sr = max(1, int(alpha_sr*num_words))

	n_ri = max(1, int(alpha_ri*num_words))

	n_rs = max(1, int(alpha_rs*num_words))



	#sr

	for _ in range(num_new_per_technique):

		a_words = synonym_replacement(words, n_sr)

		augmented_sentences.append(' '.join(a_words))



	#ri

	for _ in range(num_new_per_technique):

		a_words = random_insertion(words, n_ri)

		augmented_sentences.append(' '.join(a_words))



	#rs

	for _ in range(num_new_per_technique):

		a_words = random_swap(words, n_rs)

		augmented_sentences.append(' '.join(a_words))



	#rd

	for _ in range(num_new_per_technique):

		a_words = random_deletion(words, p_rd)

		augmented_sentences.append(' '.join(a_words))



	augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]

	shuffle(augmented_sentences)



	#trim so that we have the desired number of augmented sentences

	if num_aug >= 1:

		augmented_sentences = augmented_sentences[:num_aug]

	else:

		keep_prob = num_aug / len(augmented_sentences)

		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]



	#append the original sentence

# 	augmented_sentences.append(sentence)



	return augmented_sentences
df_train.comment_text.iloc[4]
eda(df_train.comment_text.iloc[4], alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=2)
def aug_toxic(trainDF,sr=0.3,ri=0.1,rs=0.1,rd=0.1,num_aug=4):

    aug=[]

    for i in range(trainDF.shape[0]):

        aug_textlist=eda(trainDF.comment_text.iloc[i], 

                         alpha_sr=sr,

                         alpha_ri=ri, 

                         alpha_rs=rs, 

                         p_rd=rd, 

                         num_aug=num_aug)

        aug.extend(aug_textlist)

#     return aug

    df_aug=pd.DataFrame({'comment_text':aug,'label':np.ones((len(aug),),dtype='int8')})



    return df_aug

%%time 

df_aug=aug_toxic(df_train[df_train.label==1]) # Toxic augment dataframe 



# df_augment=df_aug.append(df_train[['comment_text','label']][df_train.label==0]) # concat the augment and the non-toxic



# df_train=df_augment

df_augment=df_train[['comment_text','label']].append(df_aug)
%%time

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm



# split the dataset into training and validation datasets 

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df_augment['comment_text'], df_augment['label']

                                                                      ,test_size=0.25,random_state=1)



# label encode the target variable 

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

valid_y = encoder.fit_transform(valid_y)
valid_x.shape[0]
%%time

# word level tf-idf 

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer1=TfidfVectorizer(ngram_range=(1,2), 

#                             tokenizer=tokenize,

                      min_df=3, strip_accents='unicode', use_idf=1,

                      smooth_idf=1, sublinear_tf=1

                        ,max_features=valid_x.shape[0]

                           )

# vectorizer1 = TfidfVectorizer()

train_vectors = vectorizer1.fit_transform(train_x)

valid_vectors = vectorizer1.transform(valid_x)
print('The shape of train and test is :',

      train_vectors.shape, valid_vectors.shape)
# from imblearn.over_sampling import SMOTE

# sm = SMOTE(random_state=2)

# X_train_res, y_train_res = sm.fit_sample(train_vectors, train_y.ravel()) # 对训练集的tfidf用smote过采样

# print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

# print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



# print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))

# print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
%%time

from sklearn.linear_model import LogisticRegression

# the basic naive-bayes feature equation

def NBSVM_predict(x,TV_x):

    '''

    x: train tf-idf ,default:train_vectors

    TV_x: tf-idf of the  you want :valid vectors or  test vectors

    '''

    

    def pr(y_i, y):

        p = x[y == y_i].sum(0)

        

        return (p+1) / ((y == y_i).sum()+1)





    # fit a model for one independent at a time

    def get_mdl(y):

    #     y = y.values

        r = np.log(pr(1, y) / pr(0, y))

        m = LogisticRegression(C=4, dual=True)

        x_nb = x.multiply(r)

        return m.fit(x_nb, y), r



    labels = ["label"]

    # the variable to store predictions

    pred_valid = np.zeros((TV_x.shape[0], len(labels)))



    for i, j in enumerate(labels):

        print('fit', j)

        t = (train_y == 1)*1

        m, r = get_mdl(t)

        print(TV_x.shape)

        print(r.shape)

        pred_valid[:, i] = m.predict_proba(TV_x.multiply(r))[:, 1]





    y_pred_valid=pred_valid.flatten() # probability of predicted

    return y_pred_valid
%%time

# define input for funcition-report

y_pred_valid=NBSVM_predict(train_vectors ,valid_vectors)


from sklearn.metrics import confusion_matrix

import scikitplot as skplt

from sklearn import metrics

import scikitplot 

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score



def report(y_true,y_pred_valid):

    '''

    y_true: true label

    y_pred_valid: predict of valid or test

    '''

    

    print('The accuracy is:',accuracy_score(y_true, y_pred_valid.round()))

    print('The F1 score is %f'%metrics.f1_score(y_true, y_pred_valid.round())  )

    print('The recall socre is %f'%metrics.recall_score(y_true, y_pred_valid.round()))

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_valid)

    print("The AUC score is :",metrics.auc(fpr, tpr))

    skplt.metrics.plot_confusion_matrix(y_true, y_pred_valid.round(), normalize=True)

    

    y_probas = np.vstack(((1-y_pred_valid),y_pred_valid)).T # predicted probabilities generated by sklearn classifier

    scikitplot.metrics.plot_roc(y_true, y_probas)

    plt.show()
report(valid_y,y_pred_valid)
# input: text of df_train

y_pred_train=NBSVM_predict(train_vectors,train_vectors)
%%time

train_overall_x=vectorizer1.transform(df_train['comment_text'])

train_overall_predicted = NBSVM_predict(train_vectors,train_overall_x)
y_true=df_train.label

report(y_true,train_overall_predicted)
df_train['predict']=train_overall_predicted.round()

df_train['predict']=df_train['predict'].astype('int8')
# identity_columns = [

#     'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

#     'muslim', 'black', 'white', 'psychiatric_or_mental_illness']



identity_columns =['asian', 'latino', 'black', 'white', 'other_race_or_ethnicity',

                   'atheist', 'buddhist', 'hindu', 'jewish', 'muslim', 'christian', 

                   'other_religion', 'female', 'male', 'other_gender', 'heterosexual', 

                   'bisexual', 'transgender', 'homosexual_gay_or_lesbian', 'other_sexual_orientation',

                   'intellectual_or_learning_disability', 'physical_disability', 

                   'psychiatric_or_mental_illness', 'other_disability']

identity=(df_train[identity_columns].fillna(0).values>0.5).sum(axis=1).astype(bool).astype(np.int) 



df_train['identity']=identity
from collections import Counter

Counter(identity)
len1=df_train[np.logical_and(df_train['label']== 0,df_train['identity']==1)].shape[0]

len2=df_train[np.logical_and(df_train['label']== 0,df_train['identity']==1)][df_train['predict']==1].shape[0]

print('fpr in indentity',len2/len1*100)

# df_train[(df_train.label==0)&(df_train.predict==0)]
len3=df_train[np.logical_and(df_train['label']== 0,df_train['predict']==1)].shape[0]

len4=df_train[df_train.label==0].shape[0]

print('fpr in all',len3/len4*100)

%%time

x = train_vectors

test_x = vectorizer1.transform(df_test['comment_text'])

test_predicted = NBSVM_predict(x,test_x).round()


# from sklearn.naive_bayes import MultinomialNB,GaussianNB

# clf1 = MultinomialNB(alpha=1)

# clf1.fit(train_vectors, train_y)

# from  sklearn.metrics  import accuracy_score

# predicted = clf1.predict(valid_vectors)



# accuracy2, auc2 = evaluate(clf1, valid_vectors, valid_y)

# print("测试集正确率：%.4f%%\n" % (accuracy2* 100))

# print("测试集AUC值：%.6f\n" % (auc2))



# import sklearn

# from sklearn.metrics import classification_report



# def evaluate(model, X, y):

#     """评估数据集，并返回评估结果，包括：正确率、AUC值

#     """

#     accuracy = model.score(X, y)

#     fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, model.predict_proba(X)[:, 1], pos_label=1)

#     return accuracy, sklearn.metrics.auc(fpr, tpr)



# from sklearn.metrics import confusion_matrix

# # y_pred_valid=clf1.predict(valid_vectors)

# # confusion_matrix(valid_y, y_pred_valid)

# import scikitplot as skplt

# skplt.metrics.plot_confusion_matrix(valid_y, y_pred_valid.round(), normalize=True)



# # confusion_matrix(valid_y, y_pred_valid)

# confusion_matrix(valid_y,pred_valid.round())



# import scikitplot 

# import matplotlib.pyplot as plt



# y_true = valid_y# ground truth labels

# y_probas = clf1.predict_proba(valid_vectors)# predicted probabilities generated by sklearn classifier

# scikitplot.metrics.plot_roc(y_true, y_probas)

# plt.show()



# from sklearn import metrics

# print('The F1 score is %f'%metrics.f1_score(y_true, y_pred_valid)  )

# print('The recall socre is %f'%metrics.recall_score(y_true, y_pred_valid))



# from sklearn.metrics import classification_report

# target_names = ['Non-toxic', 'toxic']

# print(classification_report(y_true, y_pred_valid, target_names=target_names))





# test_x=vectorizer1.transform(df_test['comment_text'])

# test_predicted = clf1.predict(test_x)
df_submit = pd.read_csv('../input/sample_submission.csv')

df_submit.prediction = test_predicted
df_submit.to_csv('submission.csv', index=False)