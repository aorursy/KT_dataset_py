import pandas as pd

from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression

import numpy as np
#

# read positive and negative words.

# this is from https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107

# To keep the size of words managable, I sample every 10th word

#

# this section is precomputed - since I cannot use this code on kaggle. see important_words



# positives=""

# negatives = ""







# with open('airline-twitter-sentiment/twitter-sentiment-analysis-tutorial-201107/data/opinion-lexicon-English/positive-words.txt') as f:

#     for i, line in enumerate(f):

#         if not line.lstrip().startswith(';'):

#             #sample every 10th word to keep columns managable

#             if (i%10 == 0):

#                 positives = positives + line

                

# with open('airline-twitter-sentiment/twitter-sentiment-analysis-tutorial-201107/data/opinion-lexicon-English/negative-words.txt') as f:

#     for i, line in enumerate(f):

#         if not line.lstrip().startswith(';'):

#             #sample every 10th word to keep columns managable

#             if (i%10 == 0):

#                 negatives = negatives + line



# #print negatives



# allwords = positives + negatives

# important_words = allwords.split()

# print important_words
air = pd.read_csv('../input/Tweets.csv')

important_words =['accessable', 'accomplished', 'adaptable', 'admiringly', 'adulation', 'advocated', 'affirmation', 'agility',

       'amazed', 'amiability', 'amusing', 'appreciated', 'articulate', 'astonished', 'attraction', 'available',

       'awesomely', 'beautifully', 'benefactor', 'best', 'blessing', 'bonus', 'brave', 'brighten', 'buoyant',

       'carefree', 'charisma', 'cheerful', 'clarity', 'cleared', 'colorful', 'commitment', 'complemented', 'confident',

       'constructive', 'convienient', 'correctly', 'courtly', 'cushy', 'dawn', 'dedicated', 'delicacy', 'deservedly',

       'dextrous', 'distinctive', 'doubtless', 'earnest', 'easy', 'ecstatic', 'efficiently', 'elatedly', 'eloquently',

       'enchanting', 'endorsement', 'enhanced', 'enlighten', 'enterprising', 'enthusiastic', 'enviably', 'eulogize',

       'exaltation', 'exceeding', 'excellent', 'excites', 'exonerate', 'exuberance', 'eyecatching', 'faithfulness',

       'fantastic', 'fast-paced', 'favorite', 'feisty', 'festive', 'first-in-class', 'flourish', 'formidable',

       'freedom', 'ftw', 'gainfully', 'generosity', 'gladden', 'glistening', 'god-given', 'goood', 'grand', 'gratitude',

       'gush', 'handily', 'hardier', 'healthy', 'helping', 'holy', 'hospitable', 'humorous', 'idyllic', 'immaculately',

       'impressed', 'improving', 'inexpensive', 'innocuous', 'instantly', 'intimacy', 'invigorate', 'irresistibly',

       'joyous', 'keenly', 'laudable', 'lean', 'lighter', 'lovable', 'low-price', 'luckiest', 'luxurious',

       'magnificently', 'marvelously', 'matchless', 'meritorious', 'meticulous', 'modern', 'neat', 'nobly', 'nurturing',

       'opulent', 'outshone', 'overtakes', 'pamperedly', 'patience', 'peach', 'permissible', 'playful', 'pleasure',

       'polished', 'powerfully', 'preferable', 'pretty', 'prodigious', 'profusion', 'promoter', 'prosperous', 'prowess',

       'qualify', 'raptureous', 'reaffirm', 'reclaim', 'recovery', 'reform', 'regally', 'reliable', 'renewed',

       'resourceful', 'restful', 'reverent', 'reward', 'righteously', 'rockstar', 'sagacity', 'satisfied', 'seamless',

       'sensation', 'sharp', 'simplest', 'skillful', 'smiles', 'snazzy', 'soothingly', 'speedily', 'splendor',

       'state-of-the-art', 'steadiness', 'straightforward', 'stunningly', 'sublime', 'succes', 'suitable', 'support',

       'supurbly', 'swankiest', 'talent', 'tenaciously', 'thoughtfully', 'thriving', 'titillatingly', 'tougher',

       'triumphal', 'trusting', 'unabashed', 'undaunted', 'unforgettable', 'unselfish', 'upliftingly', 'valor',

       'victorious', 'vivid', 'well', 'well-intentioned', 'well-wishers', 'willingness', 'wisely', 'woo', 'worthy',

       'zippy', 'abominably', 'abscond', 'abuses', 'accuse', 'achey', 'addict', 'adulterated', 'affront', 'aggrieved',

       'ailing', 'allegation', 'ambivalence', 'anger', 'annoyances', 'antagonize', 'antipathy', 'apathy',

       'apprehension', 'arrogance', 'aspersions', 'atrocity', 'authoritarian', 'aweful', 'back-woods', 'bad', 'bane',

       'barren', 'batty', 'beguile', 'belligerent', 'berserk', 'betraying', 'bias', 'biting', 'blame', 'blather',

       'blindingly', 'blotchy', 'blurry', 'bombard', 'botch', 'brainless', 'breach', 'brimstone', 'bruises', 'brute',

       'bulky', 'bum', 'bunk', 'busybody', 'callous', 'cannibalize', 'carp', 'catastrophe', 'chafe', 'chatterbox',

       'cheerless', 'chore', 'clogged', 'coerce', 'comical', 'complaint', 'conceit', 'condemnable', 'confessions',

       'confrontation', 'cons', 'consternation', 'contemptuously', 'contravene', 'corrupt', 'coupists', 'crafty',

       'crashes', 'creaking', 'cringe', 'criticism', 'crowded', 'crumble', 'culpable', 'cuss', 'damn', 'dark', 'dawdle',

       'death', 'debilitating', 'deceitfully', 'decline', 'defame', 'deficient', 'degeneration', 'dejectedly', 'delude',

       'demolisher', 'denial', 'denunciation', 'deprave', 'deprive', 'desertion', 'desperate', 'despondency', 'destroy',

       'detest', 'detracts', 'deviation', 'diabolic', 'die', 'dilly-dally', 'dirt', 'disadvantageous', 'disagreement',

       'disappointment', 'disasterous', 'discomfit', 'disconsolation', 'discountenance', 'discriminate', 'disgraceful',

       'disgustingly', 'disillusion', 'disintegrates', 'disloyalty', 'disobedience', 'disown', 'displaced', 'disquiet',

       'disrespectablity', 'dissappointed', 'dissembler', 'dissing', 'distaste', 'distraught', 'disturb',

       'divisiveness', 'domineer', 'doubts', 'downhearted', 'drag', 'drastically', 'drippy', 'drowning', 'dumb', 'dust',

       'egomania', 'embarrass', 'emphatically', 'engulf', 'entrapment', 'err', 'eschew', 'exagerate', 'excessive',

       'exhausted', 'expensive', 'expunge', 'extremism', 'failed', 'fallacies', 'falsehood', 'fanaticism', 'fascist',

       'fatally', 'fatuity', 'fearfully', 'felon', 'fibber', 'figurehead', 'flagrant', 'flare', 'fleed', 'flimflam',

       'fool', 'foreboding', 'forsaken', 'frail', 'freaking', 'frenzy', 'frighten', 'fruitless', 'fucking', 'funnily',

       'futile', 'galling', 'gawk', 'gibe', 'glibly', 'goad', 'grainy', 'grievances', 'grisly', 'grudge', 'grumpily',

       'hack', 'hampered', 'harassment', 'hardheaded', 'harpy', 'haste', 'hates', 'hazard', 'heartless', 'hedonistic',

       'hells', 'hideously', 'hoard', 'hopelessness', 'horrifying', 'hubris', 'hurted', 'hypocritical', 'idiotic',

       'ignore', 'ill-sorted', 'illiterate', 'imbecile', 'immoral', 'impeach', 'imperialist', 'impinge', 'impolitic',

       'impotent', 'improbability', 'impudently', 'inaccurate', 'inadvisably', 'incapable', 'inclement', 'incompatible',

       'incongruous', 'inconsistency', 'incorrigibly', 'indecorum', 'indignantly', 'indistinguishable',

       'ineffectualness', 'ineloquently', 'inescapably', 'inexpert', 'infection', 'infiltrators', 'infraction',

       'ingratitude', 'iniquitous', 'inopportune', 'insensitive', 'insincerity', 'instability', 'insufficiency',

       'insupportably', 'intermittent', 'intolerance', 'invader', 'invisible', 'irks', 'irragularity', 'irrecoverably',

       'irrepressible', 'irritant', 'issues', 'jealous', 'jerk', 'judder', 'kill', 'kook', 'laconic', 'lame-duck',

       'lapse', 'lawbreaking', 'leaks', 'lemon', 'lewdness', 'lier', 'listless', 'loathsomely', 'loophole', 'loss',

       'ludicrous', 'lurid', 'madly', 'malevolence', 'maltreatment', 'manipulation', 'massacres', 'meddlesome',

       'mendacious', 'midget', 'misapprehend', 'mischievous', 'miserably', 'misguided', 'misleadingly', 'miss',

       'mistress', 'moan', 'molestation', 'mope', 'mortified', 'muddy', 'murky', 'nagging', 'nauseating', 'negate',

       'nepotism', 'nightmare', 'nonexistent', 'obese', 'obnoxiously', 'obsessively', 'obstructs', 'oddities',

       'offensively', 'onslaught', 'oppressively', 'outcry', 'over-acted', 'overbalanced', 'overloaded', 'overshadow',

       'overstatements', 'overwhelmingly', 'pale', 'panicked', 'pariah', 'patronize', 'peeved', 'perilously',

       'pertinaciously', 'perverted', 'petty', 'pigs', 'pitilessly', 'plight', 'poisonously', 'poor', 'prattle',

       'prejudicial', 'pretentious', 'primitive', 'profanity', 'protesting', 'punch', 'puzzling', 'quash', 'racy',

       'rampage', 'rants', 'ravage', 'reckless', 'refuses', 'regret', 'rejecting', 'remorse', 'reprehensible',

       'reprove', 'repulsing', 'restless', 'retarded', 'revert', 'rhetoric', 'rigid', 'risks', 'rough', 'ruinous',

       'rust', 'sacrificed', 'salacious', 'satirize', 'scandalized', 'scarcely', 'scathing', 'scorchingly',

       'scrambling', 'screwed-up', 'seedy', 'self-serving', 'seriousness', 'shabby', 'shamefully', 'shimmy', 'shoddy',

       'shrew', 'sick', 'silly', 'skeletons', 'skulk', 'slaughtered', 'sloooow', 'slower', 'slumpping', 'smelled',

       'smoulder', 'smuttiest', 'snarl', 'snobish', 'somber', 'spade', 'spite', 'spoiled', 'spoonfed', 'squeak',

       'stagnation', 'standstill', 'stealing', 'sticky', 'stingingly', 'straggle', 'streaky', 'stridently', 'strut',

       'stumped', 'stupify', 'subjected', 'substandard', 'sucks', 'suffering', 'sully', 'suppress', 'swamped', 'taboo',

       'tanks', 'tauntingly', 'temper', 'tepid', 'tetchily', 'threat', 'thug', 'tingled', 'torment', 'totalitarian',

       'tragically', 'trapped', 'travesty', 'trivial', 'truant', 'two-faced', 'ugly', 'unaccessible', 'unbearablely',

       'uncomfortable', 'unconvincingly', 'undercutting', 'undersized', 'uneasily', 'unexpectedly', 'unforeseen',

       'unhappily', 'uninsured', 'unkindly', 'unlucky', 'unnoticed', 'unprofitable', 'unreadable', 'unrest',

       'unsettled', 'unsteadily', 'untenable', 'unusably', 'unwell', 'upbraid', 'upseting', 'vague', 'vengefulness',

       'vibrate', 'vile', 'vindictive', 'virulence', 'vomiting', 'war-like', 'waste', 'weakening', 'weird', 'wicked',

       'wobble', 'worried', 'worsening', 'wreaked', 'wrinkled', 'yawn']
#get rid of the common words

s = set(stopwords.words('english'))



def remove_stopwords(input):

    return [i for i in input.lower().split() if i not in s]

   # print [y for x in y if y not in list(s)]



air['cleansed_text'] = air['text'].apply(lambda x: remove_stopwords(x))

print (air.head())
air.columns.values
#

#discard all neutrals

#



non_neutrals = ['positive','negative']

air_non_neutral = air[air.airline_sentiment != 'neutral']

air_non_neutral['sentiment'] = air_non_neutral['airline_sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

#print air_non_nuetral['text']



#words = air_non_nuetral['cleansed_text'].to_numpy()

#important_words =  [item for sublist in words for item in sublist]



print (len(air_non_neutral))
#

# read positive and negative words.

# this is from https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107

# To keep the size of words managable, I sample every 10th word

#

# this section is precomputed - since I cannot use this code on kaggle. see important_words



# positives=""

# negatives = ""







# with open('airline-twitter-sentiment/twitter-sentiment-analysis-tutorial-201107/data/opinion-lexicon-English/positive-words.txt') as f:

#     for i, line in enumerate(f):

#         if not line.lstrip().startswith(';'):

#             #sample every 10th word to keep columns managable

#             if (i%10 == 0):

#                 positives = positives + line

                

# with open('airline-twitter-sentiment/twitter-sentiment-analysis-tutorial-201107/data/opinion-lexicon-English/negative-words.txt') as f:

#     for i, line in enumerate(f):

#         if not line.lstrip().startswith(';'):

#             #sample every 10th word to keep columns managable

#             if (i%10 == 0):

#                 negatives = negatives + line



# #print negatives



# allwords = positives + negatives

# important_words = allwords.split()

# print important_words
#

# Now count the nunber of time important words occur in our cleaned up text

#Store the count in columns 

#



for word in important_words:

    #print word

    air_non_neutral[word] = air_non_neutral['cleansed_text'].apply(lambda s: s.count(word))
train_data = air_non_neutral.sample(frac=.8)

validation_data = air_non_neutral.drop(train_data.index)

#print len (validation_data.columns)

#print len(train_data.columns)



train_data['intercept']=1

validation_data['intercept']=1



#print len (validation_data.columns)

#print len(train_data.columns)

#

# feature_matrix is the new columsn we added (the important words)

#

feature_matrix = train_data.ix[:,17:len(train_data)]

sentiment = train_data['sentiment']


#

# ok, train the model

#

logreg = LogisticRegression()



logreg.fit(feature_matrix, sentiment)
#what is the accuracy on training 

score = logreg.score(feature_matrix, sentiment)

print ("Score is ", score)
# accuracy on validation data

feature_matrix_valid  = validation_data.ix[:,17:len(validation_data)]



sentiment_valid  = validation_data['sentiment']



#print len (validation_data.columns)

#print len(train_data.columns)



score2 = logreg.score(feature_matrix_valid, sentiment_valid)

print (score2)
#print train_data.shape

#print logreg.coef_.T.shape



result = pd.DataFrame(['intercept'] + important_words)



#result['coefficients'] = pd.Series(logreg.coef_.T.flatten(), index = result.index)

result['coefficients'] = pd.Series(logreg.coef_.T.flatten())



#print result.shape

#pd.concat([result,coeffs], axis=1)

sorted_results = result.sort_values(by='coefficients', ascending=False)
import matplotlib.pyplot as plt



#print sorted_results[['coefficients']]

sorted_results['coefficients']= sorted_results['coefficients'].apply(lambda x:  float(x))

print ("=====TOP  POSITIVE WORDS========")

print (sorted_results.head(10))

print ("====TOP 10 NEGATIVE WORDS=======")

print (sorted_results.tail(10))

#print sorted_results[['coefficients']].head()

#print sorted_results.iloc[:,0]

#print sorted_results.head()
%matplotlib inline

from pylab import rcParams

#print sorted_results.iloc[1:10,0]

#plt.figure(figsize=(10,10))

rcParams['figure.figsize'] = 5,5

#plt.subplot(211)

fig, ax = plt.subplots()

#ax.set_yticklabels(sorted_results.iloc[:,0])

ax.set_xticklabels(sorted_results.iloc[0:10][0].values,rotation="90",ha='center')

#print (type(sorted_results.iloc[0:10,1))

#print (sorted_results[[1]])

#plt.bar(range(len(sorted_results)), sorted_results[['coefficients']].values)

#plt.bar(sorted_results[['coefficients']].values,sorted_results.iloc[:,1])

#plt.barh(range(len(sorted_results.iloc[1:10,1])),sorted_results.iloc[1:10,1])

ax.bar(range(10),sorted_results.iloc[0:10]['coefficients'])