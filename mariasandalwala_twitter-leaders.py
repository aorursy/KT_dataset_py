import pandas as pd
import numpy as np
import string, re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.text import Text
stop_words = stopwords.words("english")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
import pickle
#Removing personal pronouns from stopwords because they will be used in analysis 
stop_words = stop_words[35:]
import spacy
nlp = spacy.load('en_core_web_lg')
import matplotlib.pyplot as plt
leaders = {}
names = ['arvindkejriwal','borisjohnson','donaldtrump','jairbolsonaro','justintrudeau',
         'mamata','narendramodi','rahulgandhi','sannamarin','scottmorrison']
for i in range(10):
    leaders[i+1] = pd.read_csv('../input/leader-tweets/'+names[i]+'.csv')

#Importing Sentence Structure Model

with open('../input/questionmodel/gb_Question_Model.pkl', 'rb') as file:  
    gb = pickle.load(file)
with open('../input/questionmodel/vectorizer_Question_Model.pkl', 'rb') as file:  
    vectorizer= pickle.load(file)    


#Importing Sentiment Analysis Model

with open("../input/sentimentmodel/word_features.txt", "rb") as fp:   # Unpickling
    word_features = pickle.load(fp)
with open('../input/sentimentmodel/sentimentModel.pkl', 'rb') as file:  
    classifier= pickle.load(file) 

# function to convert nltk tag to wordnet tag
def nltk_to_wordnet(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
#lemmatizing function
def lemmatize(s):
    wordnet_tagged = map(lambda x: (x[0], nltk_to_wordnet(x[1])), s)
    lemmatized = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized.append(word)
        else:        
            lemmatized.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized)
# functions needed for sentiment analysis 
def find_features(word_list):
    features = {}
    for w in word_features:
        features[w] = (w in word_list)
    return features

def sentiment(word_list):
    feats = find_features(word_list)
    return classifier.classify(feats)
# MAIN CLEANING FUNCTION
def clean(df,column):
    
    #sort dataframe according to date
    df['Date'] = df['Date'].apply(lambda s: s[3:5]+'-'+s[:2]+s[-5:])
    df['Date'] =pd.to_datetime(df.Date)
    df = df.sort_values('Date')
    
    #remove hashtags and account mentions from tweets
    df[column] = df[column].str.replace('#',"")
    df[column] = df[column].str.replace('@',"")
    
    #remove links and pictures
    df[column] = df[column].apply(lambda x:re.split('https:\/\/.*',x)[0])
    df[column] = df[column].apply(lambda x:re.split('pic.twitter',x)[0])
    
    #Makes another column that specifies the sentence structure of the tweet
    df['Type']  = df[column].apply(lambda x: gb.predict(vectorizer.transform([x]))[0])
    
    # Creating a 'Tokenized' column
    df['Tokenized'] = df[column]
    
    #convert to lower case
    df['Tokenized'] = df['Tokenized'].str.lower()
     #remove punctuations
    df['Tokenized'] = df['Tokenized'].str.replace('[^\w\s]'," ")
    #remove special characters
    df['Tokenized'] = df['Tokenized'].str.replace("\W"," ")
    #Column to find out if numbers in the tweet
    df['Numbers'] = df['Tokenized'].apply(lambda x: any(map(str.isdigit, x)))
    #remove digits
    df['Tokenized'] = df['Tokenized'].str.replace("\d+"," ")
    #remove under scores
    df['Tokenized'] = df['Tokenized'].str.replace("_"," ")
    
    #remove empty rows 
    df = df.replace(to_replace='',value=np.nan)
    df = df.dropna()
    df = df.reset_index()
    df = df.drop('index',axis =1)
    
    #tokenize string of words into array of words ==> removing stop words ==> tagging each word
    #sending tokens to get sentiment analysis 
    tokens =[]
    sentiment_list = []
    for sentence in list(df['Tokenized']):
        #tokenize + stop words removal
        word_list = [i for i in word_tokenize(sentence) if i not in (stop_words)]
        #send word list to sentiment function
        sentiment_list.append(sentiment(word_list))
        #tagging
        word_list = nltk.pos_tag(word_list)         
        tokens.append(word_list)
    df['Tokenized'] = tokens
    df['Sentiment'] = sentiment_list
    
    #lemmatize the string of words using their tags == reduce words to their root form
    df['Lemmatized']  = df['Tokenized'].apply(lambda x: lemmatize(x))   
    
    
    return df  
    
def divide_into_sentences(document):
    return [sent for sent in document.sents]

def number_of_fine_grained_pos_tags(sent):
    """
    Find all the tags related to words in a given sentence. Slightly more
    informative then part of speech tags, but overall similar data.
    Only one might be necessary. 
    For complete explanation of each tag, visit: https://spacy.io/api/annotation
    """
    tag_dict = {
    '-LRB-': 0, '-RRB-': 0, ',': 0, ':': 0, '.': 0, "''": 0, '""': 0, '#': 0, 
    '``': 0, '$': 0, 'ADD': 0, 'AFX': 0, 'BES': 0, 'CC': 0, 'CD': 0, 'DT': 0,
    'EX': 0, 'FW': 0, 'GW': 0, 'HVS': 0, 'HYPH': 0, 'IN': 0, 'JJ': 0, 'JJR': 0, 
    'JJS': 0, 'LS': 0, 'MD': 0, 'NFP': 0, 'NIL': 0, 'NN': 0, 'NNP': 0, 'NNPS': 0, 
    'NNS': 0, 'PDT': 0, 'POS': 0, 'PRP': 0, 'PRP$': 0, 'RB': 0, 'RBR': 0, 'RBS': 0, 
    'RP': 0, '_SP': 0, 'SYM': 0, 'TO': 0, 'UH': 0, 'VB': 0, 'VBD': 0, 'VBG': 0, 
    'VBN': 0, 'VBP': 0, 'VBZ': 0, 'WDT': 0, 'WP': 0, 'WP$': 0, 'WRB': 0, 'XX': 0,
    'OOV': 0, 'TRAILING_SPACE': 0}
    
    for token in sent:
        if token.is_oov:
            tag_dict['OOV'] += 1
        elif token.tag_ == '':
            tag_dict['TRAILING_SPACE'] += 1
        else:
            tag_dict[token.tag_] += 1
            
    return tag_dict

def number_of_dependency_tags(sent):
    """
    Find a dependency tag for each token within a sentence and add their amount
    to a distionary, depending how many times that particular tag appears.
    """
    dep_dict = {
    'acl': 0, 'advcl': 0, 'advmod': 0, 'amod': 0, 'appos': 0, 'aux': 0, 'case': 0,
    'cc': 0, 'ccomp': 0, 'clf': 0, 'compound': 0, 'conj': 0, 'cop': 0, 'csubj': 0,
    'dep': 0, 'det': 0, 'discourse': 0, 'dislocated': 0, 'expl': 0, 'fixed': 0,
    'flat': 0, 'goeswith': 0, 'iobj': 0, 'list': 0, 'mark': 0, 'nmod': 0, 'nsubj': 0,
    'nummod': 0, 'obj': 0, 'obl': 0, 'orphan': 0, 'parataxis': 0, 'prep': 0, 'punct': 0,
    'pobj': 0, 'dobj': 0, 'attr': 0, 'relcl': 0, 'quantmod': 0, 'nsubjpass': 0,
    'reparandum': 0, 'ROOT': 0, 'vocative': 0, 'xcomp': 0, 'auxpass': 0, 'agent': 0,
    'poss': 0, 'pcomp': 0, 'npadvmod': 0, 'predet': 0, 'neg': 0, 'prt': 0, 'dative': 0,
    'oprd': 0, 'preconj': 0, 'acomp': 0, 'csubjpass': 0, 'meta': 0, 'intj': 0, 
    'TRAILING_DEP': 0}
    
    for token in sent:
        if token.dep_ == '':
            dep_dict['TRAILING_DEP'] += 1
        else:
            try:
                dep_dict[token.dep_] += 1
            except:
                print('Unknown dependency for token: "' + token.orth_ +'". Passing.')
        
    return dep_dict

def number_of_specific_entities(sent):
    """
    Finds all the entities in the sentence and returns the amont of 
    how many times each specific entity appear in the sentence.
    """
    entity_dict = {
    'PERSON': 0, 'NORP': 0, 'FAC': 0, 'ORG': 0, 'GPE': 0, 'LOC': 0,
    'PRODUCT': 0, 'EVENT': 0, 'WORK_OF_ART': 0, 'LAW': 0, 'LANGUAGE': 0,
    'DATE': 0, 'TIME': 0, 'PERCENT': 0, 'MONEY': 0, 'QUANTITY': 0,
    'ORDINAL': 0, 'CARDINAL': 0 }
    
    entities = [ent.label_ for ent in sent.as_doc().ents]
    for entity in entities:
        entity_dict[entity] += 1
        
    return entity_dict

def sample(test_sent):
   
    parsed_test = divide_into_sentences(nlp(test_sent))
    
    # Get features
    sentence_with_features = {}
    entities_dict = number_of_specific_entities(parsed_test[0])
    sentence_with_features.update(entities_dict)
    pos_dict = number_of_fine_grained_pos_tags(parsed_test[0])
    sentence_with_features.update(pos_dict)
    
    df = pd.DataFrame(sentence_with_features, index=[0])
    
    
    df = scaler.transform(df)    
    prediction = nn_classifier_scaled.predict(df)
        
    # Run a prediction
    if prediction == 0:
        return '1'
    else:
        return '0'


with open('../input/fact-model/nn_classifier_scaled.pickle', 'rb') as f1:  
    nn_classifier_scaled = pickle.load(f1)
with open('../input/fact-model/scaler.pickle', 'rb') as f2:  
    scaler = pickle.load(f2)
covid_corpus = ['covid','corona','coronavirus','virus','pandemic','epidemic','outbreak','spread','disease','cases','patient',
            'cure','vaccine','ventilator','positive','crisis','test','social','distancing',
            'curve','peak','wuhan','china','infection','infect','quarantine','symptom','reopen','lockdown','phase']
def covid(s):
    for i in s.split():
        if i in covid_corpus:
            return '1'
    return '0'
        

h = 'stupid unintelligent foolish dumb dolt brainless fool dullard anserine idiotic idiot asinine silly blockheaded obtuse dull absurd gormless dense ridiculous slow thick ignorant boneheaded dopey crazy ludicrous lazy naive imbecile simpleton dunce blockhead weak incompetent corrupt inflexible bloated useless haphazard inept duplicative superfluous mad insane lunatic weirdo brainsick loony demented unhinged sham phony false bogus cheat imposter fraud pseudo pretender fraudulent scam deceiver feign trickster cheater phony faux manipulate bullshit forge unreal charlatan imitative falsification spurious falsehood liar stupidity cheater perjurer cheat deception prevaricator trickster gambler fibber fabricator deceitful phony fake impostor falsifier hypocrite womanizer traitor bum scoundrel idiot quitter buffoon sociopath hater troublemaker joke clown mime loose foolish thoughtless hypocritical outrageous unprofessional disgraceful inappropriate disingenuous callous absurd deceitful ridiculous despicable selfish ludicrous shameful scandalous idiotic cowardly prudent flighty irrational scatterbrained wanton frivolous insane unscrupulous flippant  arrogant dirty crappy awful rotten filthy nasty stinky shitty mediocre miserable icky darn sloppy pitiful pathetic lyin'
hate_speech = h.split()
hate_speech = list(dict.fromkeys(hate_speech))
def hate(s):
    for i in s.split():
        if i in hate_speech:
            return '1'
    return '0'

p = 'democrat republican politics politician vote reelection voter ballot poll electoral campaign presidential preelection electorate elect candidate senate party candidacy candidature opposition nomination midterm primaries turnout nominate bjp congress hindutva aap bsp conservative unionist labour labor liberal coalition alliance'
pol_speech = p.split()
def politics(s):
    for i in s.split():
        if i in pol_speech:
            return '1'
    return '0'

e = 'equality equal unity empowerment freedom gender ideal fairness unequal equally equalize dignity harmony diversity democracy pluralism inclusiveness liberty justice sexism racism disability affirmative  prejudice ableism ageism homophobia racialism segregation racial discrimination minorities bias oppression privilege'
equal_speech = e.split()
def equality(s):
    for i in s.split():
        if i in equal_speech:
            return '1'
    return '0'

c = 'health healthcare job security economy lgbt lgbtq mental medicare safety hunger environment warming brutality housing rent education'
community_speech = c.split()
def community(s):
    for i in s.split():
        if i in community_speech:
            return '1'
    return '0'

inclusive_neg = ['guy', 'guys', 'boy', 'boys', 'girl', 'girls', 'man', 'men' ,'women', 'woman']
inclusive_pos = ['we', 'our', 'ours', 'ourselves', 'mankind','humankind','people','human','humanity']
def inclusive(s):
    for i in s.split():
        if i in inclusive_neg:
            return '-1'
        if i in inclusive_pos:
            return '1'
    return '0'

m = 'Accomplish Accomplishments Achieve Act Action Active Admiration Admire Adventure Alive Ambition Ambitious Appreciate Appreciation Attain Attitude Beauty Believe Believable Bliss Breakdown Breathtaking Build Catalyst Challenge Clarity Commit Commitment Compassion Complete Concentrate Confidence Content Control Conquer Courage Create Dare Dedicate Dedication Desire Determination Determine Dream Drive Eager Earnest Empower Empowering Empowerment Encourage Encouragement Encouraging Endurance Endure Energetic Energy Enjoy Enjoyment Enthusiasm Envision Escape Excellence Experiences Faith Faithful Faithfulness Fearless Fighter Fight Finish Finisher Fire Fix Focus Forgive Freedom Fulfilment Glory Goal Goodness Gratitude Happiness Happy Harmony Honesty Honor Hope Humble Humility Imagination Imagine Impetus Improve Ineffable Initiative Inspiration Inspire Inspiring Integrity Interest Joy Joyful Joyfulness Kind Kindness Knowledge Laugh Lead Leading Learn Life Live Limitless Love Loving Mindful Mindset Mission Meaning Meaningful Memories Momentum Motivate Motivated Motivation Motive Move Movement Moving Nurture Obstacles Opportunity Optimistic Outstanding Overcome Passion Patience Peace Peaceful Peacefulness Persevere Perseverance Persist Persistence Persuade Plan Planner Positive Possibilities Power Powerful Practice Pride Prioritize Rise Role Safe Safety Satisfaction Satisfy Secure Security Self Skill Skilful Skilfully Spirit Spirited Spur Stimulus Strength  Strong Succeed Success Sustain Sustenance  Teach Teachable Time Trust Trustworthy Truth Understand Understood Value Values Versatile Will Willpower Winner Wisdom Wise Worthy Yearn Yearning Yes'
m = m.lower()
motivation_corpus = m.split()
def motivation(s):
    for i in s.split():
        if i in motivation_corpus:
            return '1'
    return '0'

feedback_corpus = [ 'feedback', 'view', 'input', 'interaction','interact','ask','comment','address','thought'
                   , 'query', 'question', 'opinion','suggestion','suggest']
def feedback(s):
    for i in s.split():
        if i in feedback_corpus:
            return '1'
    return '0'

def interrogative(s):
    if s[2:] == 'Question':
            return '1'
    return '0'

def factual(text, tokenized):
    if len(tokenized)< 3:
        return 0
    else:
        return sample(text)


def leadership(col1,col2,col3,col4):
    #return int(col1)+int(col2)+int(col3)+int(col4)
    if int(col1)+int(col2)+int(col3)+int(col4) > 0:
        return 'High'
    else:
        return 'Low'
    
def ethics(hate,col1,col2,col3,col4):
    if hate == '1':
        return 'Low'
    
    if (-1)*int(col1)+2*int(col2)+int(col3)+int(col4) >= 0:
        return 'High'
    else:
        return 'Low'
    
def analysis(df):
    
    df['Covid'] = df['Lemmatized'].apply(lambda x: covid(x))
    
    #ETHICS
    df['Hate Speech'] = df['Lemmatized'].apply(lambda x: hate(x))
    df['Political'] = df['Lemmatized'].apply(lambda x: politics(x))
    df['Promote Diversity'] = df['Lemmatized'].apply(lambda x: equality(x))
    df['Community Issues'] = df['Lemmatized'].apply(lambda x: community(x))
    df['Inclusive Language'] = df['Lemmatized'].apply(lambda x: inclusive(x))
    
    #LEADERSHIP
    df['Morale'] = df['Lemmatized'].apply(lambda x: motivation(x))
    df['Feedback'] = df['Lemmatized'].apply(lambda x: feedback(x))
    #df['Interrogative'] = df['Type'].apply(lambda x: interrogative(x))
    l,m = [],[]
    for index,row in df.iterrows():
        if len(row['Tokenized']) < 3:
            l.append('0')
            m.append('0')

        else:
            l.append(sample(row['Text']))
            m.append(interrogative(row['Type']))           
               
    df['Interrogative'] = m   
    df['Facts'] = l
    
    df['Ethics'] = df.apply(lambda x: ethics(x['Hate Speech'],x['Political'],x['Promote Diversity'],x['Community Issues'],x['Inclusive Language']),axis =1)
    df['Leadership'] = df.apply(lambda x: leadership(x['Morale'],x['Feedback'],x['Interrogative'],x['Facts']),axis =1)
    
    return df
for i in range(10):
    leaders[i+1] = clean(leaders[i+1],'Text')
    leaders[i+1] = analysis(leaders[i+1])
# Making another dictionary for graph dataframes
graphs = {}
g = {}

for i in range(1,11):
    data = leaders[i].iloc[:,[0,-2,-1]]
    low_ethics,low_leadership = [],[]
    high_ethics,high_leadership = [],[]
    for index,row in data.iterrows():
        if row['Ethics'] == 'Low':
            high_ethics.append(0)
            low_ethics.append(1)
        else:
            high_ethics.append(1)
            low_ethics.append(0)
            
        if row['Leadership'] == 'Low':
            high_leadership.append(0)
            low_leadership.append(1)
        else:
            high_leadership.append(1)
            low_leadership.append(0)
    data['High Ethics'+str(i)] = high_ethics
    data['Low Ethics'+str(i)] = low_ethics
    data['High Leadership'+str(i)] = high_leadership    
    data['Low Leadership'+str(i)] = low_leadership
    
    graphs[i] = data.iloc[:,[0,-4,-3,-2,-1]]
    g[i] = graphs[i].groupby([pd.Grouper(key='Date', freq='W-TUE')])[['High Ethics'+str(i),'Low Ethics'+str(i),'High Leadership'+str(i),'Low Leadership'+str(i)]].sum().reset_index() 
    

            
            
g[1].iloc[:,1:3]
result = pd.DataFrame()
result = result.append(g[1].iloc[:,1:3])
lead = pd.DataFrame()
lead = lead.append(g[1].iloc[:,3:])

for i in range(2,11):
    result = result.join(g[i].iloc[:,1:3]) 
    lead = lead.join(g[i].iloc[:,3:])  

result.iloc[:,13:]
lead