from bs4 import BeautifulSoup 
import requests
import pandas as pd
import nltk.data
from nltk.tokenize import RegexpTokenizer 
from nltk.tokenize import MWETokenizer
# import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
def get_StockInfo(quote):
    financeSource = 'https://www.google.com.au/search?q=ASX:{}&num=20&tbm=nws&source=lnt&tbs=sbd:1&sa=X&ved=0ahUKEwico5WXm9HXAhUMGJAKHd-CBu8QpwUIHw&biw=1167&bih=539&dpr=1.65'.format(quote.upper())
    
    page = requests.get(financeSource)
    # we can throw an exception here for the link error, possibly do it later
    
    bs = BeautifulSoup(page.content,"lxml")
    # print(bs.prettify())

    # extract data from tables in html link
    tables = bs.find_all('table')

    # store data in 4 lists: title, publish date, description and url
    titles = []
    dtimes = []
    descs = []
    urls = []

    for i in range(3,len(tables)-1):
        tbl = tables[i].find('a') # find url link in each table
        dtime = tables[i].find('span') # find publish datetime of respective url 
        title = tbl.text # get the article title
        desc = tables[i].find_all('div')[1].text # extract description of the link
        link = "https://www.google.com.au/" + tbl.get('href') # complete link from the google search
        titles.append(title)
        dtimes.append(dtime.text)
        descs.append(desc)
        urls.append(link)

    # store all data extracted into a dataframe
    df = pd.DataFrame()
    df['Title'] = titles
    df['PubDate'] = dtimes
    df['URL'] = urls
    #     df['Description'] = descs

    # split column publishdate into 2 columns named source and publishdate
    df[['Source', 'PubDate']] = df['PubDate'].str.split(' - ', expand=True)
    df = df[['Title', 'PubDate', 'Source', 'URL']] #'Description']]
    
    return df
def extract_text(link):
    text = ""
    
    # Parse the html link in the 'page' variable, and store it in Beautiful Soup format
    page = requests.get(link)
    soup = BeautifulSoup(page.content,'lxml')
    
    # remove all special character for formating in title
    title = soup.title.text.replace('\t','') # remove tab in title
    title = title.replace('\n','') # remove line adding to the title  
    
    # find all paragraph in parsed data
    content = soup.find_all('p')
    
    # extract all necessary content from the data 
    for p in content:
        para = p.text.replace('\t','')
        para = para.replace('\n','')
        text += para + "\n"
        # we should care about the headings in text, but we can do it later
        
    return (title + "\n" + text)
fin_dict = [('ansell','limited'),('lendlease','group'),('mid-cap','stock'),('healthcare','equipment'),('management','team'),('university','degree'),
                ('retirement','income'),('share','price'),('return','on','assets'),('net','income'),('cash','flow'),
                ('price','to','earnings'),('price','to','cash','flow'),('price','to','sales'),('price','to','book'),
                ('hedge','fund'),('financial','news'),('moving','average'),('book','value'),('stock','price'),('short','term'),
                ('financial','obligations'),('leverage','ratio'),('capital','gain'),('compound','interest'),('credit','score'),
                ('mutual','fund'),('net','worth'),('earnings','before','interest'),('financial','health'),('cash','in','hand'),
                ('government','bonds'),('balance','sheet'),('total','assets'),('preferred','shares'),('total','cash'),
                ('net','operating','profit'),('cash','equivalents'),('employed','capital'),('working','capital'),('current','liabilities'),
                ('current','assets'),('gross','margin'),('standard','deviation'),('asset','turnover'),('price','index'),
                ('invested','capital'),('current','value')]
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
fin_neg = []
fin_pos = []
with open('../input/fin_pos_neg.txt','r') as f:
    lines = f.read()
    vocab = ''.join(lines)
    word = ''
    for char in vocab:
        if char not in [',',' ','\n']:
            word += char

        if (char == ',') and (len(fin_neg) < 2355):
            fin_neg.append(word)
            word = ''

        if (char == ',') and (len(fin_neg) == 2355):
            fin_pos.append(word)
            word = ''

        if len(fin_pos) == 355:
            break

fin_pos.remove('')
def pos_or_neg(link):
    content = extract_text(link)
    content = content.lower()
    
    tokenizer = RegexpTokenizer(r"\w+(?:[-.]\w+)?")
    unigram_tokens = tokenizer.tokenize(content)
    
    uni_voc = list(set(unigram_tokens))
    uni_voc.extend(fin_dict)
    
    mwe_tokenizer = MWETokenizer(uni_voc)
    mwe_tokens = mwe_tokenizer.tokenize(unigram_tokens)
    
    stopwords_list = stopwords.words('english')
    stopwords_set = set(stopwords_list)
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

#     stopwords_set = set(stopwords_list)
    stopped_tokens = [w for w in mwe_tokens if w not in stopwords_set]
    
#     sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_detector.tokenize(content.strip())
    
    tagged_sents = []
    for sent in sentences:
        uni_sent = tokenizer.tokenize(sent)
        mwe_text = mwe_tokenizer.tokenize(uni_sent)
        tagged_sent = nltk.tag.pos_tag(mwe_text)
        stopped_tagged_sent = [x for x in tagged_sent if x[0] not in stopwords_set]  
        tagged_sents.append(stopped_tagged_sent)
        
    lemmatizer = WordNetLemmatizer()
    final_tokens =[]
    for tagged_set in tagged_sents:
        final_tokens = final_tokens + [lemmatizer.lemmatize(w[0], get_wordnet_pos(w[1])) for w in tagged_set ]
    
    pos_score,neg_score = 0,0
    for x in final_tokens:
        if x in fin_pos:
            pos_score += 1
        elif x in fin_neg:
            neg_score += 1
        else:
            pass
    
#     posneg = pos_score/neg_score
        
    return ('Positive: ' + str(pos_score) + ' --- ' + 'Negative: ' + str(neg_score))
#             '\n' + 'Ratio = ' + str(posneg))
info = get_StockInfo('mld')
info
for i in range(len(info.URL)):
    print(str(i) + " - " + pos_or_neg(info.URL[i]))