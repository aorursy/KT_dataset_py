import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install pubchempy
import pubchempy as pcp
df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','abstract','authors','doi','publish_time'])
print (df.shape)
#drop duplicates
#df=df.drop_duplicates()
df = df.drop_duplicates(subset='abstract', keep="first")
#drop NANs 
df=df.dropna()
# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()
#show 5 lines of the new dataframe
print (df.shape) #first 10 and first 5
df.head()
import functools
from IPython.core.display import display, HTML
from nltk import PorterStemmer
#tell the system how many sentences are needed
max_sentences=5
#
def stem_words(words):
    stemmer = PorterStemmer()
    singles=[]
    for w in words:
        singles.append(stemmer.stem(w))
    return singles
# list of lists for topic words realting to tasks
display(HTML('<h1>COVID-19 summary page vaccines and therapeutics</h1>'))
display(HTML('<h3>Table of Contents (ctrl f and search the hash tag and words below to find table</h3>'))


tasks = [['nsp12'],['RdRp']]

z=0
for terms in tasks:
    stra=' '
    stra=' '.join(terms)
    k=str(z)
    #display(HTML('<a href="#'+k+'">'+stra+'</a>'))
    display(HTML('# '+stra))
    z=z+1

# loop through the list of lists
z=0
for search_words in tasks:
    df_table = pd.DataFrame(columns = ["pub_date","authors","title","excerpt"])
    str1=''
    # a make a string of the search words to print readable search
    str1=' '.join(search_words)
    search_words=stem_words(search_words) #function to remove the ends of the words
    
    # add cov to focus the search the papers and avoid unrelated documents
    search_words.append("cov")
    
    # search the dataframe for all the keywords
    dfa=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in search_words))]
    search_words.pop()
    search_words.append("-cov-")
    dfb=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in search_words))]
    # remove the cov word for sentence level analysis
    search_words.pop()
    #combine frames with COVID and cov and drop dups
    frames = [dfa, dfb]
    df1 = pd.concat(frames)
    df1=df1.drop_duplicates()
    #I think this is the data frame we want to find
    
    display(HTML('<h3>Task Topic: '+str1+'</h3>'))
    
    display(HTML('# '+str1+' <a></a>'))
    z=z+1
    # record how many sentences have been saved for display
    # loop through the result of the dataframe search
    for index, row in df1.iterrows():
        pub_sentence=''
        sentences_used=0
        #break apart the absracrt to sentence level
        sentences = row['abstract'].split('. ')
        #loop through the sentences of the abstract
        for sentence in sentences:
            # missing lets the system know if all the words are in the sentence
            missing=0
            #loop through the words of sentence
            for word in search_words:
                #if keyword missing change missing variable
                if word not in sentence:
                    missing=missing+1
            # after all sentences processed show the sentences not missing keywords limit to max_sentences
            if missing<len(search_words) and sentences_used < max_sentences and len(sentence)<1000 and sentence!='':
                sentence=sentence.capitalize()
                if sentence[len(sentence)-1]!='.':
                    sentence=sentence+'.'
                pub_sentence=pub_sentence+'<br><br>'+sentence
        if pub_sentence!='':
            sentence=pub_sentence
            sentences_used=sentences_used+1
            authors=row["authors"].split(" ")
            link=row['doi']
            title=row["title"]
            linka='https://doi.org/'+link
            linkb=title
            sentence='<p align="left">'+sentence+'</p>'
            final_link='<p align="left"><a href="{}">{}</a></p>'.format(linka,linkb)
            to_append = [row['publish_time'],authors[0]+' et al.',final_link,sentence]
            df_length = len(df_table)
            df_table.loc[df_length] = to_append
    filename=str1+'.csv'
    #df_table.to_csv(filename,index = False)
        #display(HTML('<b>'+sentence+'</b> - <i>'+title+'</i>, '+'<a href="https://doi.org/'+link+'" target=blank>'+authors[0]+' et al.</a>'))
    #df_table=HTML(df_table.to_html(escape=False,index=False))
   # display(df_table)

#Need to now add a column to the DataFrame for the protein search term


print ("done")
#Converting the printed table from the loop into a Data Frame
df = pd.DataFrame(data=df_table)
print(df)
#!pip install -U textblob
#!python -m textblob.download_corpora
excerpts = df['excerpt']
#test dataframe of two abstracts
data_df = pd.DataFrame(
{"abstract": ["Middle East Respiratory Syndrome, caused by the MERS coronavirus (MERS-CoV), continues to cause severe respiratory disease with a high case fatality rate. To date, potential antiviral treatments for MERS-CoV have shown limited efficacy in animal studies. Here, we tested the efficacy of the broad-acting antiviral remdesivir in the rhesus macaque model of MERS-CoV infection. Remdesivir reduced the severity of disease, virus replication, and damage to the lungs when administered either before or after animals were infected with MERS-CoV. Our data show that remdesivir is a promising antiviral treatment against MERS that could be considered for implementation in clinical trials. It may also have utility for related coronaviruses such as the novel coronavirus 2019-nCoV emerging from Wuhan, China.", 
              "The emergence of the 2019 novel coronavirus (COVID-19), for which there is no vaccine or any known effective treatment created a sense of urgency for novel drug discovery approaches. One of the most important COVID-19 protein targets is the 3C-like protease for which the crystal structure is known. Most of the immediate efforts are focused on drug repurposing of known clinically-approved drugs and virtual screening for the molecules available from chemical libraries that may not work well. For example, the IC50 of lopinavir, an HIV protease inhibitor, against the 3C-like protease is approximately 50 micromolar, which is far from ideal. In an attempt to address this challenge, on January 28th, 2020 Insilico Medicine decided to utilize a part of its generative chemistry pipeline to design novel drug-like inhibitors of COVID-19 and started generation on January 30th. It utilized three of its previously validated generative chemistry approaches: crystal-derived pocked-based generator, homology modelling-based generation, and ligand-based generation. Novel druglike compounds generated using these approaches were published at www.insilico.com/ncov-sprint/. Several molecules will be synthesized and tested using the internal resources; however, the team is seeking collaborations to synthesize, test, and, if needed, optimize the published molecules"]},index=[1,2])
data_df
#how to index a specific row in 
data_df.abstract[2]
import re #Regular-Expressions
import string #Types of punctuation
import nltk

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

#StopWords Sourced from: https://gist.github.com/sebleier/554280
#all_stopwords = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]
all_stopwords = nltk.corpus.words.words()
porter=PorterStemmer()

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text) #if there's a character in [] get rid of it
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    #text = re.sub('\w*\d\w*', '', text) #remove all the digits
    
    word_tokens = word_tokenize(text) 
    
    filtered_sentence = [w for w in word_tokens if not w in all_stopwords] 
    for w in word_tokens: 
        if w not in all_stopwords: 
            filtered_sentence.append(w)
    
    stem_sentence = []
    for word in filtered_sentence:
            stem_sentence.append(porter.stem(word))
            #stem_sentence.append(word)
    print(stem_sentence)
    
    filtered2_sentence = [z for z in stem_sentence if not z in all_stopwords] 
    for z in stem_sentence: 
        if z not in all_stopwords: 
            filtered2_sentence.append(z)

    print(filtered2_sentence)
    
    text = filtered2_sentence
    
    return text

round1 = lambda x: clean_text_round1(x)
if 'is' in all_stopwords :
    print("Yes, 'the word' found in List : " , all_stopwords)
data_clean = pd.DataFrame(data_df.abstract.apply(round1))
data_df['tokenized_abstract'] = data_clean #appending tokenized_abstract to our original data frame, data_df
type(data_df.tokenized_abstract[2]) #spit out a list
data_df.tokenized_abstract[2][3]
test = data_df.tokenized_abstract[2]
test = pd.DataFrame(test)
print(test)
test.loc[:,0]
[str(i) for i in test.loc[:,0]]
def pubchem(token):
    '''Search the tokens for pubchem compound ID's token is the tokenized abstract, defined as df.tokenized_abstract'''
    for y in token:
        results = pcp.get_compounds(y, 'name')
        #if results == []:
            #continue
        print(results)
    return results
    
getID = lambda x: pubchem(x)
pubchem(data_df.tokenized_abstract[2])
results = pcp.get_compounds([str(i) for i in test.loc[:,0]], 'name')
results = pcp.get_compounds(data_df.tokenized_abstract[2], 'name')
print(results)
token = data_df.tokenized_abstract[2]
token[1]
token = data_df.tokenized_abstract[2]
data_ID = pd.DataFrame(token.apply(getID))
abstract = data_clean.abstract[1]
#data_clean.abstract = data_clean.abstract.astype(str)
print(data_clean.abstract[1])
print(data_clean.abstract[2])
stop_words = set(stopwords.words('english')) 
all_stopwords = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]
test = ["Middle East Respiratory Syndrome, caused by the MERS coronavirus (MERS-CoV), continues to cause severe respiratory disease with a high case fatality rate. To date, potential antiviral treatments for MERS-CoV have shown limited efficacy in animal studies. Here, we tested the efficacy of the broad-acting antiviral remdesivir in the rhesus macaque model of MERS-CoV infection. Remdesivir reduced the severity of disease, virus replication, and damage to the lungs when administered either before or after animals were infected with MERS-CoV. Our data show that remdesivir is a promising antiviral treatment against MERS that could be considered for implementation in clinical trials. It may also have utility for related coronaviruses such as the novel coronavirus 2019-nCoV emerging from Wuhan, China."]
words = set(nltk.corpus.words.words())
new = " ".join(w for w in nltk.wordpunct_tokenize(test) if w.lower() in words or not w.isalpha())
print(stopwords.words('english'))
print(all_stopwords)
word_tokens = word_tokenize(abstract) 
  
filtered_sentence = [w for w in word_tokens if not w in all_stopwords] 
  
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in all_stopwords: 
        filtered_sentence.append(w) 
  
print(word_tokens) 
print(filtered_sentence)

ID
len(token)
print(abstract)