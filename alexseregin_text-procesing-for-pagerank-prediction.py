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
import tensorflow as ts
from tensorflow import keras
import networkx as nx
import matplotlib.pyplot as plt

plotsize = (20,10) # for plotting later
# G = nx.read_gpickle('/home/alex/HDD/workshop/citation_network.gpickle') # quite slow
pr = pd.read_csv('/kaggle/input/covid-papers-citation-network-and-pagerank-score/pagerank.csv')
pr.columns = ['ID', 'title', 'pagerank']
node = pr.loc[27863]['title']                   # get access to an entry title in dataframe
pr.describe()
#  nx.info(G)
print('Name: \nType: DiGraph\nNumber of nodes: 1398451\nNumber of edges: 2554291\nAverage in degree:   1.8265\nAverage out degree:   1.8265')
print('Compare with Zachary\'s Karate Club average degree:' ,4.5882)
top100 = pr[pr['pagerank']>=3.7*10**-6]         
top100_by_pagerank = top100.sort_values('pagerank',ascending = False)
plt.figure(figsize=plotsize)
plt.title("Top 100 entires by pagerank")
top100_by_pagerank['pagerank'].plot(kind='bar')
plt.show()

fig, axes = plt.subplots(ncols=2,figsize=(10,5))
top100_by_pagerank['pagerank'].plot(kind='box',ax=axes[0])
top100_by_pagerank['pagerank'].plot(kind='kde',ax=axes[1],title="wtf is kde?")
plt.show()
def txt_processing(pd_frame,title='title',add_stops=[],frac_est=0.35, class_borders=(0.75,0.25)):
    """Preprocesing unit
    // under construction//: 
    Try to minimize number of libraries, 
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        
    Assumes 
    pd_frame: pandas.DataFrame, 
    title: str; the name of column containing text to process, 
    add_stops: list of str; additional stopwords
    frac_est: float; portion of the bag of words to use as training data
    class_borders: tuple of floats; define classification group borders
    
    returns:
     tuple(
     df: pandas.DataFrame; processed dataframe
     bow: {word: bag of word value}; bag of words for all entries 
     idx: {word:index}; index of the fraction of bow, set by frac_est parameter.
     )
    
    """
    
    import nltk.stem
    from nltk.corpus import stopwords
    from nltk.tokenize import RegexpTokenizer
    from sklearn.feature_extraction.text import CountVectorizer
    from tensorflow.keras.preprocessing.text import Tokenizer
    
    
    lemmer = nltk.stem.WordNetLemmatizer()    
    vectorizer = CountVectorizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = nltk.stem.SnowballStemmer("english")
    
    #tokenizer = Tokenizer()    
    #stemmer = nltk.stem.rslp.RSLPStemmer()
    #stemmer = nltk.stem.LancasterStemmer()
    
    def stopWords(words):
        '''assumes: 
        words: list
        returns:
        list wihtout stopwords
        '''
        mywords = add_stops # additional words to filter out
        wordsFiltered=[]
        for w in words:
            if w not in stopwords.words('english') and w not in mywords:
                wordsFiltered.append(w)
        return wordsFiltered
    
    def bow(corpus):
        '''for df.apply, applies sklearn.feature_extraction.text.CountVectorizer() per row
        assumes: corpus: str
        '''
        # vectorizer = CountVectorizer()
        vectorizer.fit_transform([corpus]).todense()
        return vectorizer.vocabulary_ 
    
    def Outcome(inp):
        ''' 
        Compute outcome variable
        assumes: inp: float; pagerank of an entry
        returns: -1(<low),1(>high) or 0
        '''
        if inp > df['pagerank'].quantile(class_borders[0]):
            return 1
        elif inp < df['pagerank'].quantile(class_borders[1]):
            return -1
        else:
            return 0
        
    def train_x(inp):
        """Convert bag of words into array training data
           Assumes input a dictionary, bag of words
           idx a dictionary with numbers assigned to words
           returns an np.array with values corresponding to words in the inp"""
        train_x = np.zeros(len(idx)+1)
        for key,val in inp.items():
            train_x[idx.get(key,-1*len(idx))]=val # last entry will get all the minorities, can be dropped later?
        return train_x
    
    df = pd_frame.copy()
    df['1'] = df[title].apply(tokenizer.tokenize) 
    df['2'] = df['1'].apply(lambda x: [stemmer.stem(y) for y in x])
    df['3'] = df['2'].apply(lambda x: [lemmer.lemmatize(y) for y in x])
    df['4'] = df['3'].apply(stopWords)
    df['processed'] = df['4'].apply(' '.join) 
    df['BoW'] = df['processed'].apply(bow)
    df['Outcome'] = df['pagerank'].apply(Outcome)
    
    # Build index and bag of words for all entires:
    texts = []
    
    for _,row in df.iterrows():
        texts.append(row['processed'])
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(texts).todense()
    bow = vectorizer.vocabulary_ 
    bow = {k: v for k, v in sorted(bow.items(), key=lambda item: item[1],reverse=False)}
    
    # create index
    idx={}
    for index,key in enumerate(bow):
        idx[key] = index
        if index > frac_est*len(bow):
            break
        
    df['train_x'] = df['BoW'].apply(train_x)
    df = df.drop(columns=['1', '2', '3','4'])#,'processed','Bag of Words'])
    
    return (df,bow,idx) # BoW and idx for testing, might remove later
def reshape_xy(df,x_label = 'train_x',y_label = 'Outcome'):
    """ reshapes data to numpy arrays
    Assumes df a dataframe containing dependent and independent variables
            x_label,y_label are str, names of columns containing the data
            y has to be a list of outcomes
            x has to be array-like?? output of txt_processing function
       Returns np.arrays fit to be fed to classifier"""
    y = df[y_label]
    
    x_loc = df.columns.get_loc(x_label)
    x = np.zeros((len(df),len(df.iloc[0,x_loc])))
    df = df.reset_index()
    for index,row in df.iterrows():
        
        x[index]=row[x_label]
        
    return (x,y)
import numpy as np
def optimizer(df,
              sample_size = 100,
              rand = 1,
              control_parameter = '',
              addStopWords=[],
              n_iter = 4,
              cpRange = (0.0001,1-0.0001),
              pltSize = (20,10),
              verbose = False
             ):
    '''Function to search paramter space for optimal values will increment the control parameter in number of steps(n_iter)
    lower and upper borders are conntorlled with cpRange = (lower,upper)
    //NoTES: setting cpRange to start at 0 produces errors
    parameters: 
    df: pandas.DataFrame
    sample size: int, for pandas.DataFrame.sample
    rand: 0 or 1, random_state for pandas.DataFrame.sample
    control_parameter: str, chooses parameter which values to search for.
    addStopWords: list, additional stopwords to use in  processing.
    n_iter: int, number of iterations for control parameter testing 
    cpRange: a tuple, (lower,higher) bounds on ranking of items, this affects dependent variable
    pltSize: a tuple, plt.figure(figsize=)
    verbose: bool, print tech information after the graph
    
    '''
    from sklearn.model_selection import train_test_split
    import sklearn.ensemble
    from sklearn.metrics import accuracy_score
    
    sample = df.sample(sample_size,random_state = rand)
    
    
    parameters = {
        
        'BoW fraction' : 0.4 ,
        'class_border_high' : 0.75,
        'class_border_low'  : 0.25,
        'train_test_split'  : 0.4,
        'n_estimators'      : 1,
        'min_samples_split' : 1,
        'min_samples_leaf'  : 0.15,
        
        
        
    }
    
    cp_values = np.linspace(cpRange[0],cpRange[1] ,n_iter)
    AccuracyValuesTrain,AccuracyValuesTest = [],[]
    
    for i in cp_values:
        print('setting parameter {} = {}'.format(control_parameter,i))
        parameters[control_parameter] = i
        # process the sample, generate train and test set
        processed,bow,idx = txt_processing(sample,
                                           add_stops=addStopWords,
                                           frac_est=parameters['BoW fraction'],
                                           class_borders=(parameters['class_border_high'],
                                                          parameters['class_border_low'])
                                          )
        train, test = train_test_split(processed, test_size=parameters['train_test_split'])
        train_x,train_y = reshape_xy(train)
        test_x,test_y = reshape_xy(test)


        # Create, train model, evaluate:
        
        
        model =[]
        model =  sklearn.ensemble.RandomForestClassifier(n_estimators = int(100*parameters['n_estimators']), 
                                                         min_samples_split = int(len(idx) // 10 * parameters['min_samples_split']),
                                                         min_samples_leaf = parameters['min_samples_leaf'],
                                                         verbose = False
                                                        )
        model.fit(train_x,train_y)
        trainResult = list(model.predict(train_x))
        testResult = list(model.predict(test_x))
        AccuracyValuesTrain.append(accuracy_score(train_y,trainResult))
        AccuracyValuesTest.append(accuracy_score(test_y,testResult))
        
        
    #Visualize the results: 
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=pltSize)
    plt.title("Control parameter: {}".format(control_parameter))
    plt.plot(cp_values,AccuracyValuesTrain,label='Train')
    plt.plot(cp_values,AccuracyValuesTest,label='Test')
    plt.legend(loc="upper right")
    
    plt.show()
    
    if verbose:
        print('Accuracy on train: {} \n Accuracy on test: {} \n '.format(AccuracyValuesTest,AccuracyValuesTrain))
        print(sklearn.metrics.classification_report(train_y,trainResult))
        print()
        print(sklearn.metrics.classification_report(test_y,testResult))
        
    
optimizer(pr,
          sample_size = 100,
          rand = 1,
          addStopWords = ['11','12','20','2008','2007'],
          control_parameter = '',
          cpRange = (0.1,0.4), 
          n_iter = 4,
          verbose = True)
