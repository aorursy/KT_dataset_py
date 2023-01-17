from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(filenames)
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("***************")
fileName="/kaggle/input/inputtxt/input.txt";
sns.set_style("white")

#Create Word Set Start    
wordsDf=pd.read_csv(fileName, sep=" ", header=None);
wordsDfLst = pd.read_csv(fileName, sep=" ", header=None).replace(np.nan,"").values.flatten().tolist();
wordset = set();
for word in wordsDfLst: 
    if word:
        wordset.add(word);
#Create Word Set End
#----------------------------------------------------------------#        
#Created Temporary Dictonary wordFreqDict it will be filled with map of word to frequency Start
wordFreqDict = defaultdict(int)
for word in wordsDfLst:
    if word:
        wordFreqDict[word.strip()]+=1;    
# Dictonary Creation End        
#--------------------------------------------------------------------#

#wordStats is dataframe with word to frequency map
wordStats = pd.DataFrame(list(wordFreqDict.items()),columns = ['Word','Frequency'])
#--------------------------------------------------------------------------#

#----- Create Sentence DataFrame Read from input.txt file ---------
sentenceDF = pd.read_csv(fileName, chunksize=1, header=None, encoding='utf-8');
sentenceDF = pd.concat(sentenceDF , ignore_index=True);
sentenceDF = sentenceDF.rename(columns={0: "Sentence"});
sentenceDF["sentence_length"]=sentenceDF.agg(lambda sentence:sentence.str.len());
sentenceList = sentenceDF.loc[0:].values.tolist();

is_word_sentence = pd.DataFrame(index=range(sentenceDF.Sentence.size), 
                                    columns=range(len(wordset)));
  
for i,sentence in enumerate(sentenceDF.itertuples(),start=0):
    for j,word in enumerate(wordset,start=0): 
        is_word_sentence.loc[i,j]=word in sentence.Sentence;
  

word_matrix=pd.DataFrame(index= wordset,columns=wordset,data=0);
for wi in wordset:
    for wj in wordset:
        if(wi!=wj):
            for sentence in sentenceList:
                if (str(wi) in str(sentence)) and (str(wj) in str(sentence)):
                    word_matrix.loc[wi,wj] = word_matrix.loc[wi,wj]+1;

print(is_word_sentence);
print(word_matrix);
print(wordStats);       
print(sentenceDF);
print(word_matrix.describe());

print(sentenceDF["sentence_length"].describe());    
# Plotting of Graph
wordStats.plot(x="Word",y="Frequency", kind = 'bar');
plt.xticks(rotation=45)
plt.show()
#sns.pairplot(word_matrix) 
sentenceDF.plot(x="Sentence",y="sentence_length",kind = 'bar');
plt.show()


sns.heatmap(word_matrix, center=0);
plt.show();

#sns.heatmap(is_word_sentence,annot=True);plt.show();