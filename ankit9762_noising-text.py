# importing the essential package 
import pandas as pd
import numpy as np
import random
import re

# importing the dataset  
df = pd.read_csv('/kaggle/input/brown-corpus/brown.csv')
# getting the tokenized words column and spliting it into string 
df1 = df['tokenized_text'].str.split()
df2 = df1.head(56340)     # training the model 
df3 = df1.tail(1000) 
#getting the 20 percent from the sentence and adding 20% noise , the noise which i am adding are repeating the words 
# dropping and swaping the words in the sentense 

def repeat():
    number = len(wordList)
    numb = int(number*0.20)
    for x in range(numb):
        wordList.insert((random.randint(1,len(wordList))-1),wordList[random.randint(1,len(wordList))-1])
        #print(len(wordList))
        #print(wordList)
        return wordList

def drop():
    number = len(wordList)
    numb = int(number*0.20)
    for x in range(numb):
        num = (random.randint(1,len(wordList))-1)
        wordList.remove(wordList[num])
        #print(len(wordList))
        #print(wordList)
        return wordList
    

def swap():
    def swapPositions(list, pos1, pos2): 
        list[pos1],list[pos2] = list[pos2],list[pos1] 
        return list
    
    number = len(wordList)
    numb = int(number*0.20)
    for x in range(numb):
        
        num1 = (random.randint(1,len(wordList))-1)
        num2 = (random.randint(1,len(wordList))-1)
        swapPositions(wordList, num1, num2)
        #wordList[num]
        
        #print(len(wordList))
        #print(wordList)
        return wordList
##############adding 20% noise to the dataset#########
noisy_dataset = []
for i in range(len(df2)):
    aa = df2.iloc[i]
    mystr = str(aa)
    wordList = re.sub("[^\w]", " ",  mystr).split()
    for x in range(1):
        choose_operation = random.randint(1,3)
        if choose_operation == 1:
            #print(repeat)
            repeat()
            noisy_dataset.append(wordList)
            
        if choose_operation == 2:
            #print(drop)
            drop()
            noisy_dataset.append(wordList)
            
        if choose_operation == 3:
            #print(swap)
            swap()
            noisy_dataset.append(wordList)
            
    
df5 = pd.DataFrame(noisy_dataset)    
df5['ColumnA'] = df5[df5.columns[1:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)
noissy_dataset = df5['ColumnA']  
label = df[['label']]
label_te = label.tail(1000)
label_tr = label.head(56340)

final_noisy_dataset = pd.concat([noissy_dataset, label_tr.reindex(label_tr.index)], axis=1)
#################noise free dataset #############

noisefree_data = []
for i in range(len(df2)):
    aa = df2.iloc[i]
    mystr = str(aa)
    wordList = re.sub("[^\w]", " ",  mystr).split()
    noisefree_data.append(wordList)
    
df6 = pd.DataFrame(noisefree_data)  
df6['ColumnA'] = df6[df6.columns[1:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)
noisefree_dataset = df6['ColumnA'] 
label_tr = label.head(56340)

final_noisefree_dataset = pd.concat([noisefree_dataset, label_tr.reindex(label_tr.index)], axis=1)


######################################testing dataset ######################################

testing_data = []
for i in range(len(df3)):
    aa = df3.iloc[i]
    mystr = str(aa)
    wordList = re.sub("[^\w]", " ",  mystr).split()
    testing_data.append(wordList)
    
df7 = pd.DataFrame(testing_data)  
df7['ColumnA'] = df7[df7.columns[1:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)
testing_data = df7['ColumnA']
########################
df_label = df['label']
index = df_label.tail(1000)

index_data = []
for i in range(len(index)):
    aa = index.iloc[i]
    mystr = str(aa)
    wordList = re.sub("[^\w]", " ",  mystr).split()
    index_data.append(wordList)
    
df8 = pd.DataFrame(index_data)  
df8['ColumnA'] = df8[df8.columns[1:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)
label_data = df8[0]


final_testing_dataset = pd.concat([testing_data, label_data.reindex(label_data.index)], axis=1)
########### final dataset #################
final_testing_dataset = final_testing_dataset.rename(columns={"ColumnA": "text_data", 0: "label"})
final_noisefree_dataset = final_noisefree_dataset.rename(columns={"ColumnA": "text_noisefree", "label": "label"})
final_noisy_dataset = final_noisy_dataset.rename(columns={"ColumnA": "text_noisy", "label": "label"})


print('final_testing_dataset->',final_testing_dataset)
print('final_noisefree_dataset->',final_noisefree_dataset)
print('final_noisy_dataset->',final_noisy_dataset)