import fastText
import wheel
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
from io import StringIO
import re
full_train = pd.read_csv('../input/edsa-mbti/train.csv')
test = pd.read_csv('../input/edsa-mbti/test.csv')
full_train.head()
full_train['posts'][0]
mind = []
energy = []
nature = []
tactics = []

for observation in range(len(full_train['type'])):
    ie = 2
    ns = 2
    tf = 2
    jp = 2
    grouping = full_train['type'][observation]
    if grouping[0] == 'E':
        ie = 1
    if grouping[0] == 'I':
        ie = 0
    if grouping[1] == 'N':
        ns = 1       
    if grouping[1] == 'S':
        ns = 0
    if grouping[2] == 'T':
        tf = 1
    if grouping[2] == 'F':
        tf = 0
    if grouping[3] == 'J':
        jp = 1
    if grouping[3] == 'P':
        jp = 0
    mind.append(ie)
    energy.append(ns)
    nature.append(tf)
    tactics.append(jp)
    
full_train['mind'] = mind
full_train['energy'] = energy
full_train['nature'] = nature
full_train['tactics'] = tactics

full_train.head()
#This is just some basic preprocewssing so to make the words easier to process.
#We do the same to the test dataset so that later we are classifying data that is as similar as possible.

full_train['posts'] = full_train['posts'].replace('\n',' ', regex=True).replace('\t',' ', regex=True).replace('\|\|\|',' ', regex=True).replace(':',' ', regex=True)
test['posts'] = test['posts'].replace('\n',' ', regex=True).replace('\t',' ', regex=True).replace('\|\|\|',' ', regex=True).replace(':',' ', regex=True)
def links(X):
    cleaned = ''
    x2 = X.split()
    for i in range(len(x2)):
        sep = x2[i]
        s = re.sub(r"\||^img*|^http.*|^'http.*|\"|^:www.*|^.:www.*|^www|^http\S+|^https\S+|^www\S+|^http?:\/\/.*[\r\n]*|^www?:\/\/.*[\r\n]*|^:www:\/\/.*[\r\n]*|^www*|^www.*|^ www.*^:|--|\*|^\/|\)|\(|^\\|\\|\/|\/|\+|http$|\.|png$|jpg$|jpeg$|com$", '', sep)
        s = re.sub(r'^www\S+', '', s)
        s = re.sub(r'^iphotobucket\S+', '', s)
        s = re.sub(r'\d+', '', s)
        slow = s.lower()
        cleaned = cleaned + ' ' + s
    return cleaned

#Code commented out to prevent both 'links' and 'clean' functions running as only one should be run at a time.
# timer = 0
# tempNew = []
# testNew = []
# for u in range(len(full_train['posts'])):
#     tempNew.append(links(full_train.posts[u]))
#     if ((u / len(full_train['posts']))*100) - timer >= 10:
#         timer += 10
#         print('Processing "train portion" stopwords is ' + str(timer) +'% done...')
# print('Train portion completed!!!')    
#Commented out to prevent 'links' function from being called because 'clean' function is going to run and only one should be run when processing.
# timer = 0
# for tu in range(len(test['posts'])):
#     if ((tu / len(test['posts'])*100)) - timer >= 10:
#         timer += 10
#         print('Processing "test portion" stopwords is ' + str(timer) +'% done...')
#     testNew.append(links(test.posts[tu]))
# print('Test portion completed!!!')           
def clean(X):
    cleaned = ''
    x2 = X.split()
    for i in range(len(x2)):
        sep = x2[i]
        s = re.sub(r"\||^img*|^http.*|^'http.*|\"|^:www.*|^.:www.*|^www|^http\S+|^https\S+|^www\S+|^http?:\/\/.*[\r\n]*|^www?:\/\/.*[\r\n]*|^:www:\/\/.*[\r\n]*|^www*|^www.*|^ www.*^:|--|\*|^\/|\)|\(|^\\|\\|\/|\/|\+|http$|\.|png$|jpg$|jpeg$|com$", '', sep)
        s = re.sub(r'^www\S+', '', s)
        s = re.sub(r'^iphotobucket\S+', '', s)
        s = re.sub(r'\d+', '', s)
        slow = s.lower()
        if slow not in ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp', 'isfp', 'istp', 'isfj',\
                     'istj', 'estp', 'esfp', 'estj', 'esfj','a', '"', "'", '...', ':', ';', "-", '_', \
                    'about', 'above', 'after', 'again', 'all', 'an', 'and', 'are', 'aren', "aren't", \
                    'as', 'at', 'be', 'because', 'been', 'between','both','but','by','can','during','each','few','for',\
                    'from','further', 'in','into', 'is','if', 'it',"it's",'its','itself','just','ll','m','ma',\
                    'most','mustn',"mustn't", 'needn',"needn't",'now','of','off','on','once','only',\
                    'or', 'out','shan',"shan't", 'so','some','such','than','that', '-', '_', '_____',\
                    'the', 'then', 'there','until', 've', 'while', ',' ,"''",'#', '--']:
            if slow == 'i':
                cleaned = cleaned + ' ' + s
            else:
                cleaned = cleaned + ' ' + slow
    return cleaned
timer = 0
tempNew = []
testNew = []
for u in range(len(full_train['posts'])):
    tempNew.append(clean(full_train.posts[u]))
    if ((u / len(full_train['posts']))*100) - timer >= 10:
        timer += 10
        print('Processing "train portion" stopwords is ' + str(timer) +'% done...')
print('Train portion completed!!!')    
timer = 0
for tu in range(len(test['posts'])):
    if ((tu / len(test['posts'])*100)) - timer >= 10:
        timer += 10
        print('Processing "test portion" stopwords is ' + str(timer) +'% done...')
    testNew.append(clean(test.posts[tu]))
print('Test portion completed!!!')      
all_cleaned_with_binaries = full_train.drop(['posts'], axis = 1)
all_cleaned_with_binaries['posts'] = tempNew
all_cleaned_with_binaries.head()
#Miimic operations on test corpus to keep the data similar for a better comparison.
#We also remove the "id" column because later on when we pass it to FastText to classify, FastText will require for there to be only 
#one data column.
test['posts'] = testNew
test = test.drop(['id'], axis =1)
print(test.head())
test.to_csv(r'test_final.txt', index=False, sep=' ', header=False)

#add prefix to labels for use in FastText later. FstText looks for anything beginning with "__label__" tag to identify categories for classification.

just_everything = pd.DataFrame()
just_everything['fasttext_labelled'] = '__label__' + full_train['type'].astype(str)
just_everything['content'] = tempNew
just_everything = just_everything.sample(frac=1)
#The the data set is shuffled so that later we don't need to use sklearn train_test in order to create a train and test set, we can just 
#can easily make new train test sets by saving n head rows to train and n head rows to test or n tail rows to train and n head rows to test. 
#this made more sense to me as I would be running FastText in a python shell and so this method would be easy on the fly.
just_everything = just_everything.sample(frac=1)
just_everything.head()
#Labels and data saved in a text file to be read by FastText
just_everything.to_csv(r'all_labelled.txt', index=False, sep=' ', header=False)
all_cleaned_with_type = all_cleaned_with_binaries.copy()
all_cleaned_with_binaries = all_cleaned_with_binaries.drop(['type'],axis=1)
all_cleaned_with_type.head()
#This function returns the counts of how many instances there are of each binary label in the corpus.
def minimum(lis):
    count = 0
    alt = 0
    for y in lis:
        if y == 1:
            count += 1
        else:
            alt += 1
    return min(count, alt)
def maximum(lis):
    count = 0
    alt = 0
    for y in lis:
        if y == 1:
            count += 1
        else:
            alt += 1
    return max(count, alt)
#This cell creates a dictionary containing the number of observations for each personality type.
counts = {}
temp_value = 0
temp_title = ''
data = {}
for w in range(len(all_cleaned_with_type['type'])):
    if all_cleaned_with_type['type'][w] not in counts.keys():
     #   print(all_cleaned_with_type['type'][w])
        counts[all_cleaned_with_type['type'][w]] = 1
    else:
        counts[all_cleaned_with_type['type'][w]] += 1
        
for unit in counts.keys():
    if counts[unit] > temp_value:
        temp_value = counts[unit]
        temp_title = unit
print(temp_title, temp_value)     
print(counts)
ei_min = minimum(all_cleaned_with_binaries.mind)
ei_max = maximum(all_cleaned_with_binaries.mind)
print(ei_min, ei_max)
ns_min = minimum(all_cleaned_with_binaries.energy)
ns_max = maximum(all_cleaned_with_binaries.energy)
print(ns_min, ns_max)
tf_min = minimum(all_cleaned_with_binaries.nature)
tf_max = maximum(all_cleaned_with_binaries.nature)
print(tf_min, tf_max)
jp_min = minimum(all_cleaned_with_binaries.tactics)
jp_max = maximum(all_cleaned_with_binaries.tactics)
print(jp_min, jp_max)
def data_balance_min(index, letter_a, letter_b, data, num):
    print(num, ' observations of each trait')
    select = []
    sec = []
    first = 0
    second = 0
    i = 0
    while len(select) < num:
        if index[i] == 1:
            select.append(['__label__' + letter_a,  data[i]])
        i += 1
    i = 0    
    while len(select) < (2 * num):
        if index[i] == 0:
            select.append(['__label__' + letter_b, data[i]])
        i += 1    
    select = np.array(select)
    select = pd.DataFrame(select)
    select.columns=['Label','Data']
    select = select.sample(frac=1)
    return (select)

def data_balance_max(index, letter_a, letter_b, data, num):
    print(num, ' observations of each trait')
    select = []
    sec = []
    first = 0
    second = 0
    i = 0
    while len(select) < num:
        if index[i%(len(index))] == 1:
            select.append(['__label__' + letter_a,  data[i%(len(index))]])
        i += 1
    i = 0    
    while len(select) < (2 * num):
        if index[i%(len(index))] == 0:
            select.append(['__label__' + letter_b, data[i%(len(index))]])
        i += 1    
    select = np.array(select)
    select = pd.DataFrame(select)
    select.columns=['Label','Data']
    select = select.sample(frac=1)
    return (select)
ei_frame = data_balance_max(all_cleaned_with_binaries.mind, 'E', 'I', all_cleaned_with_binaries.posts, ei_max)
print(len(ei_frame))
ei_frame.head()
#Another split of labels for a binary classification of N and S.
ns_frame = data_balance_max(all_cleaned_with_binaries.energy, 'N', 'S', all_cleaned_with_binaries.posts, ns_max)
print(len(ns_frame))
ns_frame.head()
#Another split of labels for a binary classification of T and F
tf_frame = data_balance_max(all_cleaned_with_binaries.nature, 'T', 'F', all_cleaned_with_binaries.posts, tf_max)
print(len(tf_frame))
tf_frame.head()
#Another split of labels for a binary classification of J and P.
jp_frame = data_balance_max(all_cleaned_with_binaries.tactics, 'J', 'P', all_cleaned_with_binaries.posts, jp_max)
print(len(jp_frame))
jp_frame.head()
#The binary corpuses is saved for use in FastText later.
ei_frame.to_csv(r'ei_max.txt', index=False, sep=' ', header=False)
ns_frame.to_csv(r'ns_max.txt', index=False, sep=' ', header=False)
tf_frame.to_csv(r'tf_max.txt', index=False, sep=' ', header=False)
jp_frame.to_csv(r'jp_max.txt', index=False, sep=' ', header=False)
#When called this function will take a pair of binary personality traits as input and label a dataset's observations according to the appropriate 
#binary label.

def data_binarize(index, letter_a, letter_b, data):
    select = []
    sec = []
    first = 0
    second = 0
    i = 0
    while i < len(index):
        if index[i] == 1:
            select.append(['__label__' + letter_a,  data[i]])
        i += 1
    i = 0    
    while i < len(index):
        if index[i] == 0:
            select.append(['__label__' + letter_b, data[i]])
        i += 1    
    select = np.array(select)
    select = pd.DataFrame(select)
    select.columns=['Label','Data']
    select = select.sample(frac=1)
    return (select)
ei_split = data_binarize(all_cleaned_with_binaries.mind, 'E', 'I', all_cleaned_with_binaries.posts)
ei_split.head()
ns_split = data_binarize(all_cleaned_with_binaries.mind, 'N', 'S', all_cleaned_with_binaries.posts)
ns_split.head()
tf_split = data_binarize(all_cleaned_with_binaries.mind, 'T', 'F', all_cleaned_with_binaries.posts)
tf_split.head()
jp_split = data_binarize(all_cleaned_with_binaries.mind, 'J', 'P', all_cleaned_with_binaries.posts)
jp_split.head()
#The binary corpus is saved for use in FastText later.
ei_split.to_csv(r'ei_binary.txt', index=False, sep=' ', header=False)
ns_split.to_csv(r'ns_binary.txt', index=False, sep=' ', header=False)
tf_split.to_csv(r'tf_binary.txt', index=False, sep=' ', header=False)
jp_split.to_csv(r'jp_binary.txt', index=False, sep=' ', header=False)
#The binary classifications are uploaded so that they can be formatted for the Kaggle submission.
ei = pd.read_csv('../input/binaries-and-output-files/ei_best.txt', header=None)
ns = pd.read_csv('../input/binaries-and-output-files/ns_best.txt', header=None)
tf = pd.read_csv('../input/binaries-and-output-files/tf_best.txt', header=None)
jp = pd.read_csv('../input/binaries-and-output-files/jp_best.txt', header=None)
ei.columns=['label']
ns.columns=['label']
tf.columns=['label']
jp.columns=['label']
#Here we see the first few predictions made by FastText for the N/S split.
ns.head()
#If a non-binary classification is being submitted then it is uploaded here for formatting. It is commented out because we achieved 
#better results using the individual binary classifications.
# result_whole = pd.read_csv('best_whole.txt', header=None)
# result_whole.columns=['label']
#The predictions are analyzed and converted into the appropriate biary output needed for each of the columns.
def allocate_whole(predict):
    E = []
    N = []
    T = []
    J = []
    for q in predict.label.values:
        if q[9].upper() == 'E':
            E.append(q[11:])
        if q[9].upper() == 'I':
            E.append(q[11:])
        if q[10].upper() == 'N':
            N.append(q[11:])
        if q[10].upper() == 'S':
            N.append(q[11:])
        if q[11].upper() == 'T':
            T.append(q[11:])
        if q[11].upper() == 'F':
            T.append(q[11:])
        if q[12].upper() == 'J':
            J.append(q[11:])
        if q[12].upper() == 'P':
            J.append(q[11:])    
    output = pd.DataFrame()
    output['mind'] = E
    output['energy'] = N
    output['nature'] = T
    output['tactics'] = J
    output['id'] = output.index+1
    output = output[['id', 'mind', 'energy', 'nature', 'tactics']]
    return output
#The predictions are analyzed and converted into the appropriate biary output needed for each of the columns.
def allocate_split(predict):
    log = []
    for q in predict.label.values:
        if q[-1].upper() == 'E':
            log.append(1)
        if q[-1].upper() == 'I':
            log.append(0)
        if q[-1].upper() == 'N':
            log.append(1)
        if q[-1].upper() == 'S':
            log.append(0)
        if q[-1].upper() == 'T':
            log.append(1)
        if q[-1].upper() == 'F':
            log.append(0)
        if q[-1].upper() == 'J':
            log.append(1)
        if q[-1].upper() == 'P':
            log.append(0)    
    return log
#If a non-binary submission is being processed it is formatted here and not in the function below.
#results = allocate_whole(result_whole)

output = pd.DataFrame()
output['mind'] = allocate_split(ei)
output['energy'] = allocate_split(ns)
output['nature'] = allocate_split(tf)
output['tactics'] = allocate_split(jp)
output['id'] = output.index+1
output = output[['id', 'mind', 'energy', 'nature', 'tactics']]


print(len(output))
output.head()
#The output is saved so that it can be uploaded for assesment on Kaggle.
#If a non-binary classification was processed then the 'results' DataFrame would be substituted for the 'output' DataFrame.
output.to_csv(r'submission.txt', index=False, sep=',')
#This provides a link that you can open in another tab to download the output file. This saves you from having to commit the Kaggle notebook 
#and then leaving the session to retrieve the file. Also, Kaggle seems to not be saving output files at the time of submitting this notebook.

from IPython.display import HTML

def create_download_link(title = "Download Test file", filename = "submission.txt"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)
create_download_link(filename='submission.txt')
def even_bag(index, data, num):
    print(num, ' observations of each trait')
    select2 = []

    i = 0
    while len(select2) < num:
        if index[i][0] == 'E':
            select2.append(['__label__' + 'E',  data[i]])
        i += 1
    i = 0    
    while len(select2) < (2*num):
        if index[i][0] == 'I':
            select2.append(['__label__' + 'I', data[i]])
        i += 1    
    i = 0    
    while len(select2) < (3*num):
        if index[i][1] == 'N':
            select2.append(['__label__' + 'N', data[i]])
        i += 1    
    i = 0    
    while len(select2) < (4*num):
        if index[i][1] == 'S':
            select2.append(['__label__' + 'S', data[i]])
        i += 1 
    i = 0    
    while len(select2) < (5*num):
        if index[i][2] == 'T':
            select2.append(['__label__' + 'T', data[i]])
        i += 1    
    i = 0    
    while len(select2) < (6*num):
        if index[i][2] == 'F':
            select2.append(['__label__' + 'F', data[i]])
        i += 1  
    i = 0    
    while len(select2) < (7*num):
        if index[i][3] == 'J':
            select2.append(['__label__' + 'J', data[i]])
        i += 1  
    i = 0    
    while len(select2) < (8 * num):
        if index[i][3] == 'P':
            select2.append(['__label__' + 'P', data[i]])
        i += 1    
    select2 = np.array(select2)
    select2 = pd.DataFrame(select2)
    select2.columns=['Label','Data']
    select2 = select2.sample(frac=1)
    return (select2)

def equalizer(df, label, number):
    lis = []
    x = 0
    while len(lis)<number:
        if all_cleaned_with_type['type'][x%len(all_cleaned_with_type['type'])] == label:
            lis.append([all_cleaned_with_type['type'][x%len(all_cleaned_with_type['type'])], all_cleaned_with_type['posts'][x%len(all_cleaned_with_type['type'])]])
        x += 1
    return lis    
total_bag = []
for types in counts.keys():
    total_bag = total_bag + equalizer(all_cleaned_with_type, types, temp_value)
    print(types + ' now added!')
np_total_bag = np.array(total_bag)
tot_bag = pd.DataFrame(np_total_bag)
tot_bag = tot_bag.sample(frac=1)
print(tot_bag.head())
tot_bag = tot_bag.sample(frac=1)
tot_bag[0] = '__label__' +tot_bag[0]
tot_bag.head()
#tot_bag.to_csv(r'flawed_bag.txt', index=False, sep=' ',header=None)
number = min(ei_num, ns_num, tf_num, jp_num)
even_bag = even_bag(all_cleaned_with_type['type'] , all_cleaned_with_type['posts'].values, number)
#even_bag.to_csv(r'flawed_even_bag.txt', index=False, sep=' ', header=None)
#full.to_csv('myfile.csv',index=False)
from IPython.display import HTML

def create_download_link(title = "Download Test file", filename = "fully.txt"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)
create_download_link(filename='fully.txt')