!pip install cleantext
import pandas as pd 
import calendar

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from gensim.corpora import Dictionary

from gensim import  models

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from pprint import pprint

import cleantext

from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer()
import nltk 

from tqdm import tqdm

#metadata
metadata=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
print('shape' ,metadata.shape)
print('info',metadata.info())

#journals analysis
cols = ['journal']
data=metadata[metadata[cols].notna().all(1)]
print(data.shape)
df=data['journal'].value_counts()
x=df.iloc[[0,1,2,3,4,5,6,7,8,9]]
x.plot.pie(figsize = (10,10),autopct = '%.2f%%',
 title = 'Top Ten Journals')
plt.title("Top Ten Journals", bbox={'facecolor':'0.9', 'pad':5})
plt.show()

    
#publication year analysis  
cols = ['publish_time']
data=metadata[metadata[cols].notna().all(1)]
print(data.shape)
listd=[]
for d in data['publish_time'].tolist():
    d=str(d)
    d=d.replace("-"," ")
    listd.append(str(d).strip().split()[0])
#del data['publish_time']
data['publish_year'] = listd

df=data['publish_year'].value_counts()
x=df.iloc[list(range(len(df)))]
x.plot(kind='bar',figsize=(20,20),x='year',y='number of publications')
plt.title("Number of publications per year", bbox={'facecolor':'0.9', 'pad':5})
plt.show()


#publication 2020 year analysis
#print(data['publish_time'].tolist())
dic=dict((v,k) for k,v in enumerate(calendar.month_abbr))
listd=[]
for d in data['publish_time'].tolist():
    d=str(d).strip()
    d=d.replace("-"," ")
    z=d.split()
    if len(z)>1 and z[0]=='2020':
        m=z[1]
        if m.isdigit():
            m=int(m)
            listd.append(calendar.month_name[m])
        else:
            listd.append(m)
                

xlabel=[calendar.month_name[i] for i in range(1,12)]
ylabel=[listd.count(i) for i in xlabel]
plt.figure(figsize=(10,10))
plt.barh(xlabel, ylabel)
plt.xlabel('Number of Publications', fontsize=10)
plt.ylabel('Month', fontsize=10)
#plt.title('Number of publications in each month in 2020')
plt.title("Number of publications in each month in 2020", bbox={'facecolor':'0.9', 'pad':5})

plt.show()

#author analysis
cols = ['authors']
data=metadata[metadata[cols].notna().all(1)]
print(data.shape)
df=data['authors'].value_counts()
x=df.iloc[[0,1,2,3,4,5,6,7,8,9]]
x.plot.pie(figsize = (10,10),autopct = '%.2f%%',
 title = 'Top Ten authors')
plt.title("Top Ten authors", bbox={'facecolor':'0.9', 'pad':5})
plt.show()

#get titles and abstracts
metadata['titleabstract']=metadata.title+'. '+metadata.abstract
#remove NaN from both titles and abstracts  
cols = ['titleabstract']
data=metadata[metadata[cols].notna().all(1)]
print("Total Number of papers to be processed=",len(list(data['titleabstract'])))  

print("PreProcessing Progress:")  

#processing   
listdocs=[]
tags=['NN','NNS','NNP','NNPS']
bar=tqdm(list(data['titleabstract']))
for d in bar :
    docclean=cleantext.clean(d, extra_spaces=True,stopwords=True,lowercase=True,numbers=True,punct=True)
    text = nltk.word_tokenize(docclean)
    tagged_text=nltk.pos_tag(text)
    #filter out words with length=1 or 2
    words=[]
    for w in tagged_text:
        if len(w[0])>2 and w[1] in tags:
            words.append(lemmatizer.lemmatize(w[0]))      
    listdocs.append(words)
    pass
  
dct = Dictionary(listdocs)
dct.filter_extremes(no_below=1000, no_above=0.5)
dct.save('dct.dict')
dd=[dct.doc2bow(d) for d in listdocs]
print(dct)
tdfreq=dict()

tdfreq={}

for i in range(len(dct)):
    tdfreq[dct[i]]=dct.dfs[i]

l = sorted(tdfreq.items() , reverse=True, key=lambda x: x[1])

wordcloud = WordCloud(background_color='white',
                      width=1500,
                      height=1000
                      ).generate_from_frequencies(tdfreq)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Wordcloud of the used vocabulary", bbox={'facecolor':'0.9', 'pad':5})
plt.show()


#show the topmost 50 terms
xlabel=[l[i][0] for i in range(1,50)]
ylabel=[l[i][1] for i in range(1,50)]
plt.figure(figsize=(25,15))
plt.barh(xlabel, ylabel)
plt.xlabel('Number of documents', fontsize=10)
plt.ylabel('Term', fontsize=10)
#plt.title('Number of publications in each month in 2020')
plt.title("Number of papers where the topmost 50 terms appear", bbox={'facecolor':'0.9', 'pad':5})

plt.show()

#Create Train and save doc2vec models (model0=DBOW, model1=PV-DM)
print("---------CREATE AND TRAIN DOC2VEC MODELS----------")
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(listdocs)]
model0 = Doc2Vec(documents, dm=0, vec_size=25, window=2, min_count=5, workers=4)
model1 = Doc2Vec(documents, dm=1, vec_size=25, window=2, min_count=5, workers=4)

max_epochs = 10
bar=tqdm(range(max_epochs))
for epoch in bar:
    #Train with 10 epochs
    model0.train(documents, total_examples=model0.corpus_count, epochs=model0.iter)
    model1.train(documents, total_examples=model1.corpus_count, epochs=model1.iter)
    pass
#save the models    
model0.save("doc2vecmodel0.dict")
model1.save("doc2vecmodel1.dict")
print("------MODELS CREATED AND SAVED---------")

#Testing
print("-------------------Choose a Doc2Vec Model------------")
print("1- Distributed Bag Of Words (DBOW)")
print("2- Distributed Memory (PV-DM)")
choosenmodel = input("Choose a model:")
if choosenmodel==1:
    model = Doc2Vec.load("doc2vecmodel0.dict")
else:
    model = Doc2Vec.load("doc2vecmodel1.dict")
print("------LOAD MODEL-------")
tags=['NN','NNS','NNP','NNPS']
TASKS=[
'What is known about transmission, incubation, and environmental stability?',
'What do we know about COVID-19 risk factors?',
'What do we know about virus genetics, origin, and evolution?',
'What do we know about vaccines and therapeutics?',
'What do we know about non-pharmaceutical interventions?',
'What has been published about medical care?',
'What do we know about diagnostics and surveillance?',
'What has been published about information sharing and inter-sectoral collaboration?',
'What has been published about ethical and social science considerations?']

listtasks=[]

for d in TASKS:
    docclean=cleantext.clean(d, extra_spaces=True,stopwords=True,lowercase=True,numbers=True,punct=True)
    text = nltk.word_tokenize(docclean)
    tagged_text=nltk.pos_tag(text)
    #filter out words with length=1 or 2
    words=[]
    for w in tagged_text:
        if len(w[0])>2 and w[1] in tags:
            words.append(lemmatizer.lemmatize(w[0]))      
    listtasks.append(words)

print("------------------------ TASK KEY WORDS ------------------------")
i=0
for t in listtasks:
    print("Task",i+1," :")
    print(listtasks[i])
    i=i+1
    
taskdocs = [TaggedDocument(doc, [i]) for i, doc in enumerate(listtasks)]



testdocs=[]
for t in taskdocs:
    testdocs.append(model.infer_vector(t.words))


print("----------Visualization with t-SNE----------")

from sklearn.decomposition import TruncatedSVD


doc_embeddings = model.docvecs.vectors_docs[:len(model.docvecs)]


#tsne = TSNE(perplexity=30,n_components=3, random_state=0)
#tsne = TSNE(perplexity=15)
tsne=TruncatedSVD(n_components=15, n_iter=10, random_state=42)
Y = tsne.fit_transform(doc_embeddings)
Y1 = tsne.fit_transform(testdocs)

plt.figure(figsize=(15,15))
x_coords1 = Y[:, 0]
y_coords1 = Y[:, 1]

x_coords2 = Y1[:, 0]
y_coords2 = Y1[:, 1]
# display scatter plot
plt.scatter(x_coords1, y_coords1)
plt.scatter(x_coords2, y_coords2)
plt.title("t-SNE reduction for documents embeddings visualization (tasks are red circles and blue circles are our papers", bbox={'facecolor':'0.9', 'pad':5})
plt.show()

import csv

print("----------The 10 topmost similar papers for each task----------") 
for i in range(9):
    print("********Task",i+1,"**********")
    vector = model.infer_vector(listtasks[i])
    sim = model.docvecs.most_similar([vector],topn=20)
    filename="task"+str(i)+".csv"
    csv_file=open(filename, mode='w',encoding="utf-8",newline='') 
    writer = csv.writer(csv_file)
    fieldnames = ['doi','source_x','publish_time','journal','title','abstract']
    writer.writerow(fieldnames)
    
    for elem in sim:
        row=[]
        row.append(data['doi'].tolist()[elem[0]])
        row.append(data['source_x'].tolist()[elem[0]])
        row.append(data['publish_time'].tolist()[elem[0]])
        row.append(data['journal'].tolist()[elem[0]])
        row.append(data['title'].tolist()[elem[0]])
        row.append(data['abstract'].tolist()[elem[0]])
        writer.writerow(row)
    csv_file.close()
    resultdata=pd.read_csv(filename)
    display(resultdata)

#other task
newtask = input("Enter your new task: ") 

#preprocess new task
docclean=cleantext.clean(newtask, extra_spaces=True,stopwords=True,lowercase=True,numbers=True,punct=True)
text = nltk.word_tokenize(docclean)
tagged_text=nltk.pos_tag(text)
#filter out words with length=1 or 2
words=[]
for w in tagged_text:
    if len(w[0])>2 and w[1] in tags:
        words.append(lemmatizer.lemmatize(w[0])) 

vector = model.infer_vector(words)
sim = model.docvecs.most_similar([vector],topn=20)
filename="newtask.csv"
csv_file=open(filename, mode='w',encoding="utf-8",newline='') 
writer = csv.writer(csv_file)
fieldnames = ['doi','source_x','publish_time','journal','title','abstract']
writer.writerow(fieldnames)
    
for elem in sim:
    row=[]
    
    row.append(data['doi'].tolist()[elem[0]])
    row.append(data['source_x'].tolist()[elem[0]])
    row.append(data['publish_time'].tolist()[elem[0]])
    row.append(data['journal'].tolist()[elem[0]])
    row.append(data['title'].tolist()[elem[0]])
    row.append(data['abstract'].tolist()[elem[0]])
    writer.writerow(row)

csv_file.close()
resultdata=pd.read_csv("newtask.csv")
display(resultdata)
