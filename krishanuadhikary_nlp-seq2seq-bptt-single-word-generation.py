# Importing Libaries
import requests
from torch.backends import cudnn
import re
import spacy
import torch
import nltk
from matplotlib import pyplot
import pandas as pd,numpy as np
from torch.autograd import Variable
# Setting up device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device
# torch internal optimization
cudnn.benchmark = True
response = requests.get('http://www.gutenberg.org/files/11/11-0.txt')
text = response.text
text[:100]
len(text)
# Sentence Tokenizer
sentModel = spacy.load("en_core_web_sm",disable=['parser','tagger', 'ner'])
sentModel.add_pipe(sentModel.create_pipe('sentencizer'))
# Global cleaning of the text
def cleanText(text):
    text = text.strip()
    text = re.sub(r'\n+',' ',text)
    doc = sentModel(text)
    #text = ' '.join([str(token.lemma_.strip().lower()) for token in sentModel(text) if token.lemma_.strip()])
    #text = text.replace('-pron-','')
    textList = [senttoken.text.lower() for senttoken in doc.sents]
    return ' '.join(textList)
# All Sentensed Tokenized
textRefined,tokenList = [],[] 
textRefinedTemp=cleanText(text)
textRefined = textRefinedTemp
textRefined[:100]
# Importing Libraries
import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn
# Dataset Class Builder
class datasetClass(Dataset):
    
    def __init__(self,text):
        super(datasetClass,self).__init__()
        self.textList = text.split()
        self.text,self.char2idx,self.idx2char,self.textSplit,self.textOriginal,self.trainX,self.trainY= '','','','','','',''
        self.preprocessing()
        self.encoding()
        self.x,self.y= self.segmentation(self.text)
        self.x,self.y = self.x.to(device),self.y.to(device)
    
    # Clean function
    def clean(self,text):
        #text = unidecode(text)
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z +_]')
        text = re.sub(' +',' ',text)
        text = re.sub('\n+',' ',text)
        text = REPLACE_BY_SPACE_RE.sub(' ', text)
        text = BAD_SYMBOLS_RE.sub('', text)
        text = text.lower()
        text = re.sub(' +',' ',text)
        text = re.sub('\n+',' ',text)
        text.strip()
        return text
    
    # Character Encoding 
    def encoding(self,):
        self.idx2char = {idx:token for idx,token in enumerate(set(self.text.split()))}
        self.char2idx = {token:idx for idx,token in enumerate(set(self.text.split()))}
    

    # TrainX , TrainY dataset preperation
    def segmentation(self,text,ngram=10):
        '''
         Segemention for making windowed input of size (10) and output (1) , Total = 11
         Please check the variables (self.trainX,self.trainY) for further analysis.
        '''
        self.textSplit = [self.char2idx[token] for token in text.split()]
        self.textOriginal = [token for token in text.split()]
        self.trainX,self.trainY = zip(*[((self.textOriginal[idx:idx+(ngram)]),self.textOriginal[idx+ngram]) for idx,token in enumerate(self.textSplit[:-ngram])])
        x,y = zip(*[((self.textSplit[idx:idx+(ngram)]),self.textSplit[idx+ngram]) for idx,token in enumerate(self.textSplit[:-ngram])])
        x,y = torch.tensor(x,dtype=torch.long),torch.tensor(y,dtype=torch.long)
        return x,y
    
    # Cleaning for sentences
    def preprocessing(self,):
        self.text = ' '.join([self.clean(itemStr) for itemStr in self.textList])
     
    def  __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self,):
        return len(self.y)
# Model Building 
class Model(nn.Module):
    def __init__(self,vocabSize,embedDim,targetDim,sentenceLength):
        super(Model,self).__init__()
        self.sentenceLength = sentenceLength
        self.embeddingLayer1 = nn.Embedding(vocabSize,embedDim)
        self.lstmLayer1 = nn.LSTM(embedDim,100,num_layers=1)
        self.linearLayer1 = nn.Linear(100*self.sentenceLength,targetDim)
        self.logsoftmaxLayer1 = nn.LogSoftmax(dim=1)
        self.hidden = Variable(torch.zeros(1,self.sentenceLength,100,requires_grad=True)).to(device)
        self.cell = Variable(torch.zeros(1,self.sentenceLength,100,requires_grad=True)).to(device)
        
        
    def forward(self,x,hidden,cell):
        x = self.embeddingLayer1(x)
        x, (self.hidden, self.cell) = self.lstmLayer1(x,(self.hidden,self.cell))
        x = x.view(-1,self.sentenceLength*100)
        x = self.linearLayer1(x)
        x = self.logsoftmaxLayer1(x)
        return x,(self.hidden,self.cell)
def reinitialize(hidden,state,sentence_length):
    '''
    Reinitializing the layers hidden,state
    This will be used to carry on the hidden/state till certain epochs.
    '''
    #print(hidden.shape,state.shape)
    hidden = Variable(torch.zeros(1,sentence_length,100,requires_grad=True)).to(device)
    state = Variable(torch.zeros(1,sentence_length,100,requires_grad=True)).to(device)
    return hidden,state
traintext = textRefined[:int(0.8*len(textRefined))]
testtext = textRefined[int(0.8*len(textRefined)):]
datasetClassObj = datasetClass(textRefined)
datasetClassTestObj = datasetClass(testtext)
print('X1 : ',datasetClassObj.trainX[0],'Y1 : ',datasetClassObj.trainY[0])
print('X2 : ',datasetClassObj.trainX[1],'Y2 : ',datasetClassObj.trainY[1])
print('X3 : ',datasetClassObj.trainX[2],'Y3 : ',datasetClassObj.trainY[2])
print('Length of the sentence : ',len(datasetClassObj.trainX[0]))
# Model Configs
vocabSize = len(datasetClassObj.idx2char)
targetDim = len(datasetClassObj.char2idx)
sentenceLength = 10
embedDim = 10
epochs = 100
bs = 64
dataloaderClassObj = DataLoader(datasetClassObj,batch_size=bs,shuffle=False)
dataloaderClassTestObj = DataLoader(datasetClassTestObj,batch_size=bs,shuffle=False)
# Length of the vocab
print(len(datasetClassObj.char2idx))
print(len(datasetClassTestObj.char2idx))
datasetClassObj.x,datasetClassObj.y
modelObj = Model(vocabSize,embedDim,targetDim,sentenceLength)
modelObj.to(device)
x,y = next(iter(dataloaderClassObj))
# Forward Pass
hidden,cell = reinitialize(hidden=0,state=0,sentence_length=10)
_,(_,_) = modelObj(x,hidden,cell)
# Loss Function and Optimizer
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(modelObj.parameters(), lr=0.001)
# To detect anomaly for gradient error detection 
torch.autograd.set_detect_anomaly(True)
# Statefull LSTM
modelObj.hidden,modelObj.cell = reinitialize(hidden=0,state=0,sentence_length=10)
losses = []
for epoch in range(epochs):
    modelObj.train()
    for x,y in dataloaderClassObj:
        modelObj.hidden.detach_()
        modelObj.cell.detach_()
        ypred,(modelObj.hidden,modelObj.cell) = modelObj(x,modelObj.hidden,modelObj.cell)
        loss = loss_function(ypred,y)
        loss.backward()
        with torch.no_grad():
            optimizer.step()
            modelObj.zero_grad()
    print('epoch',epoch,loss)
    losses.append(loss)
# Training Loss
lossesCpu = [float(item.to('cpu')) for item in losses]
# Training Loss Graph
pyplot.plot(lossesCpu)
# Fitment Test
hidden,cell = reinitialize(hidden=0,state=0,sentence_length=10)
percentage = []
for x,y in dataloaderClassObj:
    modelObj.eval()
    ypred,(_,_) = modelObj(x,hidden,cell)
    predictedWord = [datasetClassObj.idx2char[int(item)] for item in torch.argmax(ypred,dim=1)]
    actualWord = [datasetClassObj.idx2char[int(item)] for item in y]
    percentage.append(sum([actualWord[index]== predictedWord[index] for index,word in enumerate(actualWord)])/len(actualWord)*100)
pyplot.plot(percentage)
# Unit Test with sample of length 10(mandatory)
# For best prediction the input should be in a batch of specific batchsize 

modelObj.eval()
inpTest =  ['the','gutenberg', 'ebook', 'of', 'alices', 'adventures', 'in', 'wonderland', 'by', 'lewis']
# To get the words in the vocab 
#datasetClassObj.char2idx
hidden,cell = reinitialize(hidden=0,state=0,sentence_length=10)
modelObj.eval()
xValid = torch.tensor([datasetClassObj.char2idx[token] for token in inpTest],dtype=torch.long).to(device)

print('Convert Text to Index : ', xValid)

ypredValid,(_,_) = modelObj(xValid.unsqueeze(0),hidden,cell)
ypredValid = torch.argmax(ypredValid,dim=1)

print('Prediction : ', ypredValid,'\n\n')

print('Input in Text : ',inpTest)
print('Prediction in Text : ', datasetClassObj.idx2char[int(ypredValid.to('cpu'))])
PATH = r'/kaggle/working/SingleWordPredictor'

torch.save({
            'epochs': epochs,
            'model_state_dict': modelObj.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)
len(list(datasetClassObj.char2idx.keys()))
wordEmbedding = {}
# Embeddings
modelObj.eval()
inpTest = list(datasetClassObj.char2idx.keys())
xTest = torch.tensor([datasetClassObj.char2idx[token] for token in inpTest],dtype=torch.long).to(device)
embeddingLookupFrame = pd.DataFrame(modelObj.embeddingLayer1(xTest).to('cpu').detach().numpy(),index=inpTest)
embeddingLookupFrame.head(10)
embeddingLookupFrame.loc['impossible'].values
searchWord = embeddingLookupFrame.loc['son'].values
def distance(row,searchWord):
    return  np.linalg.norm(row.values-searchWord,ord=2)
searchSeries = embeddingLookupFrame.apply(distance,args=(searchWord,),axis=1)
searchSeries.sort_values()[:10]