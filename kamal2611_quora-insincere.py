# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np
fram = pd.read_csv("../input/train.csv")

print(fram.columns)

print(fram.groupby('target').count())



print(1225312 / (1225312 + 80810))
from nltk.tokenize import TreebankWordTokenizer

def tokenize(strq):

    rt = TreebankWordTokenizer().tokenize(strq)

    rtstr = ' '.join(rt)

    return rt , rtstr



print(tokenize('How is this day?'))
def makeVocab(datafram):

    vocab0 = list()

    vocab1 = list()

    dict0 = dict()

    dict1 = dict()

    for _ , data in datafram.iterrows():

        curTxt = data['question_text'].lower() 

        curLab = data['target']

        

        tok , curTxt = tokenize(curTxt)

        #print(tok , curTxt )

        if int(curLab) == 0:

            vocab0.append(curTxt)

            for ctk in tok:

                if ctk not in dict0:

                    dict0[ctk] = 0

                dict0[ctk] = dict0[ctk] + 1 

        elif int(curLab) == 1:

            vocab1.append(curTxt)

            for ctk in tok:

                if ctk not in dict1:

                    dict1[ctk] = 0

                dict1[ctk] = dict1[ctk] + 1    

    

    busttrm = 300

    for ky in dict0.keys():

        if ky not in dict1:

            dict0[ky] = dict0[ky] + busttrm 

        

            

    for ky in dict1.keys():

        if ky not in dict0:

            dict1[ky] = dict1[ky] + busttrm

    

    import operator

    sorted_dict0 = sorted(dict0.items(), key=operator.itemgetter(1) , reverse = True)

    sorted_dict1 = sorted(dict1.items(), key=operator.itemgetter(1), reverse = True)

    return sorted_dict0 , sorted_dict1
sorted_dict0 , sorted_dict1 = makeVocab(fram)
finalVocab = set()

cnt = 0



for wrd , _ in sorted_dict0:

    if cnt == 40000:

        break

    if True and wrd not in finalVocab:

        cnt = cnt + 1

        finalVocab.add(wrd)

        

cnt = 0

for wrd,_ in sorted_dict1:

    if cnt == 40000:

        break;

    if (True) and (wrd not in finalVocab):

        cnt = cnt + 1

        finalVocab.add(wrd)
print('openai' in finalVocab)
from nltk.tokenize import TreebankWordTokenizer

def vocabTokenize(strq):

    tr = list()

    rt = TreebankWordTokenizer().tokenize(strq)

    for trm in rt:

        if trm in finalVocab:

            tr.append(trm)

    rtstr = ' '.join(tr)

    return rtstr



print(vocabTokenize('How is this day?'))

def makeKerasSequence(datafram):

    vocab = list()

    wordlist = datafram.question_text.str.lower()

    mx = 0

    for currow in wordlist:

        cursen = vocabTokenize(currow)

        if len(cursen) > mx:

            mx = len(cursen)

        vocab.append(cursen)

    print(mx)  

    return vocab
import operator

def printCntVal(vocab):

    cntdict = dict()

    for cur in vocab:

        if len(cur) not in cntdict:

            cntdict[len(cur)] = 0 

        cntdict[len(cur)] = cntdict[len(cur)] + 1

    

    cntdict = sorted(cntdict.items(), key=operator.itemgetter(1) , reverse = True)

    print(cntdict)
from keras.models import Model , Sequential

from keras.layers import Input , Flatten , Bidirectional ,Dropout, CuDNNGRU , GlobalMaxPool1D , Dense , MaxPooling1D , Conv1D , Embedding 

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.engine.input_layer import Input

from keras.utils import to_categorical
def conv1DModel(vocab_size , outdim , maxln):

    model = Sequential()

    model.add(Embedding(vocab_size + 1, outdim))

    model.add(Conv1D(128 , 5 ,activation='relu'))

    model.add(MaxPooling1D(5))

    model.add(Conv1D(128 , 5 , activation='relu'))

    model.add(MaxPooling1D(5))

    model.add(Conv1D(128 , 5 , activation='relu'))

    model.add(MaxPooling1D(6))

    model.add(Flatten())

    model.add(Dense(128 , activation='relu'))

    model.add(Dense(1 , activation='sigmoid'))

    

    model.compile(loss = 'binary_crossentropy',

                 optimizer = 'rmsprop')

    

    return model
def conv1DModelFn(vocab_size , outdim , maxln):

    inp = Input(shape = (maxln, ) , dtype='int32')

    xt = Embedding(vocab_size + 1, outdim)(inp)

    x = Conv1D(128, 5, activation='relu')(xt)

    x = MaxPooling1D(3)(x)

    x = Conv1D(128, 5, activation='relu')(x)

    x = MaxPooling1D(3)(x)

    x = Conv1D(128, 5, activation='relu')(x)

    x = MaxPooling1D(5)(x)  # global max pooling

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)

    prd = Dense(1, activation='sigmoid')(x)

    

    model = Model(inp , prd)

    

    model.compile(loss = 'binary_crossentropy',

                 optimizer = 'rmsprop' , metrics = ['accuracy'])

    

    return model
def lstmModel(vocab_size , outdim , maxln):

    inp = Input(shape = (maxln, ) , dtype='int32')

    xt = Embedding(vocab_size + 1, outdim)(inp)

     

    #xt = Embedding(vocab_size + 1, outdim , 

    #               weights=[rpMat],trainable = False)(inp)

    #x = Dropout(.3)(xt)

    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(xt)

    x = GlobalMaxPool1D()(x)

    x = Dense(16, activation='relu')(x)

    x = Dropout(.1)(x)

    prd = Dense(1, activation='sigmoid')(x)

    

    model = Model(inp , prd)

    

    model.compile(loss = 'binary_crossentropy',

                 optimizer = 'adam' , metrics = ['accuracy'])

    

    return model
def prepareDataSequence(tokenizer , dataList , maxLn):

    dataSq = tokenizer.texts_to_sequences(dataList)

    dataSq = pad_sequences(dataSq , maxLn)

    wordInx = tokenizer.word_index

    return dataSq , wordInx
lab = fram.target.values

#lab = lab.reshape(lab.shape[0] , 1)

#lab = to_categorical(lab , 2)

print(lab.shape)
txtStor = makeKerasSequence(fram)

printCntVal(txtStor)
vocab_size = 80000

maxLn = 100

outdim = 300

tokenizer = Tokenizer(vocab_size)

tokenizer.fit_on_texts(txtStor)





'''

print(wordInx)

model = conv1DModel(vocab_size , outdim , maxLn)

print(model.summary)



model.fit(dataSq , lab , epochs = 2)

'''
dataSq , wordInx = prepareDataSequence(tokenizer , txtStor , maxLn)
def posDataInx(lab):

    inx_0 = list()

    inx_1 = list()

    cur = 0

    for clb in lab:

        if clb[0] == 1:

            inx_0.append(cur)

        elif clb[1] == 1:

            inx_1.append(cur)

        cur = cur + 1

    return inx_0, inx_1
def genData(dataSq , lab):

    inx_0 , inx_1 = posDataInx(lab)

    np.random.shuffle(inx_0)

    np.random.shuffle(inx_1)

    train_x = list()

    train_y = list()

    test_x = list()

    test_y = list()

    

    for tstCn in range(5000):

        test_x.append(dataSq[inx_0[tstCn]])

        test_y.append(lab[inx_0[tstCn]])

        

    for tstCn in range(5000):

        test_x.append(dataSq[inx_1[tstCn]])

        test_y.append(lab[inx_1[tstCn]])

        

        

    for tstCn in range(5000 , 60000):

        train_x.append(dataSq[inx_0[tstCn]])

        train_y.append(lab[inx_0[tstCn]])

        

    for tstCn in range(5000 ,len(inx_1)):

        train_x.append(dataSq[inx_1[tstCn]])

        train_y.append(lab[inx_1[tstCn]])    

    

    return train_x , train_y ,  test_x , test_y

    
print(dataSq.shape)

print(txtStor[0])

print(wordInx['did'])



#train_x , train_y , test_x , test_y = genData(dataSq , lab)





#scor = model.evaluate(test_x, test_y, batch_size=256)
model = lstmModel(len(wordInx) , outdim , maxLn)

model.fit(dataSq , lab , epochs = 2 , batch_size = 512 , class_weight={0:.4 , 1:1})
testDataFram = pd.read_csv("../input/test.csv")
testdata = makeKerasSequence(testDataFram)

testDataSq , wordInx = prepareDataSequence(tokenizer , testdata , maxLn)
printCntVal(testdata)
scor = model.predict(testDataSq, batch_size=256)

print(scor)
#yout = np.argmax(scor , axis = 1)

yout = (scor > .5).astype(int)#yout.reshape(testDataFram.shape[0])

yout = yout.reshape(yout.shape[0])

print(yout)


#yout = np.ones((testDataFram.shape[0] , ) , dtype = 'int32')

#yout = np.random.randint(2, size=testDataFram.shape[0])

print(testDataFram['qid'])

print(yout)
pred_df = pd.DataFrame({'qid':testDataFram['qid'],'prediction':yout}, columns=["qid" , "prediction"])

#print(pred_df)

pred_df.to_csv("submission.csv" , index=False)
pred_df.groupby('prediction').count()