# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
csv_data=pd.read_csv('/kaggle/input/image-captioning2040-images/file.csv')
print(csv_data.shape)
csv_data.head()
csv_data.drop(['image'],axis=1,inplace=True)
csv_data.head()
print('Image Count : ',len(os.listdir('/kaggle/input/image-captioning2040-images/file (1)/content/images')))
print((np.unique(csv_data.index)).shape)
csv_data['index']='image'+csv_data['index'].apply(str)+'.jpg'
csv_data.head()
import re
short_forms = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}
def preprocessText(text):
    
    txt=text.lower()
    txt=re.sub(r'[<>\:;?@#$%&\&\^*\(\)\\!\+/\[\]]','',txt)
    txt=' '.join([short_forms[i] if i in short_forms.keys() else i for i in txt.split()])
    txt=re.sub(r"'s'",'',txt)
    txt=re.sub('"','',txt)
    txt=re.sub(r'[\.,_-]',' ',txt)
    txt=' '.join(i for i in txt.split() if i.isalpha())
    txt=re.sub(r'[^a-zA-Z ]','',txt)
    
    return txt
csv_data['caption']='<start> '+csv_data['caption'].apply(preprocessText)+' <end>'
csv_data['caption'][0]
Images=os.listdir('/kaggle/input/image-captioning2040-images/file (1)/content/images')
Images[:5]
train_size=0.7
train_images=Images[:round(0.7*len(Images))]
test_images=Images[round(0.7*len(Images)):]
print(len(train_images),len(test_images))

train_captions=[csv_data[csv_data['index']==i]['caption'].values[0] for i in train_images]
test_captions=[csv_data[csv_data['index']==i]['caption'].values[0] for i in test_images]
print(len(train_captions),len(test_captions))
import cv2

img=cv2.imread('/kaggle/input/image-captioning2040-images/file (1)/content/images/'+train_images[5])
cvt_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.putText(cvt_img,train_captions[5],(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)
plt.imshow(cvt_img)
    
heights=[]
widths=[]
for img in train_images:
    h,w=(cv2.imread('/kaggle/input/image-captioning2040-images/file (1)/content/images/'+img,0)).shape
    heights.append(h)
    widths.append(w)

print('Median height : ',np.median(np.array(heights)),'widths : ',np.median(np.array(widths)))
median_height=400
median_width=600
max_cap_len=max([len(i.split()) for i in csv_data['caption']])
print(max_cap_len)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,BatchNormalization,Dropout,Flatten,LSTM,Embedding,Bidirectional,Input,TimeDistributed
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import plot_model,to_categorical
vgg16_model=VGG16()
vgg16_model.summary()
train_images_array=np.zeros((len(train_images),224,224,3),dtype=np.float32)
test_images_array=np.zeros((len(test_images),224,224,3),dtype=np.float32)

for i in range(len(train_images)):
    
    tr_image=cv2.imread('/kaggle/input/image-captioning2040-images/file (1)/content/images/'+train_images[i])
    cvt_tr_image=cv2.cvtColor(tr_image,cv2.COLOR_BGR2RGB)
    image=cv2.resize(cvt_tr_image,(224,224))
    train_images_array[i,:]=image/255.0
    
for i in range(len(test_images)):
    
    te_image=cv2.imread('/kaggle/input/image-captioning2040-images/file (1)/content/images/'+test_images[i])
    cvt_te_image=cv2.cvtColor(te_image,cv2.COLOR_BGR2RGB)
    image=cv2.resize(cvt_te_image,(224,224))
    test_images_array[i,:]=image/255.0
    
print(train_images_array.shape,test_images_array.shape)
     
conv_model=Model(vgg16_model.inputs,vgg16_model.layers[-2].output)
conv_model.summary()
from tqdm import tqdm

X_train_img=np.zeros((len(train_images),(conv_model.output.shape)[1]),dtype=np.float32)
X_val_img=np.zeros((len(test_images),(conv_model.output.shape)[1]),dtype=np.float32)

for i in tqdm(range(len(train_images))):
    
    img_feat=conv_model(train_images_array[i,:].reshape(1,224,224,3))
    img_feat=img_feat.numpy()
    X_train_img[i,:]=img_feat

for i in tqdm(range(len(test_images))):
    
    img_feat=conv_model(test_images_array[i,:].reshape(1,224,224,3))
    img_feat=img_feat.numpy()
    X_val_img[i,:]=img_feat
    
print(X_train.shape,X_val.shape)
tokenizer=Tokenizer(oov_token='<UNK>',)
tokenizer.fit_on_texts(train_captions)

X_train_cap=tokenizer.texts_to_sequences(train_captions)
X_val_cap=tokenizer.texts_to_sequences(test_captions)

cap_vocab_size=len(tokenizer.word_index)+1
print(cap_vocab_size)
print(len(X_train_cap),len(X_val_cap))

Inputs1,Inputs2,Outputs=[],[],[]

for i in tqdm(range(0,len(X_train_cap))):
    
    seq=X_train_cap[i]
    #print(len(seq))
    for j in range(1,len(X_train_cap[i])):
        #print(j)
        Inputs1.append(X_train_img[i])
        inp_seq,out_seq=seq[:j],seq[j]
        pad_inp_seq=pad_sequences([inp_seq],maxlen=max_cap_len)[0]
        out_seq=to_categorical(out_seq,num_classes=cap_vocab_size)
        
        Inputs2.append(pad_inp_seq)
        Outputs.append(out_seq)
        
Inputs1,Inputs2,Outputs=np.array(Inputs1),np.array(Inputs2),np.array(Outputs)

print(Inputs1.shape,Inputs2.shape,Outputs.shape)
        
    

Inputs3,Inputs4,Outputs2=[],[],[]

for i in tqdm(range(0,len(X_val_cap))):
    
    seq=X_val_cap[i]
    #print(len(seq))
    for j in range(1,len(X_val_cap[i])):
        #print(j)
        Inputs3.append(X_val_img[i])
        inp_seq,out_seq=seq[:j],seq[j]
        pad_inp_seq=pad_sequences([inp_seq],maxlen=max_cap_len)[0]
        out_seq=to_categorical(out_seq,num_classes=cap_vocab_size)
        
        Inputs4.append(pad_inp_seq)
        Outputs2.append(out_seq)
        
Inputs3,Inputs4,Outputs2=np.array(Inputs3),np.array(Inputs4),np.array(Outputs2)

print(Inputs3.shape,Inputs4.shape,Outputs2.shape)
        
    

enc_inputs=Input(shape=(4096,),name='Enc_Inputs')
dense1=Dense(300,activation='relu',name='Dense1')
enc_out=dense1(enc_inputs)
#enc_output=tf.keras.layers.RepeatVector(1,name='Enc_Output')(dense1_out)

dec_inputs=Input(shape=(max_cap_len,),name='Dec_Inputs')
embedding=Embedding(cap_vocab_size,300,trainable=True,name='Dec_Embedding')
dec_embedding=embedding(dec_inputs)

lstm=LSTM(300,name='Dec_LSTM')
lstm_out=lstm(dec_embedding)

combined_outputs=tf.add(enc_out,lstm_out)

dense2=Dense(256,activation='relu',name='Dense2')
desnse2_out=dense2(combined_outputs)

output_layer=Dense(cap_vocab_size,activation='softmax')
output=output_layer(desnse2_out)
model=Model([enc_inputs,dec_inputs],output)

model.summary()
plot_model(model,show_shapes=True)
model.compile(loss='categorical_crossentropy',optimizer='Adam')
history=model.fit([Inputs1,Inputs2],Outputs,epochs=10,batch_size=64,
          validation_data=([Inputs3,Inputs4],Outputs2))
def Image_caption(image):
    
    caption='start'
    
    plt.imshow(image)
    img=cv2.resize(image,(224,224))
    img=img/255.0
    
    img_feat=conv_model(img.reshape(1,224,224,3))
    word='start'
    
    while word!='end':
    
        start_token=tokenizer.texts_to_sequences([caption])[0]

        pad_seq=pad_sequences([start_token],maxlen=max_cap_len)

        pred_word=model.predict([img_feat,pad_seq])[0]
        word=tokenizer.index_word[np.argmax(pred_word)]
        
        caption+=' '+word
    
    return ' '.join([i for i in caption.split() if i not in ['start','end']])


Image_caption(test_images_array[0])
