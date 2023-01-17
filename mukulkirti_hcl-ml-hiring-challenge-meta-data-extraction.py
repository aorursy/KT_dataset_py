import os
import re
import subprocess
#fun to install require packge

def install(name):
    subprocess.call(['pip', 'install', name])
try: 
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    import numpy as np
except:
    install('numpy')
    install('pandas')
    install('sklearn')
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    import numpy as np
os.listdir(r'/kaggle/input/')
#extract Data from notepad file
#create datadframe with two column, column 1: Msg , column 2: category
#category has been decided as 
"""
1 : Ending of title line
2 : Bill Table Heading
3 : Bill Tables Price and Asset Liability  
4 : Rest Text labeld as 4
"""
files = []#contain list of Files
first=[]# contain Sentence which belong to categ 1
second=[]# contain Sentence which belong to categ 2
third=[]# contain Sentence which belong to categ 3
four=[]# contain Sentence which belong to categ 4


for i in os.listdir(r'/kaggle/input/'):#loop to iterate each file in a folder
    if i.endswith('.txt'):#prosess only txt file
        x=open(r'/kaggle/input/'+i,'r')#open file
        g=0
        h=0
        temp=''
        
        for j in x.read().split('\n'):
            try:
                if(i==" "):#this codition to skip all white space in bigning of file
                    continue
                
                if(re.findall('20[0-9][0-9]',j) and g==0):# this condition to find end of end line of each bill (each end line of title ends with 2019 or 2018,or 2020 or 2017
                    first.append(j)#append each end line in first list
                    g=1
                    continue
                elif(re.findall('20[0-9][0-9]',j) and g==1): # find 2nd category line which is heading of table
                    second.append(j)
                    g=2
                    continue
                elif(g==2):
                    g=3
                    continue
                elif(g==3):# find all line having Comodity and its prise 
                    if('For the year' in j):#condition to make sure no more line in the bill belong to category 3
                        g=4
                    elif(h==1 and (ord(j.split()[-1][-1]) in range(ord('0'),ord('9')+1) or j.split()[-1][-1]=="-" or j.split()[-1][-1]==')')):##if asset is in more then one line the concatinate with previous line and append to third list
                        third.append(temp+j)
                        temp=''
                    elif(ord(j.split()[-1][-1]) in range(ord('0'),ord('9')+1) or j.split()[-1][-1]=="-" or j.split()[-1][-1]==')'):#append line to third list if it belongs to it
                        third.append(j)
                    elif(h==2):#help to concatinate line which are in more then one line
                        g==4
                    else:
                        temp=j
                        h=h+1                           
                if(g==4):# add fourth category line to fourth list
                    four.append(j)
            except:
                pass
#creating dataset of each list first second third and fourth
data=pd.DataFrame(first,columns=['Msg'])
data['t']=[1]*len(first)
data2=pd.DataFrame(second,columns=['Msg'])
data2['t']=[2]*len(second)
data3=pd.DataFrame(third,columns=['Msg'])
data3['t']=[3]*len(third)
data4=pd.DataFrame(four,columns=['Msg'])
data4['t']=[4]*len(four)
data=data.append(data2,ignore_index=True)
data=data.append(data3,ignore_index=True)
data=data.append(data4,ignore_index=True)
data
# data Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Msg'])
df2=pd.DataFrame(X.todense(),columns=vectorizer.get_feature_names())
df2
#NLP Process
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#Split Data set
X_train, X_test, y_train, y_test = train_test_split(df2, data['t'],random_state = 0)
#training
clf = MultinomialNB().fit(X_train, y_train)

#prediction on test data and compute conf_matrix
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
#save model
import pickle 
saved_model = pickle.dumps(clf) 
  
# Load the pickled model 
clf = pickle.loads(saved_model) 

"""========Prediction and information extraction using traind model========="""


#function to check(\d+)integer
def check(x2):
    idx1=9999
    idx2=9999
    if(',' in x2):return True
    if('(' in x2):
        idx1=x2.find(r'(')
        idx2=x2.find(r')')
    if(idx1!=9999):
        x2=x2[idx1+1:idx2]#ok
        if(x2.isdigit()):return True
        else:return False
    else:return False 
final=[]
final_data=pd.DataFrame()#
#function to convert Sentence into Dictionary which is desire output of question
def make_dict(sc,text,temp):#sc: Date in diffrent format , Text line to be converted in Dict, Temp :temp='' if data in single line, temp= previous line data if data exist in two lines
        d={}
        x=text.split()
        sc2=[]
        #chech weather date in dd/mm/yyyy format
        for i in sc:
            if('as' in sc):
                sc2.append(sc[-1])
                break
            if(re.findall(r'[0-9][0-9]/[0-9][0-9]/[0-9][0-9][0-9][0-9]', i)):
                sc2.append(i[-4:])
            else:
                sc2.append(i)
                
        sc=sc2#now date in proper format 
        key=''    
        key2=[]# it contain all key of dict
        st=0
        i=0
        #extract keys of dict
        for jj,i in enumerate(x[::-1]):
            y=False # check string
            if(')' in i): y=check(i)
            if((i[-1].isdigit() or y or i[0]=='-') and st==0):pass
            else:
                st=1
                key2.append(i)
        #extract values
        #valh contains value of dictionary
        idx=9999
        try:
            idx=sc[::-1].index('2019')
        except:
            pass
        if(idx==9999):
            valh=np.NaN
        else:
            valh=x[::-1][idx]
            if(',' in valh):
                indx=valh.index(',')
                valh=valh[:indx]+valh[indx+1:]
            if(')' in valh ):
                valh2='-'
                for i in valh:
                    ii=[str(va) for va in range(0,10)]
                    if(i in ii+['.']):
            
                        valh2+=i
                valh=valh2
            valh2=""
            for i in valh:
                ii=[str(va) for va in range(0,10)]
                if(i in ii+['.']+['-']):
                    valh2+=i
            valh=valh2
                
                        
            if(valh==''):
                valh=x[-1]
            try:
                if(valh!='-'):
                    
                    if('.' not in valh):
                        
                        valh=int(valh)
                    else:
                        valh=float(valh)
            except:
                pass
            valh=str(valh)
        #make Dictionary by key and value  
        if(temp!=''):
            tt=''
            d1={}
            for word in temp.split():
                tt+=word+' '
            tt=tt[:-1]
            d1[tt]=valh
            d= reduce(lambda x,y: dict(x, **y), (d, d1))
        
        for i in key2[::-1]:
            key+=i+' '
        key=key[:-1]
        d[key]=valh
        d = reduce(lambda x,y: dict(x, **y), (d,d))
        #print(d)
        return d
#============================================================================
#iterate each file
#Make outout in Desire format
from functools import reduce
path='/kaggle/input/'
for i in os.listdir(path):
    if i.endswith('.txt'):
        temp=0
        temp2=""
        first=0
        second=0
        sc=[]
        final_dict={'0':0}
        x=open(path+i,'r')
        #x=open('E:\HCL ML Challenge\HCL ML Challenge Dataset\X8Y4U51F.txt','r')
        for j in x.read().split('\n'):
            #read each line and predict category in [1,2,3,4]
            if(len(j.split())==0):
                continue
            
            if('Â£' in j.split() or 'Â£' in j.split()):#replace special currency symbol in its html code as mention in Question
                new_j=''
                for i_n in j:
                    if(i_n!='Â' and i_n!='£'):
                        new_j+=i_n
                    else:
                        if(i_n=='Â'):
                            new_j+='&#194;'
                        else:
                            new_j+='&#163;'
                j=new_j
               
            X_t = vectorizer.transform([j])# vectorize current processing line for prediction
            st=clf.predict(pd.DataFrame(X_t.todense()))
            if((st[0]==1 or len(re.findall('[0-9][0-9][0-9][0-9]',j.split()[-1]))==1) and first==0):# check weathe belongs to cat 1
                if(j.split()[-1].isdigit()):
                    first=1
                    continue
            
            elif(first==1 and second==0 and (st[0]==2 or ('2020' in j))):
                sc=j.split()
                second=1
                continue
            elif(second==1):
                second=2
            #extract information from each line belong to category 3 ( line with price infromation)
            elif(second==2):
                st=clf.predict(pd.DataFrame(X_t.todense()))
                if(st[0]==3 and (j.split()[-1][-1].isdigit() or j.split()[-1][-1]==r'-' or j.split()[-1][-1]==r')')):
                    if(temp==1):
                        if('Current assets' in temp2 or 'Fixed assets' in temp2 or 'Tssued share capital' in temp2 or 'Issued share capital' in temp2):# Add line when informaton is splited in two lines,it helps to find grand total of assets and liability
                            xx=make_dict(sc,j,temp2)#ok
                            #print(xx,final_dict)
                            final_dict = reduce(lambda xy,yx: dict(xy, **yx), (xx,final_dict))#add single asset or liability to final_dict with reduce fuction ,simply add new key to dictionary
                            
                        else:
                            final.append([i,temp2])#Append all asset and liability of a single file in final list
                            j=temp2+j
                            temp=0
                            temp2=''
                    xx=make_dict(sc,j,'')
                    #print(xx,final_dict)
                    final_dict = reduce(lambda xy,yx: dict(xy, **yx), (xx,final_dict))
                elif(temp==2):
                    temp=0
                    temp2=''
                    break
                else:
                    temp=+1
                    temp2=j
        del final_dict['0']
        print([i,final_dict])
        #create Data_Frame
        conv_final_dict=pd.DataFrame(data=list([i[:-4+1-1],final_dict]))#ok
        final_data=pd.concat([final_data,conv_final_dict.T], ignore_index=True)

#Save to .csv format
final_data.columns=['Filename','Extracted Values']
final_data=final_data.sort_values(by='Filename', ascending=True)
final_data.to_csv (r'Result.csv', index = False, header=True)
#See the result file in output folder
final_output=pd.read_csv('Result.csv')
final_output
