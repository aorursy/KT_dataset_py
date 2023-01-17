
from nltk.corpus import jeita
from time import time
import numpy as np
import random
start_time=time()
print (jeita.tagged_sents()[2])
tagged_sents=jeita.tagged_sents()
print(tagged_sents[1])
train_set=tagged_sents[1:147587]
test_set=tagged_sents[147587:163986]
def get_sentences_origin(sentences_origin_set,train_set):
    dictionary=set()
    for tagged_sent in train_set:
        sentence=""
        path=[0]
        currerntposition=0;
        for tagged_word in tagged_sent:
            if tagged_word[0]!="。":
                sentence=sentence+tagged_word[0]
                #print(tagged_word[0])
                #print(currerntposition)
                currerntposition=currerntposition+len(tagged_word[0])
                #print("->%d" %currerntposition)
                path.append(currerntposition)
            else:
                #print(sentence+'\n')
                if(sentence!=""):
                    sentences_origin_set.append([sentence,path])
                sentence=""
                path=[0]
                currerntposition=0;

            dictionary.update([tagged_word[0]])
    return dictionary
sentences_origin_set=[]
t0=time()
dictionary=get_sentences_origin(sentences_origin_set,train_set)
print('done in %f' %(time()-t0))
sentences_origin_set_test=[]
t0=time()
get_sentences_origin(sentences_origin_set_test,test_set)
print('done in %f' %(time()-t0))
print("số lượng từ trong từ điển từ vựng: %d" %dictionary.__len__())
print(random.sample(dictionary,5))
print("số lượng câu thu được: %d" %len(sentences_origin_set))
print((sentences_origin_set[2]))
# st=""
# for tag in train_set[0]:
#     st=st+"|"+tag[0]
# print(st)
def getmatrix(sentence):
    matrix=np.zeros((len(sentence),len(sentence)+1))
    for i in range(0,len(sentence)):
        for j in range(i+1,len(sentence)+1):
            
            if sentence[i:j] in dictionary or (i+1)==j:
                #print(sentence[i:j])
                matrix[i][j]=1
    return matrix
def findpath(before,end):
    path=[]
    cur=end
    #print(before.shape)
    while(cur!=0):
        path.insert(0,cur)
        #print(cur)
        cur=before[cur,0]
    path.insert(0,0)
    return path
        

def shortest_path(matrix):
    lenght=matrix.shape[1]
    before=np.zeros((lenght,1),dtype=int)
    color=np.zeros((lenght,1))
    before[:]=-1
    color[:]=0
    queue=[]
    queue.insert(0,0)
    end=lenght-1
    while(len(queue)!=0):
        u=queue.pop()
        adj=[i for i in range(lenght) if matrix[u][i]==1]
        #print("%d->" %u)
        #print(adj)
        for item in adj:
            if color[item]==0:
                color[item]=1
                before[item]=u
                #print(item)
                queue.insert(0,item)
            if(item==end):
                #print("before")
                #print(before)
                #print(findpath(before,lenght-1))
                return findpath(before,lenght-1)
    
def getallanswer(sentences_origin_set):
    for sentence in sentences_origin_set:
        #print(sentence)
        matrix=getmatrix(sentence[0])
        sentence.append(shortest_path(matrix))
        #print(sentence)
def viewinganswer(sentences_origin,i,):
    print("data:")
    print(sentences_origin[i][0])
    matrix=getmatrix(sentences_origin[i][0])
    print(sentences_origin[i][1])
    print("answer:")
    print(shortest_path(matrix))
def viewing_some_answer(sentences_origin,i,j):
    for sentence in sentences_origin[i:j]:
        print("data:")
        print(sentence[0])
        matrix=getmatrix(sentence[0])
        print(sentence[1])
        print("answer:")
        print(sentence[2])
        print('='*30)
# do for all sentences
t0=time()
getallanswer(sentences_origin_set)
print("done in %f" %(time()-t0))
# do for all sentences
t0=time()
getallanswer(sentences_origin_set_test)
print("done in %f" %(time()-t0))
viewinganswer(sentences_origin_set,10)
viewing_some_answer(sentences_origin_set_test,10,12)
#thuật toán đếm số lượng từ mà thuật toán tách đúng so với câu gốc
#=> ta dựa vào vị trí đầu mút là các số trong 2 list ứng với từng câu => ta kiểm tra 2 số liên tiếp trong list 1 có nằm liên tiếp trong list 2 không. 
#nếu có thì ta tăng số lượng từ tách đúng lên một
def count_right_word(list1,list2):
    set_pair_in_list2=set()
    i=0
    for pair in zip(list2[0:len(list2)-1],list2[1:len(list2)]):
        set_pair_in_list2.update([pair])
    for pair in zip(list1[0:len(list1)-1],list1[1:len(list1)]):
        if pair in set_pair_in_list2 :
            i=i+1
    return i

def viewing_count_right_word_for_a_sentence(sentences_origin_set,i):
    print(sentences_origin_set[i][1])
    print(sentences_origin_set[i][2])
    count=count_right_word(sentences_origin_set[i][1],sentences_origin_set[i][2])
    print("số lượng từ trong câu gốc: %d" %(len(sentences_origin_set[i][1])-1))
    print("số lượng từ tách đúng    : %d" %count)
    print((float)(count/(len(sentences_origin_set[i][1])-1)))
viewing_count_right_word_for_a_sentence(sentences_origin_set,90)
def get_score(sentences_origin_set,answer_column=2):
    number_of_right_word=0
    number_of_word=0
    
    for sentence in sentences_origin_set:
        number_of_right_word=number_of_right_word+count_right_word(sentence[1],sentence[answer_column])
        number_of_word=number_of_word+len(sentence[1])-1
    
    return (float)(number_of_right_word/number_of_word)
#score in train set
t0=time()
print("train score %f" %get_score(sentences_origin_set))
print("done in %f"%(time()-t0))
#score in test set
t0=time()
print("test score %f" %get_score(sentences_origin_set_test))
print("done in %f"%(time()-t0))
viewinganswer(sentences_origin_set_test,10)
print("done in %f"%(time()-start_time))
list_cac_tu_theo_tung_cau=[]
t0=time()
for tagged_sent in train_set:
    sentence=[]
    for tagged_word in tagged_sent:
        if tagged_word[0]!="。":
            sentence.append(tagged_word[0])
        else:
            if(sentence!=[]):
                list_cac_tu_theo_tung_cau.append(sentence)
            sentence=[]
    
print('done in %f' %(time()-t0))
list_cac_tu_theo_tung_cau[1]
from copy import deepcopy
class Ngram:
    def __init__(self,ngram,laplace_smooth = False):
        self.n_gram = ngram
        self.laplace_smooth = laplace_smooth
        self.n_dict = dict()
        self.n_1_dict = dict()
        self.vocabulary = set()
    
    def fit(self,listcacstu_theocau):
        list_sentences=deepcopy(listcacstu_theocau)
        
        for list_sentence in list_sentences:
            for i in range(0,self.n_gram-1):
                list_sentence.insert(0,'s')
            list_sentence.append('e')
            
        def _count(list_sentences,n_gram=1):
            count_value_dict=dict()
            i=0
            for list_sentence in list_sentences:
                for j in range(0,len(list_sentence)-n_gram+1):
                    key=''.join(list_sentence[j:j+n_gram])
                    #print(key)
                    if key in count_value_dict:
                        count_value_dict[key] += 1
                    else:
                        count_value_dict[key] = 1
            return count_value_dict
        
        self.n_dict =_count(list_sentences,n_gram=self.n_gram)
        self.n_1_dict =_count(list_sentences,n_gram=self.n_gram-1)
        for line in listcacstu_theocau:
            self.vocabulary.update(line)
            
    def get_prob(self, prev_chars, next_char):
        prev_chars = ''.join(prev_chars[-(self.n_gram-1):]) if self.n_gram>1 else ''
        if len(self.vocabulary) == 0:
            raise Exception("Please run train first :D")   
        word = prev_chars + next_char
        #print(word)
        if word in self.n_dict:
            w_count = self.n_dict[word]
        else:
            w_count = 0
        #print(w_count)
        
        if prev_chars in self.n_1_dict:
            p_count = self.n_1_dict[prev_chars]
        else:
            p_count = 0
        #print(p_count)
        if not(self.laplace_smooth):
            if p_count == 0:
                return 0
            else:
                return w_count/p_count
        else:
            return (w_count + 1)/ (p_count + len(self.vocabulary))
    def sentence_prob(self, list_sentence):
        list_sentence=deepcopy(list_sentence)
        for i in range(0,self.n_gram-1):
            list_sentence.insert(0,'s')
        list_sentence.append('e')
        #print(list_sentence)
        prob = 1
        for i in range(0, len(list_sentence)-self.n_gram+1):
           # print(list_sentence[i:i+self.n_gram-1])
            prob *= self.get_prob(list_sentence[i:i+self.n_gram-1],list_sentence[i+self.n_gram-1])
        return prob

t0=time()
model=Ngram(ngram=2,laplace_smooth =True)
model.fit(list_cac_tu_theo_tung_cau)
print("done in %f" %(time()-t0))
model.sentence_prob(['船', 'は', '白帆', 'を', '張', 'つて', 'ノタ', 'といふ'])
model.sentence_prob(['船','は', '白帆', 'を', '張', 'つて', 'ノタ'])
model.sentence_prob(['船は',' ', '白帆', 'を', '張', 'つて', 'ノタ'])
def __recursive_find_path(before,cur,path,path_list):
    if cur==0:
        path_list.append(deepcopy(path))
        return
    for i in before[cur]:
        #print("cur %d -> %d" %(cur,i))
        path.insert(0,i)
        __recursive_find_path(before,i,path,path_list)
        path.pop(0)
        
def find_all_path(before,end):
    paths_list=[]
    path=[end]
    __recursive_find_path(before,end,path,paths_list)
    return paths_list
        

def shortest_all_path(matrix):
    lenght=matrix.shape[1]
    before = [[]] * lenght
    #print(before)
    color=np.zeros((lenght,1))
    distance=np.zeros((lenght,1))
    
    color[:]=0
    distance[:]=-1
    distance[0]=0
    
    queue=[]
    paths_list=[]
    queue.insert(0,0)
    end=lenght-1
    mindistance=-1
    while(len(queue)!=0):
        u=queue.pop()
        color[u]=2
        if u!=end:
            adj=[i for i in range(lenght) if matrix[u][i]==1]
        else:
            adj=[]
        #print("%d->" %u)
        #print(adj)
        for item in adj:
            if color[item]==0 :
                color[item]=1
                before[item]=[u]
                #print(before)
                distance[item]=distance[u]+1
                #print(item)
                queue.insert(0,item)
            else:
                if color[item]==1 and distance[item]==distance[u]+1 :
                    before[item].append(u)
                    #print(before)
            if(item==end):
                if(mindistance==-1):
                    mindistance=distance[item]
#     print("before")
#     print(before)             
    return find_all_path(before,end)
    
#test algorithm
matrix=np.zeros((6,7))
matrix[0,1:4]=1
matrix[1:4,4:6]=1
matrix[4:6,6]=1
matrix[4,5]=1
#matrix=getmatrix(sentences_origin_set[1][0])
print(sentences_origin_set[1][0])
print(matrix)
print(len(shortest_all_path(matrix)))
for i in shortest_all_path(matrix):
    print(i)
def getencoded_string(sentences_str,path):
    pre=0
    list_word=[]
    for i in path[1:]:
        #print(i)
        #print(sentences_str[pre:i])
        list_word.append(sentences_str[pre:i])
        pre=i
        
    return list_word
matrix=getmatrix(sentences_origin_set[1][0])
model.sentence_prob(getencoded_string(sentences_origin_set[1][0],shortest_all_path(matrix)[0]))
max_prob=0
max_path=[]
matrix=getmatrix(sentences_origin_set[1][0])
for i in shortest_all_path(matrix):
    curprob=model.sentence_prob(getencoded_string(sentences_origin_set[1][0],i))
    if(max_prob<curprob):
        max_prob=curprob
        max_path=i
print(max_prob)
print(max_path)
def get_n_gram_best_answer(sentences_origin_set):
    count=8000-1
    for sentence in sentences_origin_set:
        count=count+1
        max_prob=0
        max_path=[]
        matrix=getmatrix(sentence[0])
        for i in shortest_all_path(matrix):
            curprob=model.sentence_prob(getencoded_string(sentence[0],i))
            if(max_prob<curprob):
                max_prob=curprob
                max_path=i
        sentence.append(max_path)
        #if count%1000==0:
        print(count)
    return sentences_origin_set
t0=time()
answer=get_n_gram_best_answer(deepcopy(sentences_origin_set_test[8000:9080]))
print('Done in %f' %(time()-t0))
sentences_origin_set_test[8747][0]
#score in test set
t0=time()
print("test score %f" %get_score(answer,3))
print("done in %f"%(time()-t0))
