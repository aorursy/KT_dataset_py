### Download data from google drive. You need not mess with this code.

import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                
if __name__ == "__main__":
    file_id = '1e_Azf9zGvSWsDhM9PP2sfMNKC72-iWAK'
    destination = 'data.txt'
    download_file_from_google_drive(file_id, destination)
with open('data.txt', 'r') as f:
  data_raw = f.readlines()
def first_five_in_list(l):
  """
  Inputs: 
  l: Python list

  Outputs:
  l_5 : python list, first five elements of list if length of list greater than 5; None otherwise
  """
  l_5 = [] 
  if len(l) >= 5:
    for i in range(0,5):
      l_5.append(l[i])
    return l_5
  else:
    return None
def remove_trailing_newlines(s):
  """
  Function that removes all trailing newlines at the end of it
  Inputs:
    s : string

  Outputs:
    s_clean : string, string s but without newline characters at the end 
  """
  s_clean = s.strip('\n')
  return s_clean
def mapl(f, l):
  """
  Function that applies f over all elements of l
  Inputs:
    f : function, f takes elements of type t1 and returns elements of type t2
    l : list, list of elements of type t1

  Ouptuts:
    f_l : list, list of elements of type t2 obtained by applying f over each element of l
  """
  f_l = []
  for i in range(0,len(l)):
    f_l.append(f(l[i]))

  return f_l
data_clean = mapl(remove_trailing_newlines, data_raw)
def split_at_s(text, s):
  """Function that splits string text into two parts at the first occurence of string s
  Inputs:
    text: string, string to be split
    s : string, string of length 1 at which to split
  
  Outputs:
    split_text: tuple of size 2, contains text split in two (do not include the string s at which split occurs in any of the split parts) 
  """
  s1 = text[0:text.index(s)]
  s2 = text[text.index(s) + 1:len(text)]
  split_text = (s1,s2)
  return split_text
split_at_tab = lambda text: split_at_s(text,'\t')
data_clean2 = []
for i in range(0,len(data_clean)):
  data_clean2.append(split_at_tab(data_clean[i]))
import string
def remove_punctuations_and_lower(text):
  """Function that removes punctuations in a text
  Inputs:
    text: string
  Outputs:
    text_wo_punctuations
  """
  return (text.translate(str.maketrans("","", string.punctuation))).lower()
dataset = []
for i in range(0,len(data_clean2)):
  t = data_clean2[i]
  f = []
  for j in range(0,2):
   f.append(remove_punctuations_and_lower(t[j]))
  dataset.append(tuple(f))
def counter(l, f):
  """
  Function that returns a dictionary of counts of unique values obtained by applying f over elements of l
  Inputs:
    l: list; list of elements of type t
    f: function; f takes arguments of type t and returns values of type u
  
  Outputs:
    count_dict: dictionary; keys are elements of type u, values are ints
  """
  count_dict = {}
  t = []
  for i in l:
    t.append(f(i))
  r = set(t)
  for i in r:
    count_dict[i] = t.count(i)
  return count_dict
def aux_func(i):
  return(i[0])
counter(dataset,aux_func)
def random_shuffle(l):
  import random
  """Function that returns a randomly shuffled list
  Inputs:
    l: list
  Outputs:
    l_shuffled: list, contains same elements as l but randomly shuffled
  """
  random.shuffle(l)
  l_shuffled = [] 
  for i in l:
    l_shuffled.append(i)
  return l_shuffled
n = random_shuffle(dataset)
l1 = (int)(0.8 * len(n))
data_train = []
data_test = []
for i in range(0,l1):
  data_train.append(n[i])
for i in range(l1,len(n)):
  data_test.append(n[i])
vocab = []
for i in data_train:
  j = i[1].split(" ")
  for k in j:
    if k not in vocab:
      vocab.append(k)
dict_spam = {}
dict_ham = {}

for i in vocab:
  dict_spam[i] = 0
  dict_ham[i] = 0

for i in vocab:
  count = 0
  for j in data_train:
    if(j[0] == "spam"):
      t = j[1].split(" ")
      for k in t:
        if(i == k):
          count += 1
  dict_spam[i] = count

for i in vocab:
  count = 0
  for j in data_train:
    if(j[0] == "ham"):
      t = j[1].split(" ")
      for k in t:
        if(i == k):
          count +=1
  dict_ham[i] = count

dict_prob_spam = {}
dict_prob_ham = {}
sumspam = 0
sumham = 0
sumspam = sum(dict_spam.values())
sumham = sum(dict_ham.values())
for i in vocab:
  dict_prob_spam[i] = (dict_spam[i] + 1)/(len(vocab) + sumspam) 
for i in vocab:
  dict_prob_ham[i] = (dict_ham[i] + 1)/(len(vocab) + sumham)
def predict(text, dict_prob_spam, dict_prob_ham, data_train):
  """Function which predicts the label of the sms
  Inputs:
    text: string, sms
    dict_prob_spam: dictionary, contains dict_prob_spam as defined above
    dict_prob_spam: dictionary, contains dict_prob_ham as defined above
    data_train: list, list of tuples of type(label, sms), contains training dataset

  Outputs:
    prediction: string, one of two strings - either 'spam' or 'ham'
  """
  prediction = ''
  f = text.split(" ")
  d = counter(data_train,aux_func)
  spam_score = d["spam"]/len(data_train)
  ham_score = d["ham"]/len(data_train)
  for j in f:
    if j in vocab:
      spam_score = spam_score*dict_prob_spam[j]
      ham_score = ham_score*dict_prob_ham[j]
  if spam_score > ham_score:
    prediction = 'spam'
  else:
    prediction = 'ham'
  return prediction

def accuracy(data_test, dict_prob_spam, dict_prob_ham, data_train):
  """Function which finds accuracy of model
  Inputs:
    data_test: list, contains tuples of data (label, sms) 
    dict_prob_spam: dictionary, contains dict_prob_spam as defined above
    dict_prob_spam: dictionary, contains dict_prob_ham as defined above
    data_train: list, list of tuples of type(label, sms), contains training dataset


  Outputs:
    accuracy: float, value of accuracy
  """
  c = 0
  for i in data_test:
    o = i[0]
    p = predict(i[1],dict_prob_spam, dict_prob_ham, data_train)
    if p == o:
      c += 1
  accuracy = c/len(data_test)
  return accuracy
accuracy(data_test,dict_prob_spam, dict_prob_ham, data_train)