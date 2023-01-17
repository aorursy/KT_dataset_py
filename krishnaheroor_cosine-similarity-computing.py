#importing libraries



from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 
#Finding the cosine similarity between two sentences X and Y



X = ("Hi! How are you ?").lower()

Y = ("Are you from Banglore ?").lower()
#Tokenize the given sentences into tokens.

X_list = word_tokenize(X)  

Y_list = word_tokenize(Y) 
X_list, Y_list
#printing the stopwords in english

sw = stopwords.words('english')  

l1 =[];l2 =[] 
sw
## remove stop words from the string 

X_set = {w for w in X_list if not w in sw}  

Y_set = {w for w in Y_list if not w in sw} 
X_set, Y_set
# form a set containing keywords of both strings  

rvector = X_set.union(Y_set)  

for w in rvector: 

    if w in X_set: l1.append(1) # create a vector 

    else: l1.append(0) 

    if w in Y_set: l2.append(1) 

    else: l2.append(0) 

c = 0

  

# cosine formula  

for i in range(len(rvector)): 

        c+= l1[i]*l2[i] 

cosine = c / float((sum(l1)*sum(l2))**0.5) 

print("similarity: ", cosine)