import nltk

from nltk.corpus import wordnet



word1=input("Enter first word here: ")

word2=input("Enter second word here: ")



w1  = wordnet.synset(word1+".n.01")

w2  = wordnet.synset(word2+".n.01")



syns1 = wordnet.synsets(word1)

syns2 = wordnet.synsets(word2)







if w1.wup_similarity(w2) > 0.65:

    print(word1 + " and "+ word2 +"are related to each other")

else:

    print(word1 + " and "+ word2 +" are not related to each other")



print("Similarity Score is " + str(w1.wup_similarity(w2)*100) + " %")



print(syns1[0].examples())

print(syns2[0].examples())
