'''
Naive approach:
Find how many words match words on he provided lists and see which category there are more of.
Input needs to be pre-tokenized which is not a feature of the Thai language normally
'''

positive_vocab = []
negative_vocab = []
swear_words = []

with open("../input/negative-sentiment-words.txt", 'r') as f:
    for line in f:
        negative_vocab.append(line.rstrip())

with open("../input/positive-sentiment-words.txt", 'r') as f:
    for line in f:
        positive_vocab.append(line.rstrip())
        
with open("../input/swear-words.txt", 'r') as f:
    for line in f:
        swear_words.append(line.rstrip())
 

#requires tokenized input but Thai usually does not contain spaces between words

sentences = [
    #positive sentence: "I am happy because I have many friends and they are good people"
    "ฉัน มี ความสุข มาก เพราะ มี เพื่อน มาก มาย และ เป็น คน ที่ ดี",

    #negative sentence: "I am angry because I'm having a very bad day"
    "ฉัน โกรธ มาก เพราะ ฉัน มี วัน ที่ แย่ มาก",

    #neutral sentence: "Today is a Thursday in April"
    "วันนี้ เป็น วัน พฤหัส ใน เดือน เมษายน"
]

for sentence in sentences:
    neg = 0
    pos = 0
    print(sentence)
    words = sentence.split(' ')
    for word in words:
        if word in positive_vocab:
            pos = pos + 1
        if word in negative_vocab or word in swear_words:
            neg = neg + 1

    if pos > neg:
        print('positive')
    elif neg > pos:
        print('negative')
    else:
        print('neutral')