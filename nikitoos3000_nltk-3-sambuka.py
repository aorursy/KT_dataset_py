import nltk
from nltk.corpus import nps_chat

def find(word, dictionary):
    for i in range(0, len(dictionary)):
        if dictionary[i][0] == word:
            return [True, i]
    
    return [False, -1]

def most_common_user():
    words = nps_chat.words()
    dictionary = []
    for word in words:
        if len(word) > 0 and word[0] == 'U':
            flag = True
            for i in range(1, len(word)):
                if not word[i].isdigit():
                    flag = False
                    break
            if flag:
                item = find(word, dictionary)
                if item[0]:
                    dictionary[item[1]][1] += 1
                else:
                    elem = [word, 1]
                    dictionary.append(elem)
    
    maX = 0
    index = 0
    for i in range(0, len(dictionary)):
        if dictionary[i][1] >= maX:
            maX = dictionary[i][1]
            index = i
            
    print("Самый популярный пользователь:", dictionary[index][0] + ", с частотой появления -", dictionary[index][1])

most_common_user()
def getSomeInformation():
    avL = 0; count = 0; 
    for _id in nps_chat.fileids():
        chatroom = nps_chat.posts(_id)
        for chat in chatroom:
            avL += len(chat)
            count += 1
    
    print("Средняя длина сообщения:", round(avL / count))
    
getSomeInformation()