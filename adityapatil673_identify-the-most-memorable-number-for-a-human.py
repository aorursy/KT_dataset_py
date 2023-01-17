import re
from re import *
import pandas as pd

t = '2221143432453465654'
reg = r'(\d+)\1+'
#This regex identifies patterns such as 222, 4343,6565. 
re.match(reg,t)
def matching_num(num):
    import re
    regex = r"(\d+)\1+"
    test_str = num
    matches = re.finditer(regex, test_str)
    l = []
    for matchNum, match in enumerate(matches):
        l.append(match.group())
    return l
a= '8888777123'
b ='8830121233'
c='9823321518'
print(matching_num(a))
print(matching_num(b))
print(matching_num(c))
add = '../input/master_mob_score.csv'
mst= pd.read_csv(add,index_col=0)   
print (mst)
def m_score(j):
    # M_score takes a list and calculates the final score for the number
    # setting up the master for score calculation
    add = '../input/master_mob_score.csv'
    mst= pd.read_csv(add,index_col=0)   
    b=[]
    score = 0   
    #Setting up for loop for calculating scores
    for i in j:
        # X parameter here is Variety -- A number
        # Y Parameter is length of repeat -- A string
        variety =len(set(i))
        length = str(len(i))
        sc = mst.loc[variety,length]
        b.append(sc)
        score = score + sc
    return score
print(matching_num(a))
print (m_score(matching_num(a)))
print ("----------------")
print(matching_num(b))
print (m_score(matching_num(b)))
print ("----------------")
print(matching_num(c))
print (m_score(matching_num(c)))
def final_score(k) :
    # I am too lazy to call a function in a function
    return m_score(matching_num(k))

def best (j):
    #takes in a list and throws "number" & "score"
    k = []
    maxscore=0
    for i in j:
        k.append(final_score(i))
        if final_score(i) > maxscore :
            # this block is invoked when it finds the new KING in the list
            maxscore = final_score(i)
            num = i
    return num,maxscore
w = [a,b,c]
print(best(w))