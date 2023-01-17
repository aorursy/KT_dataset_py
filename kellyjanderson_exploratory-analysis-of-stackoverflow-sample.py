# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import collections

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

%matplotlib inline

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



questions = pd.read_csv("../input/Questions.csv", encoding='latin1')

answers = pd.read_csv("../input/Answers.csv", encoding='latin1')

tags = pd.read_csv("../input/Tags.csv", encoding='latin1')
ans_per_question = collections.Counter(answers['ParentId'])

answerid,noAnswers= zip(*ans_per_question.most_common())

text = "Avegrage number of answers per question ",np.mean(noAnswers)
N=20

plt.bar(range(N), noAnswers[:N], align='center', alpha=0.5)

#plt.xticks(y_pos, objects)



plt.ylabel('Number of Answers per Question')

plt.xlabel('Question Id')

plt.title('Distribution of Answers per question ')

plt.text(3,400,"Avegrage answers per question "+str(math.ceil((np.mean(noAnswers)))))



plt.show()
print("Top 2 Questions with maximum number of answers \n")

qid = answerid[:2] 



for b,id in zip(questions['Body'],questions['Id']):

    if id in qid:

        #print(id)

        print(b)

        print("................")
def plot_tags(tagCount):

    

    x,y = zip(*tagCount)



    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  

    colors = [colormap(i) for i in np.linspace(0, 0.8,50)]   



    area = [i/4000 for i in list(y)]   # 0 to 15 point radiuses

    plt.figure(figsize=(9,8))

    plt.ylabel("Number of question associations")

    for i in range(len(y)):

        plt.plot(i,y[i], marker='o', linestyle='',ms=area[i],label=x[i])



    plt.legend(numpoints=1)

    plt.show()
tagCount =  collections.Counter(list(tags['Tag'])).most_common(10)

print(tagCount)

plot_tags(tagCount)
topQuestions =list(zip(questions["Id"],questions["Score"],questions["Title"]))

topQuestions.sort(key=lambda x: x[1],reverse=True)

for id,s,t in topQuestions[:20]:

    #print("Question id:",id)

    print("Score :",s)

    print("Question Title\t:",t,'\n') 