import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools

import matplotlib.pyplot as plt

import cv2

import seaborn as sns

import os


gifts=pd.read_csv("../input/gifts.csv", encoding = "ISO-8859-1")

gifts['type'] = gifts['GiftId'].apply(lambda x: x.split('_')[0])

gifts['id'] = gifts['GiftId'].apply(lambda x: x.split('_')[1])

gifts['type'].value_counts()
def Weight(mType):

    if mType == "horse":

        return max(0, np.random.normal(5,2,1)[0])

    if mType == "ball":

        return max(0, 1 + np.random.normal(1,0.3,1)[0])

    if mType == "bike":

        return max(0, np.random.normal(20,10,1)[0])

    if mType == "train":

        return max(0, np.random.normal(10,5,1)[0])

    if mType == "coal":

        return 47 * np.random.beta(0.5,0.5,1)[0]

    if mType == "book":

        return np.random.chisquare(2,1)[0]

    if mType == "doll":

        return np.random.gamma(5,1,1)[0]

    if mType == "blocks":

        return np.random.triangular(5,10,20,1)[0]

    if mType == "gloves":

        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
gifts['weight'] = gifts['type'].apply(lambda x: Weight(x))
sns.distplot([tuple(gifts[gifts['type']=='coal']['weight'])],bins=100, label="coal");

plt.legend();
stuff = ['book', 'coal', 'bike','train','blocks','doll','horse','gloves','ball']

combs=pd.DataFrame(columns=('comb', 'bags over 50', 'bags from 0 to 50'))

for size in np.arange(3,9):

    for subset in itertools.combinations_with_replacement(stuff, size):

        a=[]

        for i in range(0,1000):

            tmp =0.0

            for j in range(0,size):

                tmp+=Weight(subset[j])

            a.append(tmp)

        t=pd.DataFrame(a,columns=['weights'])

        more50=( t[t['weights'] > 50].sum()[0]/1000)

        f0to50=(t[(t['weights']<=50)].sum()[0]/1000)

        combs=combs.append(pd.DataFrame(np.array([[subset, more50, f0to50]]),columns=('comb', 'bags over 50', 'bags from 0 to 50')), ignore_index=True)

combs.loc[ (combs['bags from 0 to 50'] >= 25)].sort_values(by=['bags from 0 to 50'], ascending=False)
coal=0

book=0

bike=0

train=0

blocks=0

doll=0

horse=0

gloves=0

ball=0

k=0

with open("../Santa_bags.csv", 'w') as f:

        f.write("Gifts \n")

        for i in range(1000):

            if horse<998:

                f.write('blocks_'+str(blocks)+' horse_'+str(horse)+' ball_'+str(ball))

                blocks+=1

                horse+=1

                ball+=1

                f.write(' blocks_'+str(blocks)+' horse_'+str(horse))

                blocks+=1

                horse+=1

                f.write(' horse_'+str(horse)+'\n')

                horse+=1

                k+=1

            elif  ball<1098:

                f.write('bike_'+str(bike)+' blocks_'+str(blocks)+' ball_'+str(ball))

                bike+=1

                blocks+=1

                ball+=1

                f.write(' ball_'+str(ball))

                ball+=1

                f.write(' ball_'+str(ball)+'\n')

                ball+=1

                k+=1

            elif horse<1000:

                f.write('blocks_'+str(blocks)+' doll_'+str(doll)+' horse_'+str(horse)+' ball_'+str(ball))

                blocks+=1

                doll+=1

                horse+=1

                ball+=1

                f.write(' blocks_'+str(blocks)+' doll_'+str(doll)+'\n')

                blocks+=1

                doll+=1

                k+=1

            elif blocks<998:

                f.write('blocks_'+str(blocks)+' doll_'+str(doll)+' gloves_'+str(gloves))

                blocks+=1

                doll+=1

                gloves+=1

                f.write(' blocks_'+str(blocks)+' doll_'+str(doll))

                blocks+=1

                doll+=1

                f.write(' doll_'+str(doll)+'\n')

                doll+=1

                k+=1

            elif  gloves<200 and train<999:

                f.write('book_'+str(book)+' train_'+str(train)+' doll_'+str(doll)+' gloves_'+str(gloves))

                book+=1

                train+=1

                gloves+=1

                doll+=1

                f.write(' train_'+str(train))

                train+=1

                f.write(' train_'+str(train)+'\n')      

                train+=1

                doll+=1

                k+=1

            elif   book<1200 and doll<854:

                f.write('book_'+str(book)+' doll_'+str(doll))

                book+=1

                doll+=1

                f.write(' book_'+str(book)+' doll_'+str(doll))    

                book+=1

                doll+=1

                f.write(' book_'+str(book)+' doll_'+str(doll))

                book+=1

                doll+=1

                f.write(' doll_'+str(doll))

                doll+=1

                f.write(' doll_'+str(doll))

                doll+=1

                f.write(' doll_'+str(doll)+'\n')

                doll+=1

                k+=1

            elif   train<999and doll<1000:

                f.write('book_'+str(book)+' train_'+str(train)+' doll_'+str(doll))

                book+=1

                train+=1

                doll+=1

                f.write(' train_'+str(train))    

                train+=1

                f.write(' train_'+str(train)+'\n')

                train+=1

                k+=1

            elif   book<1200 and doll<1000:

                f.write('book_'+str(book)+' bike_'+str(bike)+' doll_'+str(doll))

                book+=1

                bike+=1

                doll+=1

                f.write(' book_'+str(book)+' doll_'+str(doll))    

                book+=1

                doll+=1

                f.write(' book_'+str(book)+'\n')

                book+=1     

                k+=1                                       

print("coal max(166)",coal)                

print("horse max(1000)",horse)

print("book max(1200)",book)

print("bike max(500)",bike)

print("gloves max(200)",gloves)

print("train max(1000)",train)

print("ball max(1100)",ball)

print("doll max(1000)",doll)

print("blocks max(1000)",blocks)

print(k)
#35349.71148

coal=0

book=1199

bike=8

train=2

blocks=0

doll=0

horse=999

gloves=0

ball=0

k=0

with open("Santa_bags_fitted.csv", 'w') as f:

        f.write("Gifts \n")

        for i in range(1000):

            

            if horse>=1:

                f.write('blocks_'+str(blocks)+' horse_'+str(horse)+' ball_'+str(ball)+' book_'+str(book))

                blocks+=1

                horse-=1

                ball+=1

                book-=1

                f.write(' blocks_'+str(blocks)+' horse_'+str(horse))

                blocks+=1

                horse-=1

                f.write(' horse_'+str(horse)+'\n')

                horse-=1

                k+=1

            elif  ball<1098:

                f.write('bike_'+str(bike)+' blocks_'+str(blocks)+' ball_'+str(ball))

                bike+=1

                blocks+=1

                ball+=1

                f.write(' ball_'+str(ball))

                ball+=1

                f.write(' ball_'+str(ball)+'\n')

                ball+=1

                k+=1       

            elif horse>=0 :

                f.write('blocks_'+str(blocks)+' doll_'+str(doll)+' horse_'+str(horse)+' ball_'+str(ball))

                blocks+=1

                doll+=1

                horse-=1

                ball+=1

                f.write(' blocks_'+str(blocks)+' doll_'+str(doll)+'\n')

                blocks+=1

                doll+=1

                k+=1

            elif blocks<998:

                f.write('blocks_'+str(blocks)+' doll_'+str(doll)+' gloves_'+str(gloves))

                blocks+=1

                doll+=1

                gloves+=1

                f.write(' blocks_'+str(blocks)+' doll_'+str(doll))

                blocks+=1

                doll+=1

                f.write(' doll_'+str(doll)+'\n')

                doll+=1

                k+=1

            elif  gloves<200 and train<999:

                f.write('book_'+str(book)+' train_'+str(train)+' doll_'+str(doll)+' gloves_'+str(gloves))

                book-=1

                train+=1

                gloves+=1

                doll+=1

                f.write(' train_'+str(train))

                train+=1

                f.write(' train_'+str(train)+'\n')      

                train+=1

                doll+=1

                k+=1

            elif   train<998and doll<1000 and i%4==0:

                f.write('book_'+str(book)+' train_'+str(train)+' doll_'+str(doll))

                book-=1

                train+=1

                doll+=1

                f.write(' train_'+str(train)+' book_'+str(book))    

                train+=1

                book-=1

                f.write(' train_'+str(train)+'\n')

                train+=1

                k+=1

            elif   train<998and doll<1000:

                f.write('book_'+str(book)+' train_'+str(train)+' doll_'+str(doll))

                book-=1

                train+=1

                doll+=1

                f.write(' train_'+str(train))    

                train+=1

                f.write(' train_'+str(train)+'\n')

                train+=1

                k+=1

            elif   book<1200 and doll<854:

                f.write('book_'+str(book)+' doll_'+str(doll))

                book-=1

                doll+=1

                f.write(' book_'+str(book)+' doll_'+str(doll))    

                book-=1

                doll+=1

                f.write(' book_'+str(book)+' doll_'+str(doll))

                book-=1

                doll+=1

                f.write(' doll_'+str(doll))

                doll+=1

                f.write(' doll_'+str(doll))

                doll+=1

                f.write(' doll_'+str(doll)+'\n')

                doll+=1

                k+=1

            elif coal < 166:

                f.write('coal_'+str(coal)+' book_'+str(book))

                coal+=1

                book-=1

                f.write(' book_'+str(book)+'\n')

                book-=1

                k+=1    

            elif   book<1200 and doll<860:

                f.write('book_'+str(book)+' bike_'+str(bike)+' doll_'+str(doll))

                book-=1

                bike+=1

                doll+=1

                f.write(' book_'+str(book)+' doll_'+str(doll))    

                book-=1

                doll+=1

                f.write(' book_'+str(book)+'\n')

                book-=1     

                k+=1                       

print("coal max(166)",coal)                

print("horse min(-1)",horse)

print("book min(-1)",book)

print("bike max(500)",bike)

print("gloves max(200)",gloves)

print("train max(1000)",train)

print("ball max(1100)",ball)

print("doll max(1000)",doll)

print("blocks max(1000)",blocks)

print(k)