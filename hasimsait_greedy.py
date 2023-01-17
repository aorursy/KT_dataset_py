slideshowlength=999

debug=False

if slideshowlength>59994 or debug == False: #prevent keyerror set debug false to see the full score

    slideshowlength=59994

iterLimit=10**3 #sets the swap recursion limit. 10^4 max

willSwap=0 # set to 0 to disable swaps, 1 to enable

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from random import seed

from random import random

from random import choice

from random import randint

import time

start = time.perf_counter()

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def checkCommons (tags1,tags2):

    commons=0

    for key in tags1.keys():

        if key in tags2:

            commons+=1

    return commons

'''

def checkUniq (tags1,tags2):

    noncommons=0

    for key in tags1.keys():

        if key not in tags2:

            noncommons+=1

    return noncommons

'''

def points (tags1,tags2):

    comm=checkCommons(tags1,tags2)

    #unq1=checkUniq(tags1,tags2)

    unq1=len(tags1)-comm

    #unq2=checkUniq(tags2,tags1)

    unq2=len(tags2)-comm

    return min(unq1,unq2,comm)

def calcScore():

    f = open("submission.txt", "r")

    mylines=f.readlines()

    f.close()

    totalpoints=0

    line=''

    nline=''

    #could get rid of \n s here but below should work fine

    mylines[len(mylines)-1]=mylines[len(mylines)-1]+"\n"

    for i in range(1,len(mylines)-1):

        line = mylines[i][:-1]

        nline = mylines[i+1][:-1]

        totalpoints+=points(pics[line],pics[nline])

    return totalpoints



def tripleSwap(s1,s2,s3,show,pics,points):

#tripleswap(index1,index2,index3)-->0 or 1 or...or 5

#0 no possible improvement by swapping these indexes

#1 1->3 2->1 3->2 improves

#2 1->2 2->3 3->1 improves

#3 1->1 2->3 3->2 swap 2 and 3 improves

#4 1->3 2->2 3->1 swap 1 and 3 improves

#5 1->2 2->1 3->3 swap 1 and 2 improves

#return value has priorities (if 1 improves 2 won't be checked, if none improves returns 0). 

#This is done bc. if i get any improvement i will be happy, no need to push it too far

    if s1>=999 or s2>=999 or s3>=999:

        print("SOMEHOW IT INCREMENTED ITS VALUE")

        print("s1:",s1,"s2:",s2,"s3:",s3)

        return 0

    #try 1->3 2->1 3->2

    new1=points(pics[show[s1-1]],pics[show[s2]])+points(pics[show[s1+1]],pics[show[s2]])

    new2=points(pics[show[s2-1]],pics[show[s3]])+points(pics[show[s2+1]],pics[show[s3]])

    new3=points(pics[show[s3-1]],pics[show[s1]])+points(pics[show[s3+1]],pics[show[s1]])

    old1=points(pics[show[s1-1]],pics[show[s1]])+points(pics[show[s1+1]],pics[show[s1]])

    old2=points(pics[show[s2-1]],pics[show[s2]])+points(pics[show[s2+1]],pics[show[s2]])

    old3=points(pics[show[s3-1]],pics[show[s3]])+points(pics[show[s3+1]],pics[show[s3]])

    improvefor1=new1-old1

    improvefor2=new2-old2

    improvefor3=new3-old3

    if(improvefor1+improvefor2+improvefor3>0):

        print ("IMPROVED")

        return 1

    #try 1->2 2->3 3->1

    new1=points(pics[show[s1-1]],pics[show[s3]])+points(pics[show[s1+1]],pics[show[s3]])

    new2=points(pics[show[s2-1]],pics[show[s1]])+points(pics[show[s2+1]],pics[show[s1]])

    new3=points(pics[show[s3-1]],pics[show[s2]])+points(pics[show[s3+1]],pics[show[s2]])

    improvefor1=new1-old1

    improvefor2=new2-old2

    improvefor3=new3-old3

    if(improvefor1+improvefor2+improvefor3>0):

        print ("IMPROVED")

        return 2

    #try 1->1 2->3 3->2 (swap 2, wont be useful as swapping 2 didnt increase)

    new1=points(pics[show[s1-1]],pics[show[s1]])+points(pics[show[s1+1]],pics[show[s1]])

    new2=points(pics[show[s2-1]],pics[show[s3]])+points(pics[show[s2+1]],pics[show[s3]])

    new3=points(pics[show[s3-1]],pics[show[s2]])+points(pics[show[s3+1]],pics[show[s2]])

    improvefor1=new1-old1

    improvefor2=new2-old2

    improvefor3=new3-old3

    if(improvefor1+improvefor2+improvefor3>0):

        print ("IMPROVED")

        return 3

    #try 1->3 2->2 3->1 (swap 2, wont be useful as swapping 2 didnt increase)

    new1=points(pics[show[s1-1]],pics[show[s3]])+points(pics[show[s1+1]],pics[show[s3]])

    new2=points(pics[show[s2-1]],pics[show[s2]])+points(pics[show[s2+1]],pics[show[s2]])

    new3=points(pics[show[s3-1]],pics[show[s1]])+points(pics[show[s3+1]],pics[show[s1]])

    improvefor1=new1-old1

    improvefor2=new2-old2

    improvefor3=new3-old3

    if(improvefor1+improvefor2+improvefor3>0):

        print ("IMPROVED")

        return 4

    #try 1->2 2->1 3->3 (swap 2, wont be useful as swapping 2 didnt increase)TODO

    new1=points(pics[show[s1-1]],pics[show[s2]])+points(pics[show[s1+1]],pics[show[s2]])

    new2=points(pics[show[s2-1]],pics[show[s1]])+points(pics[show[s2+1]],pics[show[s1]])

    new3=points(pics[show[s3-1]],pics[show[s3]])+points(pics[show[s3+1]],pics[show[s3]])

    improvefor1=new1-old1

    improvefor2=new2-old2

    improvefor3=new3-old3

    if(improvefor1+improvefor2+improvefor3>0):

        print ("IMPROVED")

        return 5

    return 0

#read file put pics in their respective dicts

vertical_pics={}

pics={}

lengths={}

vertical_lengths={}

import os

file1 = open("/kaggle/input/hashcode-photo-slideshow/d_pet_pictures.txt","r")

mylines=file1.readlines()

file1.close()

n=int(mylines[0])

mylines.pop(0)#forgetting this cost me 4 hours :))))))))))))))))))))))))))))))))))))))))))))))))))))))))

for i in range(n):

    photo=mylines[i].split()

    if photo[0]=='V':

        photo.pop(0)

        vertical_lengths[i]=int(photo[0])#these are handled under in the horribly suboptimal while loop

        photo.pop(0)

        tags={}

        for tag in photo:

            tags[tag]=1

        vertical_pics[i]=tags

    if photo[0]=='H':

        photo.pop(0)

        lengths[str(i)]=int(photo[0])

        #TODO str may fuck up, check if it works, we use int on value so we can use max() TODO

        photo.pop(0)

        tags={}

        for tag in photo:

            tags[tag]=1

        pics[str(i)]=tags

print(len(lengths))

print(len(vertical_lengths))

pairs=0

goal=int(len(vertical_pics)/2)
#VERTICAL PAIRS aim for max tag per slide greedy

#normally you dont want to have 2*maxtagcount/3 tags since points decrease

#ie. if a photo has all tags then it will get 0 from both transitions

#there is some serious room to improve here

while(pairs<goal):

    #get the maximum amount of tags possible greedy

    max_key1 = max(vertical_lengths, key=vertical_lengths.get)

    #TODO check if it's comparing the right thing (not the photo names) TODO

    vertical_lengths.pop(max_key1)

    mymax1=vertical_pics.pop(max_key1)

    max_key2 = max(vertical_lengths, key=vertical_lengths.get)

    vertical_lengths.pop(max_key2)

    mymax2=vertical_pics.pop(max_key2)

    mymax1={**mymax1,**mymax2}#this is horrible it is x=x|y

    pics[str(max_key1)+' '+str(max_key2)]=mymax1

    lengths[str(max_key1)+' '+str(max_key2)]=len(mymax1)

    pairs+=1

    #print(pics[str(max_key1)+' '+str(max_key2)])

print(len(vertical_lengths),"= 0")#should be 0

print(pairs)

#SLIDESHOW GREEDY

#FOR SLIDE:

    #FIND THE NEXT SLIDE THAT GIVES YOU THE MOST POINTS 

    #SET IT AS THE LAST SLIDE

#REPEAT UNTILL THERE ARE NO UNUSED SLIDES LEFT

firstslide=max(lengths,key=lengths.get) #this is dumb but we will swap anyways, ideally you would add to both sides,

maxposs=lengths[firstslide]

lengths.pop(firstslide)

lastslide=firstslide

slides=""

slides+=lastslide+"\n"



totalpoints=0

i=0 

#while(len (lengths)>0):#this will prevent it from looping when there is no picture in lengths CAUSES ERROR WHEN A SLIDE THAT adds 0 points must be added

while(i<slideshowlength):#just to limit runtime,use the one above to get the max possible with this

    i+=1

    currpt=0

    maxpt=0

    maxpic=""

    #j=0

    #k=0

    for picture in lengths:

        #j+=1

        currpt=0

        if(int(lengths[picture]/2)>maxpt):#TODO WHAT IF WE CANT GET ANY POINT FOR THIS SLIDE TODO ADD A FIX AFTER FOR LOOP (IF MAXPIC=='') MAXPIC=LENGTHS[0]... 

                                            #THAT ADDS THE FIRST SLIDE AS THE NEXT SLIDE OR 

                                            #CREATE A SECOND SLIDESHOW AND ADD IT FROM THE MAX POINT TO THIS

                                            #BOTH WOULD FIX THE KEYERROR ISSUE WITH THE WHILE ANY PICS LEFT LOOP(ADDS 5 MORE SLIDES)

            #k+=1

            currpt=points(pics[picture],pics[lastslide])

            if(currpt>maxpt):

                maxpic=picture

                maxpt=currpt

            

    #print("skipped", k ,"useless pics in this iter with length/2 > maxpt comparison")

    totalpoints+=maxpt

    maxposs=int(lengths[maxpic]/2)

    lengths.pop(maxpic)

    lastslide=maxpic

    slides+=lastslide+"\n"

    #if(i%10==0):

        #print(i/10,"% maxposs of the lastslide is=",maxposs)

        #just to see the progress

#print(slides)

slides=slides[:-1]#get rid of the \n at the end

print(totalpoints,"using",i+1,"pictures.")

print("last slide added",maxpt,"points to our score.") #see the last point earned
#SWAPPER THIS IS HORRIBLE AND NEVER WORKS. WILL BE CHANGED, SWAPPING VERTICAL PICS SEEMS LIKE A MUST

     

#tripleswap(index1,index2,index3)-->0 or 1 or...or 5

#0 no possible improvement by swapping these indexes

#1 1->3 2->1 3->2 improves

#2 1->2 2->3 3->1 improves

#3 1->1 2->3 3->2 swap 2 and 3 improves

#4 1->3 2->2 3->1 swap 1 and 3 improves

#5 1->2 2->1 3->3 swap 1 and 2 improves

#return value has priorities (if 1 improves 2 won't be checked, if none improves returns 0). 

#This is done bc. if i get any improvement i will be happy, no need to push it too far



def pimpMySlideShow(show,rotation,iteration,totalImprovement):

    #this disgusts me TODO use greedy, ditch randomization 

    #(it will ruin runtime but 10^11 random swaps didn't improve)

    if(iteration>=iterLimit):

        #print("Cost function improved by",totalImprovement,"in",iteration,"iterations.")

        return show

    if(rotation==0):

        #interchanging slides within the slideshow

        s1 = randint(1, len(show)-2)#both included!

        s2 = randint(1, len(show)-2)#IndexError: list index out of range

        s3 = randint(1, len(show)-2)#both included!

        if(s1==s2==s3 or s1==s2 or s2==s3 or s1==s3):

            return pimpMySlideShow(show,0,iteration+1,totalImprovement)

        else:

            tripleswap=tripleSwap(s1,s2,s3,show,pics,points)

            if(tripleswap==1):

                #231

                temp=show[s1]

                show[s1]=show[s2]

                show[s2]=show[s3]

                show[s3]=temp                

                return pimpMySlideShow(show,0,iteration+1,totalImprovement)

            elif(tripleswap==2):

                #312

                temp=show[s2]

                show[s2]=show[s1]

                show[s1]=show[s3]

                show[s3]=temp                

                return pimpMySlideShow(show,0,iteration+1,totalImprovement)

            elif(tripleswap==3):

                #132

                temp=show[s2]

                show[s2]=show[s3]

                show[s3]=temp             

                return pimpMySlideShow(show,0,iteration+1,totalImprovement)

            elif(tripleswap==4):

                #321

                temp=show[s1]

                show[s1]=show[s3]

                show[s3]=temp                

                return pimpMySlideShow(show,0,iteration+1,totalImprovement)

            elif(tripleswap==5):

                #213

                temp=show[s1]

                show[s1]=show[s2]

                show[s2]=temp                

                return pimpMySlideShow(show,0,iteration+1,totalImprovement)

            else:

                #didnt improve, try again

                return pimpMySlideShow(show,0,iteration+1,totalImprovement)

    if(rotation==1):

        #interchanging vertical pics between pics, find vertical slides first and put their indexes in a list

        return show #TODO

    

#turn slides into a list

newslides=slides

slidelist=newslides.split('\n')

#print(slidelist)

#print("\n\n\n\n")

tic = time.perf_counter()

seed(len(slidelist))

newslidelist=pimpMySlideShow(show=slidelist,rotation=0,iteration=0,totalImprovement=0)#0 for horizontal 1 for vertical swaps

for m in range(willSwap*10**9): #10**8 takes 49 secs improves none

    newslidelist=pimpMySlideShow(newslidelist,rotation=0,iteration=m*iterLimit,totalImprovement=0)#0 for horizontal 1 for vertical swaps



#print("\n\n\n\n")

toc = time.perf_counter()

#print(newslidelist)

#SWITCH SLIDES (WHAT WE PUT IN SUBMISSION) WITH NEWSLIDELIST (SWAPPED SLIDES)

'''

swappt=0

for i in range(len(newslidelist)-1):

    swappt+=points(pics[newslidelist[i+1]],pics[newslidelist[i]])

mystr=""

for slide in newslidelist:

    mystr+=slide+"\n"

mystr=mystr[:-1]

slides=mystr

i+=1 #so that S is correct and we can comment swapping out

print("BEFORE SWAP:",totalpoints,"\nAFTER SWAP:",swappt,"\nDONE")

print("Swaps took ",toc - tic,"seconds.")

print("Program took ",toc - start,"seconds.")

'''
#PUT THE SLIDES IN SUBMISSION

f = open("submission.txt", "w")

slides=str(i+1)+"\n"+slides #add S line

f.write(slides)

f.close()

print(i+1)

print("Program took ",toc - start,"seconds.")