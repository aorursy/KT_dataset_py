import numpy as np # linear algebra

import string

import random

goal='alex'



def gen(goal):

    letters = (' '+string.ascii_lowercase )

    cmb=''

    for i in range(len(goal)):

        rnd = random.choice(letters)

        cmb+=rnd

    return cmb

   

def gencomp(n):

    maxgen=0

    maxstr=''

    for i in range(n):

        generated = gen(goal)

        maxgen1=0

        for z in range(len(generated)):

            if generated[z]==goal[z]:

                maxgen1+=1

        if maxgen1>maxgen:

            #print(maxgen1)

            maxstr=generated

            maxgen=maxgen1

            print('current iteration: ',i,' with ',maxgen,'letters for word: ',maxstr)



    return maxstr,maxgen



gencomp(1000000)

                

            

        