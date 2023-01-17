#os.getcwd()



#os.listdir('/kaggle/input/')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

os.chdir('/kaggle/')

# Any results you write to the current directory are saved as output.
import nltk

#import re

import os

#import codecs

#from sklearn import feature_extraction

#import mpld3

#import pprint

import sys,glob

from imp import reload

from nltk.corpus import stopwords



from nltk.stem.snowball import SnowballStemmer

#from nltk.stem import PorterStemmer

stemmmer = SnowballStemmer("english")

reload(sys)

class rempunct:

    def __init__(self):

        nltk.download('stopwords')

        #from nltk.corpus import stopwords

        self.swords = set(stopwords.words('english'))

        print(len(self.swords),"stopwords present!")

        

    def allfiles(self,foldername):            #returns the name of all files inside the source folder.         

            owd = os.getcwd()

            fld = foldername + "/"

            os.chdir(fld)                    #this is the name of the folder from which the file names are returned.

            arr = []                        #empty array, the names of files are appended to this array, and returned.

            for file in glob.glob("*.txt"):

                  arr.append(file)

            os.chdir(owd)

            print("All filenames extracted!")

            return arr

     

    def rem_stop(self,fname,ofilename):

        rawlines = open(fname,errors='ignore').readlines()

        lenl = len(rawlines)

        of=open(ofilename,'w',errors='ignore',)

        for r in range(lenl):    

            linex = rawlines[r]

            linex2 = "".join(c for c in linex if c not in ('!','.',':',',','?',';','``','&','-','"','(',')','[',']','0','1','2','3','4','5','6','7','8','9','‘','’','“','”'))

            linex3 = linex2.split()

            #prog=(r+1)/len(rawlines)

            for s in range(len(linex3)):    

                noword = linex3[s].lower()

                noword.replace("'","")

                noword.replace("”","")

                noword.replace("“","")

                noword.replace("’","")

                noword.replace("‘","")

                

                if noword not in self.swords:

                    of.write(noword)

                    of.write(" ")

            #self.drawProgressBar(prog)

             # stem

            of.write("\n")    



    def  rem_stemm(self,ffname,offilename):

         #rawwlines = open(ffname).readlines()

        with open(ffname,errors='ignore') as f:

             content = f.read().splitlines()

         

        lenn1=len(content)

        off=open(offilename,'w',errors='ignore')

        for r in range(lenn1):

             lineex=content[r]

             lineex2=lineex.split()

 #            progg=(r+1)/len(content)

             for s in range(len(lineex2)):

                 nooword=lineex2[s].lower()

                 off.write(stemmmer.stem(nooword))

                 off.write(" ")

             off.write("\n")

             



        

            

    def drawProgressBar(self,percent, barLen = 50):            #just a progress bar so that you dont lose patience

        sys.stdout.write("\r")

        progress = ""

        for i in range(barLen):

            if i<int(barLen * percent):

                progress += "="

            else:

                progress += " "

        sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))

        sys.stdout.flush()    

                 

                 

         

    def allremove(self):    

        array1 = self.allfiles('plagiarism-detection')

        lenv = len(array1)

        for k in range(lenv):

            progr=(k+1)/lenv

            in1 = 'plagiarism-detection/'+array1[k];

            out1 = 'stops_removed/'+array1[k]; 

            print(out1)

            out2 = 'stemm_done/'+array1[k];

            #out3 = 'final_stemm_done/'+array1[k];

            self.rem_stop(in1,out1)

            self.rem_stemm(out1,out2)

            #self.rem_fin_stop(out2,out3)

            self.drawProgressBar(progr)

        print("\nAll files done!")

         

if __name__ == '__main__':

    rp = rempunct()

    rp.allremove()

file = open('input/g2pB_taske.txt', 'r')

sent1 = file.read()

file.close()

file = open('input/orig_taske.txt', 'r')

sent2 = file.read()

file.close()

file = open('input/g2pC_taske.txt', 'r')

sent3 = file.read()

file.close()



 

Doc1 = (sent1.encode('ascii', 'ignore')).decode("utf-8")

wiki = (sent2.encode('ascii', 'ignore')).decode("utf-8")

Doc2 = (sent3.encode('ascii', 'ignore')).decode("utf-8")

def levenshteinDistance(s1, s2):

    if len(s1) > len(s2):

        s1, s2 = s2, s1



    distances = range(len(s1) + 1)

    for i2, c2 in enumerate(s2):

        distances_ = [i2+1]

        for i1, c1 in enumerate(s1):

            if c1 == c2:

                distances_.append(distances[i1])

            else:

                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))

        distances = distances_

    return distances[-1]
levenshteinDistance(Doc1, wiki)
levenshteinDistance(Doc2, wiki)
import nltk

ed_Doc1 = nltk.edit_distance(Doc1, wiki)

ed_Doc2 = nltk.edit_distance(Doc2, wiki)
ed_Doc1
ed_Doc2
Doc2
wiki