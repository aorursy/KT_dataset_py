# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import wordcloud

import json



import os

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
for dirname, _, filenames in os.walk('/kaggle/input'):

    if(''.join(dirname.split('/')[len(dirname.split('/'))-2:]) == 'noncomm_use_subsetpdf_json'):

        #print(dirname)

        print("Total files in noncomm_use_subset is",len([name for name in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, name))]))

        #for filename in filenames:

           # print(os.path.join(dirname, filename))

    
#print("length of the entire json is ",len(data))

print("data is type - ",type(data))

print("the keys of the dictionary data are ",data.keys())





print("length of  body text is", len(data['body_text']))
dataDF = pd.DataFrame(columns=['id','abstract','lenAbs'],index=range(2377))

#data
c=0

    

for dirname, _, filenames in os.walk('/kaggle/input'):

    if(''.join(dirname.split('/')[len(dirname.split('/'))-2:]) == 'noncomm_use_subsetpdf_json'):

        if(len(filenames)):

            print(len(filenames))

            for filename in filenames:

                #print(filename)

                with open(os.path.join(dirname, filename)) as f:

                    temp = json.load(f)

                if(temp['abstract']):

                    #data.loc[c] = pd.Series({'id':temp['paper_id'],'abstract':temp['abstract'][0]['text'], 'lenAbs':len(temp['abstract'][0]['text'])})

                    dataDF.loc[c] = pd.Series({'id':temp['paper_id'],'abstract':temp['abstract'][0]['text'], 'lenAbs':len(temp['abstract'][0]['text'])})

                #print(c)

                c=c+1
print("Snapshot of 4 rows  ","\n",dataDF.head(4))

print("Number of papers without an abstract",sum(dataDF.lenAbs==0))

print("remove all the papers that do not have abstracts")



dataDFFilt1 = dataDF[dataDF.lenAbs!=0]

dataDFFilt1 = dataDFFilt1.dropna()





dataDFFilt1.reset_index(drop=True, inplace=True)



dataDFFilt1.head(4)





'''***

data['abstFull']=''

for ind, absstract in zip(data.index, data.abstract):

    if(len(absstract)>1):

        concat=''

        for abstrs in range(len(absstract)):

            concat = concat +' ' + absstract[abstrs]['text']

        data.loc[ind,'abstFull'] = concat

    else:

        data.loc[ind,'abstFull'] = absstract[0]['text']

***'''
print("number of papers with abstract is :" ,len(dataDFFilt1.abstract))

text =  ' '.join([abstract for abstract in dataDFFilt1.abstract])

wc = wordcloud.WordCloud(background_color="white", max_words=100, width=2000, height=1800,random_state=1).generate(text)

# to recolour the image

plt.figure(figsize=(20,10))

plt.imshow(wc)