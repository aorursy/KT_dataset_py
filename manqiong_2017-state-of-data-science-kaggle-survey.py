import numpy as np

import pandas as pd

import scipy as sp

import csv

%matplotlib inline

import matplotlib.pylab as plt



import pandas as pd

import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns

import matplotlib.ticker as ticker
mcr=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)

mcr.head()
cr=pd.read_csv('../input/kaggle-survey-2017/conversionRates.csv', encoding="ISO-8859-1", low_memory=False)

cr.head()
fr=pd.read_csv('../input/kaggle-survey-2017/freeformResponses.csv', encoding="ISO-8859-1", low_memory=False)

fr.head()
sch=pd.read_csv('../input/kaggle-survey-2017/schema.csv', encoding="ISO-8859-1", low_memory=False)

sch.head()
sch.tail()
sch.Asked.value_counts()
sch.groupby('Asked').size()
#determine the number of categories in Aksed

Piedata=sch.Asked.value_counts() 



labels=[]

for i in Piedata.index:

    labels.append('{0}'.format(i))

labels



x = labels

y = Piedata[:]



percent = 100.*y/y.sum()



patches, texts = plt.pie(y, startangle=90, radius=1.2)

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]



sort_legend = True

if sort_legend:

    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),

                                          key=lambda x: x[2],

                                          reverse=True))



plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.),

           fontsize=8)

plt.show()

# Group by Asked on MultipleChoiceResponse and FreeFormResponse

Asked_N=range(len(sch.Asked.value_counts().index))

Askgp_mcr=[[] for z in Asked_N]

Askgp_fr=[[] for z in Asked_N]

for i in Asked_N:

    Askgp=list(sch['Column'][sch['Asked']==sch.Asked.value_counts().index[i]])    

    for j in Askgp:

        if j not in fr:

            Askgp_mcr[i].append(j)

        else:

            Askgp_fr[i].append(j)

mcr_perc_resp=[]





for i in range(len(sch.Asked.value_counts().index)):



    # Percentage of Response of each questions on MultipleChoiceResponse

    perc_resp=mcr[Askgp_mcr[i]].count()/(mcr[Askgp_mcr[i]].count()+mcr[Askgp_mcr[i]].isnull().sum())

    perc_resp_sort=perc_resp.sort_values(ascending=False)

    df_perc_resp=perc_resp_sort.to_frame() # You need to create a dataframe first, then you can append to a list.

    mcr_perc_resp.append(df_perc_resp)

    

    y = mcr_perc_resp[i].iloc[:10][0]

    N = len(y)

    x = range(N)

    width = 1/1.5



    plt.title('Percentage of Response on MultipleChoiceResponse of {0} (top 10 or less)'.format(sch.Asked.value_counts().index[i]))

    plt.xticks(x, list(mcr_perc_resp[i].index)[:10],rotation='vertical')

    ax=plt.bar(x, y, width) 



    plt.show()

 

    for j in range(len(y)):

        plt.figure(figsize=(12,8))

        ax2 = sns.countplot(x=y.index[j], data=mcr, order=mcr[y.index[j]].value_counts().index[:10])

        plt.title('{0} (top 10 or less)'.format(list(sch['Question'][sch['Column']==y.index[j]])[0]))

        plt.xlabel(y.index[j])

        plt.xticks(rotation=90)

        

        ncount = len(mcr[y.index[j]].dropna())



        for p in ax2.patches:

            x2=p.get_bbox().get_points()[:,0]

            y2=p.get_bbox().get_points()[1,1]

            ax2.annotate('{:.1f}%'.format(100.*y2/ncount), (x2.mean(), y2), 

                    ha='center', va='bottom') # set the alignment of the text





        plt.show()


