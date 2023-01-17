# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



plt.rcParams.update({'figure.max_open_warning': 0})

#sns.set(font_scale=1.5)

sns.set_context('talk')

pd.set_option('display.max_colwidth', -1)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

sns.set_color_codes("pastel")

%matplotlib inline
#Reading in the Multiple Choice Responses and the schema files

MCR=pd.read_csv('../input/multipleChoiceResponses.csv', encoding = "ISO-8859-1")

sch=pd.read_csv('../input/schema.csv', encoding = "ISO-8859-1")
def plotSelectFields(df,sch):

    '''

    This function takes in the MCR and schema files and plots 

    the fields that were selection fields in the survey. 

    These fields can be identified by the suffix "Select" in the column name.

    

    Inputs:

    df - the multiple choice response dataframe

    sch - the schema dataframe

    

    Outputs:

    Horizontal bar plots showing the counts of the top categories selected in each field    

    '''

    

    #First segregate the columns that have the 'Select' suffix

    selectCols=[colname for colname in list(df.columns) if colname[-6:]=='Select']

    

    for colname in selectCols:

        temp_dict={}

        for i in list(df[colname]):

            if type(i)==str:

                for val in i.split(','): #Multiple selections are separated by commas

                    temp_dict[val]=temp_dict.get(val,0)+1

        if len(temp_dict)==0: continue

        df_dict={'category':list(temp_dict.keys()), 'counts':list(temp_dict.values())}

        

        #Selecting the top 10 counts of categories for each column

        temp_df=pd.DataFrame(df_dict).sort_values(by='counts',ascending=False)[:10]

        

        #This statement makes sure that the plots are drawn one below another

        plt.figure(selectCols.index(colname))

        

        #Setting the size of the figure based on the no. of categories

        fig, ax = plt.subplots(figsize=(12,len(temp_df)))



        p=sns.barplot(x='counts',y='category',data=temp_df,color='b')

        sns.set_style('whitegrid')

        

        #Fetching the actual question of the column from the schema dataframe.

        #The question can end either in a '?' or a '.'

        q=sch['Question'][sch['Column']==colname].item()

        char='?'

        if q.find(char)==-1:

            char='.'

        p.axes.set_title(q[:q.find(char)+1])

        

        p.set_xlabel('')

        p.set_ylabel('')

        sns.despine()
plotSelectFields(MCR,sch)