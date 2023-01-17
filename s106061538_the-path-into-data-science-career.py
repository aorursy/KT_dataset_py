import numpy as np

import pandas as pd

import matplotlib

import seaborn as sns

matplotlib.style.use('ggplot')

import matplotlib.pyplot as plt



################################################



### target group division



def dividetargetgroup(answer):

    

    # Q5 -> career titles (no student and unemployed)



    q5data = answer.dropna(subset=['Q5'])

    studentsdata = answer[answer['Q5']=='Student']

    unemployeddata = answer[answer['Q5']=='Not employed']

    employeddata = q5data[~q5data['Q5'].isin(['Student','Not employed'])]

    

    # Q8 -> business w. ML

    

    employeddatanoML = employeddata.loc[(employeddata['Q8'] == 'No (we do not use ML methods)') | (employeddata['Q8'] == 'I do not know') ]

    employeddataML = employeddata.loc[(employeddata['Q8'] != 'No (we do not use ML methods)') & (employeddata['Q8'] != 'I do not know') ]

    employeddataML = employeddataML.dropna(subset=['Q8'])

    

    # Q9 -> job contents

    

    employeddataML_none = employeddataML[employeddataML['Q9_Part_7'] =='None of these activities are an important part of my role at work']

    employeddataML = employeddataML[employeddataML['Q9_Part_7'] != 'None of these activities are an important part of my role at work']



    employeddataML['Q9nans'] = employeddataML[['Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_8']].isnull().sum(axis=1)

    employeddataML_nan_onlyothers = employeddataML[(employeddataML['Q9nans']==7) | ((employeddataML['Q9nans']==6) & (employeddataML['Q9_Part_8']=='Other'))]

    employeddataML.drop(index=employeddataML_nan_onlyothers.reset_index()['index'].to_list(),inplace = True) # 8373 -> 7912

    employeddataML['Q9nans'] = employeddataML['Q9nans']-1

    

    return employeddataML



### Plots





def pd_plots_6groups_multi1(Qi, nsub, inputdatagroups, first, origdatatype=None):



    datatype = []

    datacount = []

    

    # nans -> find total responsdents

    

    columnlist = ['Q'+str(Qi)+'_Part_'+str(i+1) for i in range(nsub)] # create list

    inputdatagroups['Q'+str(Qi)+'nans'] = inputdatagroups[columnlist].isnull().sum(axis=1)

    groupqueslen = len(inputdatagroups['Q'+str(Qi)+'nans']<nsub)

    

    for i in range(nsub):

        

        # datatype

        

        if first:

            datatype.append(inputdatagroups['Q'+str(Qi)+'_Part_'+str(i+1)].value_counts().index.to_list()[0])

        else:

            datatype = origdatatype

        

        # datacount

        

        if len(inputdatagroups['Q'+str(Qi)+'_Part_'+str(i+1)].value_counts().to_list())>0:

            datacount.append(inputdatagroups['Q'+str(Qi)+'_Part_'+str(i+1)].value_counts().to_list()[0])

        else:

            datacount.append(0)

        

    pddata = pd.DataFrame({'counts': [x/groupqueslen*100 for x in datacount]}, index = datatype) # percentage

    

    if first:

        return pddata, datatype

    else:

        return pddata, None

    

def plots_multi_1group(inputdatagroups, inputtitle, Qi, nsub, first, origdatatype): 

    datatype = []

    datacount = []

    for i in range(nsub):

        if first:

            datatype.append(inputdatagroups['Q'+str(Qi)+'_Part_'+str(i+1)].value_counts().index.to_list()[0])

        else:

            datatype = origdatatype

        if len(inputdatagroups['Q'+str(Qi)+'_Part_'+str(i+1)].value_counts().to_list())>0:

            datacount.append(inputdatagroups['Q'+str(Qi)+'_Part_'+str(i+1)].value_counts().to_list()[0])

        else:

            datacount.append(0)

    pddata = pd.DataFrame({'counts': datacount, 'datatype': datatype})

    pddata.set_index('datatype').sort_values(by='counts').plot(kind='barh', title=inputtitle)

    

    return datatype



def plots_single_1group(inputdatagroups, inputtitle, Qi, charttype):

    

    plt.figure()

    inputdatagroups['Q'+str(Qi)].value_counts(ascending=True).plot(kind=charttype, title=inputtitle)



    return pddata

    

def plots_6groups(Qi, Q9_choice_select, nsub=None, multichoice=None, categoryorders=None):

    

    # Build table

 

    grouplist = []

    for i in range(6):

        if multichoice:

            if i==0:

                pddata, origdatatype = pd_plots_6groups_multi1(Qi, nsub, Q9_choice_select[i],1)

            else:

                pddata, _ = pd_plots_6groups_multi1(Qi, nsub, Q9_choice_select[i], 0, origdatatype)

            grouplist.append(pddata)

        else:

            grouplist.append(Q9_choice_select[i]['Q'+str(Qi)].value_counts(normalize=True)*100) # normalize within group   

    

    grouptable = pd.concat([grouplist[0],grouplist[1]],axis=1, sort=True)

    for i in range(4):

        grouptable = pd.concat([grouptable,grouplist[i+2]],axis=1, sort=True)

    grouptable.columns = targetgroupnames

    

    # Plot



    ax1 = grouptable.reindex(categoryorders).transpose().plot(kind='bar', figsize=(11,5), rot=0) 

    ax1.set(ylabel="percentage")

    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    

################################################



### Read files 



answer = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

answer_other = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')

survey_schema = pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')

answer = answer[1:]



### Target group



employeddataML = dividetargetgroup(answer)

targetgroupnames = ['Analyze data', 'Data infrastructure', 'ML prototypes', 'ML service', 'ML experimentation', 'ML research']



### Target group division



# Q9 groups



Q9_choice_none = []

Q9_choice_select = []

Q9_choice_only = []

Q9_choice_more = []



for i in range(6):

    Q9_choice_none.append(employeddataML[employeddataML['Q9_Part_'+str(i+1)].isnull()])

    Q9_choice_select.append(employeddataML.drop(index=Q9_choice_none[-1].reset_index()['index'].to_list()))

    Q9_choice_select_data = Q9_choice_select[-1]

    Q9_choice_only.append(Q9_choice_select_data[Q9_choice_select_data['Q9nans']==5])

    Q9_choice_more.append(Q9_choice_select_data[Q9_choice_select_data['Q9nans']<5])
groupnum = []

for i in range(6):

    groupnum.append(len(Q9_choice_select[i]))

groupnumpercentage = [x/len(employeddataML)*100 for x in groupnum]

groupnum = pd.DataFrame(groupnumpercentage, index = targetgroupnames).sort_values(by=0, ascending=True).rename(columns={0:'subject count percentage'})

ax = groupnum.plot(kind='barh')

_ = ax.set(xlabel="percentage")
plots_6groups(5, Q9_choice_select)
### duty 



## amounts



fig, axes = plt.subplots(nrows=1, ncols=2)

fig.set_figheight(3)

fig.set_figwidth(10)

df1 = (employeddataML['Q9nans'].value_counts(normalize=True)*100).reset_index()

df1['duty amounts'] = 6-df1['index']

df1 = df1.sort_values(by=['duty amounts'])

ax1 = df1.set_index('duty amounts').drop(columns=['index']).rename(columns={"Q9nans": "subject counts percentage"}).plot(kind='barh',ax=axes[0])

ax1.set(xlabel="percentage", ylabel="duty amounts")



## ratio of only vs more



only = []

more = []

# print('none','select','only','more')

for i in range(6):

    only.append(len(Q9_choice_only[i]))

    more.append(len(Q9_choice_more[i]))

    #print(len(Q9_choice_none[i]),len(Q9_choice_select[i]),len(Q9_choice_only[i]),len(Q9_choice_more[i]))

df = pd.DataFrame(data={'single duty': only, 'more duties': more, 'index': targetgroupnames}).set_index('index')

df = df.div(df.sum(axis=1), axis=0)*100

ax2 = df.plot(kind='barh', stacked=True, rot=0, ax=axes[1])

ax2.set(xlabel="percentage", ylabel="duty type")



plt.tight_layout()
# people with differet duty amounts vs data team size



employeddataML['duty counts'] = 6-employeddataML['Q9nans']

df3 = employeddataML.copy().rename(columns={'Q7':'employees responsible in DS'}).groupby(['employees responsible in DS','duty counts']).size().unstack(fill_value=0)

df3 = df3.reindex(['0','1-2','3-4','5-9','10-14','15-19','20+']).iloc[:, ::-1].transpose()

sns.heatmap(df3, annot=True, fmt=".1f", cmap="Blues")

plt.show()
df = employeddataML.copy().rename(columns={'Q7':'Employees responsible in DS'}).pivot_table(index='Q8', columns='Employees responsible in DS', values='duty counts',aggfunc=len)



df['Business ML Condition'] = ['Exploring ML methods', 'Well established ML methods', 'Recently starting ML methods', 'Generating insights with ML']

df = df.set_index('Business ML Condition').reindex(['Generating insights with ML', 'Recently starting ML methods', 'Exploring ML methods', 'Well established ML methods'])

df = df.reindex(columns=['0','1-2','3-4','5-9','10-14','15-19','20+'])



sns.heatmap(df, annot=True, fmt=".1f", cmap="Blues")

plt.show()
categoryorders = ['No formal education past high school', 'Some college/university study without earning a bachelor\'s degree', 'Bachelors degree ', 'Masters degree', 'Doctoral degree', 'Professional degree', 'I prefer not to answer'] # categoryrename

plots_6groups(4, Q9_choice_select)
categoryorders = ['I have never written code', '< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']

plots_6groups(15, Q9_choice_select, categoryorders=categoryorders)
categoryorders = ['< 1 years', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', '10-15 years', '20+ years'] # categoryrename

plots_6groups(23, Q9_choice_select, categoryorders=categoryorders)
plots_6groups(18, Q9_choice_select, nsub=12, multichoice=1)
plots_6groups(14, Q9_choice_select)
plots_6groups(28, Q9_choice_select, nsub=12, multichoice=1)
plots_6groups(24, Q9_choice_select, nsub=12, multichoice=1)
plots_6groups(13, Q9_choice_select, nsub=12, multichoice=1)
plots_6groups(12, Q9_choice_select, nsub=12, multichoice=1)
plots_6groups(20, Q9_choice_select, nsub=12, multichoice=1)
plots_6groups(31, Q9_choice_select, nsub=12, multichoice=1)
plots_6groups(34, Q9_choice_select, nsub=12, multichoice=1)
plots_6groups(25, Q9_choice_select, nsub=8, multichoice=1)
plots_6groups(33, Q9_choice_select, nsub=12, multichoice=1)
plots_6groups(29, Q9_choice_select, nsub=8, multichoice=1)