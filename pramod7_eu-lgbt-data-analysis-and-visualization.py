import pandas as pd

import matplotlib.pyplot as plt
#Load all the CSV files and create dataframes

csv1=pd.read_csv('../input/european-union-lgbt-survey-2012/LGBT_Survey_DailyLife.csv')

csv2=pd.read_csv('../input/european-union-lgbt-survey-2012/LGBT_Survey_Discrimination.csv')

csv3=pd.read_csv('../input/european-union-lgbt-survey-2012/LGBT_Survey_RightsAwareness.csv')

csv4=pd.read_csv('../input/european-union-lgbt-survey-2012/LGBT_Survey_TransgenderSpecificQuestions.csv')

csv5=pd.read_csv('../input/european-union-lgbt-survey-2012/LGBT_Survey_ViolenceAndHarassment.csv')

subset=pd.read_csv('../input/european-union-lgbt-survey-2012/LGBT_Survey_SubsetSize.csv')
print(subset.head())
print(subset.info())
# having our X and Y for Bar & Pie chart ready

lables = subset.columns

lables=lables[2:]

values=subset.iloc[0]

values=values[2:]



# plotting a figure

fig=plt.figure()

fig.set_figheight(4)

fig.set_figwidth(10)



# Figure A is for bar graph

a_fig=fig.add_subplot(1,2,1)

l1=plt.bar(lables, values)

l1[1].set_color('r')

plt.xticks(fontsize=7, rotation=30)

plt.xlabel('LGBT Groups')

plt.ylabel('Total Number of people')

plt.title('LGBT Metadata',fontsize=15)



# Figure B is for pie chat

b_fig=fig.add_subplot(1,2,2)

explode=(0,0.2,0,0,0)

plt.pie(values,labels=lables,shadow=True, startangle=90,explode=explode)

plt.title('LGBT Metadata',fontsize=15)

gay_countries=subset.groupby('CountryID').agg({'Gay men':'sum'})

top_10_gay_countries=gay_countries.sort_values('Gay men',ascending=False).head(11)

top_10_gay_countries=top_10_gay_countries.reset_index()

top_10_gay_x=top_10_gay_countries['CountryID'][1:]

top_10_gay_y=top_10_gay_countries['Gay men'][1:]



fig1=plt.figure()

fig1.set_figheight(4)

fig1.set_figwidth(10)



a_fig1=fig1.add_subplot(1,2,1)

l2=plt.bar(top_10_gay_x,top_10_gay_y)

l2[0].set_color('r')

plt.xticks(rotation=30,fontsize=10)

plt.xlabel('Top 10 Gay Populated EU Countries')

plt.ylabel('NO of Gay men')

plt.title('Gay Men Countries', fontsize=15)



b_fig1=fig1.add_subplot(1,2,2)

explode=(0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)

plt.pie(top_10_gay_y,labels=top_10_gay_x,shadow=True,explode=explode)

plt.title('Gay Men Countries', fontsize=15)
lesbian_women=subset.groupby('CountryID').agg({'Lesbian women':'sum'})

top_10_lesbian_countries=lesbian_women.sort_values('Lesbian women',ascending=False).head(11)

top_10_lesbian_countries=top_10_lesbian_countries.reset_index()

top_10_lesbian_x=top_10_lesbian_countries['CountryID'][1:]

top_10_lesbian_y=top_10_lesbian_countries['Lesbian women'][1:]



fig2=plt.figure()

fig2.set_figheight(4)

fig2.set_figwidth(10)



a_fig2=fig2.add_subplot(1,2,1)

l3=plt.bar(top_10_lesbian_x,top_10_lesbian_y)

l3[0].set_color('r')

plt.xticks(rotation=30,fontsize=10)

plt.xlabel('Top 10 Lesbian Populated EU Countries')

plt.ylabel('NO of Lesbian Women')

plt.title('Lesbian Women Countries', fontsize=15)



b_fig2=fig2.add_subplot(1,2,2)

explode=(0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)

plt.pie(top_10_lesbian_y,labels=top_10_lesbian_x,shadow=True,explode=explode)

plt.title('Lesbian Women Countries', fontsize=15)
transgender=subset.groupby('CountryID').agg({'Transgender':'sum'})

top_10_ts_countries=transgender.sort_values('Transgender',ascending=False).head(11)

top_10_ts_countries=top_10_ts_countries.reset_index()

top_10_ts_x=top_10_ts_countries['CountryID'][1:]

top_10_ts_y=top_10_ts_countries['Transgender'][1:]



fig3=plt.figure()

fig3.set_figheight(4)

fig3.set_figwidth(10)



a_fig3=fig3.add_subplot(1,2,1)

l4=plt.bar(top_10_ts_x,top_10_ts_y)

l4[0].set_color('r')

plt.xticks(rotation=30,fontsize=10)

plt.xlabel('Top 10 Transgender Populated EU Countries')

plt.ylabel('NO of Transgender Women')

plt.title('Transgender Countries', fontsize=15)



b_fig3=fig3.add_subplot(1,2,2)

explode=(0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)

plt.pie(top_10_ts_y,labels=top_10_ts_x,shadow=True,explode=explode)

plt.title('Transgender', fontsize=15)
LGBT=subset.groupby('CountryID').agg({'N':'sum'})

top_10_LGBT_countries=LGBT.sort_values('N',ascending=False).head(11)

top_10_LGBT_countries=top_10_LGBT_countries.reset_index()

top_10_LGBT_x=top_10_LGBT_countries['CountryID'][1:]

top_10_LGBT_y=top_10_LGBT_countries['N'][1:]



fig4=plt.figure()

fig4.set_figheight(4)

fig4.set_figwidth(10)



a_fig4=fig4.add_subplot(1,2,1)

l5=plt.bar(top_10_LGBT_x,top_10_LGBT_y)

l5[0].set_color('r')

plt.xticks(rotation=30,fontsize=10)

plt.xlabel('Top 10 LGBT Populated EU Countries')

plt.ylabel('NO of LGBT')

plt.title('LGBT Countries', fontsize=15)



b_fig4=fig4.add_subplot(1,2,2)

explode=(0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)

plt.pie(top_10_LGBT_y,labels=top_10_LGBT_x,shadow=True,explode=explode)

plt.title('LGBT', fontsize=15)
# Classifying all of the answers in to 3 classes, Widespread, rare and Dont know

Widespread=['Strongly agree', 'Agree','Most', 'All', '6 times or more in the last six months','Always open','Fairly open',

       'Very open','Very widespread', 'Fairly widespread','6','7','8','9','10','Yes' ]



Rare=['Fairly rare', 'Very rare','Disagree','Strongly disagree', 'No','None', 'A few','Never happened in the last sixth months',

       'Happened only once in the last six months','2-5 times in the last six months','selectively open','Never Open', 'Rarely Open','0','1','2','3','4']



dont_know=['Don`t know','I do not have a partner (Does not apply to me)','Single',

       'Married/in a registered partnership', 'Divorced', 'Separated',

       'Widowed', 'Living together with a partner /spouse',

       'Involved in a relationship without living together',

       'Have no relationship / do not have a partner',

       'I did not need or use any benefits or services',

       'An ethnic minority (including of migrant background)',

       'A sexual minority',

       'A minority in terms of disability (excluding diagnosis of gender dysphoria/gender identity disorder)',

       'A religious minority', 'Other minority group',

       'None of the above',

       'I read about it in a newspaper (online or printed)',

       'I received an email from an organisation or online network',

       'Somebody told me about it or sent me the link',

       'Through social media (facebook, twitter or etc.)',

       'I saw an advertisement (banner) online, please specify where:',

       'Somewhere else', 'hide LGBT identity','5','Current situation is fine']



Q1=csv1.loc[csv1['question_code']=='b1_a']

Q1['answer_1']=''
# lets analyse b1_a question response

for ix,val in enumerate(Q1.answer):

    if Q1['answer'].iloc[ix] in Widespread :

        Q1['answer_1'].iloc[ix]='Widespread'

    if Q1['answer'].iloc[ix] in Rare :

        Q1['answer_1'].iloc[ix]='Rare'

    if Q1['answer'].iloc[ix] in dont_know :

        Q1['answer_1'].iloc[ix]="Don`t know"
Q1=Q1.replace(':','0')

Q1=Q1[['CountryCode','answer_1','subset','percentage']]

Q1['percentage']=Q1['percentage'].astype(str).astype(int)



B1_a_rare=Q1.loc[Q1['answer_1']=='Rare']

B1_a_rare=B1_a_rare.groupby(['CountryCode','answer_1','subset'])['percentage'].sum()

B1_a_rare=B1_a_rare.reset_index()

B1_a_rare=B1_a_rare.sort_values('percentage',ascending=False).head(11)

print('These are the top 10 communities in the respective country feel that their politician is not abusive about LGBT')

print('\t')

print(B1_a_rare)
B1_a_WS=Q1.loc[Q1['answer_1']=='Widespread']

B1_a_WS=B1_a_WS.groupby(['CountryCode','answer_1','subset'])['percentage'].sum()

B1_a_WS=B1_a_WS.reset_index()

B1_a_WS=B1_a_WS.sort_values('percentage',ascending=False).head(10)



print('These are the top 10 communities in the respective country feel that their politicians are really verbal about LGBT')

print('\t')

print(B1_a_WS)
B1_a_nu=Q1.loc[Q1['answer_1']=="Don`t know"]

B1_a_nu=B1_a_nu.groupby(['CountryCode','answer_1','subset'])['percentage'].sum()

B1_a_nu=B1_a_nu.reset_index()

B1_a_nu=B1_a_nu.sort_values('percentage',ascending=False).head(10)



print('These are the top 10 communities in the respective countries who do not know what their politicians feel about LGBT')

print('\t')

print(B1_a_nu)
# lets analyze what LGBT from Denmark feel

B1_a_den=Q1.loc[Q1['CountryCode']=="Denmark"]

B1_a_den=B1_a_den.groupby(['CountryCode','answer_1','subset'])['percentage'].sum()

B1_a_den=B1_a_den.reset_index()

B1_a_den=B1_a_den.sort_values('percentage',ascending=False)



print('This is what LGBT from Denmark says about their politicans')

print('\t')

print(B1_a_den)
# lets analyze what LGBT from UK feel

B1_a_UK=Q1.loc[Q1['CountryCode']=="United Kingdom"]

B1_a_UK=B1_a_UK.groupby(['CountryCode','answer_1','subset'])['percentage'].sum()

B1_a_UK=B1_a_UK.reset_index()

B1_a_UK=B1_a_UK.sort_values('percentage',ascending=False)



print('This is what LGBT from UK says about their politicans')

print('\t')

print(B1_a_UK)