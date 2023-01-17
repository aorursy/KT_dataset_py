# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/datasocietypsdie2017/35017.Client0105.csv')
df.head() #load the first 5 rows of the D&S dataframe using this method

print(df.shape) #show me the shape of this dataframe 
#using the loc method, keep all rows but drop all columns other than these

df=df.loc[:,['hisp', 'birth_hisp', 'lang','q15x','q17a','q17b','q17c','q17d','q17e','q17f','q17g','q17h']]

print(df.shape) #confirm the smaller size



#create new df using the query method, filter df dataframe for respondents who identified as Hispanic and took the survey in English 

#we are comparing English vs Spanish speakers in the Latinx community, which is why we are excluding other English speakers from this analysis

dfhe = df.query('hisp == "1" & lang == "1"')

print(dfhe.shape) #see the size of this dataframe



dfsp=df[df['lang']==2] #create new dataframe of only Spanish language survey results

print(dfsp.shape) #return the shape of this new dataframe to see the change in size 
dfhe.isnull().sum() #check to see if the blank spaces that showed in the output of the previous dataframe are null values or blank spaces

#they are blank values, not 0; I assume these correspond to skipped responses (survey documentation unclear)
sk15=dfhe[dfhe['q15x']==' '].index #get the indexes (indicies?) for which column Q15 has value " ", name this output

dfHE15=dfhe.drop(sk15) #Delete that output from dataFrame so we have no skipped responses for Q15

print(dfHE15.shape) #see how many complete responses we have now

dfHE15=dfHE15.drop(['q17a','q17b','q17c','q17d','q17e','q17f','q17g','q17h'], axis=1) #drop the Q17 columns

print(dfHE15.shape)



sk15s=dfsp[dfsp['q15x']==' '].index #get the indexes (indicies?) for which column Q15 has value " ", name this output

dfS15=dfsp.drop(sk15s) #Delete that output from dataFrame so we have no skipped responses for Q15

print(dfS15.shape) #see how many complete responses we have now

dfS15=dfS15.drop(['q17a','q17b','q17c','q17d','q17e','q17f','q17g','q17h'], axis=1) #drop the Q17 columns

print(dfS15.shape) #confirm adjustment



#repeat for Q17 in both languages

sk17=dfhe[(dfhe['q17a']==' ')|(dfhe['q17b']==' ')|(dfhe['q17c']==' ')|(dfhe['q17d']==' ')|(dfhe['q17e']==' ')|(dfhe['q17f']==' ')|(dfhe['q17g']==' ')|(dfhe['q17h']==' ')].index

dfHE17=dfhe.drop(sk17) #Delete that output from dataFrame so we have no skipped responses for Q17

print(dfHE17.shape)

dfHE17=dfHE17.drop(['q15x'], axis=1) #drop the Q15 column

print(dfHE17.shape) #confirm adjustment



sk17s=dfsp[(dfsp['q17a']==' ')|(dfsp['q17b']==' ')|(dfsp['q17c']==' ')|(dfsp['q17d']==' ')|(dfsp['q17e']==' ')|(dfsp['q17f']==' ')|(dfsp['q17g']==' ')|(dfsp['q17h']==' ')].index

dfS17=dfsp.drop(sk17s) #Delete that output from dataFrame so we have no skipped responses for Q17

print(dfS17.shape)

dfS17=dfS17.drop(['q15x'], axis=1) #drop the Q15 column

print(dfS17.shape) #confirm adjustment
dfHE15.head()
#return the unique value counts of the values in column q15x, sort these in ascending order, and create a new dataframe

fifteenE=pd.DataFrame(dfHE15['q15x'].value_counts (ascending=True)) 

print(type(fifteenE)) #confirm this is a dataframe and not a series



#add a new column to this dataframe that containes the % of the whole using normalize=True

fifteenE['Percent']=dfHE15['q15x'].value_counts(normalize=True) 

dfHE15['q15x'].describe() #run some descriptive stats on this dataset; this is nominal data so there is limited analysis

#there are no answers for response option 9, would normally update the dataframe with this info, but can't figure out how to update the index so will skip for time



#clean this up for graphing 

print(fifteenE) #check the output including formatting

print(fifteenE.columns) #check column names, decide to rename them

fifteenE.rename(columns={'Index':'Response', 'q15x': 'Response_Ct'}, inplace=True) #rename them

print(fifteenE.columns) #check response 

print(fifteenE.index) #can't tell if I have correctly changed the index column name or if that's necessary

print(fifteenE) #check the output



fifE=fifteenE.sort_index()

fifE.rename(index={'1':'Very_Easy', '2': 'Somewhat_Easy', '3':'Somewhat_Difficult', '4':'Very_Difficult', '8':'DK'},inplace=True)

#fifteenE['Index']=np.where((fifteenE.Index=='3'),'Somewhat Easy',fifteenE.Index) #update response code with text

print(fifE)

fifE['Percent'] = fifE['Percent'].multiply(100).round(2) #update the Percent column by multiplying values by 100 and rounding

print(fifE)
#return the unique value counts of the values in column q15x, sort these in ascending order, and create a new dataframe

fifteenS=pd.DataFrame(dfS15['q15x'].value_counts (ascending=True)) 

print(type(fifteenS)) #confirm this is a dataframe and not a series



#add a new column to this dataframe that containes the % of the whole using normalize=True

fifteenS['Percent']=dfS15['q15x'].value_counts(normalize=True) 

dfS15['q15x'].describe() #run some descriptive stats on this dataset; this is nominal data so there is limited analysis

#there are no answers for response option 9, would normally update the dataframe with this info, but can't figure out how to update the index so will skip for time



#clean this up for graphing 

print(fifteenS) #check the output including formatting

print(fifteenS.columns) #check column names, decide to rename them

fifteenS.rename(columns={'Index':'Response', 'q15x': 'Response_Ct'}, inplace=True) #rename them

print(fifteenS.columns) #check response 

print(fifteenS.index) #can't tell if I have correctly changed the index column name or if that's necessary

print(fifteenS) #check the output



fifS=fifteenS.sort_index()

fifS.rename(index={'1':'Very_Easy', '2': 'Somewhat_Easy', '3':'Somewhat_Difficult', '4':'Very_Difficult', '8':'DK'},inplace=True)

#fifteenE['Index']=np.where((fifteenE.Index=='3'),'Somewhat Easy',fifteenE.Index) #update response code with text

print(fifS)

fifS['Percent'] = fifS['Percent'].multiply(100).round(2) #update the Percent column by multiplying values by 100 and rounding

print(fifS)
#df = pd.DataFrame({"Series Type":['S1','S1','S1','S2','S2','S3']})

##display(df)

#df2 = pd.DataFrame({"Series Type":['S1','S2','S2','S2','S3','S3','S3','S3']})

##display(df2)

#df['Channel'] = 'Carton Network'

#df2['Channel'] = 'Disney'

##combine it

#df_cmb = pd.concat([df, df2])

#display(df_cmb)

#print(df_cmb.groupby(['Series Type'])['Channel'].value_counts())

#df_cmb.groupby(['Series Type'])['Channel'].value_counts().unstack().plot(kind='bar')



fifEp=fifE.drop(['Response_Ct'], axis=1) #I'm not interested in graphing this column, drop it

fifSp=fifS.drop(['Response_Ct'], axis=1) #""

print(fifEp.shape) #confirm it worked

print(fifSp.shape) #""



fifEp['Key'] = 'e'# add a key column to the datframe with this value

fifSp['Key'] = 's'# " "



print(fifEp.columns)



df = pd.DataFrame({"Series Type":['S1','S1','S1','S2','S2','S3']})

#display(df)

df2 = pd.DataFrame({"Series Type":['S1','S2','S2','S2','S3','S3','S3','S3']})

#display(df2)

df['Channel'] = 'Carton Network'

df2['Channel'] = 'Disney'

#combine it

df_cmb = pd.concat([fifEp, fifSp])

display(df_cmb)

print(df_cmb.groupby(['Percent'])['Key'].value_counts())

df_cmb.groupby(['Percent'])['Key'].value_counts().unstack().plot(kind='bar')



print(df_cmb.groupby(['Percent'])['Key'])

#trying to remove the value_counts() argument from the code returns errors 

df_cmb.groupby(['Percent'])['Key'].plot(kind='bar') #TypeError: no numeric data to plot

df_cmb.groupby(['Percent'])['Key'].unstack().plot(kind='bar') #AttributeError: 'SeriesGroupBy' object has no attribute 'unstack'

fifEp=fifE.drop(['Response_Ct'], axis=1) #I'm not interested in graphing this column, drop it

fifSp=fifS.drop(['Response_Ct'], axis=1) #""

print(fifEp.shape) #confirm it worked

print(fifSp.shape) #""

fifEp.Percent=pd.to_numeric(fifEp.Percent) #a Google search result recommended that I try this to fix my error,, but it didn't work

fifSp.Percent=pd.to_numeric(fifSp.Percent)#" "



fifEp['Key'] = 'e'# add a key column to the datframe with this value

fifSp['Key'] = 's'# " "

DF = pd.concat([fifEp,fifSp],keys=['e','s']) #create a new pandas dataframe by combining these two referenced dataframes, reference these keys 

DFGroup = DF.groupby(['Percent','Key']) #create an object DFGroup [is this a dataframe? a groupby object?] that is the DF dataframe grouped by these two series

DFGroup.sum().unstack('Key').plot(kind='bar') #plot what is unstacked
#this attempt plots both columns, just to see if the data is plottable without getting the error in the code chunj

#I was able to plot this on my initial attempt of typing the code, but after fiddling with it, I can't get back to it.

#I swear I am loading the exact same initial code, but I still get an error

DF2 = pd.concat([fifE,fifS],keys=['e','s']) #create a new pandas dataframe

DFGroup2 = DF.groupby(['Percent','Key'])

DFGroup2.sum().unstack('Key').plot(kind='bar')
new15=pd.DataFrame({'Response':['Very_Easy', 'Somewhat_Easy', 'Somewhat_Difficult', 'Very_Difficult', 'DK'],

                    "En_Percent":[34.50, 40.70, 17.05, 6.59, 1.16],

                    "Es_Percent":[7.75, 33.80, 37.32, 20.42, 0.70]})

print(new15) #check the output

new15.plot.bar() #plot this in a bar chart

#make the plot look clearer with more specific code



ax=new15.plot(x='Response', y=['En_Percent','Es_Percent'], kind='bar', rot=50, title='Q15 Latinx Response Composition by Language')

ax.set_ylabel('% Response Composition')

ax.set_xlabel('Response')



#rename the legend if there is time

#I would also like to add labels to each specific bar height but ran out of time
# Setting the positions and width for the bars

pos = list(range(len(new15['En_Percent']))) 

width = 0.25 

    

# Plotting the bars

fig, ax = plt.subplots(figsize=(10,5))



# Create a bar with pre_score data,

# in position pos,

plt.bar(pos, 

        #using df['pre_score'] data,

        new15['En_Percent'], 

        # of width

        width, 

        # with alpha 0.5

        alpha=0.5, 

        # with color

        color='#EE3224', 

        # with label the first value in first_name

        label=new15['Response'][0]) 



# Create a bar with mid_score data,

# in position pos + some width buffer,

plt.bar([p + width for p in pos], 

        #using df['mid_score'] data,

        new15['Es_Percent'],

        # of width

        width, 

        # with alpha 0.5

        alpha=0.5, 

        # with color

        color='#F78F1E', 

        # with label the second value in first_name

        label=new15['Response'][1]) 



# Create a bar with post_score data,

# in position pos + some width buffer,

#plt.bar([p + width*2 for p in pos], 

        #using df['post_score'] data,

        #df['post_score'], 

        # of width

        #width, 

        # with alpha 0.5

        #alpha=0.5, 

        # with color

        #color='#FFC222', 

        # with label the third value in first_name

        #label=df['first_name'][2]) 



# Set the y axis label

ax.set_ylabel('Score')



# Set the chart's title

ax.set_title('Test Subject Scores')



# Set the position of the x ticks

ax.set_xticks([p + 1.5 * width for p in pos])
df1517=df[['lang','q15x','q17a','q17b','q17c','q17d','q17e','q17f','q17g','q17h']].copy() #create new dataframe with all responses to only to questions 15 and 17 

print(df1517.shape) #print this to confirm the above worked

print(df1517.groupby('lang')) #I tried to use the .groupby operation, reviewed the documentation, messed around for 90 minutes and never figured it out

df15lang=df[['lang','q15x']] #make a smaller dataframe with only these two columns

#df15lang.groupby('q15x').value_counts() #using documentation guidance,show the count of responses for q15x by language (this didn't work)

#fifcountsall=df15lang['q15x'].value_counts() #another attempt at showing the count of responses for q15x by language (also didn't work)

Endf15lang=df15lang[df['lang']==1] #create new dataframe filtering rows of lang where value is 1

Endf15lang=df15lang.drop('lang', axis=1)#remove the column 'lang' from this dataframe

Endf15lang.head()#inspect the first 5 rows of this dataframe





Endf15lang.isnull().sum() #check to see if the blank spaces that showed in the output of the previous dataframe are null values or blank spaces

#they are blank values, not 0; these correspond to skipped response
Endf15lang['q15x'].value_counts() #count the frequencies of each response to q15x in the dataframe Endflang
#I tried eng15counts=Endf15lang['q15x'].value_counts(), columns=('respons','totals') and received a "can't assign to function call" error

#I also tried using the grouby and count operations but I don't think my data was structured properly for this to work      

#my code for that was: q15perlang=df15lang.groupby(['lang','q15x']).count()



#lastly, I tried to make the dataframe using the pivot operation, but couldn't figure out how to get an excel-like "CountIF" operation to fill in for values

#this attempted code was df15lang.pivot(index='lang', columns='q15x', values=df15lang['q15x'].value_counts())
eng15counts=pd.DataFrame([["2",893],["1",849],["Skipped",650],["3",355],["4",202],["8",42],["9",9]], columns=("Responses","Totals")) #create a new dataframe

print(eng15counts)
eQ15totr=eng15counts['Totals'].sum()-eng15counts.at[2,'Totals'] #create a variable for the sum of the values in "totals" column minus the "Skipped" scalar(? is this the right word) value

print(eQ15totr)

#create the list 'perc'

perc=[eng15counts.at[0,'Totals']/eQ15totr,eng15counts.at[1,'Totals']/eQ15totr,0,eng15counts.at[3,'Totals']/eQ15totr,eng15counts.at[4,'Totals']/eQ15totr,eng15counts.at[5,'Totals']/eQ15totr, eng15counts.at[6,'Totals']/eQ15totr,]

print(perc)

#convert the list to percentage formattting at 2 decimal places: I didn't get around to this

#for x in perc:

    #x*100

    #round(2)

eng15counts['Percent']=perc

print(eng15counts)
S15lang=df15lang[df['lang']==2] #create new dataframe filtering rows of lang where value is 2

S15lang=S15lang.drop('lang', axis=1)#remove the column 'lang' from this dataframe

S15lang.head()#inspect the first 5 rows of this dataframe
S15lang['q15x'].value_counts() #count the frequencies of each response to q15x in the dataframe 
sp15counts=pd.DataFrame([["Skipped",117],["3",53],["2",48],["4",29],["1",11],["8",1],["9",0]], columns=("Responses","Totals")) #create a new dataframe

print(sp15counts)
sQ15totr=sp15counts['Totals'].sum()-sp15counts.at[0,'Totals'] #create a variable for the sum of the values in "totals" column minus the "Skipped" scalar(? is this the right word) value

print(eQ15totr)

#create the list 'perc'

sperc=[0,sp15counts.at[1,'Totals']/sQ15totr,sp15counts.at[2,'Totals']/sQ15totr,sp15counts.at[3,'Totals']/sQ15totr,sp15counts.at[4,'Totals']/sQ15totr,sp15counts.at[5,'Totals']/sQ15totr, sp15counts.at[6,'Totals']/sQ15totr]

print(sperc)

#convert the list to percentage formattting at 2 decimal places: I didn't get around to this

#for x in perc:

    #x*100

    #round(2)

sp15counts['Percent']=sperc

print(sp15counts)
E15countsr=eng15counts[eng15counts.Responses !='Skipped'] #make this new dataframe for English responses of rows associated with values in the Responses column that are not equal to "Skipped"

E15countsr.sort_values(by=['Responses'], inplace=True)

E15countsr['Percent'] = E15countsr['Percent'].multiply(100).round(2) #update the Percent column by multiplying values in this column by 100 and rounding 2 decimal places

print(E15countsr)

S15countsr=sp15counts[sp15counts.Responses !='Skipped'] #make this new dataframe for Spanish responses of rows associated with values in the Responses column that are not equal to "Skipped"

S15countsr.sort_values(by=['Responses'], inplace=True)

S15countsr['Percent'] = S15countsr['Percent'].multiply(100).round(2) #update the Percent column by multiplying values in this column by 100 and rounding 2 decimal places

print(S15countsr)



#I recognize that python is giving me these warnings. In the interest of time, I am going to move on to doing the visualizing my dataframes rather than troubleshooting these 

#/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 

#A value is trying to be set on a copy of a slice from a DataFrame



#See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

  

#/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 

#A value is trying to be set on a copy of a slice from a DataFrame.

#Try using .loc[row_indexer,col_indexer] = value instead

import matplotlib.pyplot as plt 

plt.style.use('seaborn-colorblind') #use this visual style for plots

E15countsr['Key']='English'#add a new column to this dataframe, make the value for all rows "English"

S15countsr['Key']='Spanish'#add a new column to this dataframe, make the value for all rows "Spanish"

both=pd.concat([E15countsr,S15countsr], keys=['English','Spanish']) #create a new dataframe by concatenating the two referenced dataframes

                                                            #use these referenced items as the keys

DFGroup=both.groupby(['Responses','Key']) #group the dataframe by responses and the key column values (?)

DFGroup.sum().unstack('Key').plot(kind='bar', y='Percent').set_ylabel("% of Language Responses (excludes skipped)") #I don't fully understand these instructions except that the data is unpacked and I think reshaped

                                                            #the results are then plotted into a bar graph

                                                            #Make the Y axis the percent values with label indicated above

#I tried to customize the Xaxis ticks and to add a graph label, but got stuck and ran out of time

df2=df1517.melt(id_vars=['lang']) #reshape data so that questions and their values are now in two columns

print(df2.shape) #print this to check the length of rows

e1517=df2[(df2.lang.isin(["1"]))] #filter out only english language responses. I tried to do this using loc and kept getting errors.

print(e1517)

#I also tried to simultaneously filter for english language and for specific variable values but could not get this to work in one line of code

#pivotE=e1517.pivot_table(index=['variable'],values=['value']) #returns error, "no numeric types to aggregate"

#def tally(answer):

    #for answer in e1517:

 #e1517['variable'].value_counts().plot(kind='bar')

#first, remove the skipped answers from the dataframe

e1517notsk=e1517[e1517.value !=' '] #create a new dataframe from all values in the e1517 dataframe that are not " " in the value column

print(e1517notsk.shape) #print this just to check to see if the # of rows is smaller

e17notsk=e1517notsk[e1517notsk.variable !='q15x'] #whoops also let's remove question 15a responses

cte17=pd.crosstab(e17notsk.variable, e17notsk.value, normalize=True).multiply(100).round(2) #return a crosstabulation of english responses by question and their response

                                                        #show the percentage of each combination occurring using normalize=True

                                                        #multiply the returned crosstabs by 100 and round to 2 decimal places

#Note, I think this is wrong because these percentages look much smaller than what the report referenced, but for the sake of doing the graphing exercise, I will plot the results

print(cte17)
!pip install seaborn #install this module

import seaborn as sns #import this library with this abbreviation

choices=['Friend/Peer','Family', 'Co-Worker', "Library Resource", 'Govt Website', 'Priv Website', 'Teacher', 'Other'] #create a list to update yaxis ticks

ans=['Yes', 'No', 'NA', 'Don''t Know', 'Refused'] #create a list to update xaxis tick labels

ax=sns.heatmap(cte17, cmap="YlGnBu", annot=True, cbar=False, yticklabels=choices, xticklabels=ans) #create this heatmap and set up the ax parameter

ax.set(title="Heatmap of English Language Response to ''Have you ever turned to any of the following people or places for advice to protect your personal informaiton online?''",

        xlabel= 'Respons Type (% of respondents who did not skip)',

        ylabel= 'Source of Assistance')



#I wasn't able to figure out how to split the title into two rows