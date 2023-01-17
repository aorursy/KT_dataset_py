# DATA CLEANING AND DATA VISUALIZATION
#Load StarWars csv file



starwars_p='../input/starwars-data/StarWars.csv'





# In[2]:





#installing pandas package



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns





# In[3]:





#reading the datafile(.csv) and assigning it to the variable-StarWars;encoding used as strings are unicoded, so to decode them and producing a string suitable in python as unicode literal



StarWars = pd.read_csv(starwars_p,sep=',', decimal='.', header=1,names=['Respondant ID', 'seen_any_of_6films','fan?',

                                                                        'Episode_1','Episode_2','Episode_3',

                                                                        'Episode_4','Episode_5','Episode_6',

                                                                        'rank_1','rank_2','rank_3','rank_4','rank_5',

                                                                        'rank_6','Han Solo favored?','Luke Skywalker favored?',

                                                                        'Princess Leia Organa favored?',

                                                                        'Anakin Skywalker favored?','Obi Wan Kenobi favored?',

                                                                        'Emperor Palpatine favored?','Darth Vader favored?',

                                                                        'Lando Calrissian favored?','Boba Fett favored?',

                                                                        'C-3P0 favored?','R2 D2 favored?',

                                                                        'Jar Jar Binks favored?','Padme Amidala favored?',

                                                                        'Yoda favored?','shoted_first',

                                                                        'familiar_with_the_Expanded_Universe?',

                                                                        'fan of the_Expanded_Universe?','fan of star_treck?',

                                                                        'Gender','Age','Income','Education','Location'],

                                                                        encoding='unicode_escape')





# In[4]:





#display StarWars file



StarWars





# In[4]:





#drop first and second row as it has redundant data



StarWars=StarWars.drop(StarWars.index[0])

StarWars=StarWars.drop(StarWars.index[0])





# In[5]:





#resetting index(starting from 0) and assigning it to starwars itself



StarWars=StarWars.reset_index(drop=True)





# In[7]:





#display updated StarWars file



StarWars





# In[6]:





#display all columns in starwars file



StarWars.columns





# In[7]:





#ques 1.2

#Check datatypes of all columns



StarWars.dtypes





# In[8]:





StarWars.shape





# In[9]:





#creating a dataframe of starwars



df=pd.DataFrame(StarWars)





# In[10]:





#using a 'for loop' to count all the values the columns contains



for c in df.columns:

    print ("---- %s ---" % c)

    print (df[c].value_counts())





# In[11]:





#ques 1.3

#extracting the unique values in the column for detecting any typos/impossible values

StarWars['seen_any_of_6films'].unique()





# In[12]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['fan?'].unique()





# In[13]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Episode_1'].unique()





# In[14]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Episode_2'].unique()





# In[15]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Episode_3'].unique()





# In[16]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Episode_4'].unique()





# In[17]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Episode_5'].unique()





# In[18]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Episode_6'].unique()





# In[19]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['rank_1'].unique()





# In[20]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['rank_2'].unique()





# In[21]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['rank_3'].unique()





# In[22]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['rank_4'].unique()





# In[23]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['rank_5'].unique()





# In[24]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['rank_6'].unique()





# In[25]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Han Solo favored?'].unique()





# In[26]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Luke Skywalker favored?'].unique()





# In[27]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Princess Leia Organa favored?'].unique()





# In[28]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Anakin Skywalker favored?'].unique()





# In[29]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Obi Wan Kenobi favored?'].unique()





# In[30]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Emperor Palpatine favored?'].unique()





# In[31]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Darth Vader favored?'].unique()





# In[32]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Lando Calrissian favored?'].unique() 





# In[33]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Boba Fett favored?'].unique()





# In[34]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['C-3P0 favored?'].unique()





# In[35]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['R2 D2 favored?'].unique()





# In[36]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Jar Jar Binks favored?'].unique()





# In[37]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Padme Amidala favored?'].unique()





# In[38]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Yoda favored?'].unique()





# In[39]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['shoted_first'].unique()





# In[40]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['familiar_with_the_Expanded_Universe?'].unique()





# In[41]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['fan of the_Expanded_Universe?'].unique()





# In[42]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['fan of star_treck?'].unique()





# In[43]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Gender'].unique()





# In[44]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Age'].unique()





# In[45]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Income'].unique()





# In[46]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Education'].unique()





# In[47]:





#extracting the unique values in the column for detecting any typos/impossible values

StarWars['Location'].unique()





# In[48]:





#checking and counting typos in 'fan?' column using mask



mask1=StarWars['fan?']=='Noo'

mask1





# In[49]:





#checking and counting typos in 'fan?' column using mask



mask1=StarWars['fan?']=='Yess'

mask1





# In[50]:





#checking and counting typos in 'fan of the_Expanded_Universe?' column using mask



mask2=StarWars['fan of the_Expanded_Universe?']=='Yess'

mask2





# In[51]:





#checking and counting typos in 'fan of star_treck? column using mask



mask3=StarWars['fan of star_treck?']=='Noo'

mask3





# In[52]:





#checking and counting typos in 'fan of star_treck? column using mask



mask4=StarWars['fan of star_treck?']=='yes'

mask4





# In[53]:





#checking and counting typos in 'Gender' column using mask



mask5=StarWars['Gender']=='female'

mask5





# In[54]:





#checking and counting typos in 'Gender' column using mask



mask6=StarWars['Gender']=='male'

mask6





# In[55]:





#checking and counting typos in 'Gender' column using mask



mask7=StarWars['Gender']=='F'

mask7





# In[56]:





#replacing all the typos found above by the standard value



StarWars['fan?'].replace('Noo','No',inplace=True)

StarWars['fan?'].replace('Yess','Yes',inplace=True)

StarWars['fan?'].value_counts()



StarWars['fan of the_Expanded_Universe?'].replace('Yess','Yes',inplace=True)

StarWars['fan of the_Expanded_Universe?'].value_counts()



StarWars['fan of star_treck?'].replace('Noo','No',inplace=True)

StarWars['fan of star_treck?'].replace('yes','Yes',inplace=True)

StarWars['fan of star_treck?'].value_counts()



StarWars['Gender'].replace('female','Female',inplace=True)

StarWars['Gender'].replace('male','Male',inplace=True)

StarWars['Gender'].replace('F','Female',inplace=True)

StarWars['Gender'].value_counts()





# In[57]:





#checking for extra white space and removing them and then replacing them with standard value as it has typos too



StarWars['fan of star_treck?']=StarWars['fan of star_treck?'].str.strip()

StarWars['fan of star_treck?'].replace('no','No',inplace=True)

StarWars['fan of star_treck?'].value_counts()





# In[58]:





#checking for extra white space and display the updated counts



StarWars['seen_any_of_6films']=StarWars['seen_any_of_6films'].str.strip()

StarWars['seen_any_of_6films'].value_counts()





# In[59]:





#display updated all column values



for c in df.columns:

    print ("---- %s ---" % c)

    print (df[c].value_counts())





# In[60]:





#converting all text data to upper case by using 'str.upper()' function and updating it to that particular column itself

    

StarWars['seen_any_of_6films']=StarWars['seen_any_of_6films'].str.upper()

StarWars['fan?']=StarWars['fan?'].str.upper()

StarWars['Episode_1']=StarWars['Episode_1'].str.upper()

StarWars['Episode_2']=StarWars['Episode_2'].str.upper()

StarWars['Episode_3']=StarWars['Episode_3'].str.upper()

StarWars['Episode_4']=StarWars['Episode_4'].str.upper()

StarWars['Episode_5']=StarWars['Episode_5'].str.upper()

StarWars['Episode_6']=StarWars['Episode_6'].str.upper()

StarWars['Han Solo favored?']=StarWars['Han Solo favored?'].str.upper()

StarWars['Luke Skywalker favored?']=StarWars['Luke Skywalker favored?'].str.upper()

StarWars['Princess Leia Organa favored?']=StarWars['Princess Leia Organa favored?'].str.upper()

StarWars['Anakin Skywalker favored?']=StarWars['Anakin Skywalker favored?'].str.upper()

StarWars['Obi Wan Kenobi favored?']=StarWars['Obi Wan Kenobi favored?'].str.upper()

StarWars['Emperor Palpatine favored?']=StarWars['Emperor Palpatine favored?'].str.upper()

StarWars['Darth Vader favored?']=StarWars['Darth Vader favored?'].str.upper()

StarWars['Lando Calrissian favored?']=StarWars['Lando Calrissian favored?'].str.upper()

StarWars['Boba Fett favored?']=StarWars['Boba Fett favored?'].str.upper()

StarWars['C-3P0 favored?']=StarWars['C-3P0 favored?'].str.upper()

StarWars['R2 D2 favored?']=StarWars['R2 D2 favored?'].str.upper()

StarWars['Jar Jar Binks favored?']=StarWars['Jar Jar Binks favored?'].str.upper()

StarWars['Padme Amidala favored?']=StarWars['Padme Amidala favored?'].str.upper()

StarWars['Yoda favored?']=StarWars['Yoda favored?'].str.upper()

StarWars['shoted_first']=StarWars['shoted_first'].str.upper()

StarWars['fan of the_Expanded_Universe?']=StarWars['fan of the_Expanded_Universe?'].str.upper()

StarWars['fan of star_treck?']=StarWars['fan of star_treck?'].str.upper()

StarWars['Gender']=StarWars['Gender'].str.upper()

StarWars['Age']=StarWars['Age'].str.upper()

StarWars['Income']=StarWars['Income'].str.upper()

StarWars['Education']=StarWars['Education'].str.upper()

StarWars['Location']=StarWars['Location'].str.upper()





# In[63]:





#displaying updated StarWars file



StarWars





# In[61]:





#using describe function for sanity check, for finding any impossible value

StarWars.describe()





# In[62]:





#displaying impossible value in 'Age' column

StarWars['Age']=='500'

StarWars['Age'].iloc[5:15]





# In[63]:





#selcting that particular impossible value by going to particular row no. and column

StarWars.iloc[10,34]





# In[64]:





#replacing that value with the most occuring value in the column

StarWars.iloc[10,34] = '45-60'





# In[65]:





#display of updated age column

StarWars['Age'].iloc[5:15]





# In[66]:





#displaying updated starwars columns after removing typos,white spaces and impossible values



for c in df.columns:

    print ("---- %s ---" % c)

    print (df[c].value_counts())





# In[67]:





#checking for any null values

StarWars['fan?'].isnull().value_counts()





# In[68]:





#checking for any null values

StarWars['familiar_with_the_Expanded_Universe?'].isnull().value_counts()





# In[69]:





#checking for any null values

StarWars['Han Solo favored?'].isnull().value_counts()





# In[73]:





#checking for any null values

StarWars['Luke Skywalker favored?'].isnull().value_counts()





# In[74]:





#checking for any null values

StarWars['Princess Leia Organa favored?'].isnull().value_counts()





# In[75]:





#checking for any null values

StarWars['Anakin Skywalker favored?'].isnull().value_counts()





# In[76]:





#checking for any null values

StarWars['Obi Wan Kenobi favored?'].isnull().value_counts()





# In[77]:





#checking for any null values

StarWars['Emperor Palpatine favored?'].isnull().value_counts()





# In[78]:





#checking for any null values

StarWars['Darth Vader favored?'].isnull().value_counts()





# In[79]:





#checking for any null values

StarWars['Lando Calrissian favored?'].isnull().value_counts()





# In[80]:





#checking for any null values

StarWars['Boba Fett favored?'].isnull().value_counts()





# In[81]:





#checking for any null values

StarWars['C-3P0 favored?'].isnull().value_counts()





# In[82]:





#checking for any null values

StarWars['R2 D2 favored?'].isnull().value_counts()





# In[83]:





#checking for any null values

StarWars['Jar Jar Binks favored?'].isnull().value_counts()





# In[84]:





#checking for any null values

StarWars['Padme Amidala favored?'].isnull().value_counts()





# In[85]:





#checking for any null values

StarWars['Yoda favored?'].isnull().value_counts()





# In[86]:





#checking for any null values

StarWars['fan of star_treck?'].isnull().value_counts()





# In[87]:





#counting no. of 'NO' in the column

a=StarWars['fan of star_treck?']=='YES'

a.value_counts()





# In[88]:





#checking for any null values

StarWars['Gender'].isnull().value_counts()





# In[89]:





#counting no. of 'FEMALE' in column

a=StarWars['Gender']=='FEMALE'

a.value_counts()





# In[90]:





#counting no. of 'FEMALE' in column

a=StarWars['Gender']=='MALE'

a.value_counts()





# In[91]:





#checking for any null values

StarWars['Income'].isnull().value_counts()





# In[92]:





#checking for any null values

StarWars['Education'].isnull().value_counts()





# In[93]:





#checking for any null values

StarWars['Location'].isnull().value_counts()





# In[94]:





#checking for any null values

StarWars['Age'].isnull().value_counts()





# In[95]:





#checking for any null values

StarWars['Episode_1'].isnull().value_counts()





# In[96]:





#checking for any null values

StarWars['Episode_2'].isnull().value_counts()





# In[97]:





#checking for any null values

StarWars['Episode_3'].isnull().value_counts()





# In[98]:





#checking for any null values

StarWars['Episode_4'].isnull().value_counts()





# In[99]:





#checking for any null values

StarWars['Episode_5'].isnull().value_counts()





# In[100]:





#checking for any null values

StarWars['Episode_6'].isnull().value_counts()





# In[101]:





#filling na values in the column 'fan of star_treck?' with the most occuring value whilst ignoring other observations having missing values

StarWars['fan of star_treck?']=StarWars['fan of star_treck?'].fillna('NO')





# In[102]:





#display updated starwars file

StarWars





# In[70]:





#question 2.1

# plotting pie chart of how episodes from 1 to 6 are ranked



plt.figure(0)

StarWars['rank_1'].value_counts().plot.pie(autopct='%.2f')

plt.figure(1) 

StarWars['rank_2'].value_counts().plot.pie(autopct='%.2f')

plt.figure(3)

StarWars['rank_3'].value_counts().plot.pie(autopct='%.2f')

plt.figure(4)   

StarWars['rank_4'].value_counts().plot.pie(autopct='%.2f')

plt.show()

StarWars['rank_5'].value_counts().plot.pie(autopct='%.2f')

plt.show() 

StarWars['rank_6'].value_counts().plot.pie(autopct='%.2f')

plt.show() 





# In[71]:





#plotting bar graph to see how many female/male have given rank 1 for episode 6

sw_f=StarWars['rank_6']==1

rank6_g=StarWars.loc[sw_f,'Gender'].value_counts()

f=rank6_g.plot(kind='bar')

f.grid()

rank6_g





# In[72]:





#plotting bar graph to see how many female/male have given rank 1 for episode 5

sw_f=StarWars['rank_5']==1

rank5_g=StarWars.loc[sw_f,'Gender'].value_counts()

f=rank5_g.plot(kind='bar')

f.grid()

rank5_g





# In[73]:





#plotting bar graph to see how many female/male have given rank 1 for episode 4

sw_f=StarWars['rank_4']==1

rank4_g=StarWars.loc[sw_f,'Gender'].value_counts()

f=rank4_g.plot(kind='bar')

f.grid()

rank4_g





# In[107]:





#plotting pie chart to see how different age groups has given rank 1 for episode 6



swf1=StarWars['rank_6']==1.0

countf1=StarWars.loc[swf1,'Age'].value_counts()

A=countf1.plot(kind='pie', autopct='%.2f')

A.grid()

countf1





# In[108]:





#plotting pie chart to see how different age groups has given rank 1 for episode 5



swf1=StarWars['rank_5']==1.0

countf1=StarWars.loc[swf1,'Age'].value_counts()

A=countf1.plot(kind='pie', autopct='%.2f')

A.grid()

countf1





# In[109]:





#plotting pie chart to see how different age groups has given rank 1 for episode 4



swf1=StarWars['rank_4']==1.0

countf1=StarWars.loc[swf1,'Age'].value_counts()

A=countf1.plot(kind='pie', autopct='%.2f')

A.grid()

countf1





# In[110]:





#plotting bar graph to see where people are based who have given rating 1 for episode 5(liked the most by seeing above pie chart)



swf1=StarWars['rank_5']==1.0

countf1=StarWars.loc[swf1,'Location'].value_counts()

A=countf1.plot(kind='bar')

A.grid()

countf1





# In[111]:





# question 2.2

#plotting bar graph to see those who have seen any of 6 films of 'starwars franchise' is a 'starwar franchise' fan or not



swf1=StarWars['seen_any_of_6films']=='YES'

countf1=StarWars.loc[swf1,'fan?'].value_counts()

a=countf1.plot(kind='bar')

a.grid()

countf1





# In[112]:





#plot bar graph to see people who are star treck fan are fan of Expanded_Universe as well?



swf1=StarWars['fan of star_treck?']=='YES'

countf1=StarWars.loc[swf1,'fan of the_Expanded_Universe?'].value_counts()

A=countf1.plot(kind='bar',color='orange')

A.grid()

countf1





# In[113]:





#plot pie chart to see if star war fans are familiar with the Expanded Universe?



swf1=StarWars['fan?']=='YES'

countf1=StarWars.loc[swf1,'familiar_with_the_Expanded_Universe?'].value_counts()

A=countf1.plot(kind='bar',color='red')

A.grid()

countf1





# In[77]:





swf1=StarWars['Gender']=='FEMALE'

countf1=StarWars.loc[swf1,'Han Solo favored?'].value_counts()

A=countf1.plot(kind='bar',color='red')

A.grid()

countf1





# In[114]:





#ques 2.3

#plotting graph for how much a character is favorable based on gender

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Han Solo favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[115]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Luke Skywalker favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[116]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Princess Leia Organa favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[117]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Anakin Skywalker favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[118]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Obi Wan Kenobi favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[119]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Emperor Palpatine favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[120]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Darth Vader favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[121]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Lando Calrissian favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[122]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Boba Fett favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[123]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("C-3P0 favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[124]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("R2 D2 favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[125]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Jar Jar Binks favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[126]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Padme Amidala favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[127]:





#plotting graph for how much a character is favorable based on gender



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Yoda favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Gender')

    g.set_ylabels('Number of People')





# In[128]:





#plotting graph for how much a character is favorable based on Age



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Han Solo favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[129]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Luke Skywalker favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[130]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Princess Leia Organa favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[131]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Anakin Skywalker favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[132]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Obi Wan Kenobi favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[133]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Emperor Palpatine favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[134]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Darth Vader favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[135]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Lando Calrissian favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[136]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Boba Fett favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[137]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("C-3P0 favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[138]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("R2 D2 favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[139]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Jar Jar Binks favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[140]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Padme Amidala favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[141]:





#plotting graph for how much a character is favorable based on Age

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Yoda favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Age')

    g.set_ylabels('Number of People')





# In[142]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Han Solo favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[143]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Luke Skywalker favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[144]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Princess Leia Organa favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[145]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Anakin Skywalker favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[146]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Obi Wan Kenobi favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[147]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Emperor Palpatine favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[148]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Darth Vader favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[149]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Lando Calrissian favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[150]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Boba Fett favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[151]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("C-3P0 favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[152]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("R2 D2 favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[153]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Jar Jar Binks favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[154]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Padme Amidala favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[155]:





#plotting graph to find if income has any relation with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Yoda favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Income')

    g.set_ylabels('Number of People')





# In[156]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Han Solo favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[157]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Luke Skywalker favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[158]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Princess Leia Organa favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[159]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Anakin Skywalker favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[160]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Obi Wan Kenobi favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[161]:





with sns.axes_style('whitegrid'):

    g = sns.factorplot("Emperor Palpatine favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[162]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Darth Vader favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[163]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Lando Calrissian favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[164]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Boba Fett favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[165]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("C-3P0 favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[166]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("R2 D2 favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[167]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Jar Jar Binks favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[168]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Padme Amidala favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[169]:





#plotting graph to find if Education is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Yoda favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Education')

    g.set_ylabels('Number of People')





# In[170]:





#plotting graph to see if people's location is related with character's favorability

with sns.axes_style('whitegrid'):

    g = sns.factorplot("Han Solo favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[171]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Luke Skywalker favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[172]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Princess Leia Organa favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[173]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Anakin Skywalker favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[174]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Obi Wan Kenobi favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[175]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Emperor Palpatine favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[176]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Darth Vader favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[177]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Lando Calrissian favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[178]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Boba Fett favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[179]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("C-3P0 favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[180]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("R2 D2 favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[181]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Jar Jar Binks favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[182]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Padme Amidala favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')





# In[183]:





#plotting graph to see if people's location is related with character's favorability



with sns.axes_style('whitegrid'):

    g = sns.factorplot("Yoda favored?", data=StarWars, aspect=4.0, kind='count',

                       hue='Location')

    g.set_ylabels('Number of People')
#From the pie chart we can see that episodes 6, 5 and 4 are most preferred by the viewers having 17.49%,34.61% and 24.43% respectively of rating 1 as compared to other episodes.

#So, I narrowed my focus to these 3 columns and dig deeper into it, I further calculated and plotted bar graph to see how many females and males have given rank 1 to the episode 6,5 and 4.

#a. For episode 6, 77 females gave a rating 1, 10 more than males, hence that this episode was liked by females more.

#b. For episode 5, 152 males gave rating1 and while 131 were females, this shows that this episode was liked by males much more than females.

#c. For episode 4, 112 males gave rating 1 and while only 90 females gave it 1st rating, this shows that this episode was liked by males ~56% more than the females.

# I further explored that which age group prefers which episode the most (rated 1) and plotted pie chart for visualization and to analyze.

#a. 33.33% of people gave rating 1 for episode 6 and they are between the age group '30-44'. Meaning that 33.33% of people of age group 30-44 like episode 6 the most.

#b. Whilst, 31.8% of people of age group 45-60 prefers episode 5 the most.



#c. Whereas, 34.65% of people liked episode 4 the most who belong to the 45-60 age group.

#Since episode 5 is liked the most by people, I went to a further level to find that people based in 'SOUTH ATLANTIC' have given the rating 1 the most for the episode 5. 

#Hence people in this area like the 5th episode more than any other episode by the bar graph.

#Hence, all these observations guide us to conclude that, episode 5 of Star Wars is most liked by the people. 

#And that males like this episode more than the females and people who are of age group between 45-60 prefers the episode 5 the most as compared to any other episode.

#Furthermore, ~19% of people from 'SOUTH ATLANTIC' like episode 5.

#66% of the viewers who have seen any of 6 films are Star Wars fan. And only 284 people aren't a fan amongst the viewers.

#The bar graph summarizes that a whopping 64% of people aren't familiar with the expanded universe amongst the fans of Star Wars, and thus only a minor percentage of 33.2 star war fans are familiar with it.

#we can conclude that 310 out of 549 females prefer the character 'Yoda' 'very preferably', which is 56.5% of all females. On the other hand, 62.5% of males prefer the character 'Han Solo' 'very preferably'.

#So, more than half the population of males and females prefer 'Han Solo' and 'Yoda' respectively.

#125 people of age group 18-29,157 people of age between 30-44 prefers 'Yoda'. While 180 and 151 people of age between 45-60 and above 60 respectively 'very preferably' favors 'Han Solo'.

#Thus, 'Han Solo' is the most preferred character when compared to all age groups.

#‘Han Solo’ is favored ‘very preferably’ by associate degree and bachelor degree holders in large numbers as compared to any other character of the film.

#location and character have a relationship as people from East Northcentral, West Northcentral and South Atlantic likes 'Yoda' character 'very preferably' as compared to any other character.


