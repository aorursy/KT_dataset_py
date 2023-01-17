import pandas as pd

##pandas is the library that contains lot of in-built functions that help us analyze the data



##read_csv(file) is one such in-built function. We load the data in 1 line. 

test_data = pd.read_csv ('/kaggle/input/titanic/test.csv')

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")



##test and train_data CSV files are loaded into Pandas 'dataframe' objects. Dataframes are 

##nothing but a 2 dimensional representation of the data containing rows and cols. Let us 

##start by having a very high level look at the data. The dataframe.info() function helps us.

train_data.info()
##Aah. You can see the ouput cell where the o/p of this function is printed. There are 12 

##columns for each of the 891 passengers. One of them is the 'Survived' flag. Our ML 

##algorithms are supposed to 'learn' from this 'train_data' and 

##then predict the 'Survived' col for the 'test_data' which does NOT contain this col.



##Notice that 3 of these 11 columns in train_data have missing data for - Age, Cabin & 

##Embarked. We obviously cannot drop these 3 columns entirely as they could contain 

##useful information which may help us determine if the passenger survives or not. 

##We also can't drop rows containing these missing columns as they could also contain 

##valuable info.



##Let us also look at the test data. Before that, we define a global var to print blanklines

global blankline

blankline='\n**************************************************************************\n'

print (blankline)

test_data.info()
##So we see that test data has 418 rows. It does NOT contain the Survived column. We have to 

##predict this after we 'learn' from the train_data. It has the remaining 11 columns and notice 

##that here too Age, Cabin and Fare have missing data. We now set this test data aside. We dont 

##want to mix test & training data. We focus all attention to the training data!



##Let us look at the actual data now..few of the rows. Let's use the dataframe.head() function

train_data.head()
print(train_data.describe())

##Only numeric columns are shown in describe() function. You can see in the o/p cell that 

##min, max, std deviation etc get printed.

##Mean age is 30. One crude way to fill missing ages is to make them all the age=30. But 

##let us see if there is a better way.

##The mean of the SURVIVED col shows 38% survival rate. Roughly 1 in 3 survive the Titanic

##Fare=0 needs investigation
##Let us get a bit more details on missing data

print(blankline)

print(round (train_data.isnull().sum().sort_values(ascending=False)/len(train_data)*100,1))

##The above command gives more details on the missing data. The command is explained in the

##next cell but let us analyze the o/p of this command first.

##The o/p shows what we already know but gives a better idea of the situation. With 77% 

##missing data the Cabin info seems pretty useless. Roughly 1 in 5 ages are missing and 

##we will have to guess the age..The Embarked needs to be fixed and we also have 1 fare 

##missing from the test data (as seen previously). 



##Apart from AGE and Cabin, rest of missing data is hardly couple of rows and we should not 

##pay too much time deducing how to fill them. Embarked can be the mode (remember mean, 

##medium, mode). Fare can be the mean fare.  

##Filling the missing Age values would be the key here. Should it be the mean, median or mode?

##Let us take a look at the data and decide.
print(train_data.groupby('Pclass').count())

##The groupby(col) literally implies grouping the rows of the dataset based 

##on unique values of the col. Here we have 3 unique values of Pclass so 

##there are 3 groups of rows created. Once the groups are created, we can 

##'apply' a function to it. That function acts on the groups. For e.g. we 

##can calculate the 'count' of each group or the mean etc. Let us use 'count
##We see there are 216, 184 & 491 1st, 2nd and 3rd class passengers respectively.

##Note that some columns show lesser number because of missing data. The 

##'PassengerID' column does not have any missing data and we can assume whatever 

##count is mentioned there will be the actual counts. We can ignore the counts in

##other columns. So to summarize we have 216 passengers travelling in Pclass 1, 

##184 in Pclass2 and 491 in Pclass 3 or the economy class



##Now let us focus specifically on the 'Survived' column. If we do a 'mean' of 

##the survived column (after grouping), we get the percentage of how many survived 

##in each group. Remember if person survives, then Survived col=1 else it is 0.

print(blankline, train_data [['Pclass','Survived']].groupby('Pclass').mean())



##Notice above how we give only the columns we are interested in as input to the 

##groupby function. This avoids clutter. Also notice the double braces [[ & ]]. 

##It is imp to understand what is happening here. ['Pclass','Survived'] is a Python

##list. This is passed to the external square brackets [] which is the indexer 

##operation of the Pandas dataframe object which returns a dataframe based on 

##the list passed. So train_data [['Pclass','Survived']] therefore returns a 

##dataframe object with just 2 columns. Try printing it & see.



##Note: train_data [['Pclass']] returns a "dataframe" object with just 1 column

##train_data ['Pclass'] returns a "series" object with just 1 column. You cant groupby 

##a "series" object. Again, It is very important to understand the 'type' of objects 

##being returned by each function else you could spend hours debugging the code. 

##Unfortunately in few cases the error message is not very friendly. If you dont

##believe me try printing train_data ['Pclass','Survived'].groupby('Pclass').mean()
##Let us put it in proper %age format

print(blankline, round(train_data [['Pclass','Survived']].groupby(['Pclass']).mean()*100,1))

##Do check the o/p cell

##So 2nd class passengers had twice the survival rate of 3rd class and 1st class passengers 

##had even better rates. Not a very pleasant discovery but it was on expected lines, I guess. 
##Now let us turn to Sex. Does it make a difference?

print(blankline,round(train_data [['Sex','Survived']].groupby(['Sex']).mean()*100,1))

##74% of females survived and only 19% males survived. The overall survival rate was 38.3%

##as seen earlier. So it is hoplessely skewed in favour of females
##Groupby is the one Panda command you need to know a little about because it is so useful

##Let us groupby with multiple columns

print(round(train_data [['Sex', 'Pclass','Survived']].\

                       groupby(['Pclass', 'Sex']).mean()*100,1))

##This is really useful data. Basically almost all Pclass 1 females survive and so do most 

##of Pclass 2. Pclass 3 female survival is 50% and almost all (86%) Pclass 3 males unfortunately

##do not survive. Notice how we passed only 3 cols to the groupby function - Sex, Pclass

##and Survived. We could also pass the whole dataframe as is, but this creates lot of clutter
##Lastly let us see if Embarked has any real relevant significance to survival

print(blankline, round(train_data [['Embarked', 'Sex', 'Pclass','Survived']].\

                       groupby(['Embarked', 'Pclass', 'Sex']).mean()*100,1))
##Ideally the percentages should be same as seen before but you can see some discrepancies begin

##to emerge. For e.g we see that for Embarked = Q, 100% of males in Pclass 1 and Pclass 2 expire 

##This goes against the data we just saw earlier. So let us see what is happening?



print(blankline, train_data[(train_data.Pclass==1) & (train_data.Sex=='male') & (train_data.Embarked=='Q')])

##Here we just apply a filter to the train_data dataframe. It is like saying SELECT ALL ROWS WITH

##CONDITION. Here the CONDITION we want to explore is: Pclass 1 males who embarked on port Q
##Aah! We see just 1 male passenger embarked into Pclass 1 at port Q and he unfortunately did not

##survive. We based our entire analysis of 100% death based on this single passenger and we could 

##have gone completely off track. A good idea is to also print the counts along with the %ages

##so we are not misguided by statistically insignificant data and jump to conclusions. 

print(blankline, train_data [['Embarked', 'Sex', 'Pclass','Survived']].groupby \

                       (['Embarked', 'Pclass', 'Sex']).agg(['count','mean']))

##We use the aggregate function. We pass the mean and count as list to the agg function which 

##does the rest. Let us now turn to GRAPHS which gives us a better pictorial view 
##Rather than writing a lengthy line of code, we try 'evolve' the graphs as we go 

##so it is easy to understand for the beginners

import seaborn as sns

import matplotlib.pyplot as plt

##We  need visuals for our next question - How is age linked to survival

##Let us use what is called a density plot & see the Age distribution

##We will need to import seaborn library and use its functions to plot the graph

##We also need Matplotlib which is a plotting lib for the Python programming language



sns.kdeplot(train_data['Age'], color="green")

##train_data['Age'] is just the AGE column of the Titanic data set. This is passed 

##as i/p. You can try printing train_data['Age'] independently and see for yourself.

##Now let us turn our attention to the kdeplot. Seems like while majority of folks 

##on the Titanic are in the 20-40 age range, there are lot of 10-20 range as well
##I think a histogram would serve us better. Group into 10 bins

plt.figure()   ##Tell Python this is a new fig else it will overwrite the old one

print(train_data['Age'].plot(kind='hist',bins=10))

##Much clearer now. 20-30 has max folks. There is also a sizeable 0-20 & a 30-45.
##The density plot has its uses though. For e.g. if I want to compare the age 

##distribution of male/female, here is what I would do

plt.figure()

sns.kdeplot(train_data[train_data.Sex=='female']['Age'], color="green")

sns.kdeplot(train_data[train_data.Sex=='male']['Age'], color="red", shade=True)

##We discussed what train_data[condition] returns. So train_data[train_data.Sex=='male']

##returns us a new dataframe containing all the males in the original dataframe. What is 

##this ['Age'] column next to that? Well, we are not interested in the remaining 11 columns

##We are just interested in AGE col. To retain only the columns we are interested in, we 

##just do a dataframe[col]. If we need more than one column, just pass the column names as 

##a list..e.g dataframe[[col1, col2]]. You can printing these individually to understand 

##in detail. For e.g. print train_data[train_data.Sex=='female'] first, then print 

##train_data[train_data.Sex=='female']['Age']

##Graph shows while the distribution is more or less equal, there are more females 

##around the 10 year age group
##How does survival depend on age? Let us plot violin-plot which is best suited for this

sns.violinplot(x='Survived', y='Age', data=train_data, palette={0: "r", 1: "g"});

##There is an abundance of information here. The white dot you see is the median. The 

##thick black line at the center is the first and third quartile. Violin plots show the

##distribution across the range (of age in this case) & give a visual indication of

##Outliers. Quick comparision shows that between 0-10 age survival rate is higher (green

##is fatter than red in this age range). After age 15 or so, expiry (red) just fattens out

##rapidly peaking at around 22 years of age. Green too fattens but not as rapidly as Red. 

##In this age group far too many people expire (compared to survive). This trend pretty 

##much continues. After about 65 or so, almost all expire (green is hardly a line...wheras 

##Red is fatter). Note that the shape is mirrored across 2 sides of the plot. Using the 

##Split=true option we can retain only 1 side & save some real estate
##We can squeeze in one more parameter using hue. hue needs to be a binary value parameter.

##Since Survived is a binary, let us plot it as well into the above graph

plt.figure() 

sns.violinplot(x='Pclass', y='Age', hue='Survived', data=train_data, palette={0: "r", 1: "g"});

##Too messy. Let us use the 'split' option which gives 3 figs instead of 6 (without losing info)
plt.figure() 

sns.violinplot(x='Pclass', y='Age', hue='Survived', split=True ,data=train_data, palette={0: "r", 1: "g"});

##Pclas 2 and 3 have a very sharp peak of expiry near the 20-35 age range

##Class 1 has far more survivors across all age levels except maybe after 50 or so. After 60 

##there is a sharp jump in expiry. This is not as sharp in Pclass 2 or 3. This is a curious 

##phenomenon. Let us investigate a bit
print('Pclass 1 survivors above Age 60:', round(len(train_data[(train_data['Pclass']==1) & \

    (train_data['Age']>59) & (train_data['Survived']==True)])/len(train_data[(train_data\

    ['Pclass']==1) & (train_data['Age']>59)])*100,1), '%')

print('Pclass 2 survivors above Age 60:', round(len(train_data[(train_data['Pclass']==2) & \

    (train_data['Age']>59) & (train_data['Survived']==True)])/len(train_data[(train_data \

    ['Pclass']==2) & (train_data['Age']>59)])*100,1), '%')

print('Pclass 3 survivors above Age 60:', round(len(train_data[(train_data['Pclass']==3) & \

    (train_data['Age']>59) & (train_data['Survived']==True)])/len(train_data[(train_data \

    ['Pclass']==3) & (train_data['Age']>59)])*100,1), '%')



print('Pclass1 survivors between 20-30 Age:',round(len(train_data[(train_data['Pclass']==1) \

    &(train_data['Age']>19) & (train_data['Age']<31) & (train_data['Survived']==True)])/len( \

    train_data[(train_data['Pclass']==1) & (train_data['Age']>19) \

    & (train_data['Age']<31)])*100,1),'%')

print('Pclass2 survivors between 20-30 Age:',round(len(train_data[(train_data['Pclass']==2) \

    &(train_data['Age']>19) & (train_data['Age']<31) &(train_data['Survived']==True)])/len( \

    train_data[(train_data['Pclass']==2)&(train_data['Age']>19) \

    &(train_data['Age']<31)])*100,1),'%')

print('Pclass3 survivors between 20-30 Age:',round(len(train_data[(train_data['Pclass']==3) \

    &(train_data['Age']>19) & (train_data['Age']<31) &(train_data['Survived']==True)])/len( \

    train_data[(train_data['Pclass']==3) & (train_data['Age']>19) \

    &(train_data['Age']<31)])*100,1),'%')

##The syntax is straightforward. We are just putting a bunch of conditions and calculating 

##the length of the dataframe returned each time. The denominator condition does not have 

##survived col so we get the %age. Try printing the below line:

##train_data[(train_data['Pclass']==1)&(train_data['Age']>59)&(train_data['Survived']==True)]

##Then print len of above. It gives the count. The below kind of commands

##Dataframe[(condition 1) & (condition 2) & (condition 3)] will be used over and over again 

##in 90% of the code. So familiarize yourself on the same before stepping forward.
print('Pclass 2 adult male survivors:',round(len(train_data[(train_data['Pclass']==2) & \

        (train_data['Age']>19) & (train_data['Sex']=='male') & (train_data['Survived'] \

        == True)])/len(train_data[(train_data['Pclass']==2) & (train_data['Age']>19) & \

        (train_data['Sex']=='male')])*100,1),'%')



print('Pclass 3 adult male survivors:',round(len(train_data[(train_data['Pclass']==3) & \

        (train_data['Age']>19) & (train_data['Sex']=='male') & (train_data['Survived'] \

        == True)])/len(train_data[(train_data['Pclass']==3) & (train_data['Age']>19) & \

        (train_data['Sex']=='male')])*100,1),'%')

##Let us explore the Embarked column now. We will try a new kind of plot

fg = sns.FacetGrid(train_data, row='Embarked')

##Above statement creates one row of graphs for each unique value of 'Embarked'. We have 

##to specify what data we need. This is done below

fg.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')

##x-axis is the Pclass, y-axis is the survival %age. The hue (color) category is the Sex

##Figs are small. Increase the ASPECT. Also add legend so we know what colors stand for what?

plt.figure()

fg = sns.FacetGrid(train_data, row='Embarked', aspect=2)

fg.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')

fg.add_legend()
##Did you notice something ridiculously strange happening on Embarked=C port? Those passenges

##who embarked on port C have totally reversed the odds for survival as far as Sex is concerned. 

##This is FALSE info & can quickly be verified by a query to gather the %age of survivors grouped 

##by Sex for Port C embarkation. Obviously the colors for Male & Female have got mixed up in the 

##middle row. If your warnings are suppressed you may not even notice it.. Luckily the warning 

##is clear. Please use the 'order' parameter else you may get incorrect ordering

plt.figure()

fg = sns.FacetGrid(train_data, row='Embarked', aspect=2)

fg.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', \

       hue_order=['female', 'male'], order=[1,2,3], palette={'female': "r", 'male': "g"})

fg.add_legend()

##Better! One very popular blog post (& half-a-dozen other copied copies) make this erroneous 

##observation and goes on to other topics as if it were perfectly normal data behaviour. There 

##is no point in just visualizing the data and doing nothing about it. Please dig into it 

##especially if you see patterns or anamolies which need further investigation. Survival of 

##more Males compared to Females definitely needs an investigationand we have just unravelled 

##the mystery. This observation is erroneous!
##Also Note some interesting variations between the port of embarkation. Ideally 

##one would expect the graphs to be similar to each other. In embark=S, females in pClass3 

##do extremely poorly against the other 2 classes. The difference is not so glaring in the

##other 2 ports. In Embarked=C, pClass=2 females have higher chance of survival than pclass=1.

##Embarked=Q has extremely high fatility rates for Males. For some strange reason pClass=3 

##have higher chances of survival in this port. This is so curious that we need to reverify it

print('Pclass 1 Male survivors % for Port Q embarkation :', round(len(train_data[(train_data \

    ['Pclass']==1) & (train_data['Embarked']=='Q') & (train_data['Sex']=='male')&(train_data \

    ['Survived']==True)])/len(train_data [(train_data['Pclass']==1) & (train_data['Sex']==\

    'male') & (train_data['Embarked']=='Q')])*100,1), '%')



print('Pclass 3 Male survivors % for Port Q embarkation :', round(len(train_data[(train_data \

    ['Pclass']==3) & (train_data['Embarked']=='Q') & (train_data['Sex']=='male')&(train_data \

    ['Survived']==True)])/len(train_data [(train_data['Pclass']==3) & (train_data['Sex']== \

    'male') & (train_data['Embarked']=='Q')])*100,1), '%')

##Aah - While it is true, looks like the data is statistically insignificant. Let us see...
print('Pclass 1 Males for Port Q embarkation :',len(train_data[(train_data['Pclass']==1) \

    & (train_data ['Embarked']=='Q') & (train_data['Sex']=='male') & (train_data['Survived']\

    ==True)]))

print('Pclass 1 Male survivors for Port Q embarkation:',len(train_data[(train_data['Pclass']\

    ==1) & (train_data['Embarked']=='Q') & (train_data['Sex']=='male')]))

##There you go. 1 passenger for pClass1 got on at S Port and we made an entire hypothesis 

##based on the fate of this one passenger. This is a very important lesson & need to be 

##careful as far as %ages and graphs are concerned simply because they may be statistically 

##insignificant. However considering the major variations across the 3 rows it is possible

##that port of embarkation is corelated to the survival along with pClass, Sex, age

##(above 3 were obvious though). Maybe these folks were given cabins in certain areas which

##were closer to lifeboats? So we cant drop this column
##let us check how the sex of the person could have made a difference to survival

sns.violinplot(x='Sex', y='Age', hue='Survived', split=True ,data=train_data, palette={0: "r", 1: "g"})

##If you are male your chance of survival is good upto 15 years of age else it is 

##not good. In case of female there is a decent chance of survival across all ages



##Let us merge Pclass and Sex to create a new col. I was trying to search for a graph 

##which could point 4 pieces of data when I thought of combining the sex & Pclass to 

##make it 1 col. We can now leverage Violinplots for 3 cols and get a nice view on what 

##we want to see.



train_data['PclassSex'] = train_data['Pclass'].astype(str) + train_data['Sex'] 

##In one line above, add a new column to the train_data dataframe with the data we want. 

##We need to convert Pclass which was Integer to String else it throws error. Let us 

##now plot this new col



plt.figure(figsize=(15,8))   ##Increase size of the figure

##Add cut=0 else it shows a -ve distribution. density is defined over -infinity 

##to +infinity. Just ignore the below 0 values. Also specify the order

sns.violinplot(x='PclassSex', y='Age', hue='Survived', split=True,data=train_data, cut=0, \

    palette={0: "r", 1: "g"}, order=['1male','2male','3male', '1female', '2female', '3female'])
##Let us tackle Fare...This is the last column that needs some analysis before we move 

##onto the next stage

print(train_data['Fare'].describe())

##We need to investigate the 0's. Also let us do density plot



sns.kdeplot(train_data['Fare'][train_data.Survived == 1], color="green", shade=True)

sns.kdeplot(train_data['Fare'][train_data.Survived == 0], color="red")

plt.legend(['Survived', 'Not Survived'])

##Hope by now the command train_data['Fare'][train_data.Survived == 1] is clear. Try 

##printing individually if any confusion the below command:

##train_data, train_data['Fare'] and train_data['Fare'][condition]



##There is a heavy tail. limit x axis to zoom in only on relevant information

plt.xlim(-10,250)

plt.show()

##Somewhere between 0-20 range, there is a huge spike in expiry. Post that survival 

##chances improve significantly. Between 0-20, it looks like chances of expiry is 3 times

##chances of survival. Assuming all these are pclass=3. The point where green touches red line 

##is where I believe pclas changes from 3 to 2.
##Let us get the fare=0 cases

print(train_data[(train_data['Fare']==0)])

##'Curiouser and curiouser' as Alice would say. None of them have siblings or children. 

##How can their fare be 0. All of them are middle aged males. All have embarked at one place. 

##Most likely this is the Cabin crew. 



print(train_data[(train_data['Fare']==0)].groupby('Pclass').agg('count'))

##This is split across pClasses - 5, 6 & 4 in each class



##Let us look at their ticketIDs to see if this provides a clue

print(train_data[(train_data['Fare']==0)].groupby(['Pclass', 'Ticket']).agg('count'))

##Ignoring the last number of the ticket, we can group them into 3 main categories. One for each 

##class. Looks like LINE are crew with pclass=3

##Others with Fare=0 are from pClass 2(23985x series) and pClass 1 (11205X series)

##unfortunately all of them go down. Also there are nearly 8 null values for Age. We have to fill 

##them all with the avg age of this group. This seems to be a neat find and will strengthen our 

##age prediction. Last but not the least we have to correct the fare...else this will lead the ML 

##algorithm on a diff path. So this fare=0 is to be treated as missing data! A few Kagglers have 

##done that but not too many.
##Let us plot a histogram to see if we can deduce anything else

plt.hist([train_data[train_data['Survived'] == 1]['Fare'], train_data[train_data['Survived'] == 0]['Fare']],

stacked=True, color = ['g','r'], bins = 50, label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend();

##As expected higher the fare the higher the survival chance. But this relationship is better 

##captured in the pClass. So this is like a duplicate info. Is it really required?



##Any other trends? Let us analyze the lowest fares and see their chances for survival

print(len(train_data[(train_data['Fare']<7.1) & (train_data['Fare']>0)]), blankline)

print(train_data[(train_data['Fare']<7.1) & (train_data['Fare']>0)].agg('mean'), blankline)

print(train_data[(train_data['Pclass']==3)&(train_data['Sex']=='male')].agg('mean'),blankline)

##These 23 folks seem to have abnormally abysmal chances of survival around 4%

##Even if they are all Male and Pclass=3 the survival percentage should be 13.5%. Why such a 

##glaring diff? There could be various reasons. Possibly the lowest fare tickets had cabins 

##furthest away from the lifeboats even amongst Pclass 3). This %age is statistically significant 

##and we can add a separate feature classifying these as High Risk.



##A lot of these low cost tickets seem to be single men

##When does first child make his presence..around 7.2 fare..

##Let us dig a bit more here

print(train_data[(train_data['Fare']<9)&(train_data['Age']<14)])

print(len(train_data[(train_data['Fare']<9)]))

print(len(train_data[(train_data['Age']<14)]))

##Out of 311 folks with <9 ticket, 2 are children...which means remaining 69 children on the 

##ship are in remaining 580. This is also statistically very very significant. This will be 

##very useful to fill the missing age column. Amongst the 311 folks, those wth missing ages are

##almost all adults



print(len(train_data[(train_data['Fare']<9)&(train_data['Age']>49)]))

##Not too many old people also. Bulk of the 311 folks are middle ages. How many nan's?



print(len(train_data[(train_data['Fare']<9)&(train_data['Age'].isnull())]))

##Wow. A whopping 99 of the 177 missing age group come in this range (fare<9). We have seen 

##above that there are minimal children and minimal old folks in fare<9. Based on Pclass

##(which should mostly be 3) and sex, we should calculate the avg age of this particular 

##group(fare<9) and fill in the missing 99 values for this group. This will be better fitting

##data than calculating the missing 177 age values based on any generic logic. Well, this fare 

##column is turning out to be the most interesting one and giving us a lot of information
combined=train_data.append(test_data)

train_data['PeopleInTicket']=train_data['Ticket'].map(combined['Ticket'].value_counts())

train_data['FarePerPerson']=train_data['Fare']/train_data['PeopleInTicket']



##For curiosity's sake also added a new column family count. Let us see if this tallies 

##with the PeopleInTicket

train_data['FamilyCount']=train_data['Parch']+train_data['SibSp']+1



pd.set_option("display.max_rows", None, "display.max_columns", None)

display(train_data.head())

##Wow. There we go. Let us spend a moment to analyze these new columns 

print(len(train_data[train_data['FamilyCount'] != train_data['PeopleInTicket']]))

##There are 195 rows where FamilyCount does not match with PeopleInTicket. For 

##remaining 600 odd rows these match perfectly. We will have to resolve the 

##195 rows at some point of time or other



print(len(train_data[(train_data['FarePerPerson']<7.1) & (train_data['FarePerPerson']>0) \

    & (train_data['Survived']==0)]))

print(len(train_data[(train_data['FarePerPerson']<7.1) & (train_data['FarePerPerson']>0) \

    & (train_data['Survived']==1)]))

##What is happening here? Earlier we saw that the lowest 26 fares (<7.1) were associated with 

##nearly 95% expiry Why has that changed drastically now with same calculation for 

##"FarePerPerson"? Only one reason could be that somehow groups of travellers managed to survive

##better as opposed to solo travellers. This is a very very imp observation and we need to take

##this into account during feature engg. Groups of travellers have better chance of survival 

##than solo. Another reason could be that most solo travellers were men who anyway had higher 

##chances of expiry. Let us analyze more..but before that a small deviation..
##Just for fun, I tried analyzing fares for Women to see if they differed from men. This was 

##merely a distraction and laying out (possible) proof of gender discrimination as far as fees 

##were concerned was never on my agenda but here were the observations:

print('Avg fare for solo man in Pclass 1: ', train_data[(train_data.Pclass==1) & (train_data\

    .FamilyCount==1) & (train_data.Sex=='male')]['FarePerPerson'].mean()) 

print('Avg fare for solo woman in Pclass 1: ', train_data[(train_data.Pclass==1) & (train_data\

    .FamilyCount==1) & (train_data.Sex=='female')]['FarePerPerson'].mean()) 

print('Avg fare for solo man in Pclass 2: ', train_data[(train_data.Pclass==2) & (train_data\

    .FamilyCount==1) & (train_data.Sex=='male')]['FarePerPerson'].mean()) 

print('Avg fare for solo woman in Pclass 2: ', train_data[(train_data.Pclass==2) & (train_data\

    .FamilyCount==1) & (train_data.Sex=='female')]['FarePerPerson'].mean()) 

print('Avg fare for solo man in Pclass 3: ', train_data[(train_data.Pclass==3) & (train_data\

    .FamilyCount==1) & (train_data.Sex=='male')]['FarePerPerson'].mean()) 

print('Avg fare for solo woman in Pclass 3: ', train_data[(train_data.Pclass==3) & (train_data\

    .FamilyCount==1) & (train_data.Sex=='female')]['FarePerPerson'].mean()) 



##On an avg they are higher by about 20% in Pclass 1, 8% in Pclass 2 and 4% in Pclass 3. I rest 

##my case. Maybe the men bargained more to get better deals? In any case, the fares are clearly

##more favorably disposed to males but they got more than what they bargained for at the end. 

##On a separate note, I sometimes feel we take the chivalry of these gentlemen for granted and 

##assume that the same would repeat under any circumstances but this was clearly not the case 

##in a few other parallel tragedies (woman and children had lower mortality than males in those

##incidents). So there was definitely something special about these gentlemen on the Titanic..

##or maybe it was the crew and the leadership team that made it happen. We can only speculate...
##Let us take the max of FamilyCount & PeopleInTicket and create a new column called GroupSize

train_data['GroupSize'] = train_data[['FamilyCount','PeopleInTicket']].max(axis=1)

##We are taking the max of 2 columns but what is this axis=1 business? In Pandas, we have series

##object (1 dimensional) and we have dataframes (2D or a series of series objects). axis=0 

##represents rows and axis=1 represents columns. When we use the max function, we have to specify

##the axis else by default axis=0 is considered and you will get the max Familycount and 

##PeopleInTicket across all rows of the dataframe.Try printing:

##train_data[['FamilyCount','PeopleInTicket']].max() and see for yourself



##Now let us plot the new columns we have got. We will use a barplot this time and more 

##specifically a barplot that counts frequency of the column...called countplot

plt.figure(figsize=(16, 6))

sns.countplot(x='FamilyCount', hue='Survived', data=train_data)
print('Between 2-4 familycount in Pclass 1,2: ', len(train_data[(train_data.FamilyCount \

        .between(2, 4)) & (train_data.Pclass.between(1,2))]))

print('Between 2-4 familycount in Pclass 3: ',len(train_data[(train_data.FamilyCount \

        .between(2, 4)) & (train_data.Pclass==3)]))

print('>4 familycount in Pclass 1,2: ',len(train_data[(train_data.FamilyCount>4) & \

        (train_data.Pclass.between(1,2))]))

print('>4 familycount in Pclass 3: ',len(train_data[(train_data.FamilyCount>4) & \

        (train_data.Pclass==3)]))

##Yipee! The data is as clear as it can be. There are 8 large groups in Pclass 1,2 

##VERSUS 54 large groups in Pclass 3. Obviously the chances of survival for former are high. 

##It is a corelation not a causation!!
##Now armed with this knowledge, let us do an Apple to Apple comparison. Let us exclude 

##solos and compare based on class. Let us first re-print the old graph for comparing

plt.figure(figsize=(16, 6))

sns.countplot(x='FamilyCount', hue='Survived', data=train_data[(train_data.FamilyCount > 1)])



plt.figure(figsize=(16, 6))

sns.countplot(x='FamilyCount',hue='Survived',data=train_data[(train_data.Pclass==3) \

        & (train_data.FamilyCount >1)])

##Situation looks gloomy now & quite the opposite from earlier graph. So groupsize does not 

##seem to have a very major role to play in survival. Being solo or not does but beyond that

##the size of group does not seem to matter. It could unmecessarily give some false positives

##and one must think whether this col is relevant enough to be fed to the ML model

##Even the 'solo' feature is worth a discussion. The general concession is that solo 

##travellers have higher mortality rate. Let us check this

print('Mortality rate overall: ', round(len(train_data[(train_data.FamilyCount==1) & \

    (train_data.Survived!=1)]) / len(train_data[train_data.FamilyCount==1])*100), '%')



print('Mortality rate Male: ', round(len(train_data[(train_data.FamilyCount==1) & \

    (train_data.Survived!=1) & (train_data.Sex=='male')]) / len(train_data[(train_data\

    .FamilyCount==1)& (train_data.Sex=='male')])*100), '%')



print('Mortality rate Female: ', round(len(train_data[(train_data.FamilyCount==1) & \

    (train_data.Survived!=1) & (train_data.Sex=='female')]) / len(train_data[(\

    train_data.FamilyCount==1)& (train_data.Sex=='female')])*100), '%')



##Solo females buck the trend. But you may say this is not a fair comparision as females 

##anyway have better survival rate. Let us do a final graph comparing mortality for Pclass 

##3 females between solo travellers & non-solo travellers before moving on

plt.figure(figsize=(16, 6))

sns.countplot(x='FamilyCount',hue='Survived',data=train_data[(train_data.Sex=='female') \

    & (train_data.Pclass==3)])

##We are left with analysis of Embarked and Cabin..In my view both dont have a 

##major role to play in survival. Let us create a new col called Cabin letter

train_data['CabinLetter'] = train_data['Cabin'].str[0]



##Let us analyze this a bit

print(train_data.groupby(['CabinLetter', 'Pclass', 'Sex'])['Survived'].agg(['count', 'mean']))

##A, B, C are all Pclass 1. D contains very few 2's. E contains very few 3's. F is mix of 2, 3

##and G is just 3. A has very few females. G is all females. For the rest there is a equal 

##distribution of males/females. With this in mind, the anamolies are:

##Cabin C has a slightly higher mortality rate than can be expected. This is minimal and 

##can be ignored. Cabin D & E seem to be bit lucky for males. The data is also strong. Let 

##us make a note of it but there is not much we can do because 75% of folks dont have Cabin 

##letters. At best we could drill down at ticket level & see if there are a bunch of tickets 

##that have higher survival rate in males..Maybe group dynamics are at play
##Let us now explore the 'Embarked' columne

print(train_data.groupby('Embarked')['Survived'].agg(['count','mean', 'size']))

##168 that onboarded at Port C have a disproportionate survival of 55%. Maybe there was 

##less Pclass=3 in these? Let us verify

print(train_data.groupby(['Embarked', 'Pclass'])['Survived'].agg(['count','mean']))

##There you go - There was a disproportionate amount of PClass=1 who onboarded at port C...

##almost 50%. To give a comparision PClass=1 was around 25% across all ports in the 

##train_data. Port C has twice the number of PClass=1. Naturally survival rate is disproprtionate

##when viewed across all classes. Overall 'Embarked' column does not seem to make much of a difference



##Let us do a more detailed check by bringing gender also into it

print(train_data.groupby(['Embarked', 'Pclass', 'Sex'])['Survived'].agg(['count','mean']))

##Anamolies - Port C Pclass 3 -  have higher chances of survival

##Anamolies - Port Q for Pclass 3  - females has high survival rate & low for Males

##Anamolies - Port S for Pclass 3 - females survival seems to be much lower