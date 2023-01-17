#importing header files

import pandas as pd

import pickle

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import matplotlib.colors as mcolors



#Checking a sample file

F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data1.csv")

#Dropping an unnecessary column

F=F.drop('Unnamed: 0',axis=1)

#Checking the sample data

print(F.head(5))

print(F.dtypes)
#Converting objects to their appropriate dtypes

F['subject identifier']= F['subject identifier'].astype(str)

F['matching condition']= F['matching condition'].astype(str)



#Defining cmap for graph plotting

cmap = mcolors.LinearSegmentedColormap.from_list("n",['#ffeda0','#feb24c','#f03b20'])



#Xticks Column

x=[0,5,10,15,20,25,30,35,40,45,50,55,60]



#Running a loop across the time series

for i in range(1,256):

    X=F[F['sample num']==i]



#Plotting a heatmap



    D= X.pivot_table(values='sensor value', 

                                     index='subject identifier', 

                                     columns='channel')

    s, ax = plt.subplots(figsize=(40, 20))



   

    ax=sns.heatmap(D,fmt="g", cmap=cmap,linewidths=0.20)





    #Editing the Axes

    plt.title('Voltage Readings-'+str(i),fontname='Times New Roman',fontsize=20)

    plt.xlabel('Channels(0-63)',fontname='Times New Roman' ,fontsize=10)

    plt.ylabel('')

    plt.xticks(x,(0,5,10,15,20,25,30,35,40,45,50,55,60))

    plt.yticks([])

    

    #Saving all the heatmaps

    #plt.savefig("GG\Heatmap"+ str(i)+".png")

    

    #Showing few heatmaps to see the differences according to time series

    if i<11:

        continue

    #Closing so as to save memory

    plt.close(s)

#Equating Co-Relations across all channels

for s in range(0,64):

    for j in range(0,64):

        X=F[F['channel']==s]

        Y=F[F['channel']==j]

        co=np.corrcoef(X['sensor value'],Y['sensor value'])[0,1]

        if s==0 and j==0:

            Z= pd.DataFrame(

            { 'corrcoef':co ,'channel 1': s , 'channel 2': j}, index=[s*64 +j])

        else: 

            Z = Z.append( pd.DataFrame(

            { 'corrcoef':co ,'channel 1': s , 'channel 2': j},index=[s*64 + j]))



D= Z.pivot_table(values='corrcoef', 

                                     index='channel 1', 

                                     columns='channel 2')



# Generating masks to avoid repition(As the matrix formed will be a symmetric one)

mask = np.zeros_like(D, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(18, 18))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



#Generate the heatmap

sns.heatmap(D, mask=mask, cmap=cmap,  center=0,vmax=1,vmin=-1,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



#Edit the Axes and title

plt.title('Correlation Matrix',fontname='Times New Roman',fontsize=20)

plt.xlabel('Channels(0-63)',fontname='Times New Roman' ,fontsize=10)

plt.ylabel('Channels(0-63)', fontname='Times New Roman' ,fontsize=10)

x=[0,5,10,15,20,25,30,35,40,45,50,55,60]

plt.xticks(x,(0,5,10,15,20,25,30,35,40,45,50,55,60))

plt.yticks(x,(0,5,10,15,20,25,30,35,40,45,50,55,60))

    



#Saving the image

#plt.savefig("Correlation Matrix.png")

#Loop for extracting S1 object

n=0

for i in range(1,469):

    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")

    #Converting objects to aprropriate DataTypes

    F['matching condition']= F['matching condition'].astype(str)

    F['subject identifier']= F['subject identifier'].astype(str)

    if F.loc[5,'matching condition']=='S1 obj' and F.loc[5,'subject identifier']=='a':

        n=n+1

        #Calculating Correlation for a recording

        for s in range(0,64):

            for j in range(0,64):

                if s>=j:

                    continue

                X=F[F['channel']==s]

                Y=F[F['channel']==j]

                co=np.corrcoef(X['sensor value'],Y['sensor value'])[0,1]

                if s==0 and j==1:

                    Z= pd.DataFrame(

                    { 'corrcoef':co ,'channel 1': s , 'channel 2': j}, index=[s*64 +j])

                else: 

                    Z = Z.append( pd.DataFrame(

                    { 'corrcoef':co ,'channel 1': s , 'channel 2': j},index=[s*64 + j]))



#Calculating only 10 largest corr. values from each recording

        label=Z.nlargest(10, 'corrcoef')

        

#Storing the 10 largest values in a dataframe

        if n==1:

            A= label

           

        else: 

            A = A.append(label)             

#To keep track of the loop

    if i%50==0 :

        print (i)







#Calculating the frequency of all channel pairs having largest corr. among different recordings(Top 15)

Q1=A.groupby(['channel 1','channel 2']).count()

Q1=Q1.add_suffix('_Count').reset_index()

Q1=Q1.sort_values(by=['channel 1', 'channel 2'])

Q1['Pair']=Q1['channel 1'].astype(str)+'-'+ Q1['channel 2'].astype(str)

Q1=Q1.nlargest(15, 'corrcoef_Count')



#Plotting the graph

PairFrequency= Q1[['Pair','corrcoef_Count']].groupby('Pair',as_index=False).mean().sort_values(by='corrcoef_Count',ascending=False)

sns.barplot(x='Pair', y='corrcoef_Count', data=PairFrequency)



#Editing the axes and title of the graph

plt.suptitle('Max Correlation Frequency in S1 Stimulus for alcoholics',fontname='Times New Roman' ,fontsize=15)

plt.title('(Averaged over 480 readings)',fontname='Times New Roman' ,fontsize=10)

plt.xlabel('Channel Pairs',fontname='Times New Roman' ,fontsize=10)

plt.ylabel('Count of max correlation for the pair',fontname='Times New Roman' ,fontsize=10)



#Saving the image

#plt.savefig("CorrCount(S1-A).png")
#Loop for extracting S1 object

n=0

for i in range(1,469):

    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")

    #Converting objects to aprropriate DataTypes

    F['matching condition']= F['matching condition'].astype(str)

    F['subject identifier']= F['subject identifier'].astype(str)

    if F.loc[5,'matching condition']=='S1 obj' and F.loc[5,'subject identifier']=='c':

        n=n+1

        #Calculating Correlation for a recording

        for s in range(0,64):

            for j in range(0,64):

                if s>=j:

                    continue

                X=F[F['channel']==s]

                Y=F[F['channel']==j]

                co=np.corrcoef(X['sensor value'],Y['sensor value'])[0,1]

                if s==0 and j==1:

                    Z= pd.DataFrame(

                    { 'corrcoef':co ,'channel 1': s , 'channel 2': j}, index=[s*64 +j])

                else: 

                    Z = Z.append( pd.DataFrame(

                    { 'corrcoef':co ,'channel 1': s , 'channel 2': j},index=[s*64 + j]))



#Calculating only 10 largest corr. values from each recording

        label=Z.nlargest(10, 'corrcoef')

        

#Storing the 10 largest values in a dataframe

        if n==1:

            A= label

           

        else: 

            A = A.append(label)             

#To keep track of the loop

    if i%50==0 :

        print (i)







#Calculating the frequency of all channel pairs having largest corr. among different recordings(Top 15)

Q1=A.groupby(['channel 1','channel 2']).count()

Q1=Q1.add_suffix('_Count').reset_index()

Q1=Q1.sort_values(by=['channel 1', 'channel 2'])

Q1['Pair']=Q1['channel 1'].astype(str)+'-'+ Q1['channel 2'].astype(str)

Q1=Q1.nlargest(15, 'corrcoef_Count')



#Plotting the graph

PairFrequency= Q1[['Pair','corrcoef_Count']].groupby('Pair',as_index=False).mean().sort_values(by='corrcoef_Count',ascending=False)

sns.barplot(x='Pair', y='corrcoef_Count', data=PairFrequency)



#Editing the axes and title of the graph

plt.suptitle('Max Correlation Frequency in S1 Stimulus for control',fontname='Times New Roman' ,fontsize=15)

plt.title('(Averaged over 480 readings)',fontname='Times New Roman' ,fontsize=10)

plt.xlabel('Channel Pairs',fontname='Times New Roman' ,fontsize=10)

plt.ylabel('Count of max correlation for the pair',fontname='Times New Roman' ,fontsize=10)



#Saving the image

#plt.savefig("CorrCount(S1-C).png")
#Loop for extracting S2(match) object

n=0

for i in range(1,469):

    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")

    #Converting objects to aprropriate DataTypes

    F['matching condition']= F['matching condition'].astype(str)

    F['subject identifier']= F['subject identifier'].astype(str)

    if F.loc[5,'matching condition']=='S2 match' and F.loc[5,'subject identifier']=='a':

        n=n+1

        #Calculating Correlation for a recording

        for s in range(0,64):

            for j in range(0,64):

                if s>=j:

                    continue

                X=F[F['channel']==s]

                Y=F[F['channel']==j]

                co=np.corrcoef(X['sensor value'],Y['sensor value'])[0,1]

                if s==0 and j==1:

                    Z= pd.DataFrame(

                    { 'corrcoef':co ,'channel 1': s , 'channel 2': j}, index=[s*64 +j])

                else: 

                    Z = Z.append( pd.DataFrame(

                    { 'corrcoef':co ,'channel 1': s , 'channel 2': j},index=[s*64 + j]))



#Calculating only 10 largest corr. values from each recording

        label=Z.nlargest(10, 'corrcoef')

        

#Storing the 10 largest values in a dataframe

        if n==1:

            A= label

           

        else: 

            A = A.append(label)             

#To keep track of the loop

    if i%50==0 :

        print (i)







#Calculating the frequency of all channel pairs having largest corr. among different recordings(Top 15)

Q1=A.groupby(['channel 1','channel 2']).count()

Q1=Q1.add_suffix('_Count').reset_index()

Q1=Q1.sort_values(by=['channel 1', 'channel 2'])

Q1['Pair']=Q1['channel 1'].astype(str)+'-'+ Q1['channel 2'].astype(str)

Q1=Q1.nlargest(15, 'corrcoef_Count')



#Plotting the graph

PairFrequency= Q1[['Pair','corrcoef_Count']].groupby('Pair',as_index=False).mean().sort_values(by='corrcoef_Count',ascending=False)

sns.barplot(x='Pair', y='corrcoef_Count', data=PairFrequency)



#Editing the axes and title of the graph

plt.suptitle('Max Correlation Frequency in S2(match) Stimulus for alcoholics',fontname='Times New Roman' ,fontsize=15)

plt.title('(Averaged over 480 readings)',fontname='Times New Roman' ,fontsize=10)

plt.xlabel('Channel Pairs',fontname='Times New Roman' ,fontsize=10)

plt.ylabel('Count of max correlation for the pair',fontname='Times New Roman' ,fontsize=10)



#Saving the image

#plt.savefig("CorrCount(S2 match-A).png")
#Loop for extracting S2(match) object

n=0

for i in range(1,469):

    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")

    #Converting objects to aprropriate DataTypes

    F['matching condition']= F['matching condition'].astype(str)

    F['subject identifier']= F['subject identifier'].astype(str)

    if F.loc[5,'matching condition']=='S2 match' and F.loc[5,'subject identifier']=='c':

        n=n+1

        #Calculating Correlation for a recording

        for s in range(0,64):

            for j in range(0,64):

                if s>=j:

                    continue

                X=F[F['channel']==s]

                Y=F[F['channel']==j]

                co=np.corrcoef(X['sensor value'],Y['sensor value'])[0,1]

                if s==0 and j==1:

                    Z= pd.DataFrame(

                    { 'corrcoef':co ,'channel 1': s , 'channel 2': j}, index=[s*64 +j])

                else: 

                    Z = Z.append( pd.DataFrame(

                    { 'corrcoef':co ,'channel 1': s , 'channel 2': j},index=[s*64 + j]))



#Calculating only 10 largest corr. values from each recording

        label=Z.nlargest(10, 'corrcoef')

        

#Storing the 10 largest values in a dataframe

        if n==1:

            A= label

           

        else: 

            A = A.append(label)             

#To keep track of the loop

    if i%50==0 :

        print (i)







#Calculating the frequency of all channel pairs having largest corr. among different recordings(Top 15)

Q1=A.groupby(['channel 1','channel 2']).count()

Q1=Q1.add_suffix('_Count').reset_index()

Q1=Q1.sort_values(by=['channel 1', 'channel 2'])

Q1['Pair']=Q1['channel 1'].astype(str)+'-'+ Q1['channel 2'].astype(str)

Q1=Q1.nlargest(15, 'corrcoef_Count')



#Plotting the graph

PairFrequency= Q1[['Pair','corrcoef_Count']].groupby('Pair',as_index=False).mean().sort_values(by='corrcoef_Count',ascending=False)

sns.barplot(x='Pair', y='corrcoef_Count', data=PairFrequency)



#Editing the axes and title of the graph

plt.suptitle('Max Correlation Frequency in S2(match) Stimulus for control',fontname='Times New Roman' ,fontsize=15)

plt.title('(Averaged over 480 readings)',fontname='Times New Roman' ,fontsize=10)

plt.xlabel('Channel Pairs',fontname='Times New Roman' ,fontsize=10)

plt.ylabel('Count of max correlation for the pair',fontname='Times New Roman' ,fontsize=10)



#Saving the image

#plt.savefig("CorrCount(S2 match-C).png")
#Loop for extracting S2(no match) object

n=0

for i in range(1,469):

    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")

    #Converting objects to aprropriate DataTypes

    F['matching condition']= F['matching condition'].astype(str)

    F['subject identifier']= F['subject identifier'].astype(str)

    if F.loc[5,'matching condition']=='S2 nomatch,' and F.loc[5,'subject identifier']=='a':

        n=n+1

        #Calculating Correlation for a recording

        for s in range(0,64):

            for j in range(0,64):

                if s>=j:

                    continue

                X=F[F['channel']==s]

                Y=F[F['channel']==j]

                co=np.corrcoef(X['sensor value'],Y['sensor value'])[0,1]

                if s==0 and j==1:

                    Z= pd.DataFrame(

                    { 'corrcoef':co ,'channel 1': s , 'channel 2': j}, index=[s*64 +j])

                else: 

                    Z = Z.append( pd.DataFrame(

                    { 'corrcoef':co ,'channel 1': s , 'channel 2': j},index=[s*64 + j]))



#Calculating only 10 largest corr. values from each recording

        label=Z.nlargest(10, 'corrcoef')

        

#Storing the 10 largest values in a dataframe

        if n==1:

            A= label

           

        else: 

            A = A.append(label)             

#To keep track of the loop

    if i%50==0 :

        print (i)







#Calculating the frequency of all channel pairs having largest corr. among different recordings(Top 15)

Q1=A.groupby(['channel 1','channel 2']).count()

Q1=Q1.add_suffix('_Count').reset_index()

Q1=Q1.sort_values(by=['channel 1', 'channel 2'])

Q1['Pair']=Q1['channel 1'].astype(str)+'-'+ Q1['channel 2'].astype(str)

Q1=Q1.nlargest(15, 'corrcoef_Count')



#Plotting the graph

PairFrequency= Q1[['Pair','corrcoef_Count']].groupby('Pair',as_index=False).mean().sort_values(by='corrcoef_Count',ascending=False)

sns.barplot(x='Pair', y='corrcoef_Count', data=PairFrequency)



#Editing the axes and title of the graph

plt.suptitle('Max Correlation Frequency in S2(no match) Stimulus for alcoholics',fontname='Times New Roman' ,fontsize=15)

plt.title('(Averaged over 480 readings)',fontname='Times New Roman' ,fontsize=10)

plt.xlabel('Channel Pairs',fontname='Times New Roman' ,fontsize=10)

plt.ylabel('Count of max correlation for the pair',fontname='Times New Roman' ,fontsize=10)



#Saving the image

#plt.savefig("CorrCount(S2 no match-A).png")
#Loop for extracting S2(no match) object

n=0

for i in range(1,469):

    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")

    #Converting objects to aprropriate DataTypes

    F['matching condition']= F['matching condition'].astype(str)

    F['subject identifier']= F['subject identifier'].astype(str)

    if F.loc[5,'matching condition']=='S2 nomatch,' and F.loc[5,'subject identifier']=='c':

        n=n+1

        #Calculating Correlation for a recording

        for s in range(0,64):

            for j in range(0,64):

                if s>=j:

                    continue

                X=F[F['channel']==s]

                Y=F[F['channel']==j]

                co=np.corrcoef(X['sensor value'],Y['sensor value'])[0,1]

                if s==0 and j==1:

                    Z= pd.DataFrame(

                    { 'corrcoef':co ,'channel 1': s , 'channel 2': j}, index=[s*64 +j])

                else: 

                    Z = Z.append( pd.DataFrame(

                    { 'corrcoef':co ,'channel 1': s , 'channel 2': j},index=[s*64 + j]))



#Calculating only 10 largest corr. values from each recording

        label=Z.nlargest(10, 'corrcoef')

        

#Storing the 10 largest values in a dataframe

        if n==1:

            A= label

           

        else: 

            A = A.append(label)             

#To keep track of the loop

    if i%50==0 :

        print (i)







#Calculating the frequency of all channel pairs having largest corr. among different recordings(Top 15)

Q1=A.groupby(['channel 1','channel 2']).count()

Q1=Q1.add_suffix('_Count').reset_index()

Q1=Q1.sort_values(by=['channel 1', 'channel 2'])

Q1['Pair']=Q1['channel 1'].astype(str)+'-'+ Q1['channel 2'].astype(str)

Q1=Q1.nlargest(15, 'corrcoef_Count')



#Plotting the graph

PairFrequency= Q1[['Pair','corrcoef_Count']].groupby('Pair',as_index=False).mean().sort_values(by='corrcoef_Count',ascending=False)

sns.barplot(x='Pair', y='corrcoef_Count', data=PairFrequency)



#Editing the axes and title of the graph

plt.suptitle('Max Correlation Frequency in S2(no match) Stimulus for control',fontname='Times New Roman' ,fontsize=15)

plt.title('(Averaged over 480 readings)',fontname='Times New Roman' ,fontsize=10)

plt.xlabel('Channel Pairs',fontname='Times New Roman' ,fontsize=10)

plt.ylabel('Count of max correlation for the pair',fontname='Times New Roman' ,fontsize=10)



#Saving the image

#plt.savefig("CorrCount(S2 no match-C).png")