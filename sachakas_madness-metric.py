import numpy as np 

import matplotlib as plt

import seaborn as sns

import pandas as pd 

import os

import matplotlib.pyplot as plt

%matplotlib inline

columns = {}

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        try:

            tester = pd.read_csv(os.path.join(dirname, filename))

        except:

            tester = pd.read_csv(os.path.join(dirname, filename), engine="python")

            print("USING PYTHON ENGINE")

        columns[os.path.join(dirname, filename)] = tester.columns
columns
#load needed data

MRegular = pd.read_csv("/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv")

WRegular = pd.read_csv("/kaggle/input/march-madness-analytics-2020/WDataFiles_Stage2/WRegularSeasonDetailedResults.csv")

#creating teams that we need to evaluate in regular season

MRegular_col_rows = sorted(np.unique(np.array([MRegular["WTeamID"].to_numpy(),MRegular["LTeamID"].to_numpy()])))

WRegular_col_rows = sorted(np.unique(np.array([WRegular["WTeamID"].to_numpy(),WRegular["LTeamID"].to_numpy()])))

#creating tables for results

MRegular_results  = pd.DataFrame(columns=[MRegular_col_rows],index = [MRegular_col_rows])

WRegular_results = pd.DataFrame(columns=[WRegular_col_rows],index = [WRegular_col_rows])

MRegular.groupby(by="Season")["Season"].value_counts().to_numpy()

sns.lineplot(x=np.unique(MRegular["Season"].to_numpy()),y = MRegular.groupby(by="Season")["Season"].value_counts().to_numpy())

sns.lineplot(x=np.unique(WRegular["Season"].to_numpy()),y = WRegular.groupby(by="Season")["Season"].value_counts().to_numpy())

Mfill_array = np.empty(len(MRegular_results.columns))

Mfill_array.fill(1)

MRegular_weights = pd.DataFrame([Mfill_array], columns= MRegular_results.columns)



Wfill_array = np.empty(len(WRegular_results.columns))

Wfill_array.fill(1)

WRegular_weights = pd.DataFrame([Wfill_array], columns= WRegular_results.columns)

WRegular_weights
#code block for Mregular results filling



Mshort_t = MRegular[["WTeamID","LTeamID"]].to_numpy()

Mshort_t = Mshort_t.tolist()

itera = 0



for i in MRegular_results.columns:

    for l in MRegular_results.columns:

        if [i[0],l[0]] in Mshort_t:

            MRegular_results.loc[l[0],i[0]] = 1

        else:

            MRegular_results.loc[l[0],i[0]] = 0

    itera+=1
MRegular_results
#code block for WRegular results filling



Wshort_t = WRegular[["WTeamID","LTeamID"]].to_numpy()

Wshort_t = Wshort_t.tolist()

itera = 0



for i in WRegular_results.columns:

    for l in WRegular_results.columns:

        if [i[0],l[0]] in Wshort_t:

            WRegular_results.loc[l[0],i[0]] = 1

        else:

            WRegular_results.loc[l[0],i[0]] = 0

    itera+=1
WRegular_results
itera = 0

for itera in range(20):

    for i in MRegular_results.columns:

        actual_weight = 0

        for l in MRegular_results.index:

            if MRegular_results.loc[l,i] == 1:

                actual_weight = actual_weight+MRegular_weights.loc[0,l]*1

        MRegular_weights.loc[0,i[0]] = actual_weight/len(MRegular_results.columns)

    itera+=1

            
itera = 0

for itera in range(20):

    for i in WRegular_results.columns:

        actual_weight = 0

        for l in WRegular_results.index:

            if WRegular_results.loc[l,i] == 1:

                actual_weight = actual_weight+WRegular_weights.loc[0,l]*1

        WRegular_weights.loc[0,i[0]] = actual_weight/len(WRegular_results.columns)

    itera+=1
MRegular_weights_norm = pd.DataFrame(MRegular_weights.to_numpy()*100000000000000000000000, columns=MRegular_results.columns)

WRegular_weights_norm = pd.DataFrame(WRegular_weights.to_numpy()*100000000000000000000000, columns=WRegular_results.columns)
MRegular_weights_norm
#upload neede tournament results

MTournament_results = pd.read_csv("/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv")

WTournament_results = pd.read_csv("/kaggle/input/march-madness-analytics-2020/WDataFiles_Stage2/WNCAATourneyDetailedResults.csv")

MTournament_forcheck = MTournament_results[MTournament_results["Season"]>2009]

WTournament_forcheck = WTournament_results[WTournament_results["Season"]>2009]
list_counting = []

for i in range(2,20,3):

    counter = 0

    for w,l in MTournament_forcheck[["WTeamID","LTeamID"]].to_numpy():

        if (1/i-0.1)<MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/i):

            counter+=1

            #print("NON EXPECTED: ",counter, "These Teams: ", w, l)

    list_counting.append(counter)

MTournament_non_expecting = pd.DataFrame([list_counting],columns=[str(i)+"%" for i in np.around(1/np.array(range(2,20,3))*100, decimals=0).tolist()])

#we can addgraphic that shows different graphics based on expactations results
MTournament_non_expecting.plot(kind="bar")
list_counting = []

for i in range(2,20,3):

    counter = 0

    for w,l in WTournament_forcheck[["WTeamID","LTeamID"]].to_numpy():

        if (1/i-0.1)<WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/i):

            counter+=1

            #print("NON EXPECTED: ",counter, "These Teams: ", w, l)

    list_counting.append(counter)

WTournament_non_expecting = pd.DataFrame([list_counting],columns=[str(i)+"%" for i in np.around(1/np.array(range(2,20,3))*100, decimals=0).tolist()])

#we can addgraphic that shows different graphics based on expactations results

WTournament_non_expecting.plot(kind="bar")
#df1 



x_column = "WFGM"

y_column = "WFGA"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in MTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df1 = pd.concat([saver,saveradd], ignore_index = True)



#df2



x_column = "LFGM"

y_column = "LFGA"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in MTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df2 = pd.concat([saver,saveradd], ignore_index = True)



#df3



x_column = "WFGM3"

y_column = "WFGA3"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in MTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df3 = pd.concat([saver,saveradd], ignore_index = True)



#df4



x_column = "LFGM3"

y_column = "LFGA3"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in MTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df4 = pd.concat([saver,saveradd], ignore_index = True)





#df5



x_column = "WFTM"

y_column = "WFTA"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in MTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df5 = pd.concat([saver,saveradd], ignore_index = True)



#df6



x_column = "LFTM"

y_column = "LFTA"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in MTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df6 = pd.concat([saver,saveradd], ignore_index = True)



#df7



x_column = "WOR"

y_column = "WDR"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in MTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df7 = pd.concat([saver,saveradd], ignore_index = True)







#df8

x_column = "LOR"

y_column = "LDR"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in MTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df8 = pd.concat([saver,saveradd], ignore_index = True)



#df9



x_column = "WAst"

y_column = "WTO"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in MTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df9 = pd.concat([saver,saveradd], ignore_index = True)



#df10





x_column = "LAst"

y_column = "LTO"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in MTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df10 = pd.concat([saver,saveradd], ignore_index = True)





#df11



x_column = "WStl"

y_column = "WBlk"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in MTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df11 = pd.concat([saver,saveradd], ignore_index = True)





#df12





x_column = "LStl"

y_column = "LBlk"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in MTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if MRegular_weights[(w,)].to_numpy()/MRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df12 = pd.concat([saver,saveradd], ignore_index = True)



df_names = [["WFGM","WFGA"],["LFGM","LFGA"],["WFGM3","WFGA3"],["LFGM3","LFGM3"],["WFTM","WFTA"],["LFTM","LFTA"],["WOR","WDR"],["LOR","LDR"],["WAst","WTO"],["LAst","LTO"],["WStl","WBlk"],["LStl","LBlk"]]





df_list = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12]

nrow = 6

ncol  =2

fig, axes = plt.subplots(nrow, ncol,figsize=(15,40))

count=0

for r in range(nrow):

    for c in range(ncol):

        scat  = sns.scatterplot(x="X",y="Y",hue='type',data=df_list[count], ax=axes[r,c])

        scat.set(xlabel=df_names[count][0],ylabel=df_names[count][1])

        count+=1
#df1 



x_column = "WFGM"

y_column = "WFGA"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in WTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df1 = pd.concat([saver,saveradd], ignore_index = True)



#df2



x_column = "LFGM"

y_column = "LFGA"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in WTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df2 = pd.concat([saver,saveradd], ignore_index = True)



#df3



x_column = "WFGM3"

y_column = "WFGA3"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in WTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df3 = pd.concat([saver,saveradd], ignore_index = True)



#df4



x_column = "LFGM3"

y_column = "LFGA3"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in WTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df4 = pd.concat([saver,saveradd], ignore_index = True)





#df5



x_column = "WFTM"

y_column = "WFTA"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in WTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df5 = pd.concat([saver,saveradd], ignore_index = True)



#df6



x_column = "LFTM"

y_column = "LFTA"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in WTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df6 = pd.concat([saver,saveradd], ignore_index = True)



#df7



x_column = "WOR"

y_column = "WDR"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in WTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df7 = pd.concat([saver,saveradd], ignore_index = True)







#df8

x_column = "LOR"

y_column = "LDR"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in WTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df8 = pd.concat([saver,saveradd], ignore_index = True)



#df9



x_column = "WAst"

y_column = "WTO"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in WTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df9 = pd.concat([saver,saveradd], ignore_index = True)



#df10





x_column = "LAst"

y_column = "LTO"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in WTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df10 = pd.concat([saver,saveradd], ignore_index = True)





#df11



x_column = "WStl"

y_column = "WBlk"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in WTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df11 = pd.concat([saver,saveradd], ignore_index = True)





#df12





x_column = "LStl"

y_column = "LBlk"

list_counting = []

ExpectedX = []

NONExpectedX = []

ExpectedY = []

NONExpectedY = []

K = 10

for w,l,x,y in WTournament_forcheck[["WTeamID","LTeamID",x_column,y_column]].to_numpy():

    if WRegular_weights[(w,)].to_numpy()/WRegular_weights[(l,)].to_numpy()< (1/K):

        counter+=1

        NONExpectedX.append(x)

        NONExpectedY.append(y)

    else:

        ExpectedX.append(x)

        ExpectedY.append(y)

        #print("NON EXPECTED: ",counter, "These Teams: ", w, l)



filler = np.empty(len(ExpectedX), dtype=str)

filler.fill("EXPECTED")

saver = pd.DataFrame(list(zip(ExpectedX,ExpectedY,filler.tolist())),columns=["X","Y","type"])

filler = np.empty(len(NONExpectedX),dtype=str)

filler.fill("NONEXPECTED")

saveradd = pd.DataFrame(list(zip(NONExpectedX,NONExpectedY,filler.tolist())),columns=["X","Y","type"])

df12 = pd.concat([saver,saveradd], ignore_index = True)



df_names = [["WFGM","WFGA"],["LFGM","LFGA"],["WFGM3","WFGA3"],["LFGM3","LFGM3"],["WFTM","WFTA"],["LFTM","LFTA"],["WOR","WDR"],["LOR","LDR"],["WAst","WTO"],["LAst","LTO"],["WStl","WBlk"],["LStl","LBlk"]]





df_list = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12]

nrow = 6

ncol  =2

fig, axes = plt.subplots(nrow, ncol,figsize=(15,40))

count=0

for r in range(nrow):

    for c in range(ncol):

        scat  = sns.scatterplot(x="X",y="Y",hue='type',data=df_list[count], ax=axes[r,c])

        scat.set(xlabel=df_names[count][0],ylabel=df_names[count][1])

        count+=1