import pandas as pd
%matplotlib inline
dfTrain = pd.read_csv('../input/dataset/Train.csv')
dfCampDetail = pd.read_csv('../input/dataset/Health_Camp_Detail.csv')
dfPatientProfile = pd.read_csv('../input/dataset/Patient_Profile.csv')
dfFirstCampDetail = pd.read_csv('../input/dataset/First_Health_Camp_Attended.csv')
dfSecondDetail = pd.read_csv('../input/dataset/Second_Health_Camp_Attended.csv')
dfThirdDetail = pd.read_csv('../input/dataset/Third_Health_Camp_Attended.csv')
dfTest = pd.read_csv('../input/dataset/Test.csv')
dfSecondDetail.rename(columns={"Health Score": "Health_Score"}, inplace=True)
dfTrain.head()
df_00 = dfTrain.merge(dfCampDetail, how = 'inner', on=['Health_Camp_ID'])
df_01 = df_00.merge(dfPatientProfile, how = 'left', on=['Patient_ID'])
# mostar 5 registros con los campos que tenemos hasta ahora
df_01.head().T
# campamento del primer formato
df_02 = df_01[df_01['Category1'] == 'First']
df_03 = df_02.merge(dfFirstCampDetail, how = 'left', on=['Health_Camp_ID','Patient_ID'])
df_03["Target"] = df_03["Health_Score"].apply(lambda h: 1 if h > 0 else 0)
# mostar 5 registros de los campamentos del primer formato
df_03.head().T
# campamento del segundo formato
df_04 = df_01[df_01['Category1'] == 'Second']
df_05 = df_04.merge(dfSecondDetail, how = 'left', on=['Health_Camp_ID','Patient_ID'])
df_05["Target"] = df_05["Health_Score"].apply(lambda h: 1 if h > 0 else 0)
# mostar 5 registros de los campamentos del segundo formato
df_05.head().T
# campamento del tercer formato
df_06 = df_01[df_01['Category1'] == 'Third']
df_07 = df_06.merge(dfThirdDetail, how = 'left', on=['Health_Camp_ID','Patient_ID'])
df_07["Target"] = df_07["Number_of_stall_visited"].apply(lambda h: 1 if h > 0 else 0)
# mostar 5 registros de los campamentos del tercer formato
df_07.head().T
# Concatenar los campamentos de los 3 formatos
df_08 = pd.concat([df_03,df_05,df_07])
df_08 = df_08[['Patient_ID', 'Health_Camp_ID', 'Target', 'Registration_Date', 'Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Camp_Start_Date', 'Camp_End_Date', 'Category1', 'Category2', 'Category3', 'Camp_Start_Date_t', 'Camp_End_Date_t', 'Online_Follower', 'LinkedIn_Shared', 'Twitter_Shared', 'Facebook_Shared', 'Income', 'Education_Score', 'Age', 'First_Interaction', 'City_Type', 'Employer_Category', 'Donation', 'Health_Score', 'Last_Stall_Visited_Number', 'Number_of_stall_visited']]
df_08.reset_index(drop=True, inplace=True)
pd.DataFrame(df_08.Target.value_counts(normalize=True)).reset_index(drop=False).rename(columns={"Target": "%", "index": "Target"})
df_08['Camp_Start_Date_t'] = pd.to_datetime(df_08['Camp_Start_Date'])
df_08['Camp_End_Date_t'] = pd.to_datetime(df_08['Camp_End_Date'])
df_08['Registration_Date_t'] = pd.to_datetime(df_08['Registration_Date'])
df_08["Registration_period"] = df_08.loc[pd.notnull(df_08['Registration_Date_t']), 'Registration_Date_t'].apply(lambda r: r.strftime("%Y%m"))
df_09 = df_08.groupby(["Registration_period"], as_index=False).agg({"Patient_ID": "nunique", "Target": "sum"})
df_09["assistance_%"] = df_09.apply(lambda row: (row["Target"] * 100.0)/row["Patient_ID"] , axis=1)
df_09.plot(x="Registration_period", y="assistance_%", figsize=(20, 10), ylim=(0, 100), title="Histórico del % de asistentes respecto a los registrados a los campamentos")
print("Total de registros: {}".format(dfTrain.shape[0] + dfTest.shape[0]))
dfTest