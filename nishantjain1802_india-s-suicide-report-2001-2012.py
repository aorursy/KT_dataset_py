import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/Suicides in India 2001-2012.csv")
df.info()
df.head()
df_suicide = df.loc[(df['Total'] != 0) & (~df['State'].isin(["Total (All India)", "Total (States)", "Total (Uts)"]))]
df_no_suicide = df.loc[(df['Total'] == 0) & (~df['State'].isin(["Total (All India)", "Total (States)", "Total (Uts)"]))]
df_suicide_state = df_suicide.loc[~df_suicide['State'].isin(["A & N Islands", "Chandigarh", "D & N Haveli", "Daman & Diu", "Delhi (Ut)", "Lakshadweep", "Puducherry"])]
df_suicide_state.info()
plt.subplots(figsize=(10,14))
sns.countplot(y= "State",data = df_suicide_state).set_title("Suicide Distribution by State")
plt.subplots(figsize=(10,14))
sns.countplot(y= "State", hue= "Gender" ,data = df_suicide_state, palette= "autumn").set_title("Gender Suicide Distribution by State")
plt.subplots(figsize=(20,20))
sns.countplot(y= "State", hue= "Age_group" ,data = df_suicide_state).set_title("State Suicide Distribution by Age-Group")
print(df_suicide_state.Gender.value_counts())
df_suicide_state.Gender.value_counts().plot(kind= "pie", autopct='%1.1f%%', shadow=True, figsize=(5,5))
plt.title("Gender Suicide Distribution of State")
sns.countplot(x = "Age_group", hue= "Gender", data= df_suicide_state, palette= "spring").set_title("Gender Suicide Distribution of State by Age-Group")
plt.subplots(figsize=(9,5))
sns.countplot(x = "Year", data= df_suicide_state, palette= "spring").set_title("Year Suicide Distribution of State")
print(df_suicide_state.Type_code.value_counts())
df_suicide_state.Type_code.value_counts().plot(kind= "pie", autopct='%1.1f%%', shadow=True, figsize=(7,7))
plt.title("Type-Code Distribution of State")
df_suicide_state_causes = df_suicide_state.loc[df_suicide_state["Type_code"]=='Causes']
plt.subplots(figsize=(15,12))
sns.countplot(y = "Type", data= df_suicide_state_causes).set_title("Type:'Causes' for Suicide in State")
df_suicide_state_causes = df_suicide_state.loc[df_suicide_state["Type_code"]=='Means_adopted']
plt.subplots(figsize=(15,8))
sns.countplot(y = "Type", data= df_suicide_state_causes).set_title("Type:'Means_adopted' for Suicide in State")
df_suicide_state_causes = df_suicide_state.loc[df_suicide_state["Type_code"]=='Social_Status']
plt.subplots(figsize=(10,2))
sns.countplot(y = "Type", data= df_suicide_state_causes).set_title("Type: 'Social_Status' for Suicide in State")
df_suicide_state_causes = df_suicide_state.loc[df_suicide_state["Type_code"]=='Education_Status']
plt.subplots(figsize=(10,2))
sns.countplot(y = "Type", data= df_suicide_state_causes).set_title("Type: 'Education_Status' for Suicide in State")
df_suicide_state_causes = df_suicide_state.loc[df_suicide_state["Type_code"]=='Professional_Profile']
plt.subplots(figsize=(15,5))
sns.countplot(y = "Type", data= df_suicide_state_causes).set_title("Type:'Professional_Profile' for Suicide in State")
df_suicide_ut = df_suicide.loc[df_suicide['State'].isin(["A & N Islands", "Chandigarh", "D & N Haveli", "Daman & Diu", "Delhi (Ut)", "Lakshadweep", "Puducherry"])]
df_suicide_ut.info()
sns.countplot(y= "State",data = df_suicide_ut).set_title("Suicide Distribution by UT")
sns.countplot(y= "State", hue= "Gender" ,data = df_suicide_ut).set_title("Gender Suicide Distribution by UT")
plt.subplots(figsize=(15,10))
sns.countplot(y= "State", hue= "Age_group" ,data = df_suicide_ut).set_title("UT Suicide Distribution by Age-Group")
print(df_suicide_ut.Gender.value_counts())
df_suicide_ut.Gender.value_counts().plot(kind= "pie", autopct='%1.1f%%', shadow=True, figsize=(5,5))
plt.title("Gender Suicide Distribution of UT")
sns.countplot(x = "Age_group", hue= "Gender", data= df_suicide_ut, palette= "spring").set_title("Gender Suicide Distribution of UT by Age-Group")
plt.subplots(figsize=(9,5))
sns.countplot(x = "Year", data= df_suicide_ut, palette= "spring").set_title("Year Suicide Distribution of UT")
print(df_suicide_ut.Type_code.value_counts())
df_suicide_ut.Type_code.value_counts().plot(kind= "pie", autopct='%1.1f%%', shadow=True, figsize=(7,7))
plt.title("Type-Code Distribution of UT")
df_suicide_ut_causes = df_suicide_ut.loc[df_suicide_ut["Type_code"]=='Causes']
plt.subplots(figsize=(15,12))
sns.countplot(y = "Type", data= df_suicide_ut_causes).set_title("Type:'Causes' for Suicide in UT")
df_suicide_ut_causes = df_suicide_ut.loc[df_suicide_ut["Type_code"]=='Means_adopted']
plt.subplots(figsize=(15,8))
sns.countplot(y = "Type", data= df_suicide_ut_causes).set_title("Type:'Means_adopted' for Suicide in UT")
df_suicide_ut_causes = df_suicide_ut.loc[df_suicide_ut["Type_code"]=='Social_Status']
plt.subplots(figsize=(10,2))
sns.countplot(y = "Type", data= df_suicide_ut_causes).set_title("Type: 'Social_Status' for Suicide in UT")
df_suicide_ut_causes = df_suicide_ut.loc[df_suicide_ut["Type_code"]=='Education_Status']
plt.subplots(figsize=(10,2))
sns.countplot(y = "Type", data= df_suicide_ut_causes).set_title("Type: 'Education_Status' for Suicide in UT")
df_suicide_ut_causes = df_suicide_ut.loc[df_suicide_ut["Type_code"]=='Professional_Profile']
plt.subplots(figsize=(15,5))
sns.countplot(y = "Type", data= df_suicide_ut_causes).set_title("Type:'Professional_Profile' for Suicide in UT")


