import pandas as pd

import os

import seaborn as sns

import numpy as np
list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

test_df  = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")



print("Train data shape: ",train_df.shape)

print("Test data shape: ",test_df.shape)



train_df.head()
print("Patients in train data : ",train_df.Patient.value_counts().shape[0])
sns.distplot(train_df.FVC,color="green")

print('Mean: ',np.mean(train_df.FVC))

print('Standard deviation: ',np.std(train_df.FVC))
df = pd.DataFrame()

df['Sample_count'] = train_df['Patient'].value_counts()

sns.countplot(x="Sample_count",data=df)
df = pd.DataFrame()

df_m = train_df[["Patient", "Age", "Sex", "SmokingStatus"]].drop_duplicates()

df_1 = train_df.groupby("Patient")['Weeks'].apply(lambda x: x.tolist()).reset_index()

df_2 = train_df.groupby("Patient")['FVC'].apply(lambda x: x.tolist()).reset_index()

df_3 = train_df.groupby("Patient")['Percent'].apply(lambda x: x.tolist()).reset_index()

df_12 = pd.merge(df_m,df_1,on='Patient', how='inner')

df_23 = pd.merge(df_2,df_3,on='Patient', how='inner')

df = pd.merge(df_12,df_23,on='Patient', how='inner')



print("Train data shape: ",df.shape)

df.head()

sns.distplot(df['Age'],color='magenta')
def fvc_min_max_range(x):

    return x[0]-x[-1]



df['FVC-range'] = df['FVC'].map(lambda x:fvc_min_max_range(x))

sns.distplot(df['FVC-range'],color='#37AA9C')
print(df[df['FVC-range']<0].shape[0]," patients are seen with increase in FVC values")
g = sns.pairplot(df[["Age", "FVC-range", "SmokingStatus"]], \

                 hue="SmokingStatus", corner=True)

g.fig.set_figwidth(10)

g.fig.set_figheight(10)

print(np.corrcoef(df["FVC-range"], df["Age"]))
def weeks_max_min_range(x):

    return x[-1]-x[0]



df['Weeks-range'] = df['Weeks'].map(lambda x:weeks_max_min_range(x))

sns.distplot(df['Weeks-range'],color='blue')
g = sns.pairplot(df[["Weeks-range", "FVC-range", "SmokingStatus"]], \

                 hue="SmokingStatus", corner=True)

g.fig.set_figwidth(10)

g.fig.set_figheight(10)

print(np.corrcoef(df["FVC-range"], df["Weeks-range"]))