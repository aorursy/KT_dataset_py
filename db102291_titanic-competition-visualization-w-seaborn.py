import numpy as np

import pandas as pd 

import random as rand

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error 

from IPython.display import display_html 
#Import training and testing data

df = pd.read_csv("../input/titanic/train.csv", index_col="PassengerId")
#Cabin letter

df['Cabin_let'] = df['Cabin'].str[0]



#Family size

df['Fam_size'] = [(df.Parch.iloc[i] + df.SibSp.iloc[i]) 

                          if ((df.Parch.iloc[i] + df.SibSp.iloc[i]) <= 4) 

                          else "5+" 

                          for i in range(0, len(df.Parch))]



df['Fam_size_bin'] = ["Large" if x > 4 else "Alone" if x == 0 else "Small" for x in (df.Parch + df.SibSp)]



#Title

df['Title']=df.Name.str.extract('([A-Za-z]+)\.')

df['Title_group'] = ["Mr" if x in ["Mr", "Master", "Don", "Rev", "Dr", "Major", "Sir", "Col", "Capt", "Jonkheer"] 

                     else "Mrs" if x in ["Mrs", "Miss", "Mme", "Ms", "Lady", "Mlle", "Countess"] 

                     else "None" for x in df.Title]





#Embarked/Class

df['Embark_class'] = df["Embarked"] + "_" + df["Pclass"].astype(str)



#Sex/Class

df['Sex_class'] = df['Sex'] + "_" + df["Pclass"].astype(str)



#Log Fare

df["log_Fare"] = np.log(df.Fare + 1);
cat_cols = [cname for cname in df.columns if df[cname].dtype == "object"]

num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]
df.head()
#Describe training data

df_cat = (df[cat_cols].describe()).T.sort_index()

df_num = (df[num_cols].describe()).T.sort_index()



df1_style = df_cat.style.set_table_attributes("style='display:inline'").set_caption('Categorical Variables')

df2_style = df_num.style.set_table_attributes("style='display:inline'").set_caption('Numerical Variables')



display_html(df1_style._repr_html_() + "\xa0\xa0\xa0\xa0\xa0\xa0" + df2_style._repr_html_(), raw=True)
figure(figsize=(12, 12))



#Does Pclass impact survival?

plt.subplot(2,2,1)

p1_data = df.groupby('Pclass').Survived.mean()

p1 = sns.barplot(y=p1_data.values,

                 x=p1_data.index, 

                 palette=["lightgrey","grey","black"])

p1.set(ylabel='Survival Rate', 

       xlabel=' Pclass');



#Does Embarked impact survival?

plt.subplot(2,2,2)

p2_data = df.groupby('Embarked').Survived.mean()

p2 = sns.barplot(y=p2_data.values,

                 x=p2_data.index, 

                 palette=["hotpink","mediumorchid","dodgerblue"])

p2.set(ylabel='', 

       xlabel='Embarked');



#Does Embark_class impact survival?

plt.subplot(2,2,3)

p3_data = df.groupby('Embark_class').Survived.mean()

p3 = sns.barplot(y=p3_data.values,

                 x=p3_data.index,

                 palette=["pink","hotpink","deeppink",

                          "violet","mediumorchid","darkviolet",

                          "skyblue","dodgerblue","royalblue"])

p3.set(ylabel='', 

       xlabel='Embarked_Pclass')

p3.axhline(p2_data[0], 

           xmin=0, xmax=1/3, 

           color="hotpink")

p3.axhline(p2_data[1], 

           xmin=1/3, xmax=2/3, 

           color="mediumorchid")

p3.axhline(p2_data[2], 

           xmin=2/3, xmax=1, 

           color="dodgerblue");



#Does Embark_class impact survival?

plt.subplot(2,2,4)

p4_data = df.groupby('Embark_class').Survived.mean()

p4 = sns.barplot(y=p4_data.values,

                 x=p4_data.index,

                 palette=["pink","skyblue","violet",

                          "hotpink","dodgerblue","mediumorchid",

                          "deeppink","royalblue","darkviolet"],

                 order=["C_1","S_1","Q_1",

                        "C_2","S_2","Q_2",

                        "C_3","S_3","Q_3"])

p4.set(ylabel='', 

       xlabel='Embarked_Pclass')

p4.axhline(p1_data[1], 

           xmin=0, xmax=1/3, 

           color="lightgrey")

p4.axhline(p1_data[2], 

           xmin=1/3, xmax=2/3, 

           color="grey")

p4.axhline(p1_data[3], 

           xmin=2/3, xmax=1, 

           color="black");
figure(figsize=(12, 12))



#Does SibSp (# Siblings/Spouses) impact survival?

plt.subplot(2,2,1)

p1_data = df.groupby('SibSp').Survived.mean()

p1 = sns.barplot(y=p1_data.values,

                 x=p1_data.index,

                 color="grey")

p1.set(ylabel='Survival Rate', 

       xlabel='SibSp');



#Does Parch (# Parents/Children) impact survival?

plt.subplot(2,2,2)

p2_data = df.groupby('Parch').Survived.mean()

p2 = sns.barplot(y=p2_data.values,

                 x=p2_data.index,

                 color="grey")

p2.set(ylabel='Survival Rate', 

       xlabel='Parch');



#Does Fam_size impact survival?

plt.subplot(2,2,3)

p3_data = df.groupby('Fam_size').Survived.mean()

p3 = sns.barplot(y=p3_data.values,

                 x=p3_data.index,

                 color="grey")

p3.set(ylabel='Survival Rate', 

      xlabel='Family Size');



#Does Fam_size_bin impact survival?

plt.subplot(2,2,4)

p4_data = df.groupby('Fam_size_bin').Survived.mean()

p4 = sns.barplot(y=p4_data.values,

                 x=p4_data.index,

                 color="grey",

                 order=["Alone","Small","Large"])

p4.set(ylabel='Survival Rate', 

       xlabel='Family Size Bin');
figure(figsize=(12, 12))



#Does Title impact survival?

plt.subplot(2,1,1)

p1_data = df.groupby('Title').Survived.mean()

p1 = sns.barplot(y=p1_data.values,

                x=p1_data.index,

                color="grey")

p1.set(ylabel='Survival Rate', 

      xlabel='Title');



#Does Title_group impact survival?

plt.subplot(2,1,2)

p2_data = df.groupby('Title_group').Survived.mean()

p2 = sns.barplot(y=p2_data.values,

                x=p2_data.index,

                color="grey")

p2.set(ylabel='Survival Rate', 

      xlabel='Title Group');
figure(figsize=(12, 12))

#Check distribution of Fare

plt.subplot(2,2,1)

p1 = sns.kdeplot(df["Fare"],

                 shade=True);



#Check distribution of log_Fare

plt.subplot(2,2,2)

p2 = sns.kdeplot(df["log_Fare"], 

                 shade=True);



#Does Fare affect survival?

plt.subplot(2,2,3)

p3 = sns.swarmplot(x = "Survived", 

                   y = "Fare", 

                   data = df);



#Does log_Fare affect survival?

plt.subplot(2,2,4)

p4 = sns.swarmplot(x = "Survived", 

                   y = "log_Fare", 

                   data = df);
figure(figsize=(12, 12))



#Is there a difference in the average price by class?

plt.subplot(2,2,1)

sns.swarmplot(x="Pclass", 

              y="Fare", 

              data=df,

              palette=["lightgrey", "grey", "black"]);



plt.subplot(2,2,2)

sns.swarmplot(x="Pclass", 

              y="log_Fare", 

              data=df,

              palette=["lightgrey", "grey", "black"]);



#Is there a difference in the average price by class and sex?

plt.subplot(2,2,3)

sns.swarmplot(x="Sex_class", 

              y="Fare", 

              data=df,

              order=["male_1","female_1",

                     "male_2","female_2",

                     "male_3","female_3"],

              palette=["skyblue","pink",

                       "dodgerblue","hotpink",

                       "royalblue","deeppink"]);



plt.subplot(2,2,4)

sns.swarmplot(x = "Sex_class", 

              y = "log_Fare", 

              data = df,

              order=["male_1","female_1",

                     "male_2","female_2",

                     "male_3","female_3"],

              palette=["skyblue","pink",

                       "dodgerblue","hotpink",

                       "royalblue","deeppink"]);
figure(figsize=(16, 6))



#Does Age affect survival?

plt.subplot(1,3,1)

p1=sns.swarmplot(x="Survived", 

                 y="Age",

                 data=df);



#Does Age/Sex affect survival?

plt.subplot(1,3,2)

p2=sns.swarmplot(x = "Survived", 

                 y = "Age",

                 hue="Sex",

                 palette=["dodgerblue","hotpink"],

                 data = df);



#Does Age/Class affect survival?

plt.subplot(1,3,3)

p2=sns.swarmplot(x = "Survived", 

                 y = "Age",

                 hue="Pclass",

                 palette=["lightgrey","grey","black"],

                 data = df);