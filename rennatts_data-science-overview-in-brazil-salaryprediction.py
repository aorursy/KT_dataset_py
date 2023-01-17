#Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams
#Import dataset

df = pd.read_csv('/kaggle/input/pesquisa-data-hackers-2019/datahackers-survey-2019-anonymous-responses.csv')
#let´s get more info about dataset

tab_info = pd.DataFrame(df.dtypes).T.rename(index={0:'column Type'}) 

tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))

tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.

                                       rename(index={0: 'null values (%)'}))

tab_info
df.describe()
#rename the columns that i know for sure I´m gonna need it a lot

df.rename(columns={"('P1', 'age')": "age", "('P2', 'gender')": "gender", "('P8', 'degreee_level')": "degree_level",

                   "('P16', 'salary_range')": "salary", "('P17', 'time_experience_data_science')": "time_experience_ds",

                   "('P18', 'time_experience_before')": "time_experience_before", "('P19', 'is_data_science_professional')": "ds_professional",

                   "('P22', 'most_used_proggraming_languages')": "language", "('P5', 'living_state')": "state",

                   "('P10', 'job_situation')": "job_situation", "('P12', 'workers_number')": "workers_number",

                   "('P13', 'manager')": "manager","('D3', 'anonymized_degree_area')": "degree_area", 

                   "('D4', 'anonymized_market_sector')": "market_sector"}, inplace=True)



#what is the relation between age and salary

df.groupby(["salary"])["age"].mean().sort_values(ascending=False).plot(kind='bar', figsize=(10,5), color= "red", title= "relation age & salary")

#check the degree level 

df["degree_level"].value_counts().sort_values(ascending=False).plot(kind='bar', figsize=(10,5), color= "red", title= "Degree level")

#count the range salaries 

df["salary"].value_counts().sort_values(ascending=False).plot(kind='bar', figsize=(10,5), color= "blue", title= "couting the salaries")

#considering the salaries of only those to are data science professionals

df[df.ds_professional ==1].salary.value_counts().sort_values(ascending=False).plot(kind='bar', figsize=(10,5), color= "blue", title= "ds professional salary")

#for ds professionals, let´s check the job situation

df[df.ds_professional ==1].job_situation.value_counts().sort_values(ascending=False).plot(kind='bar', figsize=(10,5), color= "blue", title= "ds professionals job situation")



#what is the relation between salary and time experience data science

df.groupby(["time_experience_before"])["salary"].value_counts().sort_values(ascending=False).nlargest(15).plot(kind='bar', figsize=(10,5), color= "red", title= "time experience before ds and salary")

#language overview

df["language"].value_counts().sort_values(ascending=False).plot(kind='bar', figsize=(10,5), color= "red", title= "most used languages")

#data manipulation on salary column

variable_split= df["salary"].str.split(" ")

df["salario1"]= variable_split.str.get(2)

df["salario2"]= variable_split.str.get(5)







#slit again

variable_split= df["salario1"].str.split("/")

df["salario1"]= variable_split.str.get(0)





variable_split= df["salario2"].str.split("/")

df["salario2"]= variable_split.str.get(0)


#replace the R$ per nan

df.loc[df['salario1'] == "R$",'salario1'] = np.nan







#in the dataset 3.000 as written as 3000. We have to change this



#replace 3000 by 3.000

df["salario2"].replace(["3000"], "3.000", inplace= True)

#remove the "." to transform it into international form

df["salario1"].replace(["1.001"], "1001", inplace= True)

df["salario1"].replace(["2.001"], "2001", inplace= True)

df["salario1"].replace(["3.001"], "3001", inplace= True)

df["salario1"].replace(["4.001"], "4001", inplace= True)

df["salario1"].replace(["6.001"], "6001", inplace= True)

df["salario1"].replace(["8.001"], "8001", inplace= True)

df["salario1"].replace(["12.001"], "12001", inplace= True)

df["salario1"].replace(["16.001"], "16001", inplace= True)



df["salario2"].replace(["2.000"], "2009", inplace= True)

df["salario2"].replace(["3.000"], "3000", inplace= True)

df["salario2"].replace(["4.000"], "4000", inplace= True)

df["salario2"].replace(["6.000"], "6000", inplace= True)

df["salario2"].replace(["8.000"], "8000", inplace= True)

df["salario2"].replace(["12.000"], "12000", inplace= True)

df["salario2"].replace(["20.000"], "20000", inplace= True)

#transform the datatype to int

#CONVERT OBJECT TO FLOAT

df["salario1"]= pd.to_numeric(df["salario1"])





df["salario2"]= pd.to_numeric(df["salario2"])       





#take the mean of both salaries

df["salario"]= (df["salario2"]+df["salario1"])/2
f,ax=plt.subplots(1,2,figsize=(12,8))

df['ds_professional'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('data science professional')

ax[0].set_ylabel('ds professional Count')

sns.countplot('ds_professional',data=df,ax=ax[1])

ax[1].set_title('ds professional')

plt.show()
#I wanna take a lot at only data science professioanals

ds_prof= df[df.ds_professional ==1]
#not ds professional

not_ds_prof= df[df.ds_professional ==0]

##take a look at gender and age related to salary of data science professionals

salary_gender_ds= ds_prof.groupby(["gender"])["salario", "age"].mean()

print(salary_gender_ds)
##take a look at gender and age related to salary of non data science professionals

salary_gender_notds= not_ds_prof.groupby(["gender"])["salario", "age"].mean()

print(salary_gender_notds)

#SALARY DISTRIBUTION PER GENDER FOR DS AND NON DS PROFESSIONALS

#DISTRIBUTION

import seaborn as sns

#SALARY DISTRIBUTION

plt.figure(figsize = (8, 6))

plt.title('Not ds professional & ds professional salary comparision')

sns.distplot(ds_prof["salario"], label="ds professional")

sns.distplot(not_ds_prof["salario"], label="Not ds professional")

plt.legend()

#Visualizing the salary and age Distribution

plt.figure(figsize = (10,10))

plt.subplot(2,2,1)

#train['Chance of Admit '].hist()

sns.distplot(df['salario'],bins=10,color='Violet',  kde_kws={"color": "g", "lw": 5, "label": "KDE"},hist_kws={"linewidth": 5,"alpha": 0.8 })

plt.subplot(2,2,2)

sns.boxplot(df['age'])
#salary per gender- non ds professionals

plt.figure(figsize=(10,10))

plt.title("salary per gender- non-ds professionals")

sns.boxplot(x='salario',y='gender',data =not_ds_prof)
#salary per gender ds professionals

plt.figure(figsize=(10,10))

plt.title("salary per gender- ds professionals")

sns.boxplot(x='salario',y='gender',data =ds_prof)
#salary x manager x age

plt.figure(figsize=(10,10))

plt.title("salario x age x manager")

sns.boxplot(x='salario',y='age',data =df,hue='manager')
#salary per gender and ds and non ds professionals comparision

index = np.arange(2)

bar_width = 0.35



fig, ax = plt.subplots()

summer = ax.bar(index, ds_prof.groupby(["gender"])["salario"].mean(), bar_width,

                label="ds professionals")





winter = ax.bar(index+bar_width, not_ds_prof.groupby(["gender"])["salario"].mean(),

                 bar_width, label="non-ds professionals")



ax.set_xlabel('gender')

ax.set_ylabel('salary')

ax.set_title('salary per gender: ds & non-ds professional')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(["female", "male"])

ax.legend()



plt.show()

#salary x age

g = sns.boxplot(x='salario', y="age", data=ds_prof, palette="Set1")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Salary according to age(ds professionals)", fontsize=15)

g.set_xlabel("salary", fontsize=12)

g.set_ylabel("age", fontsize=12)

plt.show()

#ds professionals salary x age

from matplotlib.pyplot import MaxNLocator, FuncFormatter

plt.figure(figsize = (8, 4))

plot = sns.lineplot(data = ds_prof, x = 'age', y = 'salario', markers = False)

plot.xaxis.set_major_locator(plt.MaxNLocator(19))



#degree area for ds professionals

ds_prof["degree_area"].value_counts(ascending= False).plot(kind= "bar", figsize=(10,5), title="degree for ds professionals")



#degree are for ds professionals

not_ds_prof["degree_area"].value_counts(ascending= False).plot(kind= "bar", figsize=(10,5), title="degree for non ds professionals")

#degree area with higher salaries for ds professionals

ds_prof.groupby(["degree_area"])["salario"].mean().plot(kind= "bar", figsize=(10,5), title="degree area x salary for ds professionals")

not_ds_prof.groupby(["degree_area"])["salario"].mean().plot(kind= "bar", figsize=(10,5), title="degree area x salary for non ds professionals")

ds_prof["language"].value_counts().plot(kind= "bar", figsize=(10,5), title="language x salary")
#language x salary

ds_prof.groupby(["language"])["salario"].mean().plot(kind= "bar", figsize=(10,5), title="language x salary")
#time experience data science and salary for ds professionals

ds_prof.groupby(["time_experience_ds"])["salario"].mean().plot(kind= "bar", figsize=(10,5), title="time experience x salary for ds professionals")

ds_prof.groupby(["market_sector"])["salario"].mean().plot(kind= "bar", color="red", figsize=(10,5), title="degree for ds professionals")

#market sector that hires more ds professionals

ds_prof["market_sector"].value_counts().plot(kind= "bar", figsize=(10,5), title="degree for ds professionals")

#workers number 

ds_prof["workers_number"].value_counts().plot(kind= "bar", figsize=(10,5), title="number of workers")
#workers number x salary

ds_prof.groupby(["workers_number"])["salario"].mean().plot(kind= "bar", figsize=(10,5), title="company size x ds professionals salary")

ds_prof2= ds_prof.loc[:, ["('P0', 'id')", "age", "gender", "degree_level", "time_experience_ds",

               "time_experience_before", "degree_area", "salario"]]



#dealing with missing data

median = ds_prof2['age'].median()

ds_prof2['age'].fillna(median, inplace=True)







#drop missing degree area

ds_prof2= ds_prof2.dropna(subset=["degree_area"], how="any")







ds_prof2['salario'] = ds_prof2['salario'].fillna(df.groupby('degree_area')['salario'].transform('mean'))





#drop missing degree area

ds_prof2= ds_prof2.dropna(subset=["gender"], how="any")



# Importing the dataset

X= ds_prof2.loc[:, ["age", "gender", "degree_level", "time_experience_ds",

               "time_experience_before", "degree_area"]]



Id= ds_prof2.loc[:, "('P0', 'id')"]

y = ds_prof2.loc[:, "salario"]
# Encoding categorical data"

X = pd.get_dummies(X, columns=["gender", "degree_level", "time_experience_ds", "time_experience_before", "degree_area"],\

                          prefix=["gender", "degree", "exp_ds", "exp_bef", "area"], drop_first=True)



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Fitting Decision Tree Regression to the dataset

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X, y)



# Predicting a new result

y_pred = regressor.predict(X_test)
#Checking the accuracy of the model

accu_train= np.sum(regressor.predict(X_train)== y_train)/float(y_train.size)

accu_test= np.sum(regressor.predict(X_test)== y_test)/float(y_test.size)

print("classification accu on train", accu_train)

print("classification accu on train", accu_test)