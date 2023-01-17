# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.




#importing the titanic csv file





#Print you can execute arbitrary python code

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



print("\n\nTop of the training data:")

print(train.head())



print("\n\nSummary statistics of training data")

print(train.describe())
train.to_csv('copy_of_the_training_data.csv', index=False)





train['Age'] = train['Age'].fillna(train['Age'].median())



print(train.describe())

print(train)





new_col = ['PassengerId','Survived']

collist = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']



new_train = train[collist]

pass_sur = train[new_col]

print(new_train.describe())

print(new_train.head(3))





new_train.loc[new_train["Sex"] == "male", "Sex"] = 0

new_train.loc[new_train["Sex"]=="female","Sex"]=1



print("After cleaning the data .....................")

print(new_train["Sex"].unique())

print(new_train.head(3))





print("After converting the Embarked Data **********")

new_train.loc[new_train["Embarked"] == "S", "Embarked"] = 0

new_train.loc[new_train["Embarked"] == "C", "Embarked"] = 1



new_train.loc[new_train["Embarked"] == "Q", "Embarked"] = 2

print(new_train.head(3))









print(pass_sur)



pass_sur=pass_sur[pass_sur['Survived']!=0][['PassengerId','Survived']]



        

        



print(pass_sur)



pass_sur['Survived'].dropna()



#Converting to CSV File..........



pass_sur.to_csv('Copy_of_pass_sur.csv',index=False)
#importing the library for plotting 



import matplotlib.pyplot as plt

import seaborn as sns





#Graph between no of Ages wrt gender

plt.scatter(new_train["Age"],new_train["Sex"])

#labeling the axes

plt.xlabel("Ages")

plt.ylabel("Sex")

plt.title("Ages with relative Gender")

plt.text(40,0.6,"0 for Male,1 for Female")

plt.show()
#histogram of the total embarked

plt.hist(new_train["Embarked"])

plt.xlabel("Emberked")

plt.ylabel("No of Persons")

plt.title("Total Embarked Persons")

plt.text(1,600,"S=0,C=1,Q=2")

plt.show()
sns.distplot(new_train["Embarked"])

plt.show()
#histogram of the gender in the titanic



plt.hist(new_train["Sex"])

#labeling the axes

plt.xlabel("Male and Female")

plt.ylabel("No of counts")

plt.title("No of male and female")

plt.text(.6,500,"0 for male and 1 for female")

plt.show()
#histogram of the people wrt total no of people

plt.hist(new_train["Age"],bins=80)

plt.xlabel("Ages")

plt.ylabel("No of persons")

plt.title("No of persons with their relative ages")

plt.xlim(0,80,5)

plt.ylim(0,250)

plt.show()
#histogram of the survived and not survived people

plt.hist(new_train["Survived"])

plt.title("passenger Survived")

plt.ylabel("No of passenger")

plt.xlabel("passenger who survived")

plt.text(0.6,450,"1 for survived and 0 for not")

plt.show()