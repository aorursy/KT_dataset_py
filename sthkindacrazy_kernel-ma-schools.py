%pylab inline

%matplotlib inline

#visualization result



import pandas as pd

import seaborn as sns
train = pd.read_csv("../input/MA_Public_Schools_2017.csv")

print(train.shape)

train.head()
sns.countplot(data=train, x="School Type")
#increased number of level means more assistance is required

sns.countplot(data=train, x="Accountability and Assistance Level")
sns.countplot(data=train, x="District_Accountability and Assistance Level")
# %of high needs and %of economically disadvantaged shows similar distribution

sns.distplot(train["% High Needs"], hist=False, label="Function")

sns.distplot(train["% Economically Disadvantaged"], hist=False, label="disadvantaged")
#Simply thought about relationship between average salaray and percentage of economically disadvantaged. 

sns.lmplot(data=train, x="Average Salary", y="% Economically Disadvantaged", fit_reg=False, hue="Accountability and Assistance Level")

# not quite clear. 
#District Names are too many not nice to be the factors for devision. 

sns.countplot(data=train, x="District Name")
#Here let's see plot between two factors, % economically disadvantaged and ap scores

#Limit data sets for only high schools. AP tests are usually for high school students. 

highschool = train[train["9_Enrollment"] > 0]



sns.lmplot(data=highschool, x="% Economically Disadvantaged", y="% AP_Score 3-5", fit_reg=False, hue="Accountability and Assistance Level")

sns.lmplot(data=highschool, x="% Economically Disadvantaged", y="AP_Score=5", fit_reg=False, hue="Accountability and Assistance Level")
sns.lmplot(data=highschool, x="% Economically Disadvantaged", y="SAT_Tests Taken", fit_reg=False, hue="Accountability and Assistance Level")

sns.lmplot(data=highschool, x="% Economically Disadvantaged", y="Average SAT_Reading", fit_reg=False, hue="Accountability and Assistance Level")

#Graduation Rates seems drastic change between 40%-60% of Economically Disadvantaged slot. 



sns.lmplot(data=highschool, x="% Economically Disadvantaged", y="% Graduated", fit_reg=False, hue="Accountability and Assistance Level")
#Then how about college, especially private four-year which is expected to have high tuition. 

sns.lmplot(data=highschool, x="% Economically Disadvantaged", y="% Private Four-Year", fit_reg=False, hue="Accountability and Assistance Level")
