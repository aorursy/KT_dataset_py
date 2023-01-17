import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/heart-disease/heart.csv")

df.head()
df.size
df.shape
df.describe(include="all")
df["age"]
df.columns
df.info()
# you also get features in describe function



df.describe(include="all")
#Change the sex(0,1)=(female,male)

SEX={0:"female",1:"male"}

df.sex.replace(SEX,inplace=True)

df.head()
df.sex.value_counts()
sns.distplot(df.age,hist=False,kde=True,kde_kws={"shade":True})

plt.title("KDE plot for Age using Seaborn")

plt.ylabel("Density")

plt.show()
df.age.plot(kind="kde")

plt.title("KDE plot for Age using Matplotlib")

plt.xlabel("Age")

plt.show()
plt.hist(df.chol,bins=5,rwidth=0.7)

plt.grid(linestyle="--")

plt.xlabel("Cholestrol")

plt.ylabel("Count of people")

plt.xticks([50,100,150,200,250,300,350,400,450,500])

plt.title("Histogram for showing cholestrol with Number of bins 5 using Matplotlib")

plt.show()
sns.distplot(df.chol,hist=True,kde=False,bins=5,hist_kws={"rwidth":0.7},color="blue")

plt.xlabel("Cholestrol")

plt.ylabel("Count of people")

plt.title("Histogram for showing cholestrol with Number of bins 5 using Seaborn")

plt.grid(linestyle="--")

plt.show()
plt.boxplot(df.trestbps)

plt.grid(linestyle="--")

#plt.xlabel("Cholestrol")

plt.ylabel("Resting blood pressure (in mm Hg)")

plt.title("Boxplot for showing trestbps using Matplotlib")

plt.show()
# Dark Spots indicate the outliers
sns.boxplot(df.trestbps,orient="v")

plt.grid(linestyle="--")

#plt.xlabel("Cholestrol")

plt.ylabel("Resting blood pressure (in mm Hg)")

plt.title("Boxplot for showing trestbps using Seaborn")

plt.show()
df.head()


#draw a bar plot of target by sex

sns.barplot(df.sex,df.target)

plt.grid(linestyle="--")

plt.xlabel("Gender")

plt.ylabel("Target")

plt.title("Bar plot for showing Gender and target using Seaborn")

plt.show()



#print percentages of females vs. males Heart Disease

male_with_heart_disease=df[(df.sex=="male") & (df.target==1)].sex.count()

female_with_heart_disease=df[(df.sex=="female") & (df.target==1)].sex.count()

total_male_count=df[(df.sex=="male")].sex.count()

total_female_count=df[(df.sex=="female")].sex.count()



male_with_heart_disease_percentage=male_with_heart_disease/total_male_count*100

female_with_heart_disease_percentage=female_with_heart_disease/total_female_count*100



print("Percentages of females vs. males Heart Disease is:\n","Female:",female_with_heart_disease_percentage,"\n"

     "Male:",male_with_heart_disease_percentage)
#create a subplot

sns.catplot(x="sex",y="target",data=df,kind="bar",col="sex")

plt.show()
# create bar plot using groupby

df.groupby('sex').target.value_counts().plot.bar()

plt.show()
# create count plot



sns.countplot(x="sex",data=df,hue="target")

plt.show()
sns.countplot(x="target",data=df,hue="sex")

plt.show()
# create subplot plot

sns.catplot(x="cp",y="target",data=df,kind="bar",col="cp")

plt.show()
# create bar plot using groupby

df.groupby('cp').target.value_counts().plot.bar()

plt.show()
# create count plot

sns.countplot(x="cp",data=df,hue="target")

plt.show()
sns.countplot(x="target",data=df,hue="cp")

plt.show()
# create subplot plot

sns.catplot(x="fbs",y="age",data=df,kind="violin",col="target")

plt.show()
# create violinplot plot using groupby



sns.violinplot(x="fbs",y="age",data=df,hue="target")

plt.show()
# create boxplot

sns.boxplot(x="sex",y="age",data=df)

plt.show()
#create crosstab

a=pd.crosstab(df.target,df.sex)

a

sns.scatterplot(x="age",y="oldpeak",data=df)

plt.show()