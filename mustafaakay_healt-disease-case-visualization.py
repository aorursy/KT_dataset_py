# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
from collections import Counter 
plt.style.use("seaborn-whitegrid")
import plotly as py 
import plotly.graph_objs as go 
import plotly.figure_factory as ff 
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import missingno as msno

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
heart_case=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

heart_case.info()
heart_case.isnull().values.any()
# The data which whe obtained has been visualizations as follows
# Visualization of NaN values in data 
msno.bar(heart_case)
plt.show()
import pandas_profiling as pp
pp.ProfileReport(heart_case)

plt.figure(figsize=(11,11))
sns.heatmap(heart_case.corr(),annot=True, fmt='.1f', cmap="coolwarm")
plt.show()
                        # CATEGORICAL VARIABLE
categorical_features=["sex","cp","restecg","exang","slope","thal","target","ca"]
def visualization_bar(features):
    """
    Visualization Categorical Features with Bar Plot. 
    Will be used categorical_features 
    """
    seri=heart_case[features]
    unique_variable=seri.value_counts().index
    variable_counts=seri.value_counts()
    
    plt.figure(figsize=(7,7))
    sns.barplot(unique_variable,variable_counts)
    plt.xticks(rotation=45)
    plt.title("VISUALİATİON: *{} FEATURE".format(features.upper()))
    plt.xlabel("***{}***".format(unique_variable))
    plt.ylabel("{} 's Counts".format(features))
    plt.show()
    
    print("{}".format(variable_counts))
    
    
    for i in unique_variable:
        filtre=heart_case[heart_case[features]==i]
        mean=len(filtre)/len(seri)
        print("{}: {}".format(i,round(mean,2)))
    
    
for i in categorical_features:
    visualization_bar(i)
    
               #      VISUALIZATION WITH PIE PLOTS FOR CATEGORICAL VARIABLE
for i in categorical_features:
    unique_value=heart_case[i].value_counts().index
    unique_label=list(heart_case[i].unique())
    unique_label_index=[unique_label.index(x) for x in unique_label ]
    size=[len(heart_case[heart_case[i]==x]) for x in unique_value]
    labels=[str(unique_value[x]) for x in unique_label_index]
    explode=[0 for x in unique_label_index]
    
    f,ax=plt.subplots(figsize=(9,9))
    ax.pie(size,labels=labels,explode=explode, autopct="%1.1f%%",shadow=True)
    ax.axis("equal")
    plt.title("{}".format(i.upper()))
    plt.show()
    
heart_case_male=heart_case[heart_case.sex==1]
heart_case_female=heart_case[heart_case.sex==0]
                    # DISTRIBUTION OF CATEGORICAL DATA ACCORDING TO FEMALE DATA
categorical_features=["cp","restecg","exang","slope","thal","target","ca"]
for i in categorical_features:
    seri=heart_case_female[i]
    unique_value=seri.unique()
    print("Mean of Female's Data: {} ".format(i) )
    mean_list=[]
    for a in unique_value:
        mean=round(len(heart_case_female[seri==a])/len(seri),2)
        mean_list.append(mean)
        print("{}: {}".format(a,mean,))
        
    sns.barplot(unique_value,mean_list)
    plt.title("{} Bar Plot".format(i.upper()))
    plt.show()
                        # DISTRIBUTION OF CATEGORICAL DATA ACCORDING TO MALE DATA
categorical_features=["cp","restecg","exang","slope","thal","target","ca"]
for i in categorical_features:
    seri=heart_case_male[i]
    unique_value=seri.unique()
    print("Mean of Male's Data: {}".format(i) )
    mean_list=[]
    for a in unique_value:
        mean=round(len(heart_case_male[seri==a])/len(seri),2)
        mean_list.append(mean)
        print("{}: {}".format(a,mean,))
        
    sns.barplot(unique_value,mean_list)
    plt.title("{} Bar Plot".format(i.upper()))
    plt.show()
            # FOR FEMALE
numeriacal_features=["age", "trestbps", "chol", "thalach", "oldpeak"]
for i in numeriacal_features:
    plt.figure(figsize=(20,6))
    plt.subplot(2,2,1)
    sns.swarmplot(data=heart_case_female, x="cp",y=i,hue="target",size=10)
    plt.title("{}-CP".format(i.upper()))
    plt.show()
    
    plt.figure(figsize=(20,6))
    plt.subplot(2,2,2)
    sns.countplot(x=i,data=heart_case_female,hue="target", palette="GnBu", linewidth=3)
    plt.legend(loc="upper right")
    plt.title("{}'s HISTOGRAM".format(i.upper()))
    plt.show()
    
    plt.figure(figsize=(20,6))
    plt.subplot(2,2,3)
    sns.countplot(x=i,data=heart_case,hue="target", palette="GnBu", linewidth=3)
    plt.legend(loc="upper right")
    plt.title("{}'s HISTOGRAM".format(i.upper()))
    plt.show()
    
    mean_all_data=heart_case[i].mean()
    mean=heart_case_female[i].mean()
    mean_1=heart_case_female[i][heart_case_female["target"]==1].mean()
    mean_0=heart_case_female[i][heart_case_female["target"]==0].mean()
    print("Mean of {}: {}".format(i.upper(),round(mean,2)))
    print("All Data Mean of {}: {}".format(i.upper(),round(mean_all_data,2)))
    print("Mean of {}: {} for target(1)".format(i.upper(),round(mean_1,2)))
    print("Mean of {}: {} for target(0)".format(i.upper(),round(mean_0,2)))
    
            # FOR MALE
numeriacal_features=["age", "trestbps", "chol", "thalach", "oldpeak"]
for i in numeriacal_features:
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    sns.swarmplot(data=heart_case_male, x="cp",y=i,hue="target",size=10)
    plt.title("{}-CP".format(i.upper()))
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,2)
    sns.countplot(x=i,data=heart_case_male,hue="target", palette="GnBu", linewidth=3)
    plt.legend(loc="upper right")
    plt.title("{}'s HISTOGRAM".format(i.upper()))
    plt.show()
    
    plt.figure(figsize=(20,6))
    plt.subplot(2,2,3)
    sns.countplot(x=i,data=heart_case,hue="target", palette="GnBu", linewidth=3)
    plt.legend(loc="upper right")
    plt.title("{}'s HISTOGRAM".format(i.upper()))
    plt.show()
    
    mean_all_data=heart_case[i].mean()
    mean=heart_case_male[i].mean()
    mean_1=heart_case_male[i][heart_case_male["target"]==1].mean()
    mean_0=heart_case_male[i][heart_case_male["target"]==0].mean()
    print("Mean of {}: {}".format(i.upper(),round(mean,2)))
    print("All Data Mean of {}: {}".format(i.upper(),round(mean_all_data,2)))
    print("Mean of {}: {} for target(1)".format(i.upper(),round(mean_1,2)))
    print("Mean of {}: {} for target(0)".format(i.upper(),round(mean_0,2)))
                                # NUMERICAL VARIABLE
    
numeriacal_features=["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
for i in numeriacal_features:
    plt.figure(figsize=(11,15))
    
    plt.subplot(2,2,1)
    plt.hist(heart_case[i],color="r")
    plt.title("HISTOGRAM {}".format(i.upper()))
    plt.xlabel("{}".format(i.upper()))
    
    plt.subplot(2,2,2)
    plt.scatter(heart_case[heart_case.sex==0].age, heart_case[heart_case.sex==0][i])
    plt.title("Comparasion Between Famale Age vs {}".format(i.capitalize()))
    plt.xlabel("Age")
    plt.ylabel(i)
    
    plt.subplot(2,2,3)
    plt.scatter(heart_case[heart_case.sex==1].age, heart_case[heart_case.sex==1][i])
    plt.title("Comparasion Between Male Age vs {}".format(i.capitalize()))
    plt.xlabel("Age")
    plt.ylabel(i)
    
    plt.subplot(2,2,4)
    sns.swarmplot(data=heart_case, x="cp",y=i, hue="target")
    plt.show()
# EXAMINE SEX/ NUMERIC DATAS WITH GROUP BY 
group_by_numerical=heart_case.groupby("sex")["age", "trestbps", "chol", "thalach", "oldpeak", "ca"].mean().sort_values(by="age",ascending=False)
group_by_numerical.index=["male","female"]
group_by_numerical.index.name="sex"
round(group_by_numerical,2)
# EXAMINE SEX/ NUMERIC DATAS WITH GROUP BY
group_by_numerical=heart_case.groupby("sex")["age", "trestbps", "chol", "thalach", "oldpeak", "ca"].max().sort_values(by="age",ascending=False)
group_by_numerical.index=["male","female"]
group_by_numerical.index.name="sex"
round(group_by_numerical,2)
# EXAMINE SEX/ NUMERIC DATA WITH GROUP BY
group_by_numerical=heart_case.groupby("sex")["age", "trestbps", "chol", "thalach", "oldpeak", "ca"].min().sort_values(by="age",ascending=False)
group_by_numerical.index=["male","female"]
group_by_numerical.index.name="sex"
round(group_by_numerical,2)
                        #  BOX PLOT FOR NUMERICAL VARIABLE
    
numeriacal_features=["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
for i in numeriacal_features:
    heart_case.boxplot(column=i, by="sex")
    plt.show()

numeriacal_features=["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
for i in numeriacal_features:
    heart_case.boxplot(column=i, by="cp")
    plt.show()

variable=["age","trestbps","chol","thalach","oldpeak"]
def outlier_detect(df, features):
    """
    Herhangi bir data içerisinde bulunan outlier ların testpit edilmesini sağlar.
    """
    outlier_index=[]
    for i in features:
        Q1=np.percentile(df[i],25)
        Q3=np.percentile(df[i],75)
        IQR=Q3-Q1 
        outlier_step=IQR*1.5
        filter_data=df[(df[i]<Q1-outlier_step)|(df[i]>Q3+outlier_step)]
        filter_data_index=filter_data.index
        outlier_index.extend(filter_data_index)
    
    outlier_index_count=Counter(outlier_index)
    multi_index=[i for i, v in outlier_index_count.items() if v >2 ]
    return multi_index   
outlier_detect(heart_case,["age","trestbps","chol","thalach","oldpeak"])
Age= heart_case["age"]
Trestbps=heart_case["trestbps"]/heart_case["trestbps"].max()
Chol=heart_case["chol"]/heart_case["chol"].max()
Thalach=heart_case["thalach"]/heart_case["thalach"].max()
Oldpeak=heart_case["oldpeak"]/heart_case["oldpeak"].max()
Age_Trestbps=pd.concat([Age,Trestbps,Chol,Thalach,Oldpeak],axis=1)
plt.figure(figsize=(15,11))
sns.pointplot(data=Age_Trestbps,x="age",y="trestbps")
sns.pointplot(data=Age_Trestbps, x="age",y="chol",color="r")
sns.pointplot(data=Age_Trestbps,x="age",y="thalach",color="g")
sns.pointplot(data=Age_Trestbps,x="age",y="oldpeak",color="cyan")
plt.text(45,.8, "*thalach",color="g",style="italic",size=20)
plt.text(45,.75, "*trestbps",color="b",style="italic",size=20)
plt.text(45,.7, "*chol",color="r",style="italic",size=20)
plt.text(45,.65, "*oldpeak",color="cyan",style="italic",size=20)
plt.show()