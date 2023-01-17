# Before Having a deep dive into the Data, Importing Modules





import numpy as np

import pandas as pd

import warnings

import seaborn as sns

import matplotlib.pyplot as plt



warnings.filterwarnings('ignore')
from subprocess import check_output



print(check_output(["ls", "../input/world-happiness"]).decode("utf8"))
df2015 = pd.read_csv("../input/world-happiness/2015.csv")

df2016 = pd.read_csv("../input/world-happiness/2016.csv")

df2017 = pd.read_csv("../input/world-happiness/2017.csv")
df2015.info()
df2016.info()
df2017.info()
print(df2015.columns, df2016.columns, df2017.columns, sep = " \n" )
#'Standard Error' column has been deleted from 2015 data. 





df2015 = df2015.drop('Standard Error', axis=1)
# New column names have been amended to the data set. 



df2015.columns = ['Country', 'Region', 'Happiness_Rank', 'Happiness_Score', 'Economy_GDP_per_Capita', 'Family',

       'Health_Life_Expectancy', 'Freedom', 'Trust_Government_Corruption','Generosity', 'Dystopia_Residual']
df2015["Year"] = 2015
df2015.head()
#'Upper Confidence Interval' and 'Lower Confidence Interval' column have been deleted from 2016 data set. 





df2016 = df2016.drop(['Upper Confidence Interval','Lower Confidence Interval'], axis =1 )
# New column names have been amended to the data set. 





df2016.columns = ['Country', 'Region', 'Happiness_Rank', 'Happiness_Score', 'Economy_GDP_per_Capita', 'Family',

       'Health_Life_Expectancy', 'Freedom', 'Trust_Government_Corruption','Generosity', 'Dystopia_Residual']
df2016["Year"] = 2016
df2016.head()
#'Whisker.high' and 'Whisker.low' column have been deleted from 2017 data set. 





df2017 =  df2017.drop(['Whisker.high','Whisker.low'],axis =1 )
# New column names have been amended to the data set. 





df2017.columns = ['Country', 'Happiness_Rank', 'Happiness_Score', 'Economy_GDP_per_Capita', 'Family',

       'Health_Life_Expectancy', 'Freedom', 'Generosity', 'Trust_Government_Corruption', 'Dystopia_Residual']
df2017["Year"] = 2017
df2017.head()
print(df2015.columns, df2016.columns, df2017.columns, sep = " \n" )
frames = [df2015, df2016, df2017]



Hapiness_report = pd.concat(frames,sort=True,ignore_index=True)

Hapiness_report.head()
sns.heatmap(Hapiness_report.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
Sum = Hapiness_report.isnull().sum()

Percentage = ( Hapiness_report.isnull().sum()/Hapiness_report.isnull().count() )



pd.concat([Sum,Percentage], axis =1, keys= ['Sum', 'Percentage'])

Happiness_report2 = Hapiness_report.copy()
try:

    for country in Happiness_report2.Country.unique():

        Happiness_report2.loc[Happiness_report2['Country']==str(country),

                              'Region']=Happiness_report2[Happiness_report2['Country']==str(country)].Region.mode()[0]

except IndexError:

    pass
Happiness_report2.info()
#Lets have a look at the countries which do not have region information in the data set.





Happiness_report2[Happiness_report2['Region'].isna()]
#Lets check if China exsists in the previous rows. 



Happiness_report2[Happiness_report2.Country == "China"] 
Happiness_report2.loc[[347,385], 'Region'] = "Eastern Asia"
Happiness_report2.loc[385].Region
sns.heatmap(Happiness_report2.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
sns.pairplot(Happiness_report2)
from scipy.stats import norm 





plt.figure(figsize = (20,10)) 



plt.subplot(2,5,1)

sns.distplot(Happiness_report2["Family"],fit=norm) 

plt.title("Family")



plt.subplot(2,5,2)

sns.distplot(Happiness_report2["Dystopia_Residual"],fit=norm)

plt.title("Dystopia_Residual")



plt.subplot(2,5,3)

sns.distplot(Happiness_report2["Economy_GDP_per_Capita"], fit=norm)

plt.title("Economy_GDP_per_Capita")



plt.subplot(2,5,4)

sns.distplot(Happiness_report2["Freedom"], fit=norm)

plt.title("Freedom")



plt.subplot(2,5,5)

sns.distplot(Happiness_report2["Generosity"],fit=norm)

plt.title("Generosity")



plt.subplot(2,5,6)

sns.distplot(Happiness_report2["Happiness_Rank"], fit=norm)

plt.title("Happiness_Rank")



plt.subplot(2,5,7)

sns.distplot(Happiness_report2["Happiness_Score"], fit=norm)

plt.title("Happiness_Score")



plt.subplot(2,5,8)

sns.distplot(Happiness_report2["Health_Life_Expectancy"], fit=norm)

plt.title("Health_Life_Expectancy")



plt.subplot(2,5,9)

sns.distplot(Happiness_report2["Trust_Government_Corruption"], fit=norm)

plt.title("Trust_Government_Corruption")







plt.show()
#In order to have a deep understanding of those variables, we can remove fit norm factor by using 'kde' function. 



sns.distplot(Happiness_report2["Trust_Government_Corruption"], kde=False)

plt.ylim(0,100)

plt.title("Trust_Government_Corruption")
sns.distplot(Happiness_report2["Generosity"], kde=False)

plt.ylim(0,90)

plt.title("Generosity")
sns.distplot(Happiness_report2["Health_Life_Expectancy"], kde=False)

plt.ylim(0,100)

plt.title("Health_Life_Expectancy.")
sns.distplot(Happiness_report2["Happiness_Score"], kde=False)

plt.ylim(0,100)

plt.title("Happiness_Score")

plt.show()
mean_by_year = Happiness_report2.groupby(by="Year").mean()["Happiness_Score"] 

print(mean_by_year[2015])

print(mean_by_year[2016])

print(mean_by_year[2017])
import scipy.stats as stats

from scipy.stats.mstats import winsorize

from statsmodels.stats.weightstats import ttest_ind
Happiness_Score_2015 = Happiness_report2[Happiness_report2["Year"] == 2015].Happiness_Score

Happiness_Score_2016 = Happiness_report2[Happiness_report2["Year"] == 2016].Happiness_Score 

Happiness_Score_2017 = Happiness_report2[Happiness_report2["Year"] == 2017].Happiness_Score 

stats.ttest_ind(Happiness_Score_2015, Happiness_Score_2016)

stats.ttest_ind(Happiness_Score_2016, Happiness_Score_2017)

plt.figure(figsize = (8,6))



objects = ('2015','2016','2017')

y_pos = np.arange(len(objects)) # y_pos kac tane object varsa o kdrlik bir array olusturuyor. Bar plot altina isimlerini yazar

performance =[mean_by_year[2015], mean_by_year[2016], mean_by_year[2017]]

 

plt.bar(y_pos, performance, align='center', alpha=0.6)

plt.yticks(size=15)

plt.xticks(y_pos, objects,size=15)

plt.xlabel('Year',size=15)

plt.ylabel('Happiness Score',size=15)

plt.title('Happiness Score by Years', fontsize=15)



plt.ylim(5.30,5.40)



plt.show()
mean_by_year_and_region = Happiness_report2.groupby(by=["Region", "Year"]).mean()["Happiness_Score"]
mean_by_year_and_region=mean_by_year_and_region.reset_index()
mean_by_year_and_region.head()
plt.figure(figsize=(10,8))

sns.barplot(x="Region", y="Happiness_Score", hue="Year", data=mean_by_year_and_region)

plt.xticks(rotation=90)

plt.ylim((0,10))
years_Turkey = Happiness_report2[Happiness_report2.Country == 'Turkey']['Year']
Happiness_Score_Turkey = Happiness_report2[Happiness_report2.Country == 'Turkey']['Happiness_Score']
Dystopia_Residual_Turkey = Happiness_report2[Happiness_report2.Country == 'Turkey']['Dystopia_Residual']
plt.figure(1, figsize = (8,8))

plt.plot(years_Turkey, Happiness_Score_Turkey, label = 'Year/Happiness_Score', color='blue', linewidth=5)

plt.plot(years_Turkey,Dystopia_Residual_Turkey, label = 'Year/Dystopia_Residual',color='red', linewidth=5)

plt.xlabel('Years')

plt.ylabel('scores ')

plt.xlim([2015, 2017])

plt.title('Total Happiness Score&Dystopia_Residual in Turkey')

plt.legend()

plt.show()
Happiness_report2["Economy_GDP_per_Capita"].head()
plt.figure(figsize=(15,8))

sns.scatterplot(x='Happiness_Score', y='Health_Life_Expectancy', hue='Region',data=Happiness_report2, s = Happiness_report2.Economy_GDP_per_Capita*100);

plt.xlabel('Happiness_Score',size=15)

plt.ylabel('Health_Life_Expectancy', size =10)
plt.figure(figsize=(15,8))

sns.scatterplot(x='Happiness_Score', y='Economy_GDP_per_Capita', hue='Region',data=Happiness_report2, s = Happiness_report2.Economy_GDP_per_Capita*100);

plt.xlabel('Happiness_Score',size=15)

plt.ylabel('Economy_GDP_per_Capita', size =10)
plt.figure(figsize=(15,8))

sns.scatterplot(x='Happiness_Score', y='Family', hue='Region',data=Happiness_report2, s = Happiness_report2.Economy_GDP_per_Capita*100);

plt.xlabel('Happiness_Score',size=15)

plt.ylabel('Family', size =10)
plt.figure(figsize=(15,8))

sns.scatterplot(x='Happiness_Score', y='Trust_Government_Corruption', hue='Region',data=Happiness_report2, s = Happiness_report2.Economy_GDP_per_Capita*100);

plt.xlabel('Happiness_Score',size=15)

plt.ylabel('Trust_Government_Corruption', size =10)
# As we see in previous results, 3 variables have a close relation with each other. Lets have a further reseach on those ones.  





fig, axes = plt.subplots(1,3,figsize=(20,5))

baslik_font = {'family': 'arial', 'color': 'darkred','weight': 'bold','size': 13 }



happiness_score_by_three_variables = ['Economy_GDP_per_Capita','Trust_Government_Corruption', 'Health_Life_Expectancy'] 

 

for i in range(0,3):

    

    plt.subplot(1, 3, i+1)

    plt.scatter(Happiness_report2['Happiness_Score'],Happiness_report2[happiness_score_by_three_variables[i]],c='purple', s=80)

    plt.title('Happiness Score and '+ str(happiness_score_by_three_variables[i]), fontdict=baslik_font, fontsize=13, y=1.08)

    plt.xlabel('Happiness Score',size=15)

    plt.ylabel(str(happiness_score_by_three_variables[i]),size=15)
Corr_Matrix = Happiness_report2.corr()

Corr_Matrix
Corr_Matrix.Happiness_Score.sort_values()
plt.figure(figsize=(10,10))

sns.heatmap(Corr_Matrix, cmap='bwr')

plt.title('Correlation Matrix')
graph_by_eight_variables = ['Dystopia_Residual', 'Economy_GDP_per_Capita', 'Family', 'Freedom',

               'Generosity', 'Happiness_Score', 'Health_Life_Expectancy', 'Trust_Government_Corruption'] 

plt.figure(figsize=(15,8))



for i in range(0,8):

    plt.subplot(2, 4, i+1)

    plt.boxplot(Happiness_report2[graph_by_eight_variables[i]])

    plt.title(graph_by_eight_variables[i])

    

    

    
from scipy.stats.mstats import winsorize



Happiness_report2["winsorize_Dystopia_Residual"] = winsorize(Happiness_report2["Dystopia_Residual"], (0, 0.05))

Happiness_report2["winsorize_Family"] = winsorize(Happiness_report2["Family"], (0, 0.02))

Happiness_report2["winsorize_Generosity"] = winsorize(Happiness_report2["Generosity"], (0, 0.05))

Happiness_report2["winsorize_Trust_Government_Corruption"] = winsorize(Happiness_report2["Trust_Government_Corruption"], (0, 0.03))
from sklearn.preprocessing import normalize



Happiness_report2["winsorize_Dystopia_Residual"] = normalize(np.array(Happiness_report2["winsorize_Dystopia_Residual"]).reshape(1,-1)).reshape(-1,1)

Happiness_report2["winsorize_Family"] =  normalize(np.array(Happiness_report2["winsorize_Family"]).reshape(1,-1)).reshape(-1,1)

Happiness_report2["winsorize_Generosity"] =  normalize(np.array(Happiness_report2["winsorize_Generosity"]).reshape(1,-1)).reshape(-1,1)

Happiness_report2["winsorize_Trust_Government_Corruption"] =  normalize(np.array(Happiness_report2["winsorize_Trust_Government_Corruption"]).reshape(1,-1)).reshape(-1,1)



df_first_data= Happiness_report2[['Dystopia_Residual','Economy_GDP_per_Capita',

       'Family','Freedom', 'Generosity','Happiness_Score',

       'Health_Life_Expectancy', 'Trust_Government_Corruption']]
df_winsorize = Happiness_report2[['winsorize_Dystopia_Residual',

                                  'winsorize_Family','winsorize_Generosity','winsorize_Trust_Government_Corruption']]
from sklearn.decomposition import PCA 

pca = PCA(n_components=2)

pc = pca.fit_transform(df_first_data)

print (pca.explained_variance_ratio_)
from sklearn.decomposition import PCA 

pca = PCA(n_components=2)

pc = pca.fit_transform(df_winsorize)

print (pca.explained_variance_ratio_)