# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#ouvre le fichier adult en mode lecture ('r'=read)
file = open('/kaggle/input/datalab3/files/adult.data', 'r')

def chr_int(a):
    if a.isdigit():
        return int(a)
    else:
        return 0
                
data=[]
for line in file:
     data1=line.split(', ')
     if len(data1)==15:
        data.append([chr_int(data1[0]),data1[1],chr_int(data1[2]),data1[3],chr_int(data1[4]),data1[5],data1[6],\
            data1[7],data1[8],data1[9],chr_int(data1[10]),chr_int(data1[11]),chr_int(data1[12]),data1[13],\
            data1[14]])
        
print (data[1:2])

#1.What is the obtained result? What did you ask for in the previous command? Explain.
# On obtien : [[50, 'Self-emp-not-inc', 83311, 'Bachelors', 13, 'Married-civ-spouse', 'Exec-managerial', 'Husband', 'White', 'Male', 0, 0, 13, 'United-States', '<=50K\n']]
# En effet, on a demandé d'afficher la 1ere ligne du tableau data qu'on a créé précedement.

#On met les données du fichier dans une structure DataFrame : 
df = pd.DataFrame(data) #  Two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes 

df.columns = ['age', 'type_employer', 'fnlwgt', 'education', 
                "education_num","marital", "occupation", "relationship", "race","sex",
                "capital_gain", "capital_loss", "hr_per_week","country","income"]

df.head()

#2.Describe an explain the result.
# On a défini comment on voulait structurer le DataFrame créé à partir du tableau data (les colonnes particulièrement). df.head() retourne tous les éléments (de toutes les colonnes) des 5 premières lignes du DataFrame df. 

df.tail()

#3.Describe and explain the result. Compare with the previous one
# Retourne les éléments des 5 dernières lignes du dataframe, pour toutes les lignes (de la 32556 à la 32560)

df.shape

#4. Describe an explain the result.
# (32561, 15) Il y a 32561 lignes et 15 colonnes dans le dataframe df. 

counts = df.groupby('country').size()

print (counts)

#5. How many items are there for USA? and for Mexico?
# Il y en a 29170 pour les USA et 643 pour Mexico. 

counts = df.groupby('age').size() # grouping by age
print (counts)

#6. What is the age of the most represented people?
# 20 ans, mais cela est valable que pour les 5 premiers et 5 derniers ages définis. Je n'ai pas su visualiser les ages entre 22 et 84 ans.
       
ml = df[(df.sex == 'Male')] # grouping by sex
ml.shape
ml1 = df[(df.sex == 'Male')&(df.income=='>50K\n')]
ml1.shape
       
fm =df[(df.sex == 'Female')]
fm.shape
       
fm1 =df[(df.sex == 'Female')&(df.income=='>50K\n')]
fm1.shape

#4.2 SUMMARIZE THE DATA
       
df1=df[(df.income=='>50K\n')]

print ('The rate of people with high income is: ', int(len(df1)/float(len(df))*100), '%.' )
print ('The rate of men with high income is: ', int(len(ml1)/float(len(ml))*100), '%.' )
print ('The rate of women with high income is: ', int(len(fm1)/float(len(fm))*100), '%.' )

#7. Describe an explain the result.
# The rate of people with high income is:  24 %.
#The rate of men with high income is:  30 %.
#The rate of women with high income is:  10 %. Pour obtenir ces taux, on a defini des fonctions lambda (vu dans exo1) qui utilisent les valeurs de toutes les colonnes utilies des dataframe et des sous-dataframe qu'on a créé.

print ('The average age of men is: ', ml['age'].mean(), '.' )
print ('The average age of women is: ', fm['age'].mean(), '.')
print ('The average age of high-income men is: ', ml1['age'].mean(), '.' )
print ('The average age of high-income women is: ', fm1['age'].mean(), '.')

#8. Describe an explain the result.
# Cela retourne : The average age of men is:  39.43354749885268 .
#The average age of women is:  36.85823043357163 .
#The average age of high-income men is:  44.62578805163614 .
#The average age of high-income women is:  42.125530110262936 . On a utilisé la fonction mean() qui fait la moyenne des ages des femmes en utilisant les valeurs d'ages de la categorie des femmes, de meme pour les hommes et les ages moyens de high income des 2 sexes.

ml_mu = ml['age'].mean()
fm_mu = fm['age'].mean()
ml_var = ml['age'].var()
fm_var = fm['age'].var()
ml_std = ml['age'].std()
fm_std = fm['age'].std()

print ('Statistics of age for men: mu:', ml_mu, 'var:', ml_var, 'std:', ml_std)
print ('Statistics of age for women: mu:', fm_mu, 'var:', fm_var, 'std:', fm_std)

ml_mu_hr = ml['hr_per_week'].mean()
fm_mu_hr = fm['hr_per_week'].mean()
ml_var_hr = ml['hr_per_week'].var()
fm_var_hr = fm['hr_per_week'].var()
ml_std_hr = ml['hr_per_week'].std()
fm_std_hr = fm['hr_per_week'].std()

print ('Statistics of hours per week for men: mu:', ml_mu_hr, 'var:', ml_var_hr, 'std:', ml_std_hr)
print ('Statistics of hours per week for women: mu:', fm_mu_hr, 'var:', fm_var_hr, 'std:', fm_std_hr)

#9. Describe an explain the result.
# Affiche la moyenne, variance et ecart-type de l'age des hommes, des femmes et des heures de travail par semaines pour les hommes et les femmes.

ml_median= ml['age'].median()
fm_median= fm['age'].median()

print ("Median age per men and women: ", ml_median, fm_median)

ml_median_age= ml1['age'].median()
fm_median_age= fm1['age'].median()

print ("Median age per men and women with high-income: ", ml_median_age, fm_median_age)

ml_median_hr= ml['hr_per_week'].median()
fm_median_hr= fm['hr_per_week'].median()
print ("Median hours per week per men and women: ", ml_median_hr, fm_median_hr)

#10. Describe an explain the result.
# Affiche l'age médian des hommes et des femmes, ceux des hommes et des femmes avec gros salaire et la médianne des heures travaillées par les hommes et les femmes.

import matplotlib.pyplot as plt
ml_age=ml['age']
ml_age.hist(normed=0, histtype='stepfilled', bins=20)

#10. Show the graphics and an explain the result.
#L'histograme montre la repartition des ages des hommes. Les plus représentés sont dans la plage 32-34 ans environ et il y a environ 2400 hommes qui sont dans cette plage d' age.

fm_age=fm['age']
fm_age.hist(normed=0, histtype='stepfilled', bins=10)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Female samples',fontsize=15)
plt.show()

#11. Show the graphics and an explain the result.
# L'histograme montre la repartition des ages des femmes. Les plus représentés sont dans la plage 17-23 ans environ et il y a environ 2500 femmess qui sont dans cette plage d'age. On remarque que les plages d'ages contiennent plus d'années que pour l'histograme de l'age des hommes.

import seaborn as sns
fm_age.hist(normed=0, histtype='stepfilled', alpha=.5, bins=20)   # default number of bins = 10
ml_age.hist(normed=0, histtype='stepfilled', alpha=.5, color=sns.desaturate("indianred", .75), bins=10)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Samples',fontsize=15)
plt.show()

#12. Show the graphics and an explain the result.
# affiche un histogramme de l'age des femmes en bleu et de celui des hommes en rouge en fonction du nombre d'individus de chaque age.

fm_age.hist(normed=1, histtype='stepfilled', alpha=.5, bins=20)   # default number of bins = 10
ml_age.hist(normed=1, histtype='stepfilled', alpha=.5, color=sns.desaturate("indianred", .75), bins=10)
plt.xlabel('Age',fontsize=15)
plt.ylabel('PMF',fontsize=15)
plt.show()

#13. Show the graphics and an explain the result.
# C'est un histogramme qui montre la répartition des ages des hommes et des femmes normalisé (c'est a dire en divisant le nombre d'hommes et de femmes par le nombre de samples) et donc en ordonnée on a la probabilité qu'une femme (ou un homme) est l'age des abcisses correspondant. 

ml_age.hist(normed=1, histtype='stepfilled', bins=20)

plt.xlabel('Age',fontsize=15)
plt.ylabel('Probability',fontsize=15)
plt.show()

#14. Show the graphics and an explain the result.
# affiche un histogramme normalisé (c'est a dire en divisant le nombre de hommes par le nombre de samples) de l'age des hommes et donc en ordonnée on a la probabilité qu'un homme est l'age correspondant aux abcisses.

fm_age.hist(normed=1, histtype='stepfilled', bins=20)

plt.xlabel('Age',fontsize=15)
plt.ylabel('Probability',fontsize=15)
plt.show()

#15. Show the graphics and an explain the result.
#affiche un histogramme normalisé (c'est a dire en divisant le nombre de femmes par le nombre de samples) de l'age des femmes, idem au precedant mais pour les femmes.

ml_age.hist(normed=1, histtype='step', cumulative=True, linewidth=3.5, bins=20)

plt.xlabel('Age',fontsize=15)
plt.ylabel('CDF',fontsize=15)
plt.show()

#16. Show the graphics and an explain the result.
#affiche un histogramme normalisé (c'est a dire en divisant le nombre de hommes par le nombre de samples) de l'age des hommes en probabilités cumulées.

fm_age.hist(normed=1, histtype='step', cumulative=True, linewidth=3.5, bins=20)

plt.xlabel('Age',fontsize=15)
plt.ylabel('CDF',fontsize=15)
plt.show()

#17. Show the graphics and an explain the result.
#affiche un histogramme normalisé (c'est a dire en divisant le nombre de femmes par le nombre de samples) de l'age des femmes en probabilités cumulées.

ml_age.hist(bins=10, normed=1, histtype='stepfilled', alpha=.5)   # default number of bins = 10
fm_age.hist(bins=10, normed=1, histtype='stepfilled', alpha=.5, color=sns.desaturate("indianred", .75))
plt.xlabel('Age',fontsize=15)
plt.ylabel('Probability',fontsize=15)
plt.show()

#18. Show the graphics and an explain the result.
# Histogramme normalisé (probabilités) non cumulé de l'age des hommes en bleu et celui des femmes en rouge.

ml_age.hist(normed=1, histtype='step', cumulative=True,  linewidth=3.5, bins=20)
fm_age.hist(normed=1, histtype='step', cumulative=True,  linewidth=3.5, bins=20, color=sns.desaturate("indianred", .75))
plt.xlabel('Age',fontsize=15)
plt.ylabel('CDF',fontsize=15)
plt.show()

#19. Show the graphics and an explain the result.
#Histogramme normalisé en probabilités cumulées de l'age des hommes en bleu et celui des femmes en rouge.

print ("The mean sample difference is ", ml_age.mean() - fm_age.mean())

#20. Explain the result.
# Affiche : The mean sample difference is  2.5753170652810553 : c'est la différence des moyennes d'age des femmes et des hommes. Elle est de 2,5 ans pour les echantillons de données qu'on a.

df['age'].median()
len(df[(df.income == '>50K\n') & (df['age'] < df['age'].median() - 15)])
len(df[(df.income == '>50K\n') & (df['age'] > df['age'].median() + 35)])
#df est le dataframe contenant toutes les raw data.
#on crée df2 qui est le dataframe de ces même données mais ds lequel on a enlevé les valeurs extremes.
df2 = df.drop(df.index[(df.income=='>50K\n') & (df['age']>df['age'].median() +35) & (df['age'] > df['age'].median()-15)])
df2.shape
ml1_age=ml1['age']
fm1_age=fm1['age']

ml2_age = ml1_age.drop(ml1_age.index[(ml1_age >df['age'].median()+35) & (ml1_age>df['age'].median() - 15)])

fm2_age = fm1_age.drop(fm1_age.index[(fm1_age > df['age'].median()+35) & (fm1_age > df['age'].median()- 15)])

mu2ml = ml2_age.mean()
std2ml = ml2_age.std()
md2ml = ml2_age.median()

# Computing the mean, std, median, min and max for the high-income male population

print ("Men statistics: Mean:", mu2ml, "Std:", std2ml, "Median:", md2ml, "Min:", ml2_age.min(), "Max:",ml2_age.max())

mu3ml = fm2_age.mean()
std3ml = fm2_age.std()
md3ml = fm2_age.median()

# Computing the mean, std, median, min and max for the high-income female population
print ("Women statistics: Mean:", mu2ml, "Std:", std2ml, "Median:", md2ml, "Min:", fm2_age.min(), "Max:",fm2_age.max())

print ('The mean difference with outliers is: %4.2f.'% (ml_age.mean() - fm_age.mean()))
print ("The mean difference without outliers is: %4.2f."% (ml2_age.mean() - fm2_age.mean()))

plt.figure(figsize=(13.4,5))

df.age[(df.income == '>50K\n')].plot(alpha=.25, color='blue')
df2.age[(df2.income == '>50K\n')].plot(alpha=.45,color='red')

plt.ylabel('Age')
plt.xlabel('Samples')

import numpy as np

countx,divisionx = np.histogram(ml2_age, normed=True)
county,divisiony = np.histogram(fm2_age, normed=True)

import matplotlib.pyplot as plt

val = [(divisionx[i]+divisionx[i+1])/2 for i in range(len(divisionx)-1)]

plt.plot(val, countx-county,'o-')
plt.title('Differences in promoting men vs. women')
plt.xlabel('Age',fontsize=15)
plt.ylabel('Differences',fontsize=15)
plt.show()

print ("Remember:\n We have the following mean values for men, women and the difference:\nOriginally: ", ml_age.mean(), fm_age.mean(),  ml_age.mean()- fm_age.mean()) # The difference between the mean values of male and female populations.)
print ("For high-income: ", ml1_age.mean(), fm1_age.mean(), ml1_age.mean()- fm1_age.mean()) # The difference between the mean values of male and female populations.)
print ("After cleaning: ", ml2_age.mean(), fm2_age.mean(), ml2_age.mean()- fm2_age.mean()) # The difference between the mean values of male and female populations.)

print ("\nThe same for the median:")
print (ml_age.median(), fm_age.median(), ml_age.median()- fm_age.median()) # The difference between the mean values of male and female populations.)
print (ml1_age.median(), fm1_age.median(), ml1_age.median()- fm1_age.median()) # The difference between the mean values of male and female populations.)
print (ml2_age.median(), fm2_age.median(), ml2_age.median()- fm2_age.median()), # The difference between the mean values of male and female populations.)

def skewness(x):
    res=0
    m=x.mean()
    s=x.std()
    for i in x:
        res+=(i-m)*(i-m)*(i-m)
    res/=(len(x)*s*s*s)
    return res

print ("The skewness of the male population is:", skewness(ml2_age))
print ("The skewness of the female population is:", skewness(fm2_age))

#21.Explain the result
# Affiche les statistiques pour les ages des hommes et des femmes (moyenne, variance, ecart-type).
# Affiche aussi la différence en années entre la moyenne d'age des hommes et celle des femmes, en prennant en compte les valeurs extremes dans un cas et en ne les prennant pas en compte dans l'autre (with outliers is: 2.58, without outliers is: 2.44).
# Affiche un graphe des ages en fonction du nombre de personnes (samples). En bleu on a la repartition des ages selon le nombre de pers qui gagnen plus de 50k, en prennant toutes les données (raw data) du dataframe. En rouge, c'est pareil mais en prennant seulement les données du dataframe pour lesquelles on a supprimé les éléments de valeurs extremes. 
# Affiche ensuite un graphe des differences de promotion entre hommes et femmes : pour un age inferieur à 42 ans, cette différence est négative, donc les femmes sont plus prommues. Pour un age supérieur à 42 ans environ, les hommes le sont plus que les femmes.
# Rappelle les statistiques d'ages pour les salaires élevés et pour les données sans valeurs extremes. Les personnes avec un salaire élevé sont en moyenne plus vieilles de 5-6 ans. Les hommes du fichier de données sont en moyenne plus vieux de 2 ans que les femmes et de même pour les hauts salaires. On retrouve les même tendances lorsqu'on analyse les ages médiants.
# Affiche l'assymétrie de la population des hommes et celles des femmes. Celles des femmes est supérieure. 


ml1 = df[(df.sex == 'Male')&(df.income=='>50K\n')]

ml2 = ml1.drop(ml1.index[(ml1['age']>df['age'].median() +35)&(ml1['age']> df['age'].median()- 15)])

fm2 = fm1.drop(fm1.index[(fm1['age']> df['age'].median() + 35)& (fm1['age']> df['age'].median() - 15)])

print (ml2.shape, fm2.shape)

print ("Men grouped in 3 categories:")
print ("Young:",int(round(100*len(ml2_age[ml2_age<41])/float(len(ml2_age.index)))),"%.")
print ("Elder:", int(round(100*len(ml2_age[ml2_age >44])/float(len(ml2_age.index)))),"%.")
print ("Average age:", int(round(100*len(ml2_age[(ml2_age>40) & (ml2_age< 45)])/float(len(ml2_age.index)))),"%.")

print ("Women grouped in 3 categories:")
print ("Young:",int(round(100*len(fm2_age[fm2_age <41])/float(len(fm2_age.index)))),"%.")
print ("Elder:", int(round(100*len(fm2_age[fm2_age >44])/float(len(fm2_age.index)))),"%.")
print ("Average age:", int(round(100*len(fm2_age[(fm2_age>40) & (fm2_age< 45)])/float(len(fm2_age.index)))),"%.")

print ("The male mean:", ml2_age.mean())
print ("The female mean:", fm2_age.mean())

ml2_young = len(ml2_age[(ml2_age<41)])/float(len(ml2_age.index))
fm2_young  = len(fm2_age[(fm2_age<41)])/float(len(fm2_age.index))
print ("The relative risk of female early promotion is: ", 100*(1-ml2_young/fm2_young))

ml2_elder = len(ml2_age[(ml2_age>44)])/float(len(ml2_age.index))
fm2_elder  = len(fm2_age[(fm2_age>44)])/float(len(fm2_age.index))
print ("The relative risk of male late promotion is: ", 100*ml2_elder/fm2_elder)


l = 3
x=np.arange(0,2.5,0.1)
y= 1- np.exp(-l*x)

plt.plot(x,y,'-')
plt.title('Exponential CDF: $\lambda$ =%.2f'% l ,fontsize=15)
plt.xlabel('x',fontsize=15)
plt.ylabel('CDF',fontsize=15)
plt.show()

from __future__ import division
import scipy.stats as stats

l = 3
x=np.arange(0,2.5,0.1)
y= l * np.exp(-l*x)

plt.plot(x,y,'-')
plt.title('Exponential PDF: $\lambda$ =%.2f'% l, fontsize=15)
plt.xlabel('x', fontsize=15)
plt.ylabel('PDF', fontsize=15)
plt.show()

l = 0.25

x=np.arange(0,25,0.1)
y= l * np.exp(-l*x)

plt.plot(x,y,'-')
plt.title('Exponential: $\lambda$ =%.2f' %l ,fontsize=15)
plt.xlabel('x',fontsize=15)
plt.ylabel('PDF',fontsize=15)
plt.show()

u=6 # mean
s=2 # standard deviation

x=np.arange(0,15,0.1)

y=(1/(np.sqrt(2*np.pi*s*s)))*np.exp(-(((x-u)**2)/(2*s*s)))

plt.plot(x,y,'-')
plt.title('Gaussian PDF: $\mu$=%.1f, $\sigma$=%.1f'%(u,s),fontsize=15)
plt.xlabel('x',fontsize=15)
plt.ylabel('Probability density',fontsize=15)
plt.show()

fig, ax = plt.subplots(1, 4, sharey=True, squeeze=True, figsize=(14, 5))
x = np.linspace(0, 1, 100)
for i in range(4):
    f = np.mean(np.random.random((10000, i+1)), 1)
    m, s = np.mean(f), np.std(f, ddof=1)
    fn = (1/(s*np.sqrt(2*np.pi)))*np.exp(-(x-m)**2/(2*s**2))  # normal pdf            
    ax[i].hist(f, 40, normed=True, color=[0, 0.2, .8, .6]) 
    ax[i].set_title('n=%d' %(i+1))
    ax[i].plot(x, fn, color=[1, 0, 0, .6], linewidth=5)
plt.suptitle('Demonstration of the central limit theorem for a uniform distribution', y=1.05)
plt.show()



from scipy.stats.distributions import norm

# Some random data
y = np.random.random(15) * 10
x = np.linspace(0, 10, 100)

x1 = np.random.normal(-1, 2, 15) # parameters: (loc=0.0, scale=1.0, size=None)
x2 = np.random.normal(6, 3, 10)
y = np.r_[x1, x2] # r_ Translates slice objects to concatenation along the first axis.
x = np.linspace(min(y), max(y), 100)

# Smoothing parameter
s = 0.4

# Calculate the kernels
kernels = np.transpose([norm.pdf(x, yi, s) for yi in y])

plt.plot(x, kernels, 'k:')
plt.plot(x, kernels.sum(1), 'r')
plt.plot(y, np.zeros(len(y)), 'go', ms=10)

from scipy.stats import kde

x1 = np.random.normal(-1, 0.5, 15)

# parameters: (loc=0.0, scale=1.0, size=None)

x2 = np.random.normal(6, 1, 10)
y = np.r_[x1, x2]

# r_ Translates slice objects to concatenation along the first axis.

x = np.linspace(min(y), max(y), 100)
s = 0.4   # Smoothing parameter

kernels = np.transpose([norm.pdf(x, yi, s) for yi in y])

# Calculate the kernels
density = kde.gaussian_kde(y)

plt.plot(x, kernels, 'k:')
plt.plot(x, kernels.sum(1), 'r')
plt.plot(y, np.zeros(len(y)), 'bo', ms=10)

#######21. What does the figure shows ?
#

xgrid = np.linspace(x.min(), x.max(), 200)
plt.hist(y, bins=28, normed=True)
plt.plot(xgrid, density(xgrid), 'r-')

# Create a bi-modal distribution with a mixture of Normals.

x1 = np.random.normal(-1, 2, 15) # parameters: (loc=0.0, scale=1.0, size=None)
x2 = np.random.normal(6, 3, 10)

# Append by row
x = np.r_[x1, x2]

# r_ Translates slice objects to concatenation along the first axis.
plt.hist(x, bins=18, normed=True)

density = kde.gaussian_kde(x)
xgrid = np.linspace(x.min(), x.max(), 200)
plt.hist(x, bins=18, normed=True)
plt.plot(xgrid, density(xgrid), 'r-')

#4.2 ESTIMATION

x = np.random.normal(0.0, 1.0, 10000)
a = plt.hist(x,50,normed='True')

print ('The empirical mean of the sample is ', x.mean())

NTs=200
mu=0.0
var=1.0
err = 0.0
NPs=1000
for i in range(NTs):
    x = np.random.normal(mu, var, NPs)
    err += (x.mean()-mu)**2

print ('MSE: ', err/NTs)

#######21. What do you obtain as results ?

def Cov(X, Y):
    def _get_dvis(V):
        return [v - np.mean(V) for v in V]
    dxis = _get_dvis(X)
    dyis = _get_dvis(Y)
    return np.sum([x * y for x, y in zip(dxis, dyis)])/len(X)


X = [5, -1, 3.3, 2.7, 12.2]
X= np.array(X)
Y = [10, 12, 8, 9, 11]

print ("Cov(X, X) = %.2f" % Cov(X, X))
print ("Var(X) = %.2f" % np.var(X))

print ("Cov(X, Y) = %.2f" % Cov(X, Y))

MAXN=100
MAXN=40

X=np.array([[1,9],[3, 2], [5,3],[5.5,4],[6,4],[6.5,4],[7,3.5],[7.5,3.8],[8,4],
[8.5,4],[9,4.5],[9.5,7],[10,9],[10.5,11],[11,11.5],[11.5,12],[12,12],[12.5,12],[13,10]])
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],color='b',s=120, linewidths=2,zorder=10)
plt.xlabel('Economic growth(T)',fontsize=15)
plt.ylabel('Stock market returns(T)',fontsize=15)
plt.gcf().set_size_inches((20,6))

X=np.array([[1,8],[2, 7], [3,6],[4,8],[5,8],[6,7],[7,7],[8,5],[9,5],[10,6],[11,4],[12,5],[13,3],[14,2],[15,2],[16,1]])

plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],color='b',s=120, linewidths=2,zorder=10)
plt.xlabel('World Oil Production(T)',fontsize=15)
plt.ylabel('Gasoline prices(T)',fontsize=15)
plt.gcf().set_size_inches((20,6))

def Corr(X, Y):
    assert len(X) == len(Y)
    return Cov(X, Y) / np.prod([np.std(V) for V in [X, Y]])

print ("Corr(X, X) = %.5f" % Corr(X, X))

Y=np.random.random(len(X))

print ("Corr(X, Y) = %.5f" % Corr(X, Y))

def list2rank(l):
    #l is a list of numbers
    # returns a list of 1-based index; mean when multiple instances
    return [np.mean([i+1 for i, sorted_el in enumerate(sorted(l)) if sorted_el == el]) for el in l]

l = [7, 1, 2, 5]
print ("ranks: ", list2rank(l))

def spearmanRank(X, Y):
    # X and Y are same-length lists
    print (list2rank(X) )
    print (list2rank(Y))
    return Corr(list2rank(X), list2rank(Y))

X = [10, 20, 30, 40, 1000]
Y = [-70, -1000, -50, -10, -20]
plt.plot(X,'ro')
plt.plot(Y,'go')

print ("Pearson rank coefficient: %.2f" % Corr(X, Y))
print ("Spearman rank coefficient: %.2f" % spearmanRank(X, Y))

#######Exercise: Obtain for the Anscombe's quartet [2] given in the figures bellow, the different estimators (mean, variance, covariance for each pair, Pearson's correlation and Spearman's rank correlation.

X=np.array([[10.0, 8.04,10.0, 9.14, 10.0, 7.46, 8.0, 6.58],
[8.0,6.95, 8.0, 8.14, 8.0, 6.77, 8.0, 5.76],
[13.0,7.58,13.0,8.74,13.0,12.74,8.0,7.71],
[9.0,8.81,9.0,8.77,9.0,7.11,8.0,8.84],
[11.0,8.33,11.0,9.26,11.0,7.81,8.0,8.47],
[14.0,9.96,14.0,8.10,14.0,8.84,8.0,7.04],
[6.0,7.24,6.0,6.13,6.0,6.08,8.0,5.25],
[4.0,4.26,4.0,3.10,4.0,5.39,19.0,12.50],
[12.0,10.84,12.0,9.13,12.0,8.15,8.0,5.56],
[7.0,4.82,7.0,7.26,7.0,6.42,8.0,7.91],
[5.0,5.68,5.0,4.74,5.0,5.73,8.0,6.89]])

plt.subplot(2,2,1)
plt.scatter(X[:,0],X[:,1],color='r',s=120, linewidths=2,zorder=10)
plt.xlabel('x1',fontsize=15)
plt.ylabel('y1',fontsize=15)

plt.subplot(2,2,2)
plt.scatter(X[:,2],X[:,3],color='r',s=120, linewidths=2,zorder=10)
plt.xlabel('x1',fontsize=15)
plt.ylabel('y1',fontsize=15)
plt.subplot(2,2,3)
plt.scatter(X[:,4],X[:,5],color='r',s=120, linewidths=2,zorder=10)
plt.xlabel('x1',fontsize=15)
plt.ylabel('y1',fontsize=15)

plt.subplot(2,2,4)
plt.scatter(X[:,6],X[:,7],color='r',s=120, linewidths=2,zorder=10)
plt.xlabel('x1',fontsize=15)
plt.ylabel('y1',fontsize=15)
plt.gcf().set_size_inches((10,10))


file = open('/kaggle/input/datalab3/files/adult.data', 'r')
def chr_int(a):
    if a.isdigit():
        return int(a)
    else:
        return 0
                
data=[]
for line in file:
     data1=line.split(', ')
     if len(data1)==15:
        data.append([chr_int(data1[0]),data1[1],chr_int(data1[2]),data1[3],chr_int(data1[4]),data1[5],data1[6],\
            data1[7],data1[8],data1[9],chr_int(data1[10]),chr_int(data1[11]),chr_int(data1[12]),data1[13],\
            data1[14]])
        
print (data[1:2])
import pandas as pd
df=pd.DataFrame(data) 

df.columns = ['age', 'type_employer', 'fnlwgt', 'education', 
                "education_num","marital", "occupation", "relationship", "race","sex",
                "capital_gain", "capital_loss", "hr_per_week","country","income"]

df.head()

df.tail()
df.shape
counts = df.groupby('country').size()

print (counts)
counts = df.groupby('age').size() 
print (counts)
ml = df[(df.sex == 'Male')] 
ml.shape
ml1 = df[(df.sex == 'Male')&(df.income=='>50K\n')]
ml1.shape
fm =df[(df.sex == 'Female')]
fm.shape       
fm1 =df[(df.sex == 'Female')&(df.income=='>50K\n')]
fm1.shape
df1=df[(df.income=='>50K\n')]

print ('The rate of people with high income is: ', int(len(df1)/float(len(df))*100), '%.' )
print ('The rate of men with high income is: ', int(len(ml1)/float(len(ml))*100), '%.' )
print ('The rate of women with high income is: ', int(len(fm1)/float(len(fm))*100), '%.' )
print ('The average age of men is: ', ml['age'].mean(), '.' )
print ('The average age of women is: ', fm['age'].mean(), '.')
print ('The average age of high-income men is: ', ml1['age'].mean(), '.' )
print ('The average age of high-income women is: ', fm1['age'].mean(), '.')
ml_mu = ml['age'].mean()
fm_mu = fm['age'].mean()
ml_var = ml['age'].var()
fm_var = fm['age'].var()
ml_std = ml['age'].std()
fm_std = fm['age'].std()

print ('Statistics of age for men: mu:', ml_mu, 'var:', ml_var, 'std:', ml_std)
print ('Statistics of age for women: mu:', fm_mu, 'var:', fm_var, 'std:', fm_std)
ml_mu_hr = ml['hr_per_week'].mean()
fm_mu_hr = fm['hr_per_week'].mean()
ml_var_hr = ml['hr_per_week'].var()
fm_var_hr = fm['hr_per_week'].var()
ml_std_hr = ml['hr_per_week'].std()
fm_std_hr = fm['hr_per_week'].std()

print ('Statistics of hours per week for men: mu:', ml_mu_hr, 'var:', ml_var_hr, 'std:', ml_std_hr)
print ('Statistics of hours per week for women: mu:', fm_mu_hr, 'var:', fm_var_hr, 'std:', fm_std_hr)
ml_median= ml['age'].median()
fm_median= fm['age'].median()

print ("Median age per men and women: ", ml_median, fm_median)
ml_median_age= ml1['age'].median()
fm_median_age= fm1['age'].median()

print ("Median age per men and women with high-income: ", ml_median_age, fm_median_age)
ml_median_hr= ml['hr_per_week'].median()
fm_median_hr= fm['hr_per_week'].median()
print ("Median hours per week per men and women: ", ml_median_hr, fm_median_hr)
import matplotlib.pyplot as plt
ml_age=ml['age']
ml_age.hist(normed=0, histtype='stepfilled', bins=20)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Male samples',fontsize=15)
plt.show()
fm_age=fm['age']
fm_age.hist(normed=0, histtype='stepfilled', bins=10)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Female samples',fontsize=15)
plt.show()
import seaborn as sns
fm_age.hist(normed=0, histtype='stepfilled', alpha=.5, bins=20)   
ml_age.hist(normed=0, histtype='stepfilled', alpha=.5, color=sns.desaturate("indianred", .75), bins=10)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Samples',fontsize=15)
plt.show()
fm_age.hist(normed=1, histtype='stepfilled', alpha=.5, bins=20)   
ml_age.hist(normed=1, histtype='stepfilled', alpha=.5, color=sns.desaturate("indianred", .75), bins=10)
plt.xlabel('Age',fontsize=15)
plt.ylabel('PMF',fontsize=15)
plt.show()
ml_age.hist(normed=1, histtype='stepfilled', bins=20)

plt.xlabel('Age',fontsize=15)
plt.ylabel('Probability',fontsize=15)
plt.show()
fm_age.hist(normed=1, histtype='stepfilled', bins=20)

plt.xlabel('Age',fontsize=15)
plt.ylabel('Probability',fontsize=15)
plt.show()
ml_age.hist(normed=1, histtype='step', cumulative=True, linewidth=3.5, bins=20)

plt.xlabel('Age',fontsize=15)
plt.ylabel('CDF',fontsize=15)
plt.show()
fm_age.hist(normed=1, histtype='step', cumulative=True, linewidth=3.5, bins=20)

plt.xlabel('Age',fontsize=15)
plt.ylabel('CDF',fontsize=15)
plt.show()
ml_age.hist(bins=10, normed=1, histtype='stepfilled', alpha=.5)   
fm_age.hist(bins=10, normed=1, histtype='stepfilled', alpha=.5, color=sns.desaturate("indianred", .75))
plt.xlabel('Age',fontsize=15)
plt.ylabel('Probability',fontsize=15)
plt.show()
ml_age.hist(normed=1, histtype='step', cumulative=True,  linewidth=3.5, bins=20)
fm_age.hist(normed=1, histtype='step', cumulative=True,  linewidth=3.5, bins=20, color=sns.desaturate("indianred", .75))
plt.xlabel('Age',fontsize=15)
plt.ylabel('CDF',fontsize=15)
plt.show()
print ("The mean sample difference is ", ml_age.mean() - fm_age.mean())
df['age'].median()
len(df[(df.income == '>50K\n') & (df['age'] < df['age'].median() - 15)])
len(df[(df.income == '>50K\n') & (df['age'] > df['age'].median() + 35)])
df2 = df.drop(df.index[(df.income=='>50K\n') & (df['age']>df['age'].median() +35) & (df['age'] > df['age'].median()-15)])
df2.shape
ml1_age=ml1['age']
fm1_age=fm1['age']

ml2_age = ml1_age.drop(ml1_age.index[(ml1_age >df['age'].median()+35) & (ml1_age>df['age'].median() - 15)])

fm2_age = fm1_age.drop(fm1_age.index[(fm1_age > df['age'].median()+35) & (fm1_age > df['age'].median()- 15)])

mu2ml = ml2_age.mean()
std2ml = ml2_age.std()
md2ml = ml2_age.median()
print ("Men statistics: Mean:", mu2ml, "Std:", std2ml, "Median:", md2ml, "Min:", ml2_age.min(), "Max:",ml2_age.max())

mu3ml = fm2_age.mean()
std3ml = fm2_age.std()
md3ml = fm2_age.median()
print ("Women statistics: Mean:", mu2ml, "Std:", std2ml, "Median:", md2ml, "Min:", fm2_age.min(), "Max:",fm2_age.max())

print ('The mean difference with outliers is: %4.2f.'% (ml_age.mean() - fm_age.mean()))
print ("The mean difference without outliers is: %4.2f."% (ml2_age.mean() - fm2_age.mean()))

plt.figure(figsize=(13.4,5))

df.age[(df.income == '>50K\n')].plot(alpha=.25, color='blue')
df2.age[(df2.income == '>50K\n')].plot(alpha=.45,color='red')

plt.ylabel('Age')
plt.xlabel('Samples')
import numpy as np

countx,divisionx = np.histogram(ml2_age, normed=True)
county,divisiony = np.histogram(fm2_age, normed=True)

import matplotlib.pyplot as plt

val = [(divisionx[i]+divisionx[i+1])/2 for i in range(len(divisionx)-1)]

plt.plot(val, countx-county,'o-')
plt.title('Differences in promoting men vs. women')
plt.xlabel('Age',fontsize=15)
plt.ylabel('Differences',fontsize=15)
plt.show()
print ("Remember:\n We have the following mean values for men, women and the difference:\nOriginally: ", ml_age.mean(), fm_age.mean(),  ml_age.mean()- fm_age.mean()) # The difference between the mean values of male and female populations.)
print ("For high-income: ", ml1_age.mean(), fm1_age.mean(), ml1_age.mean()- fm1_age.mean())
print ("After cleaning: ", ml2_age.mean(), fm2_age.mean(), ml2_age.mean()- fm2_age.mean()) 

print ("\nThe same for the median:")
print (ml_age.median(), fm_age.median(), ml_age.median()- fm_age.median()) 
print (ml1_age.median(), fm1_age.median(), ml1_age.median()- fm1_age.median())
print (ml2_age.median(), fm2_age.median(), ml2_age.median()- fm2_age.median()),
def skewness(x):
    res=0
    m=x.mean()
    s=x.std()
    for i in x:
        res+=(i-m)*(i-m)*(i-m)
    res/=(len(x)*s*s*s)
    return res

print ("The skewness of the male population is:", skewness(ml2_age))
print ("The skewness of the female population is:", skewness(fm2_age))
ml1 = df[(df.sex == 'Male')&(df.income=='>50K\n')]

ml2 = ml1.drop(ml1.index[(ml1['age']>df['age'].median() +35)&(ml1['age']> df['age'].median()- 15)])

fm2 = fm1.drop(fm1.index[(fm1['age']> df['age'].median() + 35)& (fm1['age']> df['age'].median() - 15)])

print (ml2.shape, fm2.shape)
print ("Men grouped in 3 categories:")
print ("Young:",int(round(100*len(ml2_age[ml2_age<41])/float(len(ml2_age.index)))),"%.")
print ("Elder:", int(round(100*len(ml2_age[ml2_age >44])/float(len(ml2_age.index)))),"%.")
print ("Average age:", int(round(100*len(ml2_age[(ml2_age>40) & (ml2_age< 45)])/float(len(ml2_age.index)))),"%.")
print ("Women grouped in 3 categories:")
print ("Young:",int(round(100*len(fm2_age[fm2_age <41])/float(len(fm2_age.index)))),"%.")
print ("Elder:", int(round(100*len(fm2_age[fm2_age >44])/float(len(fm2_age.index)))),"%.")
print ("Average age:", int(round(100*len(fm2_age[(fm2_age>40) & (fm2_age< 45)])/float(len(fm2_age.index)))),"%.")

print ("The male mean:", ml2_age.mean())
print ("The female mean:", fm2_age.mean())
ml2_young = len(ml2_age[(ml2_age<41)])/float(len(ml2_age.index))
fm2_young  = len(fm2_age[(fm2_age<41)])/float(len(fm2_age.index))
print ("The relative risk of female early promotion is: ", 100*(1-ml2_young/fm2_young))

ml2_elder = len(ml2_age[(ml2_age>44)])/float(len(ml2_age.index))
fm2_elder  = len(fm2_age[(fm2_age>44)])/float(len(fm2_age.index))
print ("The relative risk of male late promotion is: ", 100*ml2_elder/fm2_elder)
l = 3
x=np.arange(0,2.5,0.1)
y= 1- np.exp(-l*x)

plt.plot(x,y,'-')
plt.title('Exponential CDF: $\lambda$ =%.2f'% l ,fontsize=15)
plt.xlabel('x',fontsize=15)
plt.ylabel('CDF',fontsize=15)
plt.show()

from __future__ import division
import scipy.stats as stats

l = 3
x=np.arange(0,2.5,0.1)
y= l * np.exp(-l*x)

plt.plot(x,y,'-')
plt.title('Exponential PDF: $\lambda$ =%.2f'% l, fontsize=15)
plt.xlabel('x', fontsize=15)
plt.ylabel('PDF', fontsize=15)
plt.show()
l = 0.25

x=np.arange(0,25,0.1)
y= l * np.exp(-l*x)

plt.plot(x,y,'-')
plt.title('Exponential: $\lambda$ =%.2f' %l ,fontsize=15)
plt.xlabel('x',fontsize=15)
plt.ylabel('PDF',fontsize=15)
plt.show()

u=6 
s=2 

x=np.arange(0,15,0.1)

y=(1/(np.sqrt(2*np.pi*s*s)))*np.exp(-(((x-u)**2)/(2*s*s)))

plt.plot(x,y,'-')
plt.title('Gaussian PDF: $\mu$=%.1f, $\sigma$=%.1f'%(u,s),fontsize=15)
plt.xlabel('x',fontsize=15)
plt.ylabel('Probability density',fontsize=15)
plt.show()

fig, ax = plt.subplots(1, 4, sharey=True, squeeze=True, figsize=(14, 5))
x = np.linspace(0, 1, 100)
for i in range(4):
    f = np.mean(np.random.random((10000, i+1)), 1)
    m, s = np.mean(f), np.std(f, ddof=1)
    fn = (1/(s*np.sqrt(2*np.pi)))*np.exp(-(x-m)**2/(2*s**2))            
    ax[i].hist(f, 40, normed=True, color=[0, 0.2, .8, .6]) 
    ax[i].set_title('n=%d' %(i+1))
    ax[i].plot(x, fn, color=[1, 0, 0, .6], linewidth=5)
plt.suptitle('Demonstration of the central limit theorem for a uniform distribution', y=1.05)
plt.show()

from scipy.stats.distributions import norm

y = np.random.random(15) * 10
x = np.linspace(0, 10, 100)

x1 = np.random.normal(-1, 2, 15) 
x2 = np.random.normal(6, 3, 10)
y = np.r_[x1, x2]
x = np.linspace(min(y), max(y), 100)

s = 0.4

kernels = np.transpose([norm.pdf(x, yi, s) for yi in y])

plt.plot(x, kernels, 'k:')
plt.plot(x, kernels.sum(1), 'r')
plt.plot(y, np.zeros(len(y)), 'go', ms=10)
from scipy.stats import kde

x1 = np.random.normal(-1, 0.5, 15)

x2 = np.random.normal(6, 1, 10)
y = np.r_[x1, x2]

x = np.linspace(min(y), max(y), 100)
s = 0.4  

kernels = np.transpose([norm.pdf(x, yi, s) for yi in y])

density = kde.gaussian_kde(y)

plt.plot(x, kernels, 'k:')
plt.plot(x, kernels.sum(1), 'r')
plt.plot(y, np.zeros(len(y)), 'bo', ms=10)
xgrid = np.linspace(x.min(), x.max(), 200)
plt.hist(y, bins=28, normed=True)
plt.plot(xgrid, density(xgrid), 'r-')
x1 = np.random.normal(-1, 2, 15)
x2 = np.random.normal(6, 3, 10)

x = np.r_[x1, x2]

plt.hist(x, bins=18, normed=True)
density = kde.gaussian_kde(x)
xgrid = np.linspace(x.min(), x.max(), 200)
plt.hist(x, bins=18, normed=True)
plt.plot(xgrid, density(xgrid), 'r-')
x = np.random.normal(0.0, 1.0, 10000)


print ('The empirical mean of the sample is ', x.mean())
a = plt.hist(x,50,normed='True')
NTs=200
mu=0.0
var=1.0
err = 0.0
NPs=1000
for i in range(NTs):
    x = np.random.normal(mu, var, NPs)
    err += (x.mean()-mu)**2
print ('MSE: ', err/NTs)
def Cov(X, Y):
    def _get_dvis(V):
        return [v - np.mean(V) for v in V]
    dxis = _get_dvis(X)
    dyis = _get_dvis(Y)
    return np.sum([x * y for x, y in zip(dxis, dyis)])/len(X)


X = [5, -1, 3.3, 2.7, 12.2]
X= np.array(X)
Y = [10, 12, 8, 9, 11]

print ("Cov(X, X) = %.2f" % Cov(X, X))
print ("Var(X) = %.2f" % np.var(X))

print ("Cov(X, Y) = %.2f" % Cov(X, Y))
MAXN=100
MAXN=40

X=np.array([[1,9],[3, 2], [5,3],[5.5,4],[6,4],[6.5,4],[7,3.5],[7.5,3.8],[8,4],
[8.5,4],[9,4.5],[9.5,7],[10,9],[10.5,11],[11,11.5],[11.5,12],[12,12],[12.5,12],[13,10]])

plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],color='b',s=120, linewidths=2,zorder=10)
plt.xlabel('Economic growth(T)',fontsize=15)
plt.ylabel('Stock market returns(T)',fontsize=15)
plt.gcf().set_size_inches((20,6))
X=np.array([[1,8],[2, 7], [3,6],[4,8],[5,8],[6,7],[7,7],[8,5],[9,5],[10,6],[11,4],[12,5],[13,3],[14,2],[15,2],[16,1]])

plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],color='b',s=120, linewidths=2,zorder=10)
plt.xlabel('World Oil Production(T)',fontsize=15)
plt.ylabel('Gasoline prices(T)',fontsize=15)
plt.gcf().set_size_inches((20,6))
def Corr(X, Y):
    assert len(X) == len(Y)
    return Cov(X, Y) / np.prod([np.std(V) for V in [X, Y]])

print ("Corr(X, X) = %.5f" % Corr(X, X))

Y=np.random.random(len(X))

print ("Corr(X, Y) = %.5f" % Corr(X, Y))
def list2rank(l):
    #l is a list of numbers
    # returns a list of 1-based index; mean when multiple instances
    return [np.mean([i+1 for i, sorted_el in enumerate(sorted(l)) if sorted_el == el]) for el in l]

l = [7, 1, 2, 5]
print ("ranks: ", list2rank(l))
def spearmanRank(X, Y):
    print (list2rank(X) )
    print (list2rank(Y))
    return Corr(list2rank(X), list2rank(Y))

X = [10, 20, 30, 40, 1000]
Y = [-70, -1000, -50, -10, -20]
plt.plot(X,'ro')
plt.plot(Y,'go')

print ("Pearson rank coefficient: %.2f" % Corr(X, Y))
print ("Spearman rank coefficient: %.2f" % spearmanRank(X, Y))
X=np.array([[10.0, 8.04,10.0, 9.14, 10.0, 7.46, 8.0, 6.58],
[8.0,6.95, 8.0, 8.14, 8.0, 6.77, 8.0, 5.76],
[13.0,7.58,13.0,8.74,13.0,12.74,8.0,7.71],
[9.0,8.81,9.0,8.77,9.0,7.11,8.0,8.84],
[11.0,8.33,11.0,9.26,11.0,7.81,8.0,8.47],
[14.0,9.96,14.0,8.10,14.0,8.84,8.0,7.04],
[6.0,7.24,6.0,6.13,6.0,6.08,8.0,5.25],
[4.0,4.26,4.0,3.10,4.0,5.39,19.0,12.50],
[12.0,10.84,12.0,9.13,12.0,8.15,8.0,5.56],
[7.0,4.82,7.0,7.26,7.0,6.42,8.0,7.91],
[5.0,5.68,5.0,4.74,5.0,5.73,8.0,6.89]])
plt.subplot(2,2,1)
plt.scatter(X[:,0],X[:,1],color='r',s=120, linewidths=2,zorder=10)
plt.xlabel('x1',fontsize=15)
plt.ylabel('y1',fontsize=15)
plt.subplot(2,2,2)
plt.scatter(X[:,2],X[:,3],color='r',s=120, linewidths=2,zorder=10)
plt.xlabel('x2',fontsize=15)
plt.ylabel('y2',fontsize=15)
plt.subplot(2,2,3)
plt.scatter(X[:,4],X[:,5],color='r',s=120, linewidths=2,zorder=10)
plt.xlabel('x3',fontsize=15)
plt.ylabel('y3',fontsize=15)
plt.subplot(2,2,4)
plt.scatter(X[:,6],X[:,7],color='r',s=120, linewidths=2,zorder=10)
plt.xlabel('x4',fontsize=15)
plt.ylabel('y4',fontsize=15)
plt.gcf().set_size_inches((10,10))
x1=X[:,0],X[:,1]
x2=X[:,2],X[:,3]
x3=X[:,4],X[:,5]
x4=X[:,6],X[:,7]
print ('The empirical mean of the sample is ', x1.mean())
print ("Cov(x1, x1) = %.2f" % Cov(x1, x1))
print ("Var(x1) = %.2f" % np.var(x1))
print ("Cov(x1, x2) = %.2f" % Cov(x1, x2))
print ("Corr(x1, x2) = %.5f" % Corr(x1, x2))
print ("Pearson rank coefficient: %.2f" % Corr(x1, x2))
print ("Spearman rank coefficient: %.2f" % spearmanRank(x1, x2))