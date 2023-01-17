import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('/kaggle/input/Admission_Predict.csv')
df.info()
df.head()
df.mean()
df.describe()
df.quantile(q=0.25)
df.quantile(q=0.50)
corr=df.corr()
corr
plt.figure(figsize=(10,10))
sns.heatmap(corr, cmap='plasma', vmax=1, vmin=-1, annot=True)
df.head()
sns.lineplot(df['GRE Score'], df['Chance of Admit '])
sns.scatterplot(df['GRE Score'], df['Chance of Admit '])
sns.pairplot(df[['GRE Score', 'Chance of Admit ']], kind='reg')
plt.figure(figsize=(10,10))
sns.distplot(df['GRE Score'])
df['GRE Score'].hist(bins=50, figsize=(10,10))
sns.boxplot(df['TOEFL Score'])
df.skew()
plt.figure(figsize=(10,10))
mode = df['TOEFL Score'].mode()
plt.axvline(df['TOEFL Score'].mean(), label='mean', color='green')
plt.axvline(df['TOEFL Score'].median(), label='median', color='blue')
plt.axvline(mode[0], label='mode_0', color='red')
plt.legend()
plt.xlabel('TOFEFL Score')
plt.ylabel('Frequency')
plt.hist(df['TOEFL Score'], bins=100,color='lightblue')

# Create boxplot for column="TOEFL Score"
df.boxplot(column="TOEFL Score",figsize=(8,8))

# create text(x=0.74, y=22.25, s="3rd Quartile")like Median, 1st Quartile,Min,Max,IQR:
plt.text(x=0.74, y=df['TOEFL Score'].quantile(q=0.75), s="3rd Quartile")
plt.text(x=0.8, y=df['TOEFL Score'].quantile(q=0.50), s="Median")
plt.text(x=0.75, y=df['TOEFL Score'].quantile(q=0.25), s="1st Quartile")
plt.text(x=0.9, y=df['TOEFL Score'].min(), s="Min")
plt.text(x=0.9, y=df['TOEFL Score'].max(), s="Max")
plt.text(x=0.7, y=df['TOEFL Score'].median()+1, s="IQR", rotation=90, size=25)
import scipy.stats as stats

#convert pandas DataFrame object to numpy array and sort
h = np.asarray(df['GRE Score'])
h1 = sorted(h) 

 
#use the scipy stats module to fit a normal distirbution with same mean and standard deviation
fit = stats.norm.pdf(h1, np.mean(h1), np.std(h1)) 
 
figure, axe = plt.subplots(1,1)
#plot both series on the histogram
df['GRE Score'].plot.hist(ax= axe, label='Actual distribution')
axe.plot(h1,fit,'-',linewidth = 2,label="Normal distribution with same mean and var")
#plt.hist(h1, bins = 100,label="Actual distribution")      
axe.legend()
#plt.show()
 