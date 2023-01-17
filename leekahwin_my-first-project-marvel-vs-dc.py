import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
dc = pd.read_csv('../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv')
dc.head()
marvel = pd.read_csv('../input/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv')
marvel.head()
sns.set_style("dark")
plt.subplots(1,2,figsize = (15,3))
plt.subplot(121)
sns.countplot(x= 'ALIGN',hue = 'SEX',data = dc)
plt.legend(loc='upper right')
plt.title('DC - Nature of Characters, Genders and Count ')

plt.subplot(122)
sns.countplot(x= 'ALIVE',hue = 'SEX',data = dc)
plt.legend(loc='upper right')
plt.title('DC - Living of Characters, Genders and Count ')
sns.set_style("dark")
plt.subplots(1,2,figsize = (15,3))
plt.subplot(121)
sns.countplot(x= 'ALIGN',hue = 'SEX',data = marvel)
plt.legend(loc='upper left')
plt.title('Marvel - Nature of Characters, Genders and Count ')

plt.subplot(122)
sns.countplot(x= 'ALIVE',hue = 'SEX',data = marvel)
plt.legend(loc='upper right')
plt.title('Marvel - Living of Characters, Genders and Count ')
sns.set_style("dark")
plt.subplots(1,2, figsize = (20,10))

plt.subplot(121)
sns.countplot(x= 'ALIGN',hue = 'EYE',data = dc)
plt.legend(loc='upper right')
plt.title('DC - Nature of Characters, AND APPEARANCE ')

plt.subplot(122)
sns.countplot(x= 'ALIGN',hue = 'EYE',data = marvel)
plt.legend(loc='upper right')
plt.title('MARVEL - Nature of Characters AND APPEARANCE ')