# get necessary libs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

plt.figure(figsize=(16,6))
%matplotlib inline

data = pd.read_csv('../input/kagglers/liste.csv',sep=';')
data.head()
data.pays.value_counts()[:10]
nodata = int(data.pays.isna().sum()/len(data)*100)
print(str(nodata),"% of Kagglers did'nt give their country")
plt.figure(figsize=(16,6))
data.pays.value_counts()[:10].plot.bar()
population = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv',sep=',')
population = population[['Country (or dependency)','Population (2020)']]
population = population.rename(columns={"Country (or dependency)": "pays", "Population (2020)": "population"})
population = population.set_index('pays')
population = population.iloc[:,0]
population.head()
top_countries = data.pays.value_counts()
top_countries = top_countries.divide(population)*1000000
df = top_countries.sort_values(ascending=False)[:30]

df.plot(kind='barh',    # Plot a bar chart
        figsize=(16,16))

mostPopulated = population[population>1000000]
top_countries = data.pays.value_counts()
top_countries = top_countries.divide(mostPopulated)*1000000
df = top_countries.sort_values(ascending=False)[:30]

df.plot(kind='barh',    # Plot a bar chart
        figsize=(16,16))
universities = pd.read_html('http://www.shanghairanking.com/ARWU-Statistics-2019.html', header=0, index_col=0)
nbUniversities = universities[1]['501-1000'].replace('—',0).astype(int)
nbKagglers = data.pays.value_counts()
KagglersVSuniversities = pd.DataFrame(dict(s1 = nbKagglers, s2 = nbUniversities)).reset_index().dropna()
plt.figure(figsize=(16,10))
# let us see if there seem to be a correlation between number of top 1000 universities and number of KAggles
sns.scatterplot(x="s1", y="s2", data=KagglersVSuniversities)
plt.title('Kagglers VS universities')
# Set x-axis label
plt.xlabel('nb Kagglers')
# Set y-axis label
plt.ylabel('nb Universities')
plt.ylim(0, 40)
plt.xlim(0, 100)
from scipy.stats import pearsonr
# calculate Pearson's correlation
corr, _ = pearsonr(KagglersVSuniversities.s1,KagglersVSuniversities.s2 )
print('Pearsons correlation: %.3f' % corr)