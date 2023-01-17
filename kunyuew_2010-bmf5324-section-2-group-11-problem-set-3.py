import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # visualization
housingprice = pd.read_csv('../input/korea-housing/housing price.csv')

jeonseprice=pd.read_csv('../input/jeonse/Housing Jeonse.csv')

housingprice.info()

jeonseprice.info()
housingprice.head(10)
jeonseprice.head(10)
jeonseprice.drop([len(jeonseprice)-1],inplace=True)

jeonseprice.info()
housing=housingprice.melt(id_vars=['Conversion','Epidemic'],value_vars=['Seoul','Gyeonggi','Incheon','Busan','Daegu','Gwangju','Daejeon','Ulsan','Sejong','Gangwon','Chungbuk','Chungnam','Jeonbuk','Jeonnam','Gyeongbuk','Gyeongnam','Jeju'])

housing=housing.dropna(axis=0,subset = ['value'])

housing['Epidemic'].fillna(0, inplace=True)

housing.info()
housing.head(10)
jeonse=jeonseprice.melt(id_vars=['Conversion','Epidemic'],value_vars=['Seoul','Gyeonggi','Incheon','Busan','Daegu','Gwangju','Daejeon','Ulsan','Sejong','Gangwon','Chungbuk','Chungnam','Jeonbuk','Jeonnam','Gyeongbuk','Gyeongnam','Jeju'])

jeonse=jeonse.dropna(axis=0,subset = ['value'])

jeonse['Epidemic'].fillna(0, inplace=True)

jeonse.info()
epi_housing=housing[housing['Epidemic']==1]



epi_housing.info()

epi_jeonse=jeonse[jeonse['Epidemic']==1]



epi_jeonse.info()
normal_housing=housing[housing['Epidemic']==0]

normal_housing.info()
normal_jeonse=jeonse[jeonse['Epidemic']==0]

normal_jeonse.info()
plt.hist(epi_housing['value'], bins = 100)

plt.show()
plt.hist(normal_housing['value'], bins = 100)

plt.show()
from scipy.stats import t

from scipy import stats



n_epi_housing=epi_housing['value'].count()

n_normal_housing=normal_housing['value'].count()

df=n_epi_housing+n_normal_housing-2

sample_mean_epi_housing = epi_housing['value'].mean()

sample_mean_normal_housing=normal_housing['value'].mean()

sample_mean_delta=sample_mean_epi_housing-sample_mean_normal_housing

var_epi_housing=epi_housing['value'].var()

var_normal_housing=normal_housing['value'].var()



sample_standard_error = np.sqrt(((n_epi_housing-1)*var_epi_housing+(n_normal_housing-1)*var_normal_housing)/df)

tt = sample_mean_delta/(sample_standard_error*np.sqrt(1/n_epi_housing+1/n_normal_housing))

pval = stats.t.sf(np.abs(tt), df)*2  # two-sided pvalue = Prob(abs(t)>tt)



print("Epidemic: Point estimate : " + str(sample_mean_epi_housing))

print("Normal: Point estimate : " + str(sample_mean_normal_housing))

print("Standard error weighted: " + str(sample_standard_error))

print("Degree of freedom : " + str(df))

print("T-statistic : " + str(tt))



# p value test 

alpha = 0.05

print("P-value : " + str(pval))

if pval <= alpha: 

    print('There is a difference in the housing price change ratio between epidemic and nomal periods (reject H0)') 

else: 

    print('There is no difference in the housing price change ratio between epidemic and nomal periods (fail to reject H0)') 
stats.levene(epi_housing['value'],normal_housing['value'])
stats.ttest_ind(epi_housing['value'],normal_housing['value'],equal_var=False)
confidence_level = 0.95



confidence_interval = t.interval(confidence_level, df, sample_mean_delta, sample_standard_error)



print("Point estimate : " + str(sample_mean_delta))

print("Confidence interval (0.025, 0.975) : " + str(confidence_interval))