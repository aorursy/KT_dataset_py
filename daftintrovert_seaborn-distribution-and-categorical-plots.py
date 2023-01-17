import seaborn as sns
%matplotlib inline
tips = sns.load_dataset('tips')
tips.head()
sns.distplot(tips['total_bill'],kde = False,bins = 30,color = 'red') #displot helps us to visualize the distribution.

#it is multivariate.
sns.jointplot(x ='total_bill',y = 'tip',data = tips,kind = 'hex') #kind = 'hex'
sns.jointplot(x ='total_bill',y = 'tip',data = tips,kind = 'kde') #kind = 'kde'
sns.jointplot(x ='total_bill',y = 'tip',data = tips) #default is scatter
sns.jointplot(x ='total_bill',y = 'size',data = tips,kind = 'kde')
sns.pairplot(tips) #make a combination of two numerical data rows and permute them in graphs
sns.pairplot(tips, hue = 'sex') #hue is added only for the categorical data like sex and smoker
sns.pairplot(tips,hue = 'smoker',palette = 'coolwarm')
sns.rugplot(tips['total_bill'])
sns.distplot(tips['total_bill']) #kde = kernel density estimation plot
sns.kdeplot(tips['total_bill']) #for plotting actual kde plot.
import seaborn as sns

%matplotlib inline

tips = sns.load_dataset('tips')

tips.head()
import numpy as np
sns.barplot(x = 'sex',y = 'total_bill',data = tips,estimator = np.std) #it gives the estimator of total_bill given by male and female
sns.countplot(x = 'sex',data = tips)
sns.boxplot(x = 'day',y = 'total_bill',data = tips)
sns.boxplot(x = 'day',y = 'total_bill',data = tips,hue = 'smoker') #adding hue = smoker to split boxplot into two subparts
sns.violinplot(x = 'day',y = 'total_bill',data = tips)
sns.violinplot(x = 'day',y = 'total_bill',data = tips,hue = 'sex')
sns.violinplot(x = 'day',y = 'total_bill',data = tips,hue = 'sex',split = True)
sns.stripplot(x = 'day',y = 'total_bill',data = tips,jitter = True,hue = 'sex',split = True)
sns.swarmplot(x = 'day',y = 'total_bill',data = tips) #not used for large datasets!
sns.violinplot(x = 'day',y = 'total_bill',data = tips)

sns.swarmplot(x = 'day',y = 'total_bill',data = tips,color = 'black')

sns.factorplot(x = 'day',y = 'total_bill',data = tips,kind = 'bar') #it takes argument 'kind'
sns.factorplot(x = 'day',y = 'total_bill',data = tips,kind = 'violin')