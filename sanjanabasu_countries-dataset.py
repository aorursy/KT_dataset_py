import pandas as pd
data=pd.read_csv('../input/countries-1/countries1.csv')
data.head(10)
data.describe()
import matplotlib.pyplot as plt
pd.pivot_table(data,index=['year','continent'],aggfunc='mean')
data1=pd.pivot_table(data,index=['continent','year'],aggfunc='mean')
data1
import seaborn as sns
sns.pairplot(data)
data.plot(kind='scatter',x='lifeExpectancy',y='gdpPerCapita')

data['lifeExpectancy'].skew()
data['lifeExpectancy'].plot(kind='kde')
pd.pivot_table(data,index=['country'])
df1=pd.DataFrame(pd.qcut(data['year'],q=5))
df1
sns.barplot(x='continent',hue=df1.year,y='lifeExpectancy',data=data)
data[data.country=='Afghanistan']
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
demo = pd.read_csv("../input/countries-1/countries1.csv")
demo2007 = demo[demo.year==2007]
demo2007=demo2007.drop('year',axis=1)
demo2007['loggdp'] = np.log10(demo2007['gdpPerCapita'])
demo2007['logpop'] = np.log10(demo2007['population'])
demo2007['life'] = demo2007['lifeExpectancy']**2
demo2007 = demo2007.drop('gdpPerCapita',axis=1)
demo2007 = demo2007.drop('lifeExpectancy',axis=1)
demo2007 = demo2007.drop('population',axis=1)
demo2007.head()
