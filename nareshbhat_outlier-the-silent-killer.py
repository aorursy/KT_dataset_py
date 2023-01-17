import numpy as np

import scipy.stats as stats

x = np.array([12,13,14,19,21,23])

y = np.array([12,13,14,19,21,23,45])

def grubbs_test(x):

    n = len(x)

    mean_x = np.mean(x)

    sd_x = np.std(x)

    numerator = max(abs(x-mean_x))

    g_calculated = numerator/sd_x

    print("Grubbs Calculated Value:",g_calculated)

    t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)

    g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))

    print("Grubbs Critical Value:",g_critical)

    if g_critical > g_calculated:

        print("From grubbs_test we observe that calculated value is lesser than critical value, Accept null hypothesis and conclude that there is no outliers\n")

    else:

        print("From grubbs_test we observe that calculated value is greater than critical value, Reject null hypothesis and conclude that there is an outliers\n")

grubbs_test(x)

grubbs_test(y)
import pandas as pd

import numpy as np

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

out=[]

def Zscore_outlier(df):

    for i in df: 

        z = (i-np.mean(df))/np.std(df)

        if np.abs(z) > 3: 

            out.append(i)

    print("Outliers:",out)

Zscore_outlier(train['LotArea'])
import pandas as pd

import numpy as np

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

out=[]

def ZRscore_outlier(df):

    med = np.median(df)

    ma = stats.median_absolute_deviation(df)

    for i in df: 

        z = (0.6745*(i-med))/ (np.median(ma))

        if np.abs(z) > 3: 

            out.append(i)

    print("Outliers:",out)

ZRscore_outlier(train['LotArea'])
import pandas as pd

import numpy as np

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

out=[]

def iqr_outliers(df):

    for i in df:

        q1 = df.quantile(0.25)

        q3 = df.quantile(0.75)

        iqr = q3-q1

        Lower_tail = q1 - 1.5 * iqr

        Upper_tail = q3 + 1.5 * iqr

        if i > Upper_tail or i < Lower_tail:

            out.append(i)

    print("Outliers:",out)

iqr_outliers(train['LotArea'])
import pandas as pd

import numpy as np

train = pd.read_csv('../input/titanic/train.csv')

out=[]

def Winsorization_outliers(df):

    for i in df:

        q1 = np.percentile(df , 1)

        q3 = np.percentile(df , 99)

        if i > q3 or i < q1:

            out.append(i)

    print("Outliers:",out)

Winsorization_outliers(train['Fare'])
import pandas as pd

from sklearn.cluster import DBSCAN

train = pd.read_csv('../input/titanic/train.csv')

def DB_outliers(df):

    outlier_detection = DBSCAN(eps = 2, metric='euclidean', min_samples = 5)

    clusters = outlier_detection.fit_predict(df.values.reshape(-1,1))

    data = pd.DataFrame()

    data['cluster'] = clusters

    print(data['cluster'].value_counts().sort_values(ascending=False))

DB_outliers(train['Fare']) 
from sklearn.ensemble import IsolationForest

import numpy as np

import pandas as pd

train = pd.read_csv('../input/titanic/train.csv')

train['Fare'].fillna(train[train.Pclass==3]['Fare'].median(),inplace=True)

def Iso_outliers(df):

    iso = IsolationForest( behaviour = 'new', random_state = 1, contamination= 'auto')

    preds = iso.fit_predict(df.values.reshape(-1,1))

    data = pd.DataFrame()

    data['cluster'] = preds

    print(data['cluster'].value_counts().sort_values(ascending=False))

Iso_outliers(train['Fare']) 
import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from statsmodels.graphics.gofplots import qqplot

train = pd.read_csv('../input/titanic/train.csv')

def Box_plots(df):

    plt.figure(figsize=(10, 4))

    plt.title("Box Plot")

    sns.boxplot(df)

    plt.show()

Box_plots(train['Age'])



def hist_plots(df):

    plt.figure(figsize=(10, 4))

    plt.hist(df)

    plt.title("Histogram Plot")

    plt.show()

hist_plots(train['Age'])



def scatter_plots(df1,df2):

    fig, ax = plt.subplots(figsize=(10,4))

    ax.scatter(df1,df2)

    ax.set_xlabel('Age')

    ax.set_ylabel('Fare')

    plt.title("Scatter Plot")

    plt.show()

scatter_plots(train['Age'],train['Fare'])



def dist_plots(df):

    plt.figure(figsize=(10, 4))

    sns.distplot(df)

    plt.title("Distribution plot")

    sns.despine()

    plt.show()

dist_plots(train['Fare'])



def qq_plots(df):

    plt.figure(figsize=(10, 4))

    qqplot(df,line='s')

    plt.title("Normal QQPlot")

    plt.show()

qq_plots(train['Fare'])



import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

train = pd.read_csv('../input/cost-of-living/cost-of-living-2018.csv')

sns.boxplot(train['Cost of Living Index'])

plt.title("Box Plot before outlier removing")

plt.show()

def drop_outliers(df, field_name):

    iqr = 1.5 * (np.percentile(df[field_name], 75) - np.percentile(df[field_name], 25))

    df.drop(df[df[field_name] > (iqr + np.percentile(df[field_name], 75))].index, inplace=True)

    df.drop(df[df[field_name] < (np.percentile(df[field_name], 25) - iqr)].index, inplace=True)

drop_outliers(train, 'Cost of Living Index')

sns.boxplot(train['Cost of Living Index'])

plt.title("Box Plot after outlier removing")

plt.show()
#Scalling

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn import preprocessing

train = pd.read_csv('../input/cost-of-living/cost-of-living-2018.csv')

plt.hist(train['Cost of Living Index'])

plt.title("Histogram before Scalling")

plt.show()

scaler = preprocessing.StandardScaler()

train['Cost of Living Index'] = scaler.fit_transform(train['Cost of Living Index'].values.reshape(-1,1))

plt.hist(train['Cost of Living Index'])

plt.title("Histogram after Scalling")

plt.show()
#Log Transformation

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

train = pd.read_csv('../input/cost-of-living/cost-of-living-2018.csv')

sns.distplot(train['Cost of Living Index'])

plt.title("Distribution plot before Log transformation")

sns.despine()

plt.show()

train['Cost of Living Index'] = np.log(train['Cost of Living Index'])

sns.distplot(train['Cost of Living Index'])

plt.title("Distribution plot after Log transformation")

sns.despine()

plt.show()
#cube root Transformation

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

train = pd.read_csv('../input/titanic/train.csv')

plt.hist(train['Age'])

plt.title("Histogram before cube root Transformation")

plt.show()

train['Age'] = (train['Age']**(1/3))

plt.hist(train['Age'])

plt.title("Histogram after cube root Transformation")

plt.show()
#Box-transformation

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import scipy

train = pd.read_csv('../input/cost-of-living/cost-of-living-2018.csv')

sns.boxplot(train['Rent Index'])

plt.title("Box Plot before outlier removing")

plt.show()

train['Rent Index'],fitted_lambda= scipy.stats.boxcox(train['Rent Index'] ,lmbda=None)

sns.boxplot(train['Rent Index'])

plt.title("Box Plot after outlier removing")

plt.show()
#mean imputation

import pandas as pd

import numpy as np

train = pd.read_csv('../input/titanic/train.csv')

sns.boxplot(train['Age'])

plt.title("Box Plot before mean imputation")

plt.show()

for i in train['Age']:

    q1 = train['Age'].quantile(0.25)

    q3 = train['Age'].quantile(0.75)

    iqr = q3-q1

    Lower_tail = q1 - 1.5 * iqr

    Upper_tail = q3 + 1.5 * iqr

    if i > Upper_tail or i < Lower_tail:

            train['Age'] = train['Age'].replace(i, np.mean(train['Age']))

sns.boxplot(train['Age'])

plt.title("Box Plot after mean imputation")

plt.show()   
#median imputation

import pandas as pd

import numpy as np

train = pd.read_csv('../input/titanic/train.csv')

sns.boxplot(train['Age'])

plt.title("Box Plot before median imputation")

plt.show()

for i in train['Age']:

    q1 = train['Age'].quantile(0.25)

    q3 = train['Age'].quantile(0.75)

    iqr = q3-q1

    Lower_tail = q1 - 1.5 * iqr

    Upper_tail = q3 + 1.5 * iqr

    if i > Upper_tail or i < Lower_tail:

            train['Age'] = train['Age'].replace(i, np.median(train['Age']))

sns.boxplot(train['Age'])

plt.title("Box Plot after median imputation")

plt.show()            

#Zero value imputation

import pandas as pd

import numpy as np

train = pd.read_csv('../input/titanic/train.csv')

sns.boxplot(train['Age'])

plt.title("Box Plot before Zero value imputation")

plt.show()

for i in train['Age']:

    q1 = train['Age'].quantile(0.25)

    q3 = train['Age'].quantile(0.75)

    iqr = q3-q1

    Lower_tail = q1 - 1.5 * iqr

    Upper_tail = q3 + 1.5 * iqr

    if i > Upper_tail or i < Lower_tail:

            train['Age'] = train['Age'].replace(i, 0)

sns.boxplot(train['Age'])

plt.title("Box Plot after Zero value imputation")

plt.show()            
