%matplotlib inline

import pandas as pd

import numpy as np

import statistics

import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from scipy import stats

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold   #For K-fold cross validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import metrics
payroll = pd.read_csv("../input/data.csv")
payroll = payroll[pd.notnull(payroll['Payroll Department'])]

payroll.rename(columns={'Projected Annual Salary' : 'Annual_sal'}, inplace = True)

payroll.rename(columns={'Job Class Title' : 'Job_title'}, inplace = True)

payroll.rename(columns={'Base Pay' : 'Base_Pay'}, inplace = True)
#Removing $ 

for i in ['Annual_sal','Q1 Payments','Q2 Payments','Q3 Payments','Q4 Payments','Payments Over Base Pay',

          'Total Payments','Base_Pay','Permanent Bonus Pay','Longevity Bonus Pay','Temporary Bonus Pay','Overtime Pay',

          'Other Pay & Adjustments','Other Pay (Payroll Explorer)','Average Health Cost','Average Dental Cost',

          'Average Basic Life','Average Benefit Cost']:

    payroll[i] = payroll[i].str.replace('$','')
payroll = payroll[payroll.Annual_sal != 0]  

payroll = payroll[payroll.Base_Pay != 0]

payroll.Annual_sal = payroll.Annual_sal.astype(float)

payroll.Base_Pay = payroll.Base_Pay.astype(float)
plt.figure(figsize = (12,6))

sns.distplot(payroll.Annual_sal,color = 'darkgreen')
payroll_2015 = payroll[payroll.Year ==2015]

payroll_2016 = payroll[payroll.Year ==2016]
pop_mean_2015 = payroll_2015['Annual_sal'].mean()

pop_std_2015 = statistics.stdev(payroll_2015.Annual_sal)

print("Population Mean: "+str(pop_mean_2015))

print("Population Standard Deviation: "+str(pop_std_2015))
print("Population Mean: "+str(payroll_2016['Annual_sal'].mean()))

payroll_2016_sample = payroll_2016.sample(frac=0.10)

sample_mean_2016 = payroll_2016_sample['Annual_sal'].mean()

print("Sample Mean: "+str(sample_mean_2016))

sample_std_2016 = statistics.stdev(payroll_2016_sample.Annual_sal)

print("Sample Standard Deviation: "+str(sample_std_2016))

pop_std_2016= statistics.stdev(payroll_2016.Annual_sal)

print("Population Standard Deviation: "+str(pop_std_2016))
import math

# Confidence Level 95 %  for one sided Normal curve

zscore_critical = 1.65 

# Calculate the test statistics 

zscore_test_stat = ((sample_mean_2016 - pop_mean_2015)*math.sqrt(8916))/sample_std_2016

print(zscore_test_stat)
# we are basically checking the true value of the population characteristics

pop_mean_2016 = payroll_2016['Annual_sal'].mean()

pop_std_2016 = statistics.stdev(payroll_2016.Annual_sal)



zscore_error = ((pop_mean_2016 - pop_mean_2015)/pop_std_2016)

print(zscore_error)
#Calculating the Sample Parameters**

payroll_2014 = payroll[payroll.Year ==2014]

payroll_2015 = payroll[payroll.Year ==2015]

# Creating Sample distribution for T statistics

payroll_t_2015_sample = payroll_2015.sample(frac=0.00062)
payroll_t_2015_sample = payroll_2015.sample(frac=0.00062)

N = len(payroll_t_2015_sample)

sample_mean_2015 = payroll_t_2015_sample['Annual_sal'].mean()

sample_std_2015 = statistics.stdev(payroll_t_2015_sample.Annual_sal)

pop_std_2014  = statistics.stdev(payroll_2014.Annual_sal)

pop_mean_2014 = payroll_2014['Annual_sal'].mean()
# Confidence Level 95 %  for one sided T curve

t_critical = 1.311



# Calculate the test statistics 

tscore_test_stat = ((sample_mean_2015 - pop_mean_2014)*math.sqrt(N))/sample_std_2015



print(tscore_test_stat)
payroll_2014 = payroll[payroll.Year ==2014]

#Payroll info for the job of Electrician across three years

payroll_2014_elec = payroll_2014[payroll_2014.Job_title == 'Electrician']

payroll_2015_elec = payroll_2015[payroll_2015.Job_title == 'Electrician']

payroll_2016_elec = payroll_2016[payroll_2016.Job_title == 'Electrician']



#Considering 47% of the Data

sample_elec_2014 = payroll_2014_elec.sample(frac=0.47)

sample_elec_mean_2014 = sample_elec_2014['Base_Pay'].mean()

print("Sample Mean 2014 "+str(sample_elec_mean_2014))



#Considering 41% of the Data

sample_elec_2015 = payroll_2015_elec.sample(frac=0.41)

sample_elec_mean_2015 = sample_elec_2015['Base_Pay'].mean()

print("Sample Mean 2015 "+str(sample_elec_mean_2015))



#Considering 22% of the Data

sample_elec_2016 = payroll_2016_elec.sample(frac=0.22)

sample_elec_mean_2016 = sample_elec_2016['Base_Pay'].mean()

print("Sample Mean 2016 "+str(sample_elec_mean_2016))

# Creating the Samples of the base pays over three years

sam_1 = sample_elec_2014.Base_Pay

sam_2 = sample_elec_2015.Base_Pay

sam_3 = sample_elec_2016.Base_Pay
f, p = stats.f_oneway(sam_1, sam_2, sam_3 )

print ('F value:', f)

print ('P value:', p, '\n')
# Transform the qualitative data into vectors

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(stop_words = 'english')

dtm = vect.fit_transform(payroll.Job_title)
#Split the data into training and testing datasets

from sklearn.model_selection import train_test_split



X = dtm

y = payroll.Annual_sal



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100)
from sklearn.ensemble import RandomForestRegressor



clf = RandomForestRegressor()

clf.fit(X_train, y_train)
from sklearn.metrics import r2_score, mean_squared_error



pred_train = clf.predict(X_train)

pred_test = clf.predict(X_test)



print('Root mean Score Training: {}'.format(r2_score(y_train, pred_train)))

print('Root mean Score Testing: {}'.format(r2_score(y_test, pred_test)))
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

plt.plot(np.arange(len(pred_train)), y_train - pred_train,'o')

plt.xlabel('Training')

plt.axhline(0)

plt.subplot(1,2,2)

plt.plot(np.arange(len(pred_test)), y_test - pred_test,'o')

plt.xlabel('Testing')

plt.axhline(0)

plt.tight_layout()
# Selecting the features and creating train test split

y = payroll["Average Benefit Cost"]

X = payroll[['Annual_sal','Q1 Payments','Q2 Payments','Q3 Payments','Q4 Payments']].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

ranked_suburbs = coeff_df.sort_values("Coefficient", ascending = False)

print(ranked_suburbs)
pred_train = lm.predict(X_train)

pred_test = lm.predict(X_test)

print('Root mean Score Training: {}'.format(r2_score(y_train, pred_train)))

print('Root mean Score Testing: {}'.format(r2_score(y_test, pred_test)))