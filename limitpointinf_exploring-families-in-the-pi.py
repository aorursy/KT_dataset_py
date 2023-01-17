import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import math

from sklearn import preprocessing, cross_validation, svm

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

import graphviz



sns.set()



df = pd.read_csv('../input/Family Income and Expenditure.csv')



#ECDF function

def ecdf(data):

    n=len(data)

    x=np.sort(data)

    y=np.arange(1,n+1)/n

    return x,y



#target

col_name = 'Total Household Income'

income_thousands = df[col_name]/1000
#histogram

plt.figure(figsize=(11, 7), dpi=200)

plt.hist(income_thousands, bins=250)

plt.xlabel('Thousands')

plt.ylabel('Count')

plt.show()
#get theoretical and real means/std-deviation

mean = np.mean(income_thousands)



#generate theoretical samples

samples = np.random.exponential(mean, size=len(income_thousands))



#get ecdf of both

x_theor, y_theor = ecdf(samples)

x, y = ecdf(income_thousands)



#plot

plt.figure(figsize=(11, 7), dpi=200)

m_size = 3

c_theor = 'red'

c_real = 'green'

plt.plot(x_theor, y_theor, marker='.', linestyle='none',ms=m_size, color=c_theor)

plt.plot(x, y, marker='.', linestyle='none',ms=m_size, color=c_real)

plt.xlabel('Total Household Income (thousands)')

plt.ylabel('ECDF')

plt.legend(('theoretical: red', 'real: green'))



plt.show()



print("The mean of income distribution in thousands is: " + str(round(mean,3)))
p_list = [25,50,75]

percentiles = np.percentile(income_thousands,p_list)



for i in range(len(percentiles)):

    print(str(p_list[i]) + "th percentile: " + str(percentiles[i]))

#create percentages

wealth_perc = income_thousands/sum(income_thousands)



#plot wealth distribution

plt.figure(figsize=(11, 7), dpi=200)

plt.plot(income_thousands, wealth_perc, marker='.', linestyle='none')

plt.xlabel('Total Household Income (thousands)')

plt.ylabel('Percentage of Wealth')

plt.show()
print("The amount of households with income greater than 9 million PHP: " 

      + str(sum(income_thousands > 9000)))



income_sorted = income_thousands.sort_values(ascending=False)

wealth_perc = income_sorted/sum(income_sorted)

iw = pd.DataFrame(dict(income=income_sorted, percentage=wealth_perc)).reset_index()

iw = iw.drop(['index'],1)



count = 0

tot_perc = 0

for i in range(len(iw.index)):

    tot_perc += iw.percentage[i]

    if tot_perc <= 0.50:

        count += 1

        if count == 3:

            print("These households control " + str(round(tot_perc*100,3)) + "% of the wealth in our sample")

    else:

        break

print("Number of wealthy households that give that control 50% of the sample's wealth is: " 

      + str(count))

print("Percentage of sample: " + str(round(count/len(iw.index) * 100,2)) + "%")     

keep = [x for x in df.columns if df[x].dtype == 'int64']

num_df = df[keep]



X = np.array(num_df.drop([col_name],1))

y = np.array(num_df[col_name])



accuracy = []

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) 

clf = LinearRegression(n_jobs = -1)

clf.fit(X_train, y_train)

accuracy = clf.score(X_test,y_test)



print(accuracy)
#filter out non-numeric columns

cat_col_name = [x for x in df.columns if df[x].dtype != 'int64']

print(cat_col_name)



#convert non-numeric columns to categoricals

for i in cat_col_name:

    df[i] = df[i].astype('category')

le = preprocessing.LabelEncoder()

le.fit(df.Region)

print(list(le.classes_))

print(le.transform(df.Region) )

print(len(df.Region.cat.categories))



#split train-test

X_train, y_train, X_test, y_test = train_test_split(df.iloc[:,1:],df[col_name], random_state=42)





#clf = DecisionTreeClassifier(random_state=0)

#clf.fit(X_train, y_train)


