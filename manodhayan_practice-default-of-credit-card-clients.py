# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
creditCard_csv = '/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv'



creditCard_df = pd.read_csv(creditCard_csv)

target = 'default.payment.next.month'

creditCard_df.head()

creditCard_df.info()
creditCard_df.describe()
def drawCountOnBar(axes, orient = "v"):

    for p in axes.patches:

        if orient == "v":

            height = p.get_height()

            axes.text(x = p.get_x()+p.get_width()/2., y = height + 1 ,s = height ,ha="center")

        else:

            width = p.get_width()

            axes.text(x = p.get_x() + width, y = p.get_y() + p.get_height()/2 ,s = width ,ha="left")
defaultPaymentsCount = creditCard_df[target].value_counts()

defaultPaymentsCount = defaultPaymentsCount.rename(index = {0: "Non - Default", 1: "Default"})

axes = defaultPaymentsCount.nlargest().plot(kind='bar', title='default payment distribution')

axes.set_xlabel('default payment')

axes.set_ylabel('Count')

drawCountOnBar(axes)



print(defaultPaymentsCount.nlargest())

print("default.payment.next.month is {}%".format(100 * defaultPaymentsCount[1]/sum(defaultPaymentsCount)))
fig=plt.figure(figsize=(20,15))

axes=fig.add_axes([0,0,0.8,0.8])

sns.heatmap(creditCard_df.corr(),annot=True,cmap="viridis")
column = 'LIMIT_BAL'

limitBalanceCounts = creditCard_df[column].value_counts()



axes = limitBalanceCounts.nlargest(n=10).plot(kind='barh', title='Credit limit issued to Customers (Top 10)')

axes.set_xlabel('Count')

axes.set_ylabel('Credit limit')

drawCountOnBar(axes, orient = "h")
print("Correlation of LIMIT BAL: {}".format(creditCard_df.corr()[target][column]))
axes = limitBalanceCounts.nsmallest(n=10).plot(kind='barh', title='Credit limit issued to Customers (Bottom 10)')

axes.set_xlabel('Count')

axes.set_ylabel('Credit limit')

drawCountOnBar(axes, orient = "h")
sns.boxplot(x = target, y= column, data = creditCard_df)
column = 'SEX'

genderCounts = creditCard_df[column].value_counts()

genderCounts = genderCounts.rename(index = {1: "Male", 2: "Female"})



axes = genderCounts.nlargest().plot(kind='barh', title='Gender distribution')

axes.set_xlabel('Count')

axes.set_ylabel('Gender')



drawCountOnBar(axes, orient = "h")
axes = sns.countplot(x=target, data=creditCard_df, hue='SEX');



axes.set_title("Gender - Default Payment Next month")

axes.set_ylabel("Count")

axes.set_xlabel("Default Payment Next month")

axes.set_xticklabels(['Yes','No'])

drawCountOnBar(axes, orient = "v")

column = 'EDUCATION'

educationCounts = creditCard_df[column].value_counts()



educationCounts = educationCounts.rename(index = {1 : "graduate school", 2 : "university", 3 : "high school", 4 : "others", 5 : "unknown", 6 : "unknown"})

axes = educationCounts.nlargest(n=10).plot(kind='bar', title='Stats of Education Categories')

axes.set_xlabel('Count')

axes.set_ylabel('Education')

drawCountOnBar(axes, orient = "v")

plt.show()





axes = sns.countplot(x=column, data=creditCard_df, hue='SEX');

axes.set_title('Stats of Education Categories by Sex')

drawCountOnBar(axes, orient = "v")
column = 'MARRIAGE'

educationCounts = creditCard_df[column].value_counts()



educationCounts = educationCounts.rename(index = {1 : "Married", 2 : "Single", 3 : "Others"})

axes = educationCounts.nlargest(n=10).plot(kind='barh', title='Stats of categories in Marital Status')

axes.set_xlabel('Count')

axes.set_ylabel('Marriage')

drawCountOnBar(axes, orient = "h")

plt.show()
column = 'AGE'

print(creditCard_df[column].describe())



educationCounts = creditCard_df[column].value_counts()



axes = educationCounts.nlargest(n=10).plot(kind='barh', title='Stats of Clients by Age (Top 10)')

axes.set_xlabel('Count')

axes.set_ylabel('Age')

drawCountOnBar(axes, orient = "h")

plt.show()



axes = educationCounts.nsmallest(n=10).plot(kind='barh', title='Stats of Clients by Age (Lower 10)')

axes.set_xlabel('Count')

axes.set_ylabel('Age')

drawCountOnBar(axes, orient = "h")

plt.show()
Pay = ["PAY_0"] + ["PAY_{}".format(index) for index in range(2, 7)]



for pay in Pay:

    plot_graph = sns.FacetGrid(creditCard_df, col=target)

    plot_graph.map(sns.countplot, pay)

    plt.show()