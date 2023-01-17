# This data set is a set with data from a website that tracks user data and whether they clicked an ad on the site.   Using this information

# we will see if we can predict whether a user will click an ad based on some of their characteristics.  



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns
import pandas as pd

df = pd.read_csv("../input/advertising.csv")
### Data exploration!



df.head()
df.info()
df.describe()
###  All the fields seem to be filled.   None of the data points seem to be wildly out of place.   

###  However we'll need to alter the "Gender" and "Clicked on Ad" fields so they are binary.  This will allow logistic regression

genders = {'male':1,'female':0}

df['Gender'] = df['Gender'].map(genders)
clicked = {'no':0,'yes':1}

df['Clicked on Ad'] = df['Clicked on Ad'].map(clicked)
df['Clicked on Ad'].unique()
df['Gender'].unique()
### Good!   Now we are binary we'll look at some visualizations. 



sns.set_style('whitegrid')

sns.distplot(df['Age'], bins = 40, kde = False)
# A nice even distribution.   We can work with this.  

sns.jointplot(df['Age'],df['Area Income'])
# No trend is standing out

sns.jointplot(df['Age'],df['Daily Time Spent on Site'], color= 'red')
# No trend is standing out

sns.jointplot(df['Daily Internet Usage'],df['Daily Time Spent on Site'], color='green', kind='kde')
# Ok, now there seems to be a distinction happening above.  We'll see if males and females are drastically different.

sns.pairplot(df, hue='Gender', palette='bwr')

# Males and females seem to be pretty similar.  We'll colorize based on whether someone clicked on the ad

sns.pairplot(df, hue='Clicked on Ad', palette='bwr')
# Now there is a clear difference between the two emerging.  Before we start machine learning lets see what correlates most with clicking on an ad

df.corr()['Clicked on Ad'].sort_values()
# It seems odd the longer someone is on a website, the less likely they are to click an ad.  Well, knowing that lets try to predict who will click them.

#  We won't worry about the time stamps, location, or the ad main line.   At some point we could probably investigate the ad line with NLP.

X = df[['Daily Time Spent on Site', 'Age', 'Area Income',

       'Daily Internet Usage']]

y = df['Clicked on Ad']
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
log_model = LogisticRegression()
log_model.fit(X_train,y_train)
predictions = log_model.predict(X_test)
# Now that we have our model, we can make our predictions

print(classification_report(y_test,predictions))
#  With our predictions above, we can see there is about an 88% accuracy rate in the model.  We will next take a look false positives and false negatives.



print(confusion_matrix(y_test,predictions))
# Based on our model we are a little more accurate in predicting a click. I would need to see the website. It seems odd to me the longer

# someone is on the site the less likely they are to click on an ad. There might be an issue with different browsers like OS vs Android.  

# That would be the first thing I would investigate.  It doesn't suprise me, however, the more internet a person uses, the less likely they are to

# click an ad. 