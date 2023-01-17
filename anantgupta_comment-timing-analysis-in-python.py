# Importing Libraries 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import networkx as nx

import matplotlib.pyplot as plt

import datetime

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Read the input files

comments=pd.read_csv("../input/comment.csv")

likes=pd.read_csv("../input/like.csv")

members=pd.read_csv("../input/member.csv")

posts=pd.read_csv("../input/post.csv")
def getDayOfWeek(timeArg):

	dayOfWeek=datetime.datetime.strptime(timeArg, "%Y-%m-%d %H:%M:%S").weekday()

	return(dayOfWeek)

	#return('Monday' if dayOfWeek==0 else 'Tuesday' if dayOfWeek==1 else 'Wednesday' if dayOfWeek==2 else 'Thursday' if dayOfWeek==3 else 'Friday' if dayOfWeek==4 else 'Saturday' if dayOfWeek==5 else 'Sunday')



def getDayOfWeekfromIndex(dayOfWeek):

	return('Monday' if dayOfWeek==0 else 'Tuesday' if dayOfWeek==1 else 'Wednesday' if dayOfWeek==2 else 'Thursday' if dayOfWeek==3 else 'Friday' if dayOfWeek==4 else 'Saturday' if dayOfWeek==5 else 'Sunday')





# Let us see the comment distribution by Day of the week

commentTimings=pd.merge(comments[['gid','pid','cid','msg','id','name']],posts[['pid','gid','timeStamp','name']],left_on=['pid','gid'],right_on=['pid','gid'])

commentTimings.columns=['gid','pid','cid','response','id','commentedBy','timeStamp','postedBy']

commentTimings['Hour']=commentTimings['timeStamp'].map(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour) 

commentTimings['DayOfWeek']=commentTimings['timeStamp'].map(getDayOfWeek)

commentTimingPlot=commentTimings.groupby(['DayOfWeek'])['response'].count()

commentTiming=pd.DataFrame(commentTimingPlot.values,columns=['NumberofComments'])

commentTiming['DayOfWeek']=commentTimingPlot.index.values

commentTiming['DayOfWeek']=commentTiming['DayOfWeek'].map(getDayOfWeekfromIndex)

N=len(commentTiming)

plt.figure()

plt.gcf().subplots_adjust(bottom=0.35)

p1=plt.bar(range(N),commentTiming['NumberofComments'])

plt.ylabel('Number of Comments')

plt.title('Day wise distribution')

plt.xticks(range(N),commentTiming['DayOfWeek'],rotation=75)

plt.show()

# Let us do the same analysis with Hours

p1=commentTimings.groupby(['Hour'])['response'].count().plot()

plt.ylabel('Number of Comments')

plt.title('Hour wise distribution')

plt.show()
# We will now be doing a regression model for predicting the number of likes or comments a person will get on average



# Inputs HOUR, DAY OF WEEK

# Output : Number of Comments



# I will be breaking it into test and train data set

commentsGroup=commentTimings.groupby(['Hour','DayOfWeek'])['response'].count()

commentsdf=pd.DataFrame(commentsGroup.values,columns=['Count'])

commentsdf['Val1']=commentsGroup.index.values

commentsdf['Day']=commentsdf.apply(lambda x:x['Val1'][1],axis=1)

commentsdf['Hour']=commentsdf.apply(lambda x:x['Val1'][0],axis=1)

del(commentsdf['Val1'])





import statsmodels.api as sm

X=commentsdf[['Day','Hour']]

Y=commentsdf['Count']



# Train and Test Set

from sklearn.cross_validation import train_test_split

from sklearn import datasets, linear_model

import matplotlib.patches as mpatches

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)



regr=linear_model.LinearRegression()

regr.fit(X_train ,Y_train)

print("Residual sum of squares: %.2f"%np.mean((regr.predict(X_test) - Y_test) ** 2))



plt.figure()

plt.scatter(X_test['Day'], X_test['Hour'], s=Y_test,c='r',alpha=0.5)

plt.scatter(X_test['Day'], X_test['Hour'], s=regr.predict(X_test),c='b',alpha=0.5)

plt.xlabel('Day of Week')

plt.ylabel('Hour of the Day')

plt.title('Difference between regression and actual')

red_patch = mpatches.Patch(color='red', label='Actual Data')

blue_patch = mpatches.Patch(color='blue', label='Regressed Data')

plt.legend(handles=[red_patch,blue_patch])

plt.show()



# We can see that the regression has failed miserable in a few cases. Especially before noon time. After noon, the predictions have been correct

# This means that we have to add some other features. Let us hunt for them



# Let me try with polynomial regression

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

degree=3

model = make_pipeline(PolynomialFeatures(degree), Ridge())

model.fit(X_train ,Y_train)

print("Residual sum of squares using Polynomial of degree 3: %.2f"%np.mean((model.predict(X_test) - Y_test) ** 2))



plt.figure()

plt.scatter(X_test['Day'], X_test['Hour'], s=Y_test,c='r',alpha=0.5)

plt.scatter(X_test['Day'], X_test['Hour'], s=model.predict(X_test),c='b',alpha=0.5)

plt.xlabel('Day of Week')

plt.ylabel('Hour of the Day')

plt.title('Difference between polynomial regression and actual')

red_patch = mpatches.Patch(color='red', label='Actual Data')

blue_patch = mpatches.Patch(color='blue', label='Regressed Data')

plt.legend(handles=[red_patch,blue_patch])

plt.show()



# Wow good improvement. Let us try with degree 4

# Let me try with polynomial regression

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

degree=4

model = make_pipeline(PolynomialFeatures(degree), Ridge())

model.fit(X_train ,Y_train)

print("Residual sum of squares using Polynomial of degree 4: %.2f"%np.mean((model.predict(X_test) - Y_test) ** 2))



plt.figure()

plt.scatter(X_test['Day'], X_test['Hour'], s=Y_test,c='r',alpha=0.5)

plt.scatter(X_test['Day'], X_test['Hour'], s=model.predict(X_test),c='b',alpha=0.5)

plt.xlabel('Day of Week')

plt.ylabel('Hour of the Day')

plt.title('Difference between polynomial regression and actual')

red_patch = mpatches.Patch(color='red', label='Actual Data')

blue_patch = mpatches.Patch(color='blue', label='Regressed Data')

plt.legend(handles=[red_patch,blue_patch])

plt.show()



# Not a lot of improvement between 3 and 4



# Let us try something else