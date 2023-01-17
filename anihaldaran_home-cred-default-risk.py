# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
main_file_path1 = '../input/application_train.csv' # this is the path to the training data that you will use
ATRdata = pd.read_csv(main_file_path1)
ATRdata.dtypes #shows the columns in the application_train.csv table with their corresponding data types

#selecting multiple columns from the applicaton_training dataframe
ATRcolumns_of_interest = ['NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE' ]
ATRcolumns_of_data = ATRdata[ATRcolumns_of_interest]
print(ATRcolumns_of_data)
ATRcolumns_of_data.describe() #displays the count, unique, top value and frequency for the columns of interest in the application_training.csv
main_file_path2 = '../input/application_test.csv' # this is the path to the test data that you will use
ATdata = pd.read_csv(main_file_path2)
ATdata.dtypes #shows the columns in the application_test.csv table with their corresponding data types
#selecting multiple columns from the applicaton_test dataframe
ATcolumns_of_interest = ['NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_HOUSING_TYPE' ]
ATcolumns_of_data = ATdata[ATcolumns_of_interest]
print(ATcolumns_of_data)
ATcolumns_of_data.describe() #displays the count, unique, top value and frequency for the columns of interest in the application_est.csv
import seaborn as sns #for seaborn plotting
sns.jointplot(x='TARGET', y='DAYS_REGISTRATION', data=ATRdata[ATRdata['TARGET'] < 100000]) #jointplot for TARGET (dependent variable) and DAYS_REGISTRATION  (independent variable) for scatter graph and histogram
import seaborn as sns #for seaborn plotting
Bx = ATRdata[ATRdata.NAME_EDUCATION_TYPE.isin(ATRdata.NAME_EDUCATION_TYPE.value_counts().head(3).index)]

sns.boxplot(
    x='NAME_EDUCATION_TYPE',
    y='DAYS_REGISTRATION',
    data=Bx
) #box plot for education type and days of registration
ATRdata.head(5)
from sklearn.tree import DecisionTreeRegressor # used to make predictions from certain data
#factors that will predict the TARGET
desired_factors = ['DAYS_REGISTRATION','AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'DAYS_ID_PUBLISH','DAYS_EMPLOYED','DAYS_BIRTH']

#set my model to DecisionTree
model1 = DecisionTreeRegressor()

#set prediction data to factors that will predict, and set target to TARGET
train_data = ATRdata[desired_factors]
test_data = ATdata[desired_factors]
target = ATRdata.TARGET

#fitting model with prediction data and telling it my target for the application_test.csv data
model1.fit(train_data, target)

model1.predict(test_data)
submission1 = pd.DataFrame({'SK_ID_CURR': ATdata.SK_ID_CURR ,'TARGET': model1.predict(test_data)})

submission1.to_csv('submission1.csv', index=False)
main_file_path3 = '../input/bureau.csv' # this is the path to the test data that you will use
Bdata = pd.read_csv(main_file_path3)
Bdata.head() #shows the first 5 rows of the dataset bureau.csv
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go #for plotly graphs
dtt = Bdata.head(800).assign(n=0).groupby(['DAYS_CREDIT_UPDATE', 'DAYS_CREDIT'])['n'].count().reset_index()
dtt = dtt[dtt["DAYS_CREDIT"] < 2000]
ver = dtt.pivot(index='DAYS_CREDIT', columns='DAYS_CREDIT_UPDATE', values='n').fillna(0).values.tolist()
iplot([go.Surface(z=ver)])
#plotly Surface (the most impressive feature)
#shows the distribution of days of credit against daus of credit update
from plotnine import * #plotline graphs

#shows the most common credit types according to the days of credit
(ggplot(Bdata.head(50))
         + aes('DAYS_CREDIT', 'CREDIT_TYPE')
         + geom_bin2d(bins=20)
         + ggtitle("common credit types according to the days of credit")
)
#The plotnine equivalent of a hexplot, a two-dimensional histogram, is geom_bin2d
x_train = ATRdata['NAME_EDUCATION_TYPE']
x_test = ATdata['NAME_EDUCATION_TYPE']
y=ATRdata['TARGET']
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
text_clf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,10), max_features=10000, lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False, tokenizer=TweetTokenizer().tokenize, stop_words='english')),
                         ('clf', LogisticRegression(random_state=17, C=1.8))])
from sklearn.model_selection import RandomizedSearchCV
parameters = {
               'clf__C': np.logspace(.1,1,10),
 }
gs = RandomizedSearchCV(text_clf, parameters, n_jobs=-1, verbose=3)
text_clf.fit(x_train, y)
predicted = text_clf.predict(x_test)
ATdata['TARGET'] = predicted
submission2 = ATdata[["SK_ID_CURR","TARGET"]]
submission2.to_csv("submission2.csv", index = False)
from plotnine import * #plotline graphs
dta5 = ATRdata.head(1000)

(ggplot(dta5)
     + aes('TARGET', 'DAYS_REGISTRATION')
     + aes(color='TARGET')
     + geom_point()
     + stat_smooth()
     + facet_wrap('NAME_EDUCATION_TYPE')
)
#applying faceting with the categorical variable NAME_EDUCATION_TYPE
from plotnine import * #plotline graphs
dta4 = ATRdata.head(1000)

(
    ggplot(dta4)
        + aes('DAYS_REGISTRATION', 'DAYS_ID_PUBLISH')
        + geom_point()
        + aes(color='DAYS_REGISTRATION')
        + stat_smooth()
)
#plots a line of best fit (logistic regression) along the scatter graph with coloured points
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go #for plotly graphs

iplot([go.Scatter(x=ATRdata.head(1000)['DAYS_REGISTRATION'], y=ATRdata.head(1000)['DAYS_ID_PUBLISH'], mode='markers')])
#basic plotly scatter graph
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go #for plotly graphs

iplot([go.Histogram2dContour(x=ATRdata.head(500)['DAYS_REGISTRATION'], 
                             y=ATRdata.head(500)['DAYS_ID_PUBLISH'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=ATRdata.head(1000)['DAYS_REGISTRATION'], y=ATRdata.head(1000)['DAYS_ID_PUBLISH'], mode='markers')])
# KDE plot (what plotly refers to as a Histogram2dContour) and scatter plot of the same data.
import seaborn as sns

sns.lmplot(x='DAYS_REGISTRATION', y='DAYS_ID_PUBLISH',  markers=['o', 'x', '*'], hue='NAME_EDUCATION_TYPE', 
           data=ATRdata.loc[ATRdata['NAME_EDUCATION_TYPE'].isin(['Higher education', 'Lower secondary', 'Incomplete higher'])], 
           fit_reg=False)
#multivariate scatter plot with markers
model1.predict(test_data)
submission3 = pd.DataFrame({'SK_ID_CURR': ATdata.SK_ID_CURR ,'TARGET': model1.predict(test_data)})

submission3.to_csv('submission3.csv', index=False)