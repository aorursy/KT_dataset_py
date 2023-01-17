import re

import pandas as pd 

from IPython.display import display

import seaborn as sns

sns.set(style="white")
df = pd.read_csv('../input/WinnersInterviewBlogPosts.csv')
df.head()
len(df)
## blatantly stealing code from my high-scoring peers

df.index = pd.to_datetime(df['publication_date'])

del(df['publication_date'])
type('svm')
# Copying code, but adding some of my own thoughts to make it 'better'

def check_for(the_string_obj):

    check = 0

    if type(the_string_obj) == list:

        for item in the_string_obj:

            check += df['content'].str.count(item,flags=re.IGNORECASE)

    elif type(the_string_obj) == str:            

        check += df['content'].str.count(the_string_obj,flags=re.IGNORECASE)

    return check



df['SVM'] = check_for(['SVM','support vector machine']) > 0

df['Random Forest'] = check_for(['random forest','randomforest']) > 0

df['Neural Net'] = check_for(['deep learning','neural net','CNN']) > 0

df['GBM'] = check_for(['gbm','gradient boosting machine','xgboost']) > 0

df['python'] = check_for('python') > 0

df['lasso'] = check_for('lasso') > 0

df['ensemble'] = check_for('ensemble') > 0

df['logistic'] = check_for('logistic regression') > 0

df['wine'] = check_for(['wine','beer','alcohol','drunk']) > 0

df['hadoop'] = check_for(['hadoop', 'hive', 'spark']) > 0

df['R'] = check_for([' R ', 'CRAN']) > 0



winners_by_method = df[['SVM','Random Forest','Neural Net',

                        'GBM', 'python','lasso', 'ensemble',

                        'logistic','wine', 'hadoop', 'R']].sum()/len(df)

ax = winners_by_method.plot(kind='bar')

ax.set_ylabel("% of posts")
#notice that we had few posts in 2013 and 2014

df_groupby = df.groupby(df.index.year)

df_groupby['title'].count()

#group 2013 and 2014 because we had so few posts for those two years

df_groupby_sum = df_groupby[['SVM','Random Forest','Neural Net',

                        'GBM', 'python','lasso', 'ensemble',

                        'logistic','wine', 'hadoop','R']].sum()

df_groupby_sum[df_groupby_sum.index == 2013] = df_groupby_sum[df_groupby_sum.index == 2013].values + df_groupby_sum[df_groupby_sum.index == 2014].values

df_groupby_sum = df_groupby_sum.drop([2010,2014])

df_groupby_sum = df_groupby_sum.set_index([['2011','2012','2013 & 2014','2015','2016']])



df_groupby_count = df_groupby[['SVM','Random Forest','Neural Net',

                        'GBM', 'python','lasso', 'ensemble',

                        'logistic','wine', 'hadoop','R']].count()

df_groupby_count[df_groupby_count.index == 2013] = df_groupby_count[df_groupby_count.index == 2013].values + df_groupby_count[df_groupby_count.index == 2014].values

df_groupby_count = df_groupby_count.drop([2010,2014])

df_groupby_count = df_groupby_count.set_index([['2011','2012','2013 & 2014','2015','2016']])
ax1 = (df_groupby_sum/df_groupby_count).plot()

ax1.set_ylabel = '% of posts'