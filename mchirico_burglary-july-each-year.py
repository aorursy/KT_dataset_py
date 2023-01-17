import numpy as np

import pandas as pd

import datetime



import seaborn as sns

sns.set(style="white", color_codes=True)



import matplotlib.pyplot as plt

import matplotlib.lines as mlines

import sqlite3







df = pd.read_csv("../input/crime.csv")

c = sqlite3.connect(":memory:")

df.to_sql("crime",c)
sql="""

select strftime('%Y-%m', dispatch_date_time) month,

sum('Burglary Non-Residential' == text_general_code) as 'Burglary Non-Residential',

sum('Burglary Residential' == text_general_code) as 'Burglary Residential'

from crime 

where text_general_code like 'Burglary%'

and strftime('%m', dispatch_date_time) = '07'

group by month

order by dispatch_date_time



"""



d = pd.read_sql(sql, c)

d
# Convert Month to month-datetime

import datetime

d['month-datetime'] = d['month'].apply(lambda x: datetime.datetime.strptime(x+'-01 00:00:00','%Y-%m-%d %H:%M:%S') )
from sklearn import linear_model



def plotCat(cat='Burglary Residential',title=''):



    fig, ax = plt.subplots()



    ax.spines["top"].set_visible(False)    

    ax.spines["bottom"].set_visible(False)    

    ax.spines["right"].set_visible(False)    

    ax.spines["left"].set_visible(False)  





    ax.get_xaxis().tick_bottom()    

    ax.get_yaxis().tick_left() 

    plt.xticks(fontsize=12) 





    ax.plot(d['month-datetime'], d[cat],'k')

    ax.plot(d['month-datetime'], d[cat],'ro')



    # Adjust Data

    Y = d[cat].values.reshape(-1,1)

    X=np.arange(Y.shape[0]).reshape(-1,1)



    # Build Linear Fit

    model = linear_model.LinearRegression()

    model.fit(X,Y)

    m = model.coef_[0][0]

    c = model.intercept_[0]

    ax.plot(d['month-datetime'],model.predict(X), color='blue',

         linewidth=2)

    blue_line = mlines.Line2D([], [], color='blue', label='Linear Fit: y = %2.2fx + %2.2f' % (m,c))





    # Robustly fit linear model with RANSAC algorithm

    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),random_state=23)

    model_ransac.fit(X, Y)



    mr = model_ransac.estimator_.coef_[0][0]

    cr = model_ransac.estimator_.intercept_[0]



    ax.plot(d['month-datetime'],model_ransac.predict(X), color='green',

             linewidth=2)



    green_line = mlines.Line2D([], [], color='green', label='RANSAC Fit: y = %2.2fx + %2.2f' % (mr,cr))





    ax.legend(handles=[blue_line,green_line], loc='best')

    ax.set_title(title)

    fig.autofmt_xdate()

    plt.show()

    

plotCat(cat='Burglary Residential',title='Burglary Residential (July Months)')







sql="""

select strftime('%Y-%m', dispatch_date_time) month,

sum('Burglary Non-Residential' == text_general_code) as 'Burglary Non-Residential',

sum('Burglary Residential' == text_general_code) as 'Burglary Residential'

from crime 

where text_general_code like 'Burglary%'

and strftime('%m', dispatch_date_time) = '06'

group by month

order by dispatch_date_time



"""



d = pd.read_sql(sql, c)

d
# Convert Month to month-datetime

import datetime

d['month-datetime'] = d['month'].apply(lambda x: datetime.datetime.strptime(x+'-01 00:00:00','%Y-%m-%d %H:%M:%S') )
plotCat(cat='Burglary Residential',title='Burglary Residential (June Months)')