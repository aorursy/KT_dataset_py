import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import pearsonr
#
# read DATA into DataFrame object
#
df = pd.read_csv("../input/seattleWeather_1948-2017.csv", encoding = "ISO-8859-1")

#
# convert to SI units
#
df['PRCP'] = df['PRCP'].apply(lambda x: round(x * 25.4, 1))
df['TMAX'] = df['TMAX'].apply(lambda x: round((x - 32) * 5 / 9))
df['TMIN'] = df['TMIN'].apply(lambda x: round((x - 32) * 5 / 9))

#
# drop NAs
#
df = df.dropna(how='any')
#
# show outliers for Tmin and Tmax columns
#
plt.boxplot(x=[df['TMAX'], df['TMIN']], labels=['TMAX', 'TMIN'])
plt.ylabel('T in C')
plt.show()

#
# remove outliers in the Tmin and Tmax columns
# outliers are defined as values below or above mean -/+ 2.5 times standard deviation
#
z = 2.5
avg_tmax = df['TMAX'].mean()
std_tmax = df['TMAX'].std()
low_tmax_cutoff = avg_tmax - z * std_tmax
high_tmax_cutoff = avg_tmax + z * std_tmax
df = df.drop(df[df['TMAX']<low_tmax_cutoff].index) #removes 66 rows
df = df.drop(df[df['TMAX']>high_tmax_cutoff].index) #removes 94 rows
avg_tmin = df['TMIN'].mean()
std_tmin = df['TMIN'].std()
low_tmin_cutoff = avg_tmin - z * std_tmin
high_tmin_cutoff = avg_tmin + z * std_tmin
df = df.drop(df[df['TMIN']<low_tmin_cutoff].index) #removes 189 rows
df = df.drop(df[df['TMIN']>high_tmin_cutoff].index) #remove 3 rows
plt.boxplot(x=[df['TMAX'], df['TMIN']], labels=['TMAX', 'TMIN'])
plt.ylabel('T in C')
plt.show()

#
# TMIN and TMAX co-vary. Graph shows correlation
# and value of Pearson's r coefficient
#
def show_correlation_Tmin_Tmax(df):
    str = 'Pearson r = {:.3f}'.format(pearsonr(df['TMAX'], df['TMIN'])[0])
    style.use('fivethirtyeight')
    plt.scatter(df['TMAX'], df['TMIN'], s=1.5)
    plt.ylabel('TMIN')
    plt.xlabel('TMAX')
    plt.figtext(0.1, 0.9, str)
    plt.show()
    return
show_correlation_Tmin_Tmax(df)

#
# Add index corresponding to the date.
#
df['ind'] = pd.to_datetime(df['DATE'])
df.set_index('ind', inplace=True)

#
# Polar projection graph to show seasonality of the data
# 
# Each bar corresponds to the rain frequency for the indicated 
# month and in a given year, from 1948 to 2017.There are thus
# 70 bars per month, for a total of 840 monthly averages shown.
# 
# The rain frequency is equal to the number of days it rained 
# divided by the total number of days in the month.  
#
rain_freq = df['RAIN'].resample('M').agg(lambda x: round(x.sum()/x.count(), 2))
tmax = round(df['TMAX'].resample('M').mean(), 1)
tmax_monthly_averages = []
rain_monthly_freq = []
for i in range(0, 12):
    tmax_monthly_averages += [ tmax[a] for a in range(i, len(tmax), 12) ]
    rain_monthly_freq += [ rain_freq[b] for b in range(i, len(rain_freq), 12) ]

def polar_projection(radii, title, color, rticks, max_rtick):
    n = len(radii)
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    width = (2 * np.pi) / n
    months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax = plt.subplot(111, projection='polar')
    ax.set_title(title, loc='center', fontsize=12)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(range(0,360,30), labels=months_labels, fontsize=10)
    ax.set_theta_zero_location('N')
    ax.set_rgrids(rticks, fontsize=10, angle=165.0)
    ax.bar(theta, radii, width=width, bottom=min(radii), color=color, alpha=1.0)
    # set_rmax() must be called after bar() since the latter sets rmax to a calculated value
    ax.set_rmax(max_rtick)
    for i in range(0, n, 70):
        ax.annotate("1948", xy=(theta[i+8], 0.92*max_rtick), fontsize=6)
        ax.annotate("2017", xy=(theta[i+62], 0.92*max_rtick), fontsize=6)
    plt.show()
    return

polar_projection(rain_monthly_freq, 'Monthly Rain Frequency', color='b', rticks=[ 0, 0.2, 0.4, 0.6, 0.8, 1.0 ], max_rtick=1.1)

#
# similar projection for TMAX
#
min_tick = round(min(tmax) / 5) * 5
max_tick = (ceil(max(tmax) / 5)) * 5
interval = round((max_tick - min_tick) / 5)
rticks = [ x for x in range(min_tick, max_tick+1, interval) ]
polar_projection(tmax_monthly_averages, 'Average TMax', color='r', rticks=rticks, max_rtick=max_tick+interval)

#
# splitting of data into Dry (June-September) and Rainy (October-May) seasons
#
df_rainy = df[(df.index.month <= 5) | (df.index.month >= 10)]
df_dry = df[(df.index.month >= 6) & (df.index.month <= 9)]

#
# generation of the dependent variable (does it rain?  True or False = X)
# and the independent variable (predictor = Tmax = y)
# Tmin is not used because it is highly correlated to Tmax
#
# X1, y1 = year round
# X2, y2 = rainy season
# X3, y3 = dry season
#

df = df.reset_index()
df_rainy = df_rainy.reset_index()
df_dry = df_dry.reset_index()

X1 = df.drop(['ind', 'DATE', 'PRCP', 'RAIN', 'TMIN'], axis=1)
X2 = df_rainy.drop(['ind', 'DATE', 'PRCP', 'RAIN', 'TMIN'], axis=1)
X3 = df_dry.drop(['ind', 'DATE', 'PRCP', 'RAIN', 'TMIN'], axis=1)

y1 = df['RAIN'].values.astype('int')
y2 = df_rainy['RAIN'].values.astype('int')
y3 = df_dry['RAIN'].values.astype('int')
#
# Results of Logistic Regression for the 3 datasets
#
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def glm_logistic_regression(name_test, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    c = confusion_matrix(y_test, prediction)
    n = len(X_test)
    print(name_test)
    print('n in test sample=', n)
    print('---accuracy:', round((c[0,0]+c[1,1])/n, 4))
    print('---false positives (predicted rain when there was none):', round(c[0,1]/n, 4))
    print('---false negatives (failed to predict it would rain):', round(c[1,0]/n, 4))
    print()
    return

glm_logistic_regression('YEAR ROUND', X1, y1)
glm_logistic_regression('RAINY SEASON - OCTOBER TO MAY', X2, y2)
glm_logistic_regression('DRY SEASON - JUNE TO SEPTEMBER', X3, y3)
