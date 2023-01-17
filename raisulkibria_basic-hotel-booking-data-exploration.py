import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
df = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
df.head()
df.describe(include = 'all')
df.dtypes
df.isna().sum()
df['country']=df['country'].fillna('NRF')
df['agent'] = df['agent'].fillna(0.0)
df['company'] = df['company'].fillna(0.0)
df['children'] = df['children'].fillna(int(df['children'].mean()))
df.isna().sum()
corr_matrix = df.corr()
corr_matrix
df.columns
df.plot(kind = 'box', figsize = (9, 5))
plt.show()
fig = plt.figure(figsize=(12, 4))
year_freq = df['arrival_date_year'].value_counts().sort_values().to_dict()
year, freq = zip(*year_freq.items())
ytick = []
for i in range(len(year)):
    month_f = df[['arrival_date_year', 'arrival_date_month']]
    month_f = month_f.groupby(['arrival_date_year']).get_group(year[i])
    month_f = pd.Categorical(month_f['arrival_date_month'], categories = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', "December"])
    month_f = month_f.value_counts().to_dict()
    month, m_freq = zip(*month_f.items())
    ytick.append(m_freq)
    plt.plot(month, m_freq)
ytick = np.array(ytick).reshape(3,12)
for j in range(3):
    for i in range(12):
        if ytick[j, i]!=0:
            plt.text(i, ytick[j, i], str(ytick[j, i]))

plt.legend(year)
plt.show()
country_freq = df['country'].value_counts().iloc[0:10].to_dict()
country, freq = zip(*country_freq.items()) 
plt.bar(country, freq)
plt.xticks(country)
peak = plt.plot(np.arange(10), freq, 'r.')
for i in range(len(freq)):
    plt.text(i, freq[i]+1000, str(freq[i]))
plt.show()
fig = plt.figure(figsize=(5,6))
wait = df[['days_in_waiting_list','arrival_date_day_of_month']]
wait = wait.groupby(['arrival_date_day_of_month']).sum().to_dict()
x, y = zip(*wait['days_in_waiting_list'].items())
plt.barh(x, y)
plt.yticks(x)
plt.show()
dist_chnl = df['distribution_channel'].value_counts()
dist_chnl.iloc[:4].plot.pie(figsize=(5,5), fontsize =10)
fig = plt.figure(figsize=(60,7))
c = df[['country', 'is_canceled']].groupby('country').sum().sort_values(by='is_canceled', ascending=False).to_dict()
country_freq = df['country'].value_counts().to_dict()
for _, (ctr, cnc) in enumerate(c['is_canceled'].items()):
    for _, (ctr2, freq) in enumerate(country_freq.items()):
        if (ctr == ctr2):
            cnc = cnc/freq
            if(cnc>.5):
                plt.bar(ctr, cnc)
                print(ctr, cnc)
dt_type = df.dtypes.to_dict()
cat = []
for _, (i, j) in enumerate(dt_type.items()):
    if j == object:
        cat.append(str(i))
print(cat, len(cat))
for col in cat:
    df[col] = df[col].astype('category')
    df[col] = df[col].cat.codes
df.dtypes
X = df[['hotel', 'lead_time','arrival_date_year', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month', 'adults', 'children', 'babies',
       'country', 'is_repeated_guest', 'previous_cancellations', 'booking_changes', 'deposit_type', 'agent', 'company', 'days_in_waiting_list', 'customer_type']]
Y = df['is_canceled']
X.head()
print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2)
print(X_train.shape)
print(X_test.shape)

print(Y_train.shape)
print(Y_test.shape)
clf1 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf1.fit(X_train, Y_train)
score1 = clf1.score(X_test, Y_test)
score1
clf2 =  make_pipeline(StandardScaler(), MLPClassifier(alpha=.001, max_iter=2000))
clf2.fit(X_train, Y_train)
score2 = clf2.score(X_test, Y_test)
score2
clf3 = RandomForestClassifier()
clf3.fit(X_train, Y_train)
score3 = clf3.score(X_test, Y_test)
score3