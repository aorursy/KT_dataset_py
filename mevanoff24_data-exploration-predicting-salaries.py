%matplotlib inline
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
%matplotlib inline
conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()

def read_from_db():
    c.execute("SELECT * FROM salaries LIMIT 10")
    for row in c.fetchall():
        print(row)
        
        
# read_from_db()      
      
c.close()
conn.close()
dat = pd.read_csv('../input/Salaries.csv')
dat.drop(['Notes', 'Agency'], axis = 1, inplace = True)
dat['Event'] = 1
dat.head(2)
dat[['Year', 'TotalPay']].groupby('Year').mean()
year2011 = dat[dat.Year == 2011]
year2012 = dat[dat.Year == 2012]
year2013 = dat[dat.Year == 2013]
year2014 = dat[dat.Year == 2014]

plt.figure(figsize=(10,4))
ax = plt.boxplot([year2011.TotalPay, year2012.TotalPay, year2013.TotalPay, year2014.TotalPay])
plt.ylim(0, 250000)
plt.title('Boxplot of Total Pay By Year')
plt.tight_layout()
def find_job_title(row):
    
    police_title = ['police', 'sherif', 'probation', 'sergeant', 'officer', 'lieutenant']
    fire_title = ['fire']
    transit_title = ['mta', 'transit']
    medical_title = ['anesth', 'medical', 'nurs', 'health', 'physician', 'orthopedic', 'pharm', 'care']
    court_title = ['court', 'legal']
    automotive_title = ['automotive', 'mechanic', 'truck']
    engineer_title = ['engineer', 'engr', 'eng', 'program']
    general_laborer_title = ['general laborer', 'painter', 'inspector', 'carpenter', 
                             'electrician', 'plumber', 'maintenance']
    aide_title = ['aide', 'assistant', 'secretary', 'attendant']
    
    for police in police_title:
        if police in row.lower():
            return 'police'    
    for fire in fire_title:
        if fire in row.lower():
            return 'fire'
    for aide in aide_title:
        if aide in row.lower():
            return 'assistant'
    for transit in transit_title:
        if transit in row.lower():
            return 'transit'
    for medical in medical_title:
        if medical in row.lower():
            return 'medical'
    if 'airport' in row.lower():
        return 'airport'
    if 'worker' in row.lower():
        return 'social worker'
    if 'architect' in row.lower():
        return 'architect'
    for court in court_title:
        if court in row.lower():
            return 'court'
    if 'major' in row.lower():
        return 'mayor'
    if 'librar' in row.lower():
        return 'library'
    if 'guard' in row.lower():
        return 'guard'
    if 'public' in row.lower():
        return 'public works'
    if 'attorney' in row.lower():
        return 'attorney'
    if 'custodian' in row.lower():
        return 'custodian'
    if 'account' in row.lower():
        return 'account'
    if 'garden' in row.lower():
        return 'gardener'
    if 'recreation' in row.lower():
        return 'recreation leader'
    for automotive in automotive_title:
        if automotive in row.lower():
            return 'automotive'
    for engineer in engineer_title:
        if engineer in row.lower():
            return 'engineer'
    for general_laborer in general_laborer_title:
        if general_laborer in row.lower():
            return 'general laborer'
    if 'food serv' in row.lower():
        return 'food service'
    if 'clerk' in row.lower():
        return 'clerk'
    if 'porter' in row.lower():
        return 'porter' 
    if 'analy' in row.lower():
        return 'analyst'
    if 'manager' in row.lower():
        return 'manager'
    else:
        return 'other'
    
dat['CombJobTitle'] = dat['JobTitle'].map(find_job_title)
plt.figure(figsize=(16,5))
sns.countplot('CombJobTitle', data = dat)
plt.xticks(rotation = 45)
plt.tight_layout()
plt.figure(figsize=(16,5))
sns.countplot('CombJobTitle', data = dat, hue = 'Year')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.figure(figsize=(16,5))
sns.countplot('CombJobTitle', data = dat, hue = 'Status')
plt.xticks(rotation = 45)
plt.tight_layout()
def find_name(name):
    name_map = dat.EmployeeName.map(lambda x: 1 if name in x.lower() else 0)
    df = (dat[name_map == 1])
    print('Total Amount of {} in DataSet: {}'.format(name.upper(), len(df)))
    print('{} Total Pay: {}'.format(name.upper(), df.TotalPay.sum()))
    print('Average Pay for {}: {}'.format(name.upper(), df.TotalPay.mean()))
    plt.figure(figsize=(16,4))
    sns.countplot('CombJobTitle', data = df)
    plt.title('All {} Job Titles'.format(name.upper()))
    plt.tight_layout()
find_name('lee')
find_name('wong')
find_name('chan')
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(stop_words = 'english')
dtm = vect.fit_transform(dat.JobTitle)
from sklearn.cross_validation import train_test_split

X = dtm
y = dat.TotalPay

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100)
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor()
clf.fit(X_train, y_train)
from sklearn.metrics import r2_score, mean_squared_error

pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)

print('root mean_squared_error train / test: {} / {}'.format(
    np.sqrt(mean_squared_error(y_train, pred_train)), np.sqrt(mean_squared_error(y_test, pred_test))))
print('r2_score train / test: {} / {}'.format(
    r2_score(y_train, pred_train), r2_score(y_test, pred_test)))
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.plot(np.arange(len(pred_train)), y_train - pred_train,'o')
plt.axhline(0)
plt.subplot(1,2,2)
plt.plot(np.arange(len(pred_test)), y_test - pred_test,'o')
plt.axhline(0)
plt.tight_layout()