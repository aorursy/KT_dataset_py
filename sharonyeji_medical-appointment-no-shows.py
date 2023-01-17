import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

from scipy.interpolate import interp1d

from datetime import date

import calendar
sb.set_style("whitegrid")



"""sns.set_style preset themes: darkgrid, whitegrid, 

dark, white, ticks

"""



aptdata = pd.read_csv('../input/KaggleV2-May-2016.csv')

aptdata=aptdata.rename(columns = {'Neighbourhood':'Neighborhood','Hipertension':'Hypertension','Handcap':'Handicap'})
aptdata.info()

'No Null Values'
aptdata.head()
aptdata['ScheduledDay']=pd.to_datetime(aptdata['ScheduledDay'])

aptdata['AppointmentDay']=pd.to_datetime(aptdata['AppointmentDay'])

aptdata['daystillapt']=(aptdata['AppointmentDay']-aptdata['ScheduledDay']).dt.days

aptdata.loc[aptdata['daystillapt']<0,'daystillapt']=0
aptdata['WeekdayNum']=aptdata['AppointmentDay'].dt.weekday



'''edit data so that numerical analysis can be done'''

aptdata.loc[aptdata['Age']<0,'Age']=0

aptdata.loc[aptdata['No-show']=='No','Show']=1

aptdata.loc[aptdata['No-show']=='Yes','Show']=0

aptdata.loc[aptdata['Gender']=='F','Sex']=1

aptdata.loc[aptdata['Gender']=='M','Sex']=0

aptdata['Neighborhood'] = aptdata.Neighborhood.astype('category')

aptdata['Neighborhood'] = aptdata['Neighborhood'].cat.codes
aptdata.head()
plt.figure(figsize=(12,12))

sb.heatmap(aptdata.iloc[:, 2:].corr(), annot=True, square=True, cmap='BuPu')

plt.show()
aptdata['No-show'].value_counts(normalize=True).plot.bar(figsize=(10,10), title= 'No-Shows')
def probStatus(dataset,variable):

    df=pd.crosstab(index=dataset[variable],columns=dataset['No-show'])

    df['probShowUp']=df['No']/(df['Yes']+df['No'])

    df=df.reset_index()

    return df
df=probStatus(aptdata,'Age')

x0=df['Age']

y0=df['probShowUp']

plt.plot(x0,y0,'o',label='Data')

x=np.linspace(0,100,30)

options = ('slinear','cubic')

for o in options:

    f=interp1d(x0,y0,kind=o)

    plt.plot(x,f(x),label=o)



plt.legend()

plt.show()
df=probStatus(aptdata,'daystillapt')

x0=df['daystillapt']

y0=df['probShowUp']

plt.plot(x0,y0,'o',label='Data')

x=np.linspace(0,120,30)

options = ('slinear','quadratic',2)

for o in options:

    f=interp1d(x0,y0,kind=o)

    plt.plot(x,f(x),label=o)



plt.legend()

plt.show()
def probStatusVariable(variable):

    rows=[]

    for item in variable:

        for level in aptdata[item].unique():

            row = {'Condition': item}

            total = len(aptdata[aptdata[item] == level])

            n = len(aptdata[(aptdata[item] == level) & (aptdata.Show == 1)])

            row.update({'Level': level, 'Probability': n/total})

            rows.append(row)

    return pd.DataFrame(rows)
sb.barplot(data = probStatusVariable(['Diabetes', 'Hypertension']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')

plt.ylabel('Probability')

plt.show()
data = aptdata.drop(['PatientId','AppointmentID','Gender','ScheduledDay','AppointmentDay','No-show'],1)

data = np.log1p(data)

y = data['Show']

data = data.drop('Show',1)

data = data.values



from sklearn.model_selection import train_test_split



X_train,X_test,y_train, y_test= train_test_split(data, y ,test_size = .3, random_state= 42)
X_train.shape ,X_test.shape ,y_train.shape , y_test.shape
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from sklearn import preprocessing

lab_enc = preprocessing.LabelEncoder()

y_train_encoded = lab_enc.fit_transform(y_train)

rf = RandomForestClassifier()

rf.fit(X_train,y_train_encoded)

rf_pred=rf.predict(X_train)

accuracy_score(pd.DataFrame(rf.predict(X_train)),y_train_encoded)
predictions = rf.predict(X_test)

y_test_encoded = lab_enc.fit_transform(y_test)

rf_score= accuracy_score(pd.DataFrame(rf.predict(X_test)),y_test_encoded)

y=predictions[:100]

show_test = aptdata.Show[100000:]

show_test=show_test[:100]

show_test=show_test.reset_index()

plt.scatter(show_test['index'], show_test['Show'], c='b', label = 'data') #blue is true value

plt.scatter(show_test['index'], y, c='r', label = 'prediction') #red is predicted value

rf_score
from sklearn.metrics import classification_report, confusion_matrix

print("=== Confusion Matrix ===")

print(confusion_matrix(y_test_encoded, predictions))

print('\n')

print("=== Classification Report ===")

print(classification_report(y_test_encoded, predictions))

print('\n')
list(zip(aptdata.drop(['PatientId','AppointmentID','Gender','ScheduledDay','AppointmentDay','No-show', 'Show'],1), rf.feature_importances_))
gb= GradientBoostingClassifier(learning_rate=0.1, n_estimators=120

                            ,max_depth=10, min_samples_split= 8,

                               max_features='sqrt', 

                                    subsample=0.8, random_state=42)

gb.fit(X_train,y_train_encoded)

gb_pre=gb.predict(X_train)

accuracy_score(pd.DataFrame(gb.predict(X_train)),y_train_encoded)
gb.predict(X_test)

gb_score=accuracy_score(pd.DataFrame(gb.predict(X_test)),y_test_encoded)

gb_score
predictions = gb.predict(X_test)

gb_score= accuracy_score(pd.DataFrame(gb.predict(X_test)),y_test_encoded)

y=predictions[:100]

show_test = aptdata.Show[100000:]

show_test=show_test[:100]

show_test=show_test.reset_index()

plt.scatter(show_test['index'], show_test['Show'], c='b', label = 'data') #blue is true value

plt.scatter(show_test['index'], y, c='r', label = 'prediction') #red is predicted value

gb_score
print("=== Confusion Matrix ===")

print(confusion_matrix(y_test_encoded, predictions))

print('\n')

print("=== Classification Report ===")

print(classification_report(y_test_encoded, predictions))

print('\n')
list(zip(aptdata.drop(['PatientId','AppointmentID','Gender','ScheduledDay','AppointmentDay','No-show', 'Show'],1), rf.feature_importances_))
algorithms = [rf_score,gb_score]

names= ['Random Forest','Gradient Boosting']

final = pd.DataFrame([names,algorithms]).T

final.columns =['Algorithms', 'Accuracy Score'] 
final