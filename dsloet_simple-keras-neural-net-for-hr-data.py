# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from keras.models import Sequential

from keras.layers import Dense

import numpy

numpy.random.seed(7)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/core_dataset.csv')
df['Date of Termination'] = df['Date of Termination'].fillna("None")
df = df[df.Position.notnull()]
HispLat_map = {'Yes': 1, 'yes': 1, 'No': 0, 'no': 0}

df['Hispanic/Latino'] = df['Hispanic/Latino'].replace(HispLat_map)
MaritalDesc_map = {'Divorced': 0, 'Married': 1, 'Separated': 2,'Single': 3, 'widowed': 4}

df['MaritalDesc'] = df['MaritalDesc'].replace(MaritalDesc_map)
PerformanceScore_map = {'N/A- too early to review': 2,

                        'Needs Improvement': 1,

                        'Fully Meets': 2,

                        '90-day meets': 2,

                        'Exceeds': 3,

                        'Exceptional': 4,

                        'PIP': 1}

df['Performance Score'] = df['Performance Score'].replace(PerformanceScore_map)
Sex_map = {'Male': 0,

           'male': 0,

           'Female': 1,

           'female': 1}

df['Sex'] = df['Sex'].replace(Sex_map)
RaceDesc_map = {'American Indian or Alaska Native': 0,

                'Asian': 1,

                'Black or African American': 2,

                'Hispanic': 3,

                'Two or more races': 4,

                'White': 5}

df['RaceDesc'] = df['RaceDesc'].replace(RaceDesc_map)

CitizenDesc_map = {'Eligible NonCitizen': 0,

                   'Non-Citizen': 0,

                   'US Citizen': 1}

df['CitizenDesc'] = df['CitizenDesc'].replace(CitizenDesc_map)
EmploymentStatus_map = {'Active': 0,

                        'Future Start': 1,

                        'Leave of Absence': 2,

                        'Terminated for Cause': 3,

                        'Voluntarily Terminated': 4}

df['Employment Status'] = df['Employment Status'].replace(EmploymentStatus_map)
Department_map = {'Admin Offices': 0,

                  'Executive Office': 1,

                  'IT/IS': 2,

                  'Production       ': 3,

                  'Sales': 4,

                  'Software Engineering': 5,

                  'Software Engineering     ': 5}

df['Department'] = df['Department'].replace(Department_map)
del df['DOB']

del df['Date of Hire']
le = preprocessing.LabelEncoder()
le.fit(df['State'])

State1 = pd.DataFrame({'State1': le.transform(df['State'])})

df = pd.concat([df, State1], axis=1)

del df['State']
le.fit(df['Position'])

Position1 = pd.DataFrame({'Position1': le.transform(df['Position'])})

df = pd.concat([df, Position1], axis=1)

del df['Position']
le.fit(df['Manager Name'])

ManagerName1 = pd.DataFrame({'ManagerName1': le.transform(df['Manager Name'])})

df = pd.concat([df, ManagerName1], axis=1)

del df['Manager Name']
le.fit(df['Employee Source'])

EmployeeSource1 = pd.DataFrame({'EmployeeSource1': le.transform(df['Employee Source'])})

df = pd.concat([df, EmployeeSource1], axis=1)

del df['Employee Source']
le.fit(df['Reason For Term'])

ReasonForTerm1 = pd.DataFrame({'ReasonForTerm1': le.transform(df['Reason For Term'])})

df = pd.concat([df, ReasonForTerm1], axis=1)

del df['Reason For Term']
del df['Employee Name']
del df['Date of Termination']

del df['Zip']
Performance1 = pd.DataFrame({'Performance1': df['Performance Score']})

df = pd.concat([df, Performance1], axis=1)

del df['Performance Score']
PerfHigh = pd.DataFrame({'PerfHigh': df['Performance1']})

df = pd.concat([df, PerfHigh], axis=1)

PerfHigh_map = {2: 0,

                  1: 0,

                  3: 1,

                  4:1}

df['PerfHigh'] = df['PerfHigh'].replace(PerfHigh_map)
df.head()
del df['Performance1']
del df['Employee Number']
del df['Hispanic/Latino']
df.head()
df.shape
df.info()
df.to_csv('dataset.csv', header=0)
dataset = numpy.genfromtxt("dataset.csv", delimiter=',', skip_header=1)
dataset.shape
X = dataset[:,0:14]

Y = dataset[:,14]
X.shape
model = Sequential()

model.add(Dense(12, input_dim=14, activation='relu'))

model.add(Dense(14, activation='relu'))

model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=450, batch_size=15, verbose=0)
scores = model.evaluate(X, Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.summary()
model.get_config()
from pydot import graphviz

from keras.utils import plot_model

plot_model(model, to_file='model.png')