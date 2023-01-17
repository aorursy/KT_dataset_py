import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_excel('../input/Career.xlsx')

data.describe(include='all')
# from sklearn.preprocessing import LabelEncoder

data.college = data.college.astype('category') #.map({'IITs':0,'NITs':1,'IIITs':2,'Private':3,'Government':4})

data.year = data.year.astype('category') #.map({'1st':1,'2nd':2,'3rd':3,'4th':4})

data.stream = data.stream.map(

                {'Computer science and engineering,':'COMP','Electronics and communication engineering,':'ELEC',

                 'Information technology,':'COMP','Mechanical engineering,':'MECH','Civil engineering,':'MECH',

                 'Mechatronics engineering,':'MECH','Electrical and electronics and engineering':'ELEC',

                 'Environmental engineering,':'CHEM','Architecture and construction engineering,':'MECH',

                 'Highway engineering,':'MECH','Computer engineering,':'COMP','Marine engineering,':'COMP''ELEC',

                 'Automobile engineering,':'MECH','Aeronautical engineering,':'MECH','Aerospace engineering,':'MECH',

                 'Telecommunication engineering,':'ELEC','Electronics and communication engineering,':'ELEC',

                 'Agricultural engineering,':'CHEM','Production and industrial engineering,':'MECH',

                 'Chemical engineering,':'CHEM','Electrical engineering,':'ELEC','Instrumental engineering,':'ELEC',

                 'Mining engineering,':'CHEM','Architectural':'MECH','Biological Science':'CHEM','Bio-Medical':'CHEM',

                 'Biological engineering':'CHEM','Nuclear engineering':'CHEM','Systems engineering':'COMP','Others':-1

                })

data.cgpa = data.cgpa.astype('category') #.map({'<6':0,'6-8':1,'>8':2})

data['12th'] = data['12th'].astype('category') #.map({'Private School':0,'Public School':1})

data['10th'] = data['10th'].astype('category') #.map({'Private School':0,'Public School':1})

data.career = data.career.astype('category') #.map({'Public Sector Job':0,'Private Sector job':1,'Defence':2,'Higher study':3,'Entrepreneurship':4,

#                 'Family business':5})

data.parents = data.parents.astype('category') #.map({'Business':0,'Public Sector':1,'Private Sector':2,'Defence':2,'Self-Employed':3})

data.income = data.income.astype('category') #.map({'< 3 lakhs':0,'3-6 lakhs':1,'6-10 lakhs':2,'> 10 lakhs':3})

data.state = data.state.map(

    {'Rajasthan':'W','Delhi':'N','Haryana':'N','Punjab':'N','Chandigarh':'N','Andhra Pradesh':'S',

     'Telangana':'S','Arunachal Pradesh':'E','Assam':'E','Bihar':'C','Chhattisgarh':'C','Goa':'S',

     'Gujarat':'W','Himachal Pradesh':'N','Jammu and Kashmir':'N','Jharkhand':'C','Karnataka':'S',

     'Kerala':'S','Madhya Pradesh':'C','Maharashtra':'C','Manipur':'E','Meghalaya':'E','Mizoram':'E',

     'Nagaland':'E','Odisha':'E','Sikkim':'E','Tamil Nadu':'S','Tripura':'E','Uttar Pradesh':'N',

     'Uttarakhand':'N','West Bengal':'E','Andaman and Nicobar Islands':'S',

     'Dadra and Nagar Haveli':'W','Daman and Diu':'W','Lakshadweep':'S','Pondicherry':'S',

     'Abroad':-1

    }).astype('category')

# data.state = stateEncoder.transform(data.state)

data.locality = data.locality.astype('category') #.map({'Rural':0,'Urban':1})
data.head()
data.describe()
print(data.college.value_counts())

pd.crosstab(data.college, data.career,normalize='index').style.background_gradient()
print(data.year.value_counts())

pd.crosstab(data.year, data.career,normalize='index').style.background_gradient()
print(data.cgpa.value_counts())

pd.crosstab(data.cgpa, data.career,normalize='index').style.background_gradient()
print(data.parents.value_counts())

pd.crosstab(data.parents, data.career,normalize='index').style.background_gradient()
print(data.income.value_counts())

pd.crosstab(data.income, data.career,normalize='index').style.background_gradient()
print(data.locality.value_counts())

pd.crosstab(data.locality, data.career,normalize='index').style.background_gradient()
print(data.state.value_counts())

pd.crosstab(data.state, data.career,normalize='index').style.background_gradient()
print(data.stream.value_counts())

pd.crosstab(data.stream, data.career,normalize='index').style.background_gradient()
print(data['12th'].value_counts())

pd.crosstab(data['12th'], data.career,normalize='index').style.background_gradient()
print(data['10th'].value_counts())

pd.crosstab(data['10th'], data.career,normalize='index').style.background_gradient()