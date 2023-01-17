# Import dependencies



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt
titanic=pd.read_csv('../input/titanic/train.csv')

heart_diseases=pd.read_csv('../input/heart-disease-uci/heart.csv')

iris=pd.read_csv('../input/iris/Iris.csv')

diabetes=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

pokemon=pd.read_csv('../input/pokemon/Pokemon.csv')

heart_failure=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

corona_virus = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

latest_covid_cases = pd.read_csv('../input/novel-covid19-dataset/cases_country.csv')
plt.plot(iris['PetalLengthCm'], iris['PetalWidthCm'], linestyle='none', marker='o', color='b')

plt.show()
x = np.linspace(0, 20, 1000)

y = np.cos(x)



plt.plot(x,y, color='b', linestyle='--')

plt.show()
plt.hist(heart_diseases['age'])

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Age distribution.')
bar_data = pokemon['Type 1'].value_counts().reset_index()

bar_data['err'] = pokemon['Type 1'].value_counts().std()
plt.bar(bar_data['index'], bar_data['Type 1'])

plt.xticks(rotation=90)

plt.xlabel('Ability')

plt.ylabel('Count')

plt.title("Abilities of Pokemons")
plt.figure(figsize=(10,6))

plt.barh(bar_data['index'], bar_data['Type 1'])

plt.xlabel('Ability')

plt.ylabel('Count')

plt.title("Abilities of Pokemons")
plt.bar(bar_data['index'], bar_data['Type 1'], yerr=bar_data['err'])

plt.xticks(rotation=90)

plt.xlabel('Ability')

plt.ylabel('Count')

plt.title("Abilities of Pokemons")
plt.figure(figsize=(10,6))

plt.boxplot(heart_diseases['chol'])

plt.ylabel('Value')

plt.title("Simple Box Plot")
plt.figure(figsize=(10,6))

plt.boxplot([diabetes['Age'], diabetes['BMI']])

plt.ylabel('Value')

plt.title("Box Plot")
plt.figure(figsize=(10,6))

plt.boxplot([diabetes['Age'], diabetes['BMI'], diabetes['BloodPressure'], diabetes['Glucose']])

plt.xticks([1, 2, 3, 4], ['Age', 'BMI', 'BloodPressure', 'Glucose'])

plt.ylabel('Value')

plt.title("Box Plot")
plt.figure(figsize=(10,6))

plt.boxplot([diabetes['Age'], diabetes['BMI'], diabetes['BloodPressure'], diabetes['Glucose']], vert=False)

plt.yticks([1, 2, 3, 4], ['Age', 'BMI', 'BloodPressure', 'Glucose'])

plt.xlabel('Value')

plt.title("Box Plot")
temp = corona_virus.groupby('ObservationDate')['Confirmed'].sum().reset_index()
fig = plt.figure(figsize=(20,12))

plt.fill_between(temp['ObservationDate'][:30], temp['Confirmed'][:30], color='lightblue')

plt.xticks(rotation=90)

plt.show()
india = corona_virus[corona_virus['Country/Region']=='India'].groupby('ObservationDate')['Confirmed'].sum().reset_index()

us = corona_virus[corona_virus['Country/Region']=='US'].groupby('ObservationDate')['Confirmed'].sum().reset_index()

brazil = corona_virus[corona_virus['Country/Region']=='Brazil'].groupby('ObservationDate')['Confirmed'].sum().reset_index()
fig = plt.figure(figsize=(20,12))

plt.stackplot(temp['ObservationDate'][-50:], us['Confirmed'][-50:],india['Confirmed'][-50:], 

              brazil['Confirmed'][-50:], labels=['US', 'India', 'Brazil'])

plt.xticks(rotation=90)

plt.legend(loc='upper left')



plt.show()
temp = latest_covid_cases.sort_values('Confirmed', ascending= False)[['Country_Region','Confirmed']]
plt.stem(temp['Country_Region'][:10], temp['Confirmed'][:10])

plt.xticks(rotation=90)

plt.show()
plt.hlines(y=temp['Country_Region'][:10][::-1], xmin=0, xmax=temp['Confirmed'][:10][::-1], color='skyblue')

plt.plot(temp['Confirmed'][:10][::-1], temp['Country_Region'][:10][::-1], 'D')

plt.xticks(rotation=90)

plt.show()