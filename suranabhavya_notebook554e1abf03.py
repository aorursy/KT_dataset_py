import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("../input/farming-analytical-data/ForAnalysis-WeeklySurvey-Farmer(PART1)Modified - Student.csv")
df.head()
df.shape
df.describe()
df.boxplot()
df.info()
df.isnull()
df.isnull().sum()
x = df['Year ']
print(x)
df.machinery_type
df.machinery_type[df.machinery_type == 'Plow'].count()
df.machinery_type[df.machinery_type == 'Tractor'].count()
df.machinery_type[df.machinery_type == 'Spray_machine'].count()
df.machinery_type[df.machinery_type == 'Cultivator'].count()
df.machinery_type[df.machinery_type == 'Rotavator'].count()
df.machinery_type[df.machinery_type == 'Bulls'].count()
gb = df.groupby('machinery_type')['machinery_type'].count()
print(gb)
plt.figure(figsize=(16,5))
plt.plot(gb, 'ro')
plt.show()
x = []
for i in gb.keys():
    x.append(i)
print(x)
y = []
for i in gb:
    y.append(i)
print(y)
plt.pie(y, labels=x, autopct='%1.1f%%',shadow=True)
plt.axis('equal')
plt.show()
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(x)), y, align='center', alpha=0.5)
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.ylabel('No. of Machinery Type used')
plt.title('Machinery Type Usage')
plt.show()
gb_bulls = df.groupby('Year ')['machinery_type_Bulls'].count()
print(gb_bulls)
x_bulls = []
for i in gb_bulls.keys():
    x_bulls.append(i)
print(x_bulls)
y_bulls = []
for i in gb_bulls:
    y_bulls.append(i)
print(y_bulls)
plt.figure(figsize=(16,5))
plt.plot(gb_bulls, 'red')
plt.show()
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(x_bulls)), y_bulls, align='center', alpha=0.5, color='green')
plt.xticks(np.arange(len(x_bulls)), x_bulls, rotation=90)
plt.ylabel('no.of Machinery type Bull')
plt.title('Machinery Type Bull Usage')
plt.show()
gb_pesticide = df.groupby('Year ')['pesticide_types_num'].count()
gb_pesticide
x_pesticide = []
for i in gb_pesticide.keys():
    x_pesticide.append(i)
print(x_pesticide)
y_pesticide = []
for i in gb_pesticide:
    y_pesticide.append(i)
print(y_pesticide)
plt.figure(figsize=(16,5))
plt.plot(gb_pesticide, 'red')
plt.show()
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(x_pesticide)), y_pesticide, align='center', alpha=0.5, color='blue')
plt.xticks(np.arange(len(x_pesticide)), x_pesticide, rotation=90)
plt.ylabel('no.of pesticides used')
plt.title('Pesticide Usage')
plt.show()
gb_pesticide = df.groupby(['Year ','Month'])['pesticide_types_num'].count()
gb_pesticide
seasons = []
seasons.extend(['Nov/2016 - Mar/2017','Nov/2017 - Mar/2018','Nov/2018 - Mar/2019','Nov/2019 - Mar/2020','Nov/2020 - Mar/2021'])
print(seasons)
pesticide_s1 = gb_pesticide[2016,'November'] + gb_pesticide[2016,'December'] + gb_pesticide[2017,'January'] + gb_pesticide[2017,'February'] + gb_pesticide[2017,'March']
pesticide_s2 = gb_pesticide[2017,'November'] + gb_pesticide[2017,'December'] + gb_pesticide[2018,'January'] + gb_pesticide[2018,'February'] + gb_pesticide[2018,'March']
pesticide_s3 = gb_pesticide[2018,'November'] + gb_pesticide[2018,'December'] + gb_pesticide[2019,'January'] + gb_pesticide[2019,'February'] + gb_pesticide[2019,'March']
pesticide_s4 = gb_pesticide[2019,'November'] + gb_pesticide[2019,'December'] + gb_pesticide[2020,'January'] + gb_pesticide[2020,'February'] + gb_pesticide[2020,'March']
pesticide_s5 = gb_pesticide[2020,'November'] + gb_pesticide[2020,'December'] + gb_pesticide[2021,'January'] + gb_pesticide[2021,'February'] + gb_pesticide[2021,'March']
no_pesticides = []
no_pesticides.extend((pesticide_s1,pesticide_s2,pesticide_s3,pesticide_s4,pesticide_s5))
print(no_pesticides)
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(seasons)), no_pesticides, align='center', alpha=0.5, color='red')
plt.xticks(np.arange(len(seasons)), seasons)
plt.ylabel('no.of pesticides used')
plt.title('Pesticide Usage')
plt.show()
gb_fertilizer = df.groupby(['Year ','Month'])['fertilizer_types_num'].count()
gb_fertilizer
fertilizer_s1 = gb_fertilizer[2016,'November'] + gb_fertilizer[2016,'December'] + gb_fertilizer[2017,'January'] + gb_fertilizer[2017,'February'] + gb_fertilizer[2017,'March']
fertilizer_s2 = gb_fertilizer[2017,'November'] + gb_fertilizer[2017,'December'] + gb_fertilizer[2018,'January'] + gb_fertilizer[2018,'February'] + gb_fertilizer[2018,'March']
fertilizer_s3 = gb_fertilizer[2018,'November'] + gb_fertilizer[2018,'December'] + gb_fertilizer[2019,'January'] + gb_fertilizer[2019,'February'] + gb_fertilizer[2019,'March']
fertilizer_s4 = gb_fertilizer[2019,'November'] + gb_fertilizer[2019,'December'] + gb_fertilizer[2020,'January'] + gb_fertilizer[2020,'February'] + gb_fertilizer[2020,'March']
fertilizer_s5 = gb_fertilizer[2020,'November'] + gb_fertilizer[2020,'December'] + gb_fertilizer[2021,'January'] + gb_fertilizer[2021,'February'] + gb_fertilizer[2021,'March']
no_fertilizers = []
no_fertilizers.extend((fertilizer_s1,fertilizer_s2,fertilizer_s3,fertilizer_s4,fertilizer_s5))
print(no_fertilizers)
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(seasons)), no_fertilizers, align='center', alpha=0.5, color='orange')
plt.xticks(np.arange(len(seasons)), seasons)
plt.ylabel('no.of fertilizers used')
plt.title('Fertilizers Usage')
plt.show()
gb_herbicide = df.groupby(['Year ','Month'])['herbicide_types_num'].count()
gb_herbicide
herbicide_s1 = gb_herbicide[2016,'November'] + gb_herbicide[2016,'December'] + gb_herbicide[2017,'January'] + gb_herbicide[2017,'February'] + gb_herbicide[2017,'March']
herbicide_s2 = gb_herbicide[2017,'November'] + gb_herbicide[2017,'December'] + gb_herbicide[2018,'January'] + gb_herbicide[2018,'February'] + gb_herbicide[2018,'March']
herbicide_s3 = gb_herbicide[2018,'November'] + gb_herbicide[2018,'December'] + gb_herbicide[2019,'January'] + gb_herbicide[2019,'February'] + gb_herbicide[2019,'March']
herbicide_s4 = gb_herbicide[2019,'November'] + gb_herbicide[2019,'December'] + gb_herbicide[2020,'January'] + gb_herbicide[2020,'February'] + gb_herbicide[2020,'March']
herbicide_s5 = gb_herbicide[2020,'November'] + gb_herbicide[2020,'December'] + gb_herbicide[2021,'January'] + gb_herbicide[2021,'February'] + gb_herbicide[2021,'March']
no_herbicides = []
no_herbicides.extend((herbicide_s1,herbicide_s2,herbicide_s3,herbicide_s4,herbicide_s5))
print(no_herbicides)
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(seasons)), no_herbicides, align='center', alpha=0.5, color='violet')
plt.xticks(np.arange(len(seasons)), seasons)
plt.ylabel('no.of herbicides used')
plt.title('Herbicides Usage')
plt.show()
gb_bulls = df.groupby(['Year ','Month'])['machinery_type_Bulls'].count()
gb_bulls
bulls_s1 = gb_bulls[2016,'November'] + gb_bulls[2016,'December'] + gb_bulls[2017,'January'] + gb_bulls[2017,'February'] + gb_bulls[2017,'March']
bulls_s2 = gb_bulls[2017,'November'] + gb_bulls[2017,'December'] + gb_bulls[2018,'January'] + gb_bulls[2018,'February'] + gb_bulls[2018,'March']
bulls_s3 = gb_bulls[2018,'November'] + gb_bulls[2018,'December'] + gb_bulls[2019,'January'] + gb_bulls[2019,'February'] + gb_bulls[2019,'March']
bulls_s4 = gb_bulls[2019,'November'] + gb_bulls[2019,'December'] + gb_bulls[2020,'January'] + gb_bulls[2020,'February'] + gb_bulls[2020,'March']
bulls_s5 = gb_bulls[2020,'November'] + gb_bulls[2020,'December'] + gb_bulls[2021,'January'] + gb_bulls[2021,'February'] + gb_bulls[2021,'March']
no_bulls = []
no_bulls.extend((bulls_s1,bulls_s2,bulls_s3,bulls_s4,bulls_s5))
print(no_bulls)
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(seasons)), no_bulls, align='center', alpha=0.5, color='magenta')
plt.xticks(np.arange(len(seasons)), seasons)
plt.ylabel('no.of machinery bulls used')
plt.title('Machinery Bulls Usage')
plt.show()
gb_tractor = df.groupby(['Year ','Month'])['machinery_type_Tractor'].count()
gb_tractor
tractor_s1 = gb_tractor[2016,'November'] + gb_tractor[2016,'December'] + gb_tractor[2017,'January'] + gb_tractor[2017,'February'] + gb_tractor[2017,'March']
tractor_s2 = gb_tractor[2017,'November'] + gb_tractor[2017,'December'] + gb_tractor[2018,'January'] + gb_tractor[2018,'February'] + gb_tractor[2018,'March']
tractor_s3 = gb_tractor[2018,'November'] + gb_tractor[2018,'December'] + gb_tractor[2019,'January'] + gb_tractor[2019,'February'] + gb_tractor[2019,'March']
tractor_s4 = gb_tractor[2019,'November'] + gb_tractor[2019,'December'] + gb_tractor[2020,'January'] + gb_tractor[2020,'February'] + gb_tractor[2020,'March']
tractor_s5 = gb_tractor[2020,'November'] + gb_tractor[2020,'December'] + gb_tractor[2021,'January'] + gb_tractor[2021,'February'] + gb_tractor[2021,'March']
no_tractor = []
no_tractor.extend((tractor_s1,tractor_s2,tractor_s3,tractor_s4,tractor_s5))
print(no_tractor)
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(seasons)), no_tractor, align='center', alpha=0.5, color='magenta')
plt.xticks(np.arange(len(seasons)), seasons)
plt.ylabel('no.of Tractor used')
plt.title('Tractor Usage')
plt.show()
gb_plow = df.groupby(['Year ','Month'])['machinery_type_Plow'].count()
gb_plow
plow_s1 = gb_plow[2016,'November'] + gb_plow[2016,'December'] + gb_plow[2017,'January'] + gb_plow[2017,'February'] + gb_plow[2017,'March']
plow_s2 = gb_plow[2017,'November'] + gb_plow[2017,'December'] + gb_plow[2018,'January'] + gb_plow[2018,'February'] + gb_plow[2018,'March']
plow_s3 = gb_plow[2018,'November'] + gb_plow[2018,'December'] + gb_plow[2019,'January'] + gb_plow[2019,'February'] + gb_plow[2019,'March']
plow_s4 = gb_plow[2019,'November'] + gb_plow[2019,'December'] + gb_plow[2020,'January'] + gb_plow[2020,'February'] + gb_plow[2020,'March']
plow_s5 = gb_plow[2020,'November'] + gb_plow[2020,'December'] + gb_plow[2021,'January'] + gb_plow[2021,'February'] + gb_plow[2021,'March']
no_plow = []
no_plow.extend((plow_s1,plow_s2,plow_s3,plow_s4,plow_s5))
print(no_plow)
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(seasons)), no_plow, align='center', alpha=0.5, color='blue')
plt.xticks(np.arange(len(seasons)), seasons)
plt.ylabel('no.of Plow used')
plt.title('Plow Usage')
plt.show()
gb_sprayMachine = df.groupby(['Year ','Month'])['machinery_type_Spray_machine'].count()
gb_sprayMachine
sprayMachine_s1 = gb_sprayMachine[2016,'November'] + gb_sprayMachine[2016,'December'] + gb_sprayMachine[2017,'January'] + gb_sprayMachine[2017,'February'] + gb_sprayMachine[2017,'March']
sprayMachine_s2 = gb_sprayMachine[2017,'November'] + gb_sprayMachine[2017,'December'] + gb_sprayMachine[2018,'January'] + gb_sprayMachine[2018,'February'] + gb_sprayMachine[2018,'March']
sprayMachine_s3 = gb_sprayMachine[2018,'November'] + gb_sprayMachine[2018,'December'] + gb_sprayMachine[2019,'January'] + gb_sprayMachine[2019,'February'] + gb_sprayMachine[2019,'March']
sprayMachine_s4 = gb_sprayMachine[2019,'November'] + gb_sprayMachine[2019,'December'] + gb_sprayMachine[2020,'January'] + gb_sprayMachine[2020,'February'] + gb_sprayMachine[2020,'March']
sprayMachine_s5 = gb_sprayMachine[2020,'November'] + gb_sprayMachine[2020,'December'] + gb_sprayMachine[2021,'January'] + gb_sprayMachine[2021,'February'] + gb_sprayMachine[2021,'March']
no_sprayMachine = []
no_sprayMachine.extend((sprayMachine_s1,sprayMachine_s2,sprayMachine_s3,sprayMachine_s4,sprayMachine_s5))
print(no_sprayMachine)
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(seasons)), no_sprayMachine, align='center', alpha=0.5, color='green')
plt.xticks(np.arange(len(seasons)), seasons)
plt.ylabel('no.of spray machines used')
plt.title('Spray Machine Usage')
plt.show()
gb_cultivator = df.groupby(['Year ','Month'])['machinery_type_Cultivator'].count()
gb_cultivator
cultivator_s1 = gb_cultivator[2016,'November'] + gb_cultivator[2016,'December'] + gb_cultivator[2017,'January'] + gb_cultivator[2017,'February'] + gb_cultivator[2017,'March']
cultivator_s2 = gb_cultivator[2017,'November'] + gb_cultivator[2017,'December'] + gb_cultivator[2018,'January'] + gb_cultivator[2018,'February'] + gb_cultivator[2018,'March']
cultivator_s3 = gb_cultivator[2018,'November'] + gb_cultivator[2018,'December'] + gb_cultivator[2019,'January'] + gb_cultivator[2019,'February'] + gb_cultivator[2019,'March']
cultivator_s4 = gb_cultivator[2019,'November'] + gb_cultivator[2019,'December'] + gb_cultivator[2020,'January'] + gb_cultivator[2020,'February'] + gb_cultivator[2020,'March']
cultivator_s5 = gb_cultivator[2020,'November'] + gb_cultivator[2020,'December'] + gb_cultivator[2021,'January'] + gb_cultivator[2021,'February'] + gb_cultivator[2021,'March']
no_cultivator = []
no_cultivator.extend((cultivator_s1,cultivator_s2,cultivator_s3,cultivator_s4,cultivator_s5))
print(no_cultivator)
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(seasons)), no_cultivator, align='center', alpha=0.5, color='red')
plt.xticks(np.arange(len(seasons)), seasons)
plt.ylabel('no.of cultivator used')
plt.title('Cultivator Usage')
plt.show()
gb_rotavator = df.groupby(['Year ','Month'])['machinery_type_Rotavator'].count()
gb_rotavator
rotavator_s1 = gb_rotavator[2016,'November'] + gb_rotavator[2016,'December'] + gb_rotavator[2017,'January'] + gb_rotavator[2017,'February'] + gb_rotavator[2017,'March']
rotavator_s2 = gb_rotavator[2017,'November'] + gb_rotavator[2017,'December'] + gb_rotavator[2018,'January'] + gb_rotavator[2018,'February'] + gb_rotavator[2018,'March']
rotavator_s3 = gb_rotavator[2018,'November'] + gb_rotavator[2018,'December'] + gb_rotavator[2019,'January'] + gb_rotavator[2019,'February'] + gb_rotavator[2019,'March']
rotavator_s4 = gb_rotavator[2019,'November'] + gb_rotavator[2019,'December'] + gb_rotavator[2020,'January'] + gb_rotavator[2020,'February'] + gb_rotavator[2020,'March']
rotavator_s5 = gb_rotavator[2020,'November'] + gb_rotavator[2020,'December'] + gb_rotavator[2021,'January'] + gb_rotavator[2021,'February'] + gb_rotavator[2021,'March']
no_rotavator = []
no_rotavator.extend((rotavator_s1,rotavator_s2,rotavator_s3,rotavator_s4,rotavator_s5))
print(no_rotavator)
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(seasons)), no_rotavator, align='center', alpha=0.5, color='blue')
plt.xticks(np.arange(len(seasons)), seasons)
plt.ylabel('no.of rotavator used')
plt.title('Rotavator Usage')
plt.show()
gb_cropDamage = df.groupby(['Year ','Month'])['crop_damage_type'].count()
gb_cropDamage
cropDamage_s1 = gb_cropDamage[2016,'November'] + gb_cropDamage[2016,'December'] + gb_cropDamage[2017,'January'] + gb_cropDamage[2017,'February'] + gb_cropDamage[2017,'March']
cropDamage_s2 = gb_cropDamage[2017,'November'] + gb_cropDamage[2017,'December'] + gb_cropDamage[2018,'January'] + gb_cropDamage[2018,'February'] + gb_cropDamage[2018,'March']
cropDamage_s3 = gb_cropDamage[2018,'November'] + gb_cropDamage[2018,'December'] + gb_cropDamage[2019,'January'] + gb_cropDamage[2019,'February'] + gb_cropDamage[2019,'March']
cropDamage_s4 = gb_cropDamage[2019,'November'] + gb_cropDamage[2019,'December'] + gb_cropDamage[2020,'January'] + gb_cropDamage[2020,'February'] + gb_cropDamage[2020,'March']
cropDamage_s5 = gb_cropDamage[2020,'November'] + gb_cropDamage[2020,'December'] + gb_cropDamage[2021,'January'] + gb_cropDamage[2021,'February'] + gb_cropDamage[2021,'March']
no_cropDamage = []
no_cropDamage.extend((cropDamage_s1,cropDamage_s2,cropDamage_s3,cropDamage_s4,cropDamage_s5))
print(no_cropDamage)
plt.figure(figsize=(16,5))
plt.bar(np.arange(len(seasons)), no_cropDamage, align='center', alpha=0.5, color='green')
plt.xticks(np.arange(len(seasons)), seasons)
plt.ylabel('no.of Crop Damage')
plt.title('Season wise Crop damage')
plt.show()
gb_date_fertilizer = df.groupby(['Year ','SubmissionDate.1'])['fertilizer_types_num'].count()
gb_date_fertilizer
dateFertilizers = []
for i in gb_date_fertilizer[2016]:
    dateFertilizers.append(i)
dateFertilizers.extend((gb_date_fertilizer[2017,'01-02-2017'],gb_date_fertilizer[2017,'02-01-2017'],gb_date_fertilizer[2017,'03-01-2017']))
print(dateFertilizers)
dates = []
for i in gb_date_fertilizer[2016].keys():
    dates.append(i)
for i in range(3):
    dates.append((gb_date_fertilizer[2017].keys()[i]))
print(dates)
plt.figure(figsize=(16,5))
plt.plot(dates,dateFertilizers,color='red',marker='o')
plt.title('One Season Statistics of Fertilizers Usage')
plt.xlabel('Dates in one Season')
plt.ylabel('No.of fertilizers used')
plt.show()
gb_date_pesticide = df.groupby(['Year ','SubmissionDate.1'])['pesticide_types_num'].count()
gb_date_pesticide
datePesticide = []
for i in gb_date_pesticide[2016]:
    datePesticide.append(i)
datePesticide.extend((gb_date_pesticide[2017,'01-02-2017'],gb_date_pesticide[2017,'02-01-2017'],gb_date_pesticide[2017,'03-01-2017']))
print(datePesticide)
dates = []
for i in gb_date_fertilizer[2016].keys():
    dates.append(i)
for i in range(3):
    dates.append((gb_date_fertilizer[2017].keys()[i]))
print(dates)
plt.figure(figsize=(16,5))
plt.plot(dates,datePesticide,color='green',marker='o')
plt.title('One Season Statistics of Pesticide Usage')
plt.xlabel('Dates in one Season')
plt.ylabel('No.of Pesticides used')
plt.show()
gb_date_herbicide = df.groupby(['Year ','SubmissionDate.1'])['herbicide_types_num'].count()
gb_date_herbicide
dateHerbicide = []
for i in gb_date_herbicide[2016]:
    dateHerbicide.append(i)
dateHerbicide.extend((gb_date_herbicide[2017,'01-02-2017'],gb_date_herbicide[2017,'02-01-2017'],gb_date_herbicide[2017,'03-01-2017']))
print(dateHerbicide)
dates = []
for i in gb_date_herbicide[2016].keys():
    dates.append(i)
for i in range(3):
    dates.append((gb_date_herbicide[2017].keys()[i]))
print(dates)
plt.figure(figsize=(16,5))
plt.plot(dates,dateHerbicide,color='blue',marker='o')
plt.title('One Season Statistics of Herbicide Usage')
plt.xlabel('Dates in one Season')
plt.ylabel('No.of Herbicides used')
plt.show()
gb_date_bulls = df.groupby(['Year ','SubmissionDate.1'])['machinery_type_Bulls'].count()
gb_date_bulls
dateBulls = []
for i in gb_date_bulls[2016]:
    dateBulls.append(i)
dateBulls.extend((gb_date_bulls[2017,'01-02-2017'],gb_date_bulls[2017,'02-01-2017'],gb_date_bulls[2017,'03-01-2017']))
print(dateBulls)
dates = []
for i in gb_date_bulls[2016].keys():
    dates.append(i)
for i in range(3):
    dates.append((gb_date_bulls[2017].keys()[i]))
print(dates)
plt.figure(figsize=(16,5))
plt.plot(dates,dateBulls,color='magenta',marker='o')
plt.title('One Season Statistics of Machinery type Bull Usage')
plt.xlabel('Dates in one Season')
plt.ylabel('No.of Machinery type Bulls used')
plt.show()
gb_date_tractor = df.groupby(['Year ','SubmissionDate.1'])['machinery_type_Tractor'].count()
gb_date_tractor
dateTractor = []
for i in gb_date_tractor[2016]:
    dateTractor.append(i)
dateTractor.extend((gb_date_tractor[2017,'01-02-2017'],gb_date_tractor[2017,'02-01-2017'],gb_date_tractor[2017,'03-01-2017']))
print(dateTractor)
dates = []
for i in gb_date_tractor[2016].keys():
    dates.append(i)
for i in range(3):
    dates.append((gb_date_tractor[2017].keys()[i]))
print(dates)
plt.figure(figsize=(16,5))
plt.plot(dates,dateTractor,color='orange',marker='o')
plt.title('One Season Statistics of Machinery type Tractor Usage')
plt.xlabel('Dates in one Season')
plt.ylabel('No.of Machinery type Tractor used')
plt.show()
modifiedDF = df.loc[df.farmer_id=='FAR1'] 
modifiedDF
gb_farm_fertilizer = modifiedDF.groupby(['Year ','SubmissionDate.1'])['fertilizer_types_num'].count()
gb_farm_fertilizer
farmFertilizer = []
for i in gb_farm_fertilizer[2016]:
    farmFertilizer.append(i)
farmFertilizer.extend((gb_farm_fertilizer[2017,'01-02-2017'],gb_farm_fertilizer[2017,'02-01-2017'],gb_farm_fertilizer[2017,'03-01-2017']))
print(farmFertilizer)
dates = []
for i in gb_farm_fertilizer[2016].keys():
    dates.append(i)
for i in range(3):
    dates.append((gb_farm_fertilizer[2017].keys()[i]))
print(dates)
plt.figure(figsize=(16,5))
plt.plot(dates,farmFertilizer,color='red',marker='o')
plt.title('One Season Statistics of Fertilizer Usage of Farmer-1')
plt.xlabel('Dates in one Season')
plt.ylabel('No.of Fertilizers used')
plt.show()
gb_farm_pesticide = modifiedDF.groupby(['Year ','SubmissionDate.1'])['pesticide_types_num'].count()
gb_farm_pesticide
farmPesticide = []
for i in gb_farm_pesticide[2016]:
    farmPesticide.append(i)
farmPesticide.extend((gb_farm_pesticide[2017,'01-02-2017'],gb_farm_pesticide[2017,'02-01-2017'],gb_farm_pesticide[2017,'03-01-2017']))
print(farmPesticide)
dates = []
for i in gb_farm_pesticide[2016].keys():
    dates.append(i)
for i in range(3):
    dates.append((gb_farm_pesticide[2017].keys()[i]))
print(dates)
plt.figure(figsize=(16,5))
plt.plot(dates,farmPesticide,color='green',marker='o')
plt.title('One Season Statistics of Pesticide Usage of Farmer-1')
plt.xlabel('Dates in one Season')
plt.ylabel('No.of Pesticides used')
plt.show()
gb_farm_herbicide = modifiedDF.groupby(['Year ','SubmissionDate.1'])['herbicide_types_num'].count()
gb_farm_herbicide
farmHerbicide = []
for i in gb_farm_herbicide[2016]:
    farmHerbicide.append(i)
farmHerbicide.extend((gb_farm_herbicide[2017,'01-02-2017'],gb_farm_herbicide[2017,'02-01-2017'],gb_farm_herbicide[2017,'03-01-2017']))
print(farmHerbicide)
dates = []
for i in gb_farm_herbicide[2016].keys():
    dates.append(i)
for i in range(3):
    dates.append((gb_farm_herbicide[2017].keys()[i]))
print(dates)
plt.figure(figsize=(16,5))
plt.plot(dates,farmHerbicide,color='blue',marker='o')
plt.title('One Season Statistics of Herbicide Usage of Farmer-1')
plt.xlabel('Dates in one Season')
plt.ylabel('No.of Herbicides used')
plt.show()
gb_farm_bulls = modifiedDF.groupby(['Year ','SubmissionDate.1'])['machinery_type_Bulls'].count()
gb_farm_bulls
farmBulls = []
for i in gb_farm_bulls[2016]:
    farmBulls.append(i)
farmBulls.extend((gb_farm_bulls[2017,'01-02-2017'],gb_farm_bulls[2017,'02-01-2017'],gb_farm_bulls[2017,'03-01-2017']))
print(farmBulls)
dates = []
for i in gb_farm_bulls[2016].keys():
    dates.append(i)
for i in range(3):
    dates.append((gb_farm_bulls[2017].keys()[i]))
print(dates)
plt.figure(figsize=(16,5))
plt.plot(dates,farmBulls,color='magenta',marker='o')
plt.title('One Season Statistics of Machinery type Bulls Usage of Farmer-1')
plt.xlabel('Dates in one Season')
plt.ylabel('No.of Machinery type Bulls used')
plt.show()
gb_farm_plow = modifiedDF.groupby(['Year ','SubmissionDate.1'])['machinery_type_Plow'].count()
gb_farm_plow
farmPlow = []
for i in gb_farm_plow[2016]:
    farmPlow.append(i)
farmPlow.extend((gb_farm_plow[2017,'01-02-2017'],gb_farm_plow[2017,'02-01-2017'],gb_farm_plow[2017,'03-01-2017']))
print(farmPlow)
dates = []
for i in gb_farm_plow[2016].keys():
    dates.append(i)
for i in range(3):
    dates.append((gb_farm_plow[2017].keys()[i]))
print(dates)
plt.figure(figsize=(16,5))
plt.plot(dates,farmPlow,color='orange',marker='o')
plt.title('One Season Statistics of Machinery type Plow Usage of Farmer-1')
plt.xlabel('Dates in one Season')
plt.ylabel('No.of Machinery type Plow used')
plt.show()
