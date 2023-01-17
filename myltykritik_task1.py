import pandas as pd
breeds = pd.read_csv('../input/breed_labels.csv')
colors = pd.read_csv('../input/color_labels.csv')
states = pd.read_csv('../input/state_labels.csv')
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')

train.head()
breed_dict = dict(zip(breeds['BreedID'], breeds['BreedName']))
# тут пишут, что есть более быстрый способ, но интуитивно менее понятный:
# https://stackoverflow.com/questions/17426292/what-is-the-most-efficient-way-to-create-a-dictionary-of-two-pandas-dataframe-co

# pd.Series(breeds['BreedName'].values, index=breeds['BreedID']).to_dict()
animal_dict = {1: 'Dog', 2: 'Cat'}
train['Breed1_new'] = train['Breed1'].map(breed_dict)
train['Type'] = train['Type'].map(animal_dict)
train.groupby(['Type', 'Breed1_new', 'FurLength'])['PetID'].count()
#Разделим датафрейм на кошек и собак и рассмотрим их раздельно:
train_cats = train[train.Type != 'Dog']
train_dogs = train[train.Type != 'Cat']

#Для каждого распределения попробуем построить визуализацию
import seaborn as sns
#Создаем для кошек датафреймы для описания распределений скорости adoption в разрезах различных категориальных фич
ms_train_cats = pd.DataFrame(train_cats.groupby([ 'AdoptionSpeed', 'MaturitySize'])['PetID'].count()).reset_index()
fr_train_cats = pd.DataFrame(train_cats.groupby(['AdoptionSpeed', 'FurLength'])['PetID'].count()).reset_index()
vc_train_cats = pd.DataFrame(train_cats.groupby(['AdoptionSpeed', 'Vaccinated'])['PetID'].count()).reset_index()
dw_train_cats = pd.DataFrame(train_cats.groupby(['AdoptionSpeed', 'Dewormed'])['PetID'].count()).reset_index()
st_train_cats = pd.DataFrame(train_cats.groupby(['AdoptionSpeed', 'Sterilized'])['PetID'].count()).reset_index()
h_train_cats = pd.DataFrame(train_cats.groupby(['AdoptionSpeed', 'Health'])['PetID'].count()).reset_index()
age_train_cats = pd.DataFrame(train_cats.groupby(['AdoptionSpeed', 'Age'])['PetID'].count()).reset_index()
#Добавляем колонку в датафреймы с разрезами с долей питомцев от общего числа для каждого значения категориальной фичи
ms_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993
fr_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993
vc_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993
dw_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993
st_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993
h_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993
age_train_cats['PetID_Share'] = ms_train_cats['PetID']/14993
#распределение размеров по количеству питомцев
train_cats.groupby(['MaturitySize'])['PetID'].count()
sns.distplot(train_cats['MaturitySize'])
#распределение размеров питомцев в разрезе adoption скорости
train_cats.groupby(['MaturitySize', 'AdoptionSpeed'])['PetID'].count()
# Хитмап по размеру питомца
pv_ms_train_cats = ms_train_cats.pivot_table(values='PetID_Share', index='MaturitySize',   columns='AdoptionSpeed')
sns.heatmap(pv_ms_train_cats, cmap='inferno_r')
#распределение длины шерсти по количеству питомцев
train_cats.groupby(['FurLength'])['PetID'].count()
sns.distplot(train_cats['FurLength'])
#распределение длины шерсти питомцев в разрезе adoption скорости
train_cats.groupby(['AdoptionSpeed', 'FurLength'])['PetID'].count()
#Хитмап по длине шерсти питомца
pv_fr_train_cats = fr_train_cats.pivot_table(values='PetID_Share', index='FurLength',   columns='AdoptionSpeed')
sns.heatmap(pv_fr_train_cats, cmap='inferno_r')
#распределение статуса вакцинации по количеству питомцев
train_cats.groupby(['Vaccinated'])['PetID'].count()
sns.distplot(train_cats['Vaccinated'])
#распределение статуса вакцинации в разрезе adoption скорости
train_cats.groupby(['AdoptionSpeed', 'Vaccinated'])['PetID'].count()
pv_vc_train_cats = vc_train_cats.pivot_table(values='PetID_Share', index='Vaccinated',   columns='AdoptionSpeed')
sns.heatmap(pv_vc_train_cats, cmap='inferno_r')
#распределение статусов применения противоглистных препаратов по количеству питомцев
train_cats.groupby(['Dewormed'])['PetID'].count()
sns.distplot(train_cats['Dewormed'])
#распределение статуса применения противоглистных препаратов в разрезе adoption скорости
train_cats.groupby(['AdoptionSpeed', 'Dewormed'])['PetID'].count()
pv_dw_train_cats = dw_train_cats.pivot_table(values='PetID_Share', index='Dewormed',   columns='AdoptionSpeed')
sns.heatmap(pv_dw_train_cats, cmap='inferno_r')
#распределение статусов стерилизации по количеству питомцев
train_cats.groupby(['Sterilized'])['PetID'].count()
sns.distplot(train_cats['Sterilized'])
#распределение статуса стерилизации в разрезе adoption скорости
train_cats.groupby(['AdoptionSpeed', 'Sterilized'])['PetID'].count()
pv_st_train_cats = st_train_cats.pivot_table(values='PetID_Share', index='Sterilized',   columns='AdoptionSpeed')
sns.heatmap(pv_st_train_cats, cmap='inferno_r')
#распределение статусов здоровья по количеству питомцев
train_cats.groupby(['Health'])['PetID'].count()
sns.distplot(train_cats['Health'])
#распределение статуса здоровья в разрезе adoption скорости
train_cats.groupby(['AdoptionSpeed', 'Health'])['PetID'].count()
pv_h_train_cats = h_train_cats.pivot_table(values='PetID_Share', index='Health',   columns='AdoptionSpeed')
sns.heatmap(pv_h_train_cats, cmap='inferno_r')
sns.distplot(train_cats['Age'])
pv_age_train_cats = age_train_cats.pivot_table(values='PetID_Share', index='Age',   columns='AdoptionSpeed')
sns.heatmap(pv_age_train_cats, cmap='inferno_r')
#Создаем для собак датафреймы для описания распределений скорости adoption в разрезах различных категориальных фич
ms_train_dogs = pd.DataFrame(train_dogs.groupby([ 'AdoptionSpeed', 'MaturitySize'])['PetID'].count()).reset_index()
fr_train_dogs = pd.DataFrame(train_dogs.groupby(['AdoptionSpeed', 'FurLength'])['PetID'].count()).reset_index()
vc_train_dogs = pd.DataFrame(train_dogs.groupby(['AdoptionSpeed', 'Vaccinated'])['PetID'].count()).reset_index()
dw_train_dogs = pd.DataFrame(train_dogs.groupby(['AdoptionSpeed', 'Dewormed'])['PetID'].count()).reset_index()
st_train_dogs = pd.DataFrame(train_dogs.groupby(['AdoptionSpeed', 'Sterilized'])['PetID'].count()).reset_index()
h_train_dogs = pd.DataFrame(train_dogs.groupby(['AdoptionSpeed', 'Health'])['PetID'].count()).reset_index()
age_train_dogs = pd.DataFrame(train_dogs.groupby(['AdoptionSpeed', 'Age'])['PetID'].count()).reset_index()
#Добавляем колонку в датафреймы с разрезами с долей питомцев от общего числа для каждого значения категориальной фичи
ms_train_dogs['PetID_Share'] = ms_train_dogs['PetID']/14993
fr_train_dogs['PetID_Share'] = ms_train_dogs['PetID']/14993
vc_train_dogs['PetID_Share'] = ms_train_dogs['PetID']/14993
dw_train_dogs['PetID_Share'] = ms_train_dogs['PetID']/14993
st_train_dogs['PetID_Share'] = ms_train_dogs['PetID']/14993
h_train_dogs['PetID_Share'] = ms_train_dogs['PetID']/14993
age_train_dogs['PetID_Share'] = ms_train_dogs['PetID']/14993
#распределение размеров по количеству питомцев
train_dogs.groupby(['MaturitySize'])['PetID'].count()
sns.distplot(train_dogs['MaturitySize'])
#распределение размеров питомцев в разрезе adoption скорости
train_dogs.groupby(['MaturitySize', 'AdoptionSpeed'])['PetID'].count()
# Хитмап по размеру питомца
pv_ms_train_dogs = ms_train_dogs.pivot_table(values='PetID_Share', index='MaturitySize',   columns='AdoptionSpeed')
sns.heatmap(pv_ms_train_dogs, cmap='inferno_r')
#распределение длины шерсти по количеству питомцев
train_dogs.groupby(['FurLength'])['PetID'].count()
sns.distplot(train_dogs['FurLength'])
#распределение длины шерсти питомцев в разрезе adoption скорости
train_dogs.groupby(['AdoptionSpeed', 'FurLength'])['PetID'].count()
#Хитмап по длине шерсти питомца
pv_fr_train_dogs = fr_train_dogs.pivot_table(values='PetID_Share', index='FurLength',   columns='AdoptionSpeed')
sns.heatmap(pv_fr_train_dogs, cmap='inferno_r')
#распределение статуса вакцинации по количеству питомцев
train_dogs.groupby(['Vaccinated'])['PetID'].count()
sns.distplot(train_dogs['Vaccinated'])
#распределение статуса вакцинации в разрезе adoption скорости
train_dogs.groupby(['AdoptionSpeed', 'Vaccinated'])['PetID'].count()
pv_vc_train_dogs = vc_train_dogs.pivot_table(values='PetID_Share', index='Vaccinated', columns='AdoptionSpeed')
sns.heatmap(pv_vc_train_dogs, cmap='inferno_r')
#распределение статусов применения противоглистных препаратов по количеству питомцев
train_dogs.groupby(['Dewormed'])['PetID'].count()
sns.distplot(train_dogs['Dewormed'])
#распределение статуса применения противоглистных препаратов в разрезе adoption скорости
train_dogs.groupby(['AdoptionSpeed', 'Dewormed'])['PetID'].count()
pv_dw_train_dogs = dw_train_dogs.pivot_table(values='PetID_Share', index='Dewormed',   columns='AdoptionSpeed')
sns.heatmap(pv_dw_train_dogs, cmap='inferno_r')
#распределение статусов стерилизации по количеству питомцев
train_dogs.groupby(['Sterilized'])['PetID'].count()
sns.distplot(train_dogs['Sterilized'])
#распределение статуса стерилизации в разрезе adoption скорости
train_dogs.groupby(['AdoptionSpeed', 'Sterilized'])['PetID'].count()
pv_st_train_dogs = st_train_dogs.pivot_table(values='PetID_Share', index='Sterilized',   columns='AdoptionSpeed')
sns.heatmap(pv_st_train_dogs, cmap='inferno_r')
#распределение статусов здоровья по количеству питомцев
train_dogs.groupby(['Health'])['PetID'].count()
sns.distplot(train_dogs['Health'])
#распределение статуса здоровья в разрезе adoption скорости
train_dogs.groupby(['AdoptionSpeed', 'Health'])['PetID'].count()
pv_h_train_dogs = h_train_dogs.pivot_table(values='PetID_Share', index='Health',   columns='AdoptionSpeed')
sns.heatmap(pv_h_train_dogs, cmap='inferno_r')
sns.distplot(train_dogs['Age'])
pv_age_train_dogs = age_train_dogs.pivot_table(values='PetID_Share', index='Age',   columns='AdoptionSpeed')
sns.heatmap(pv_age_train_dogs, cmap='inferno_r')




