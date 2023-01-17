import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

%matplotlib inline
data = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
data = data.replace('?', np.nan)
data.isna().sum()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))

ax1.barh(data['workclass'].value_counts().keys(), 
        data['workclass'].value_counts().values,
       )
ax1.set_title('"Workclass" value counts')


ax2.barh(data['occupation'].value_counts().keys(), 
        data['occupation'].value_counts().values,
       )
ax2.set_title('"Occupation" value counts')

fig.tight_layout()
plt.show()
occupation_workclass_none = len(data[(data['occupation'].isna()) & (data['workclass'].isna())])
print(f'Строк с незаполненными "occupation" и "workclass": {occupation_workclass_none}')
data = data.dropna()
data['good.salary'] = data['income'].apply(lambda x: x == '>50K')
print('Типы данных: ')
data.dtypes
print('Уникальные категории по номинальным признакам: ')
nominal_features = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for i, nf in enumerate(nominal_features):
    print(f'{i+1}) {nf} - {len(data[nf].unique())}')
    if len(data[nf].unique()) < 15:
        for unique in data[nf].unique():
            print(f'\t{unique}')
male_count = len(data[data['sex'] == 'Male'])
female_count = len(data[data['sex'] == 'Female'])
plt.bar(['Мужчины', 'Женщины'], [male_count, female_count])
plt.title('Колличество людей по половому признаку')
plt.show()
male_good_salary_count = len(data[(data['sex'] == 'Male') & (data['good.salary'])])
female_good_salary_count = len(data[(data['sex'] == 'Female') & (data['good.salary'])])

male_good_salary_propotion = male_good_salary_count / male_count
female_good_salary_propotion = female_good_salary_count / female_count
plt.bar(['Мужчины', 'Женщины'],
       [male_good_salary_propotion, female_good_salary_propotion])
plt.title('Доля хорошо зарабатывающих людей по половому признаку')
plt.show()
# Окончившие только подготовительные классы и elementary school
dropouts = data[(1 <= data['education.num']) & (data['education.num'] <= 3)]
dropouts_count = len(dropouts)
dropouts_good_salary_count = len(dropouts[dropouts['good.salary']])
dropouts_good_salary_propotion = dropouts_good_salary_count/dropouts_count

# Окончившие middle school и high grad
school = data[(4 <= data['education.num']) & (data['education.num'] <= 9)]
school_count = len(school)
school_good_salary_count = len(school[school['good.salary']])
school_good_salary_propotion = school_good_salary_count/school_count

# Окончившие колледжи и проф училища
college = data[(10 <= data['education.num']) & (data['education.num'] <= 12)]
college_count = len(college)
college_good_salary_count = len(college[college['good.salary']])
college_good_salary_propotion = college_good_salary_count/college_count


# Окончившие бакалавриат и магистратуру
graduates = data[(13 <= data['education.num']) & (data['education.num'] <= 14)]
graduates_count = len(graduates)
graduates_good_salary_count = len(graduates[graduates['good.salary']])
graduates_good_salary_propotion = graduates_good_salary_count/graduates_count

# Окончившие докторнатуру
postgraduates = data[(15 <= data['education.num']) & (data['education.num'] <= 16)]
postgraduates_count = len(postgraduates)
postgraduates_good_salary_count = len(postgraduates[postgraduates['good.salary']])
postgraduates_good_salary_propotion = postgraduates_good_salary_count/postgraduates_count
plt.bar(x=['dropouts', 'school', 'college', 'graduates', 'postgraduates'],
       height=[dropouts_good_salary_propotion, school_good_salary_propotion, 
               college_good_salary_propotion, graduates_good_salary_propotion, 
               postgraduates_good_salary_propotion])
plt.title('Доля хорошо зарабатывающих людей по уровню образования')
plt.show()
white_race_count = len(data[data['race'] == 'White'])
white_race_good_salary_count = len(data[(data['race'] == 'White') & (data['good.salary'])])
white_race_good_salary_propotion = white_race_good_salary_count / white_race_count

not_white_race_count = len(data[data['race'] != 'White'])
not_white_race_good_salary_count = len(data[(data['race'] != 'White') & (data['good.salary'])])
not_white_race_good_salary_propotion = not_white_race_good_salary_count / not_white_race_count
plt.bar(x=['White', 'Not white'],
       height=[white_race_good_salary_propotion, not_white_race_good_salary_propotion])
plt.title('Доля хорошо зарабатывающих людей по рассе')
plt.show()
maried_females = data[(data['sex'] == 'Female') & (data['relationship'] == 'Wife') & (data['age'] > 30)]
maried_females_count = len(maried_females)
maried_females_good_salary_count = len(maried_females[maried_females['good.salary']])
maried_females_good_salary_propotion = maried_females_good_salary_count / maried_females_count


not_maried_females = data[(data['sex'] == 'Female') & (data['relationship'] != 'Wife') & (data['age'] > 30)]
not_maried_females_count = len(not_maried_females)
not_maried_females_good_salary_count = len(not_maried_females[not_maried_females['good.salary']])
not_maried_females_good_salary_propotion = not_maried_females_good_salary_count / not_maried_females_count
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7))

ax1.bar(x=['Замужние', 'Незамужние'],
       height=[maried_females_count, not_maried_females_count])
ax1.set_title('Колличество женщин по семейному положению')

ax2.bar(x=['Замужние', 'Незамужние'],
       height=[maried_females_good_salary_propotion, not_maried_females_good_salary_propotion])
ax2.set_title('Доля хорошо зарабатывающих женщин по семейному положению')

fig.tight_layout()
plt.show()
def сount_ocupation_gsp(occupation: str) -> float:
    """
    Вычисляет долю людей указанной сферы деятельности, получающих хорошую зарплату
    
    :param occupation: сфера деятельности
    """
    occupation_count = len(data[data['occupation'] == occupation])
    occupation_gs_count = len(data[(data['occupation'] == occupation) & (data['good.salary'])])
    
    return occupation_gs_count / occupation_count
occupations = data['occupation'].value_counts().keys()
occupation_сounts = data['occupation'].value_counts().values
ocupation_good_salary_propotions = [сount_ocupation_gsp(o) for o in occupations]
plt.plot(occupation_сounts,
         ocupation_good_salary_propotions)
plt.xlabel('Колличество человек, занимающихся данным видом деятельности')
plt.ylabel('Доля людей, получающих хорошую зарплату')
plt.title('Зависимость доли людей с хорошей зарплатой от конкурентности в сфере их деятельности')
fig.tight_layout()
males = data[data['sex'] == 'Female']
females = data[data['sex'] == 'Male']
fig, ax = plt.subplots(1,1)

ax.boxplot([males['education.num'], females['education.num']])
ax.set_xticklabels(['Мужчины', 'Женщины'])
ax.set_title('Уровень образования')

plt.show()
whites = data[data['race'] == 'White']
not_whites = data[data['race'] != 'White']
fig, ax = plt.subplots(1,1)

ax.boxplot([whites['education.num'], not_whites['education.num']])
ax.set_xticklabels(['Белокожие', 'Не белокожие'])
ax.set_title('Уровень образования')

plt.show()