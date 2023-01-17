import numpy as np

np.random.seed(1227)

import pandas as pd



# Graph

import matplotlib.pyplot as plt



# Read dataset

data = pd.read_csv("../input/train.csv")

print(data.columns.values)

print(data.head(5))



male_color = "#0d86ff"

female_color = "#ff5e5e"

survival_color = "#fbc900"

died_color = "#6f6f6f"

class_1_color = "#009999"

class_2_color = "#ff6600"

class_3_color = "#cc3300"

all_color = "#9900ff"



class_color_array = [class_1_color, class_2_color, class_3_color]

gender_color_array = [male_color, female_color]

survival_color_array= [died_color, survival_color]



%matplotlib inline
# print(data["Survived"].isnull().any())

# No null values



# Create array survived by gender  

survival = [0, 1]

genders = ["male", "female"]

survival_distrubution = np.array(np.zeros(len(survival)))

for i in survival:

    survival_distrubution[i] = len(data[data.Survived == i].values)



# print(survival_distrubution)

    

gender_survived = np.array(np.zeros(len(survival)*len(genders)))

for i in range(0, len(survival)):

    for j in range(0, len(genders)):

        gender_survived[2*i + j] = len(data[(data.Survived == i) & (data.Sex == genders[j])].values)
# Gender percentage

genders = [len(data[data.Sex == "male"].values), len(data[data.Sex == "female"].values)]

fig, ax = plt.subplots(figsize=(10, 10))

size = 1

ax.pie(survival_distrubution, radius=1, colors=gender_color_array, labels=["Males", "Females"], autopct='%1.2f%%',

       wedgeprops=dict(width=size, edgecolor='w'))

ax.set(aspect="equal", title='Gender')

plt.show()
# Survived by gender

genders = [len(data[data.Sex == "male"].values), len(data[data.Sex == "female"].values)]

survival_rate = [len(data[data.Survived == 0].values), len(data[data.Survived == 1].values)]

survival_rates = [0 , 0]

survival_rates[0] = (len(data[(data.Sex == "male") & (data.Survived == 1) ].values)/genders[0]) * 100

survival_rates[1] = (len(data[(data.Sex == "female") & (data.Survived == 1) ].values)/genders[1]) * 100



died_array = [0, 0]

died_array[0] = (len(data[(data.Survived == 0) & (data.Sex == "male")]) / survival_rate[0])*100

died_array[1] = (len(data[(data.Survived == 0) & (data.Sex == "female")]) / survival_rate[0])*100



survival_array = [0, 0]

survival_array[0] = (len(data[(data.Survived == 1) & (data.Sex == "male")]) / survival_rate[1])*100

survival_array[1] = (len(data[(data.Survived == 1) & (data.Sex == "female")]) / survival_rate[1])*100



male_array = [0, 0]

male_array[0] = (len(data[(data.Survived == 0) & (data.Sex == "male")])/genders[0])*100

male_array[1] = (len(data[(data.Survived == 1) & (data.Sex == "male")])/genders[0])*100



female_array = [0, 0]

female_array[0] = (len(data[(data.Survived == 0) & (data.Sex == "female")])/genders[1])*100

female_array[1] = (len(data[(data.Survived == 1) & (data.Sex == "female")])/genders[1])*100



fig, ax = plt.subplots(figsize=(10, 5))

index = np.arange(len(survival_rates))

bar_width = 0.5

opacity = 1

rects1 = ax.bar(index[0], survival_rates[0], bar_width,

                alpha=opacity, color=male_color, label='Males')

rects1 = ax.bar(index[1], survival_rates[1], bar_width,

                alpha=opacity, color=female_color, label='Females')

for i in range(0, len(index)):

    ax.text(index[i] - (bar_width/8), survival_rates[i] + 1, str(round(survival_rates[i], 2)) + ' %', fontweight='bold')

ax.set_xlabel('Gender')

ax.set_ylabel('% survived')

ax.set_xticks(index)

ax.set_xticklabels(('Males', 'Females'))

ax.set_title('% survived by gender')

ax.legend()

fig.tight_layout()

plt.show()
# Survived percentage by gender

fig, ax = plt.subplots(figsize=(10, 10))

size = 0.3

ax.pie(survival_distrubution, radius=1, colors=survival_color_array, labels=["Died", "Survived"],

       wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(gender_survived, radius=1 - size, colors=gender_color_array*2,

       labels=["Males died", "Females Died", "Males survived", "Females survivevd"],

       autopct='%1.2f%%', wedgeprops=dict(width=size, edgecolor='w'), pctdistance=0.75)

ax.set(aspect="equal", title='Survived by gender') 

plt.show()



fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].pie(died_array, colors=gender_color_array, labels=["Males", "Females"], autopct='%1.2f%%',

       wedgeprops=dict(width=1, edgecolor='w'))

ax[0, 0].set(aspect="equal", title='Died')

ax[0, 1].pie(survival_array, colors=gender_color_array, labels=["Males", "Females"], autopct='%1.2f%%',

       wedgeprops=dict(width=1, edgecolor='w'))

ax[0, 1].set(aspect="equal", title='Survived')

ax[1, 0].pie(male_array, colors=survival_color_array, labels=["Died", "Survived"], autopct='%1.2f%%',

       wedgeprops=dict(width=1, edgecolor='w'))

ax[1, 0].set(aspect="equal", title='Males')

ax[1, 1].pie(female_array, colors=survival_color_array, labels=["Died", "Survived"], autopct='%1.2f%%',

       wedgeprops=dict(width=1, edgecolor='w'))

ax[1,1].set(aspect="equal", title='Females')

plt.show()
# Class by gender

genders_array = ["male", "female"]

survived_array = [0, 1]

classes = [1, 2, 3]

gender_class = np.array(np.zeros(len(genders) * len(classes)))

for i in range(0, len(genders)):

    for j in range(0, len(classes)):

        stevec = (i * 2) + (j + i)

        gender_class[stevec] = len(data[(data.Sex == genders_array[i]) & (data.Pclass == classes[j])].values)



        

class_gender_distribution = [ [0, 0],

                              [0, 0], 

                              [0, 0] ]



for i in range(0, len(classes)):

    for j in range(0, len(genders_array)):

        class_gender_distribution[i][j] = (len(data[(data.Sex == genders_array[j]) & (data.Pclass == classes[i])])/len(data[data.Pclass == classes[i]]))*100 

            

fig, ax = plt.subplots(figsize=(10, 10))

size = 0.3

ax.pie(genders, radius=1, colors=gender_color_array, labels=["Male", "Female"], wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(gender_class, radius=1 - size, colors=class_color_array*2,

       labels=["Males #1", "Males #2", "Males #3", "Females #1", "Females #2", "Females #3"],

       autopct='%1.2f%%', wedgeprops=dict(width=size, edgecolor='w'), pctdistance=0.75)

ax.set(aspect="equal", title='Classes by gender')

plt.show()





fig, ax = plt.subplots(1, 3, figsize=(10, 10))

ax[0].pie(class_gender_distribution[0], radius=1, colors=gender_color_array, labels=["Males", "Females"], autopct='%1.2f%%',

       wedgeprops=dict(width=1, edgecolor='w'))

ax[0].set(aspect="equal", title='Class 1')

ax[1].pie(class_gender_distribution[1], colors=gender_color_array, labels=["Males", "Females"], autopct='%1.2f%%',

       wedgeprops=dict(width=1, edgecolor='w'))

ax[1].set(aspect="equal", title='Class 2')

ax[2].pie(class_gender_distribution[2], colors=gender_color_array, labels=["Males", "Females"], autopct='%1.2f%%',

       wedgeprops=dict(width=1, edgecolor='w'))

ax[2].set(aspect="equal", title='Class 3')

plt.show()
# Classes by survival

survival = [0, 1]

classes = [1, 2, 3]

survival_class = np.array(np.zeros(len(survival) * len(classes)))

for i in range(0, len(survival)):

    for j in range(0, len(classes)):

        stevec = (i * 2) + (j + i)

        survival_class[stevec] = len(data[(data.Survived == survival[i]) & (data.Pclass == classes[j])].values)

        

class_survival_distribution = [ [0, 0],

                                [0, 0],

                                [0, 0] ]



for i in range(0, len(classes)):

    for j in range(0, len(survived_array)):

        class_survival_distribution[i][j] = (len(data[(data.Survived == survived_array[j]) & (data.Pclass == classes[i])])/len(data[data.Pclass == classes[i]]))*100 



fig, ax = plt.subplots(figsize=(10, 10))

size = 0.3

ax.pie(survival_distrubution, radius=1, colors=survival_color_array, labels=["Died", "Survived"],

       wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(survival_class, radius=1 - size, colors=class_color_array,

       labels=["Died #1", "Died #2", "Died #3", "Survived #1", "Survived #2", "Survived #3"],

       autopct='%1.2f%%', wedgeprops=dict(width=size, edgecolor='w'), pctdistance=0.75)



ax.set(aspect="equal", title='Survival by class')

plt.show()



fig, ax = plt.subplots(1, 3, figsize=(10, 10))

ax[0].pie(class_survival_distribution[0], radius=1, colors=survival_color_array, labels=["Died", "Survived"], autopct='%1.2f%%',

       wedgeprops=dict(width=1, edgecolor='w'))

ax[0].set(aspect="equal", title='Class 1')

ax[1].pie(class_survival_distribution[1], colors=survival_color_array, labels=["Died", "Survived"], autopct='%1.2f%%',

       wedgeprops=dict(width=1, edgecolor='w'))

ax[1].set(aspect="equal", title='Class 2')

ax[2].pie(class_survival_distribution[2], colors=survival_color_array, labels=["Died", "Survived"], autopct='%1.2f%%',

       wedgeprops=dict(width=1, edgecolor='w'))

ax[2].set(aspect="equal", title='Class 3')

plt.show()
# Survival by gender and class

n_groups = 3

percentage = np.array(np.zeros((len(genders_array), n_groups)))

for i in range(0, len(genders_array)):

    for j in range(0, n_groups):

        percentage[i][j] = len(

            data[(data.Sex == genders_array[i]) & (data.Survived == 1) & (data.Pclass == classes[j])]) / len(data[(data.Sex == genders_array[i]) & (data.Pclass == classes[j])])



means_men = percentage[0]*100

means_women = percentage[1]*100



fig, ax = plt.subplots(figsize=(10, 5))

index = np.arange(n_groups)

bar_width = 0.35

opacity = 1

error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, means_men, bar_width,

                alpha=opacity, color=male_color, error_kw=error_config, label='Males')

rects2 = ax.bar(index + bar_width, means_women, bar_width,

                alpha=opacity, color=female_color, error_kw=error_config, label='Females')

for i in range(0, len(index)):

    ax.text(index[i] - (bar_width/4), means_men[i] + 1, str(round(means_men[i], 2)) + ' %', fontweight='bold')

for i in range(0, len(index)):

    ax.text(index[i] - (bar_width/4) + bar_width, means_women[i] + 1, str(round(means_women[i], 2)) + ' %', fontweight='bold')

ax.set_xlabel('Classes')

ax.set_ylabel('% survived')

ax.set_title('% survived by class and gender')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(('Class 1', 'Class 2', 'Class 3'))

ax.legend()

fig.tight_layout()

plt.show()
# ----- AGE ANALYSIS  -----

data_age = data.copy()



# check for null values

# print(data["Age"].isnull().any())

# Replace null values with rounded to int mean

data_age["Age"].fillna(int(round(data_age["Age"].mean())), inplace=True)

# Replace anyone who is less than 1 years old to be 1 years old

data_age[data_age.Age < 1] = 1

# or most frequent value

# data_age_mode["Age"].fillna(data_age_mode["Age"].value_counts().index[0], inplace=True)



# Create age bins

age_bins = []

for i in range(0, 90, 5):

    age_bins.append(i)



age_distribution = pd.cut(data_age["Age"].values, age_bins)

categories = age_distribution.categories.format()



# Age distribution plot

fig, ax = plt.subplots(figsize=(20, 5))

ax.bar(age_distribution.categories.format(), age_distribution.value_counts().values)

ax.set(xlabel='Age bins', ylabel='No. people',

       title='No. people by ages')

ax.grid()

plt.show()



# Survival by age

survived_age_distribution = pd.cut(data_age[data.Survived == 1]["Age"].values, age_bins).value_counts().values



# Age/Survived distribution plot

fig, ax = plt.subplots(figsize=(20, 5))

ax.bar(categories, survived_age_distribution)

ax.set(xlabel='Age bins', ylabel='No. people',

       title='No. people by ages who survived')

ax.grid()

plt.show()



age_distribution_array = pd.cut(data_age["Age"].values, age_bins).value_counts().values

survived_age_distribution_array = pd.cut(data_age[data.Survived == 1]["Age"].values, age_bins).value_counts().values

procentage_survival_by_age = np.array(np.zeros(len(age_distribution.value_counts().values)))

for i in range(0, len(age_distribution.value_counts().values)):

    if(age_distribution_array[i] != 0):

        procentage_survival_by_age[i] = (survived_age_distribution_array[i] / age_distribution_array[i])*100

        

# Age/Survived procentage distribution plot

fig, ax = plt.subplots(figsize=(20, 5))

ax.bar(categories, procentage_survival_by_age)

ax.set(xlabel='Age bins', ylabel='% survived',

       title='% survived by age')

ax.grid()

plt.show()
# Box plot

survived_ages = data_age[data_age.Survived == 1]["Age"].values

died_ages = data_age[data_age.Survived == 0]["Age"].values



box_plot = [died_ages, survived_ages]



fig1, ax1 = plt.subplots(figsize=(10, 10))

ax1.set_title('Ages bar survival')

ax1.boxplot(box_plot)

plt.grid()

ax1.set_yticks([step for step in range(1, 90, 5)])

ax1.set_ylabel('Age')

ax1.set_xticklabels(('Died', 'Survived'))

plt.show()

# Survival by gender and class

n_groups = len(categories)

# survived = pd.DataFrame(columns=['Age', 'Sex'])



genders = ["male", "female"]

survived_by_gender = [[]] * len(genders)

all_by_gender = [[]] * len(genders)

for i in range(0, len(genders)):

    survived_by_gender[i] = pd.cut(data_age[(data_age.Sex == genders[i]) & (data_age.Survived == 1)]["Age"].values, age_bins).value_counts().values

    all_by_gender[i] = pd.cut(data_age[data_age.Sex == genders[i]]["Age"].values, age_bins).value_counts().values 



gender_age_procentage = np.array(np.zeros((len(genders), len(categories))))   

for i in range(0, len(genders)):

    for j in range(0, len(categories)):

        if(all_by_gender[i][j] != 0):

            gender_age_procentage[i][j] = (survived_by_gender[i][j] / all_by_gender[i][j]) * 100

    

fig, ax = plt.subplots(figsize=(25, 10))

index = np.arange(n_groups)

bar_width = 0.35

opacity = 1

error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, gender_age_procentage[0], bar_width,

                alpha=opacity, color=male_color, error_kw=error_config, label='Males')

rects2 = ax.bar(index + bar_width, gender_age_procentage[1], bar_width,

                alpha=opacity, color=female_color, error_kw=error_config, label='Females')

for j in range(0, 2):

    for i in range(0, len(index)):

         ax.text(index[i] - (bar_width/3) + j*bar_width, gender_age_procentage[j][i] + 1, str(round(gender_age_procentage[j][i], 2)) + ' %', fontweight='bold')

ax.set_xlabel('Age bins')

ax.set_ylabel('% survived')

ax.set_title('% of people that survived')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(categories)

ax.legend()

fig.tight_layout()

plt.show()
# Check the surviving man

surviving_passenger = data_age[(data_age.Survived == 1) & (data_age.Age > 74) & (data_age.Age < 81)].values

# print(surviving_passenger)



print("Name: " + str(surviving_passenger[0][3]))

print("Class: " + str(surviving_passenger[0][2]))

print("Age: " + str(surviving_passenger[0][5]))

print("Cabin: " + str(surviving_passenger[0][10]))

if(surviving_passenger[0][11] == "S"):

    print("Port of embarkment: Southampton")

elif(surviving_passenger[0][11] == "C"):

    print("Port of embarkment: Cherbourg")

else:

    print("Port of embarkment: Queenstown")
n_groups = len(categories)

genders = ["male", "female"]

all_by_gender = [[]] * len(genders)

for i in range(0, len(genders)):

    all_by_gender[i] = pd.cut(data_age[data_age.Sex == genders[i]]["Age"].values, age_bins).value_counts().values

    

fig, ax = plt.subplots(figsize=(20, 10))

index = np.arange(n_groups)

bar_width = 0.35

opacity = 1

error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, all_by_gender[0], bar_width,

                alpha=opacity, color=male_color, error_kw=error_config, label='Males')

rects2 = ax.bar(index + bar_width, all_by_gender[1], bar_width,

                alpha=opacity, color=female_color, error_kw=error_config, label='Females')

for j in range(0, 2):

    for i in range(0, len(index)):

         ax.text(index[i] - (bar_width/3) + j*bar_width, all_by_gender[j][i] + 1, all_by_gender[j][i], fontweight='bold')

ax.set_xlabel('Age groups')

ax.set_ylabel('No. people')

ax.set_title('No. of people by gender and age group')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(categories)

ax.legend()

fig.tight_layout()

plt.show()
# ----- EMBARKED ANALYSIS  -----

emb_data = data.copy()



# Check for null values

# print(emb_data["Embarked"].isnull().any())

# Replace null values with most frequent value

emb_data["Embarked"].fillna(emb_data["Embarked"].mode()[0], inplace=True)

# No null values

# print(emb_data["Embarked"].isnull().any())
ports = np.unique(emb_data["Embarked"].values)

pass_by_port = np.array(np.zeros(len(ports)))

survived_by_port = np.array(np.zeros(len(ports)))

percentage_by_port = np.array(np.zeros(len(ports)))



for i in range(0, len(ports)):

    pass_by_port[i] = len(emb_data[emb_data.Embarked == ports[i]].values)

    survived_by_port[i] = len(emb_data[(emb_data.Embarked == ports[i]) & (emb_data.Survived == 1) ].values)

    percentage_by_port[i] = (survived_by_port[i] / pass_by_port[i]) * 100

    

fig, ax = plt.subplots(figsize=(20, 10))

index = np.arange(len(ports))

bar_width = 0.35

opacity = 1

error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, pass_by_port, bar_width,

                alpha=opacity, color=all_color, error_kw=error_config, label='All')

rects2 = ax.bar(index + bar_width, survived_by_port, bar_width,

                alpha=opacity, color=survival_color, error_kw=error_config, label='Survived')



for i in range(0, len(pass_by_port)):

    ax.text(index[i] - (bar_width/5), pass_by_port[i] + 1, pass_by_port[i], fontweight='bold')

    

for i in range(0, len(survived_by_port)):

    ax.text(index[i] - (bar_width/5) + bar_width, survived_by_port[i] + 1, survived_by_port[i], fontweight='bold')

ax.set_xlabel('Ports')

ax.set_ylabel('No. people')

ax.set_title('No. of people by port')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(ports)

ax.legend()

fig.tight_layout()

plt.show()



fig, ax = plt.subplots(figsize=(10, 10))

index = np.arange(len(ports))

rects1 = ax.bar(index, percentage_by_port,

                alpha=opacity, color=survival_color, error_kw=error_config, label='Percentage')

for i in range(0, len(percentage_by_port)):

    ax.text(index[i] - (bar_width/4), percentage_by_port[i] + 1, str(round(percentage_by_port[i], 2)) + ' %', fontweight='bold')

ax.set_xlabel('Ports')

ax.set_ylabel('% people')

ax.set_title("% survived by port")

ax.set_yticks([i for i in range(0, 105, 5)])

ax.set_xticks(index)

ax.set_xticklabels(ports)

ax.legend()

fig.tight_layout()

plt.show()
# Gender by port

ports = np.unique(emb_data["Embarked"].values)

pass_by_port = np.array(np.zeros(len(ports)))

genders = ["male", "female"]



gender_by_port = np.array(np.zeros(len(genders) * len(ports)))



for i in range(0, len(genders)):

    for j in range(0, len(ports)):

        stevec = (i * 2) + (j + i)

        gender_by_port[stevec] = len(emb_data[(emb_data.Sex == genders[i]) & (emb_data.Embarked == ports[j])].values)

    

fig, ax = plt.subplots(figsize=(10, 10))

index = np.arange(3)

bar_width = 0.35

opacity = 1

error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, gender_by_port[:3], bar_width,

                alpha=opacity, color=male_color, error_kw=error_config, label='Males')

rects2 = ax.bar(index + bar_width, gender_by_port[3:], bar_width,

                alpha=opacity, color=female_color, error_kw=error_config, label='Females')

for j in range(0, 2):

    for i in range(0, 3):

        ax.text(index[i] - (bar_width/4) + bar_width*j, gender_by_port[i + j*3] + 1, gender_by_port[i+j*3], fontweight='bold')

ax.set_xlabel('Ports')

ax.set_ylabel('No. of passengers')

ax.set_title('No. of passengers by port and gender')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(('C', 'Q', 'S'))

ax.legend()

fig.tight_layout()

plt.show()
classes = [1, 2, 3]

class_by_port = np.array(np.zeros(len(ports) * len(classes)))



for i in range(0, len(ports)):

    for j in range(0, len(classes)):

        stevec = (i * 2) + (j + i)

#         print(str(ports[i]) + " " + str(classes[j]))

        class_by_port[stevec] = len(emb_data[(emb_data.Embarked == ports[i]) & (emb_data.Pclass == classes[j])].values)

        

fig, ax = plt.subplots(figsize=(10, 5))

index = np.arange(3)

bar_width = 0.2

opacity = 1

error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, class_by_port[:3], bar_width,

                alpha=opacity, color=class_1_color, error_kw=error_config, label='Class 1')

rects2 = ax.bar(index + bar_width, class_by_port[3:6], bar_width,

                alpha=opacity, color=class_2_color, error_kw=error_config, label='Class 2')

rects3 = ax.bar(index + (bar_width+bar_width), class_by_port[6:], bar_width,

                alpha=opacity, color=class_3_color, error_kw=error_config, label='Class 3')

for j in range(0, len(index)):

    for i in range(0, len(index)):

        ax.text(index[i] - (bar_width/4) + bar_width*j, class_by_port[i + j*3] + 1, class_by_port[i+j*3], fontweight='bold')

ax.set_xlabel('Ports')

ax.set_ylabel('No. of passengers')

ax.set_title('No. of passengers by class and port')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(('C', 'Q', 'S'))

ax.legend()

fig.tight_layout()

plt.show()
# ----- SIBLINGS ANALYSIS  -----

data_sib = data.copy()



# check for null values

# print(data["SibSp"].isnull().any())



all_with_families = len(data_sib[data_sib.SibSp > 0].values)

all_without_families = len(data_sib[data_sib.SibSp == 0].values)



survived = [0, 0]

survived[0] = (len(data_sib[(data_sib.SibSp > 0) & (data_sib.Survived == 1)].values) / all_with_families) * 100

survived[1] =  (len(data_sib[(data_sib.SibSp == 0) & (data_sib.Survived == 1)].values) / all_with_families) * 100



fig, ax = plt.subplots(figsize=(10, 10))

index = np.arange(2)

opacity = 1

rects1 = ax.bar(index, survived,

                alpha=opacity, color=survival_color, label='Percentage')

for i in range(0, len(survived)):

    ax.text(index[i]-0.05, survived[i] + 1, str(round(survived[i], 2)) + ' %', fontweight='bold')

ax.set_ylabel('% survived')

ax.set_title("% survived by family size")

ax.set_yticks([i for i in range(0, 105, 5)])

ax.set_xticks(index)

ax.set_xticklabels(["Families", "Singles"])

ax.legend()

fig.tight_layout()

plt.show()
# survival by size of the family

all_family_sizes = np.unique(data_sib["SibSp"].values)

# print(all_family_sizes)



survival_by_sibsp = np.array(np.zeros((len(all_family_sizes), 3)))

for family_size in range(0, len(all_family_sizes)):

    survival_by_sibsp[family_size][0] = len(data_sib[data_sib.SibSp == all_family_sizes[family_size]].values)

    survival_by_sibsp[family_size][1] = len(data_sib[(data_sib.SibSp == all_family_sizes[family_size]) & (data_sib.Survived == 0)].values)

    survival_by_sibsp[family_size][2] = len(data_sib[(data_sib.SibSp == all_family_sizes[family_size]) & (data_sib.Survived == 1)].values) 

    

fig, ax = plt.subplots(figsize=(25, 10))

index = np.arange(3)

bar_width = 0.1

opacity = 1

error_config = {'ecolor': '0.3'}

for i in range(0, len(survival_by_sibsp)):

    rects = ax.bar(index+i*(bar_width), survival_by_sibsp[i], bar_width,

                    alpha=opacity, error_kw=error_config, label='Family size ' + str(all_family_sizes[i]))

    for j in range(0, len(index)):

        ax.text((index[j]+i*(bar_width))-(bar_width/4), survival_by_sibsp[i][j] + 10, survival_by_sibsp[i][j], fontweight='bold')

ax.set_ylabel('No. of passengers')

ax.set_title('No. of passengers family size and outcome')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(("All", "Died", "Survived"))

ax.legend()

fig.tight_layout()

plt.show()





print(data_sib[data_sib.SibSp == 8][["Pclass", "Sex", "Age"]])

print(data_sib[data_sib.SibSp == 5][["Pclass", "Sex", "Age"]])