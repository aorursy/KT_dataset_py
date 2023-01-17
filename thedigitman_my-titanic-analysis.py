import pandas as pd

import numpy as np



# for min_max scaling

from mlxtend.preprocessing import minmax_scaling



# plotting modules

import seaborn as sns

import missingno

import matplotlib.pyplot as plt



print("Libraries imported !")

input_file = '../input/titanic.csv'

df = pd.read_csv(input_file)

df.head(10)
print(f"{len(df.index)} records in the dataframe.\n")

total_passengers = len(df.index)

df.info()

df.count()
df.isnull().sum()
# Drop the 'Cabin' column

df.drop(columns="Cabin", inplace=True)

# Put placeholder values in 'Age' and 'Embarked'. 999 for 'Age' and XX for 'Embarked'



df["Age"].fillna(999, inplace = True)

df["Embarked"].fillna('XX', inplace = True)



df.isnull().sum()



fig = plt.figure(figsize=(20,1))

sns.countplot(y='Sex', data=df)

total_male, total_female = df.Sex.value_counts()

print(df.Sex.value_counts())
survived = df[df.Survived == 1]

survived_male, survived_female = survived.Sex.value_counts()



plt.bar(2, df.Sex.value_counts()['male'], label='Male')

plt.bar(2, survived.Sex.value_counts()['male'], label='Male Survived')

plt.bar(1, df.Sex.value_counts()['female'], label='Female')

plt.bar(1, survived.Sex.value_counts()['female'], label='Female Survived')





plt.ylabel('Count')

plt.xticks([])

plt.title("Persons Survived on Titanic")



plt.legend()

plt.show()

raw_data = pd.DataFrame([0, total_male, total_female, survived_male, survived_female, total_passengers])



scaled_data = minmax_scaling(raw_data, columns = [0])



plt.bar(2, scaled_data.iloc[1], label='Male')

plt.bar(2, scaled_data.iloc[3], label='Male Survived')

plt.bar(1, scaled_data.iloc[2], label='Female')

plt.bar(1, scaled_data.iloc[4], label='Female Survived')



plt.ylabel('Count')

plt.xticks([])

plt.title("Persons Survived on Titanic (scaled)")



plt.legend()

plt.show()



print(f"Male: {scaled_data.iloc[1] * 100}%")

print(f"Male Survived: {scaled_data.iloc[3] * 100}%")

print(f"Female: {scaled_data.iloc[2] * 100}%")

print(f"Female Survived: {scaled_data.iloc[4] * 100}%")



def plot_pie(df, index, title):

    ax[index].pie(df, shadow=True, autopct='%1.1f%%')

    ax[index].set_title(title)

    ax[index].axis('equal')



def get_ports_data(df):

    return [df.Embarked.value_counts()['S'], df.Embarked.value_counts()['C'], df.Embarked.value_counts()['Q']]

    

ports = get_ports_data(df)

ports_sur = get_ports_data(survived)



fig1, ax = plt.subplots(1, 2)

plot_pie(ports, 0, 'Ports of Embarkment')

plot_pie(ports_sur, 1, 'Survivors Ports of Embarkment')

labels = 'Southhampton', 'Cherbourg', 'Queenstown'

ax[1].legend(labels,

          title="Port of Embarkment",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))



plt.show()

print(df.Embarked.value_counts())

def plot_bar(df, index, title):

    ax[index].bar([0,1,2], df, tick_label=['1st', '2nd', '3rd'], color=['red', 'blue', 'green'])

    ax[index].set_title(title)



def get_classes_data(df):

    return [df.Pclass.value_counts()[1], df.Pclass.value_counts()[2], df.Pclass.value_counts()[3]]

    



classes = get_classes_data(df)

classes_surv = get_classes_data(survived)

    

fig1, ax = plt.subplots(1,2)



plot_bar(classes, 0, "Passenger Classes")

plot_bar(classes_surv, 1, "Survivors Classes")



plt.show()



print(df.Pclass.value_counts())
def get_scaled_data(df):

    raw_classes = pd.Series([0, df.Pclass.value_counts()[1], df.Pclass.value_counts()[2], df.Pclass.value_counts()[3], total_passengers])

    scaled_classes = minmax_scaling(raw_classes, columns = [0])

    scaled_classes_flat = [float(scaled_classes[1]), float(scaled_classes[2]), float(scaled_classes[3]), float(scaled_classes[4])]

    return scaled_classes_flat

    

def plot_bar_s(df, index, title):

    ax2[index].bar([0,1,2,3], df, tick_label=['1st', '2nd', '3rd', ''], color=['red', 'blue', 'green', 'white'])

    ax2[index].set_title(title)

    

classes_all = get_scaled_data(df)

classes_surv = get_scaled_data(survived)



fig1, ax2 = plt.subplots(1,2)

plot_bar_s(classes_all, 0, "Passenger Classes")

plot_bar_s(classes_surv, 1, "Survivors Classes")

plt.show()





def create_table(dfr):

    

    new_table = pd.DataFrame(0, columns=['1st', '2nd', '3rd', 'Total'], index=['Southhampton', 'Cherbourg', 'Queenstown', 'Total'])

    for pclass in ['1st', '2nd', '3rd']:

        for port in ['Southhampton', 'Cherbourg', 'Queenstown']:

            new_table[pclass][port] = dfr.PassengerId[df.Embarked == port[0]][dfr.Pclass == int(pclass[0])].count()

            new_table['Total'][port]  += new_table[pclass][port]

            new_table[pclass]['Total'] += new_table[pclass][port]

            new_table['Total']['Total'] += new_table[pclass][port]

    return new_table



total_table = create_table(df)

surv_table = create_table(survived)



surv_rate_table = pd.DataFrame(0, columns=['1st', '2nd', '3rd', 'Total'], index=['Southhampton', 'Cherbourg', 'Queenstown', 'Total'])

for pclass in ['1st', '2nd', '3rd', 'Total']:

    for port in ['Southhampton', 'Cherbourg', 'Queenstown', 'Total']:

        surv_rate_table[pclass][port] = (surv_table[pclass][port] / total_table[pclass][port]) * 100



print("Total Passengers")

print(total_table)

print("\nSurviving Passengers")

print(surv_table)

print("\nSurviving Passengers %")

print(surv_rate_table)