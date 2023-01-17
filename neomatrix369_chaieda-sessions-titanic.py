# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Import library and dataset

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid", font_scale=1.75)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import math



import matplotlib.pyplot as plt



# prettify plots\n

plt.rcParams['figure.figsize'] = [20.0, 5.0]

    

%matplotlib inline



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
training_dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

print("Column count:", len(training_dataset.columns))

training_dataset.dtypes
sensible_columns = ['Survived', 'Died', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
training_dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

training_dataset
!pip install -U missingno
import missingno as msno
msno.bar(training_dataset)
sns.heatmap(training_dataset.isnull(), yticklabels=False, cbar=False,cmap='viridis')
def expand_embark_acronym(embarked):

    result = []

    mapping = {

            "C": "Cherbourg",

            "S": "Southampton",

            "Q": "Queenstown"

    }    

    for each in embarked.values:

        if len(str(each)) > 1:

            result.append(each)

        else:        

            if each in mapping:

                result.append(mapping[each])

            else:

                result.append("Unknown")

    return result



def expand_pclass_acronym(pclass):

    result = []

    mapping = {

            1: "1st class",

            2: "2nd class",

            3: "3rd class"

    }    

    for each in pclass.values:

        if len(str(each)) > 1:

            result.append(each)

        else:

            if each in mapping:

                result.append(mapping[each])

            else:

                result.append("Unknown")

    return result



def is_a_minor(age):

    if math.isnan(age):

        return "Unknown"

    

    if age < 18:

        return "Under 18 (minor)"

    

    return "Adult"



# See https://help.healthycities.org/hc/en-us/articles/219556208-How-are-the-different-age-groups-defined-

def apply_age_groups(age):

    result = []

    mapping = {

            1: "Infant",      # Infants: <1

           13: "Child",       # Children: <18, <11 or K - 7th grade

           18: "Teen",        # Teens: 13-17 (Teens, who are not Adults)

           66: "Adult",       # Adults: 20+ (includes adult teens: 18+)

           123: "Elderly"     # Elderly: 65+ (123 is the oldest age known till date)

    }    

    for each_age in age.values:

        if type(each_age) == str:

            result.append(category)

        else:

            category = "Unknown"

            if each_age != np.nan:

                for each_age_range in mapping:

                    if  each_age < each_age_range:

                        category = mapping[each_age_range]

                        break

            result.append(category)

    return result



def apply_age_ranges(age):

    result = []

    mapping = {

            6: "00-05 years",

           12: "06-11 years",     

           19: "12-18 years",

           31: "19-30 years",

           41: "31-40 years",

           51: "41-50 years",

           61: "51-60 years",

           71: "61-70 years",

           81: "71-80 years",

           91: "81-90 years",

           124: "91+ years",  # (123 is the oldest age known till date)

    }

            

    for each_age in age.values:

        if type(each_age) == str:

            result.append(category)

        else:

            category = "Unknown"

            if each_age != np.nan:

                for each_age_range in mapping:

                    if  each_age < each_age_range:

                        category = mapping[each_age_range]

                        break

            result.append(category)

    return result



def is_married_of_single(names, ages, sexes):

    result = []

    for name, age, sex in zip(names.values, ages.values, sexes.values):

        if age < 18:

            result.append("Not of legal age")

        else:

            if ('Mrs.' in name) or ('Mme.' in name):

                result.append("Married")

            elif ('Miss.' in name) or ('Ms.' in name) or ('Lady' in name) or ('Mlle.' in name):

                result.append("Single")

            else:

                result.append("Unknown")

    

    return result



def apply_travel_companions(siblings_spouse, parent_children):

    result = []

    for siblings_spouse_count, parent_children_count in zip(siblings_spouse.values, parent_children.values):

        if (siblings_spouse_count > 0) and (parent_children_count > 0):

            result.append("Parent/Children & Sibling/Spouse")

        else:

            if (siblings_spouse_count > 0):

                result.append("Sibling/Spouse")

            elif (parent_children_count > 0):

                result.append("Parent/Children")

            else:

                result.append("Alone")

    

    return result



def apply_fare_ranges(fare):

    result = []

    mapping = {

           11: "£000 - 010",

           21: "£011 - 020",     

           41: "£020 - 040",

           81: "£041 - 080",

          101: "£081 - 100",

          201: "£101 - 200",

          301: "£201 - 300",

          401: "£301 - 400",

          515: "£401 & above"  # in this case the max fare is around £512

    }    

    for each_fare in fare.values:

        if type(each_fare) == str:

            result.append(category)

        else:

            category = "Unknown"

            if each_fare != np.nan:

                for each_fare_range in mapping:

                    if  each_fare < each_fare_range:

                        category = mapping[each_fare_range]

                        break

            result.append(category)



    return result



def were_in_a_cabin_or_not(row):

    if type(row) is str:

        return "In a Cabin"

    return "Not in a Cabin"
## Loading the table again to regenerate the feature engineered columns from scratch

training_dataset = pd.read_csv('/kaggle/input/titanic/train.csv')



## Survived (or Died)

training_dataset['Died'] = abs(1 - training_dataset['Survived'])



## Embarked: Place of embarkation

training_dataset['Embarked'] = expand_embark_acronym(training_dataset['Embarked'])



# Pclass: Passenger Class

training_dataset['Pclass'] = expand_pclass_acronym(training_dataset['Pclass'])



# Age

training_dataset['Adult_or_minor'] = training_dataset['Age'].apply(is_a_minor)



females_filter = training_dataset['Sex'] == 'female'

adult_filter = training_dataset['Adult_or_minor'] == '2. Adult'



training_dataset['Marital_status'] = is_married_of_single(training_dataset['Name'], training_dataset['Age'], training_dataset['Sex']) 

training_dataset['Age_group'] = apply_age_groups(training_dataset['Age'])

training_dataset['Age_ranges'] = apply_age_ranges(training_dataset['Age'])



# SibSp and Parch: Sibling/Spouse counts, Parent/Children counts

training_dataset['Travel_companion'] = apply_travel_companions(training_dataset['SibSp'], training_dataset['Parch'])



# Fare: ticket fare across the different classes

training_dataset['Fare_range'] = apply_fare_ranges(training_dataset['Fare'])



# Cabin: ticket holder has a cabin or not

training_dataset['In_Cabin'] = training_dataset['Cabin'].apply(were_in_a_cabin_or_not)

training_dataset['Cabin'] = training_dataset['Cabin'].fillna('No cabin')
training_dataset
training_dataset[sensible_columns].describe()
### This is novel use of the percentiles param of the describe function

training_dataset[sensible_columns].describe(percentiles=np.arange(10)/10.0)
def print_count_of_passengers(dataset):

    total_ticket_holders = dataset.shape[0]

    siblings_count = dataset['SibSp'].sum()

    parents_children_count = dataset['Parch'].sum()



    print("siblings_count:", siblings_count)

    print("parents_children_count:", parents_children_count)

    print("total_ticket_holders:", total_ticket_holders)

    print("total (siblings, parents and children count):", siblings_count + parents_children_count)



    grand_total = total_ticket_holders + siblings_count + parents_children_count

    print("grand total (ticket holders, siblings, parents, children count):", grand_total)

    

    return grand_total



training_dataset_passengers_count = print_count_of_passengers(training_dataset)
g = sns.countplot(x=training_dataset['Survived'])

plt.legend(loc='upper right')

g.set(xlabel="Survival", xticklabels=["Died", "Survived"]) # "0=Died", "1=Survived"
training_dataset.pivot_table(values=['Survived', 'Died'], index=['Sex'], aggfunc=np.mean)
gender_pivot_table = training_dataset.pivot_table(values=['Survived', 'Died'], index=['Sex'], aggfunc=np.sum)

gender_pivot_table
gender_pivot_table.plot(kind='barh')
training_dataset.pivot_table(values=['Survived', 'Died'], index=['Pclass'], aggfunc=np.mean)
passenger_class_pivot_table = training_dataset.pivot_table(values=['Survived', 'Died'], index=['Pclass'], aggfunc=np.sum)

passenger_class_pivot_table
passenger_class_pivot_table.plot(kind='barh')

plt.ylabel('Passenger Class')
training_dataset.pivot_table(values=['Survived', 'Died'], index=['Sex', 'Pclass'], aggfunc=np.mean)
gender_passenger_class_pivot_table = training_dataset.pivot_table(values=['Survived', 'Died'], index=['Sex', 'Pclass'], aggfunc=np.sum)

gender_passenger_class_pivot_table
g = sns.catplot(x="Survived", hue="Pclass", col='Sex', data=training_dataset.sort_values(by='Pclass'), kind='count')

g.set(xticklabels=['Died', 'Survived'], xlabel="Survival")
g = sns.catplot(x="Survived", y="Fare", data=training_dataset, kind="bar");

g.set(xticklabels=['Died', 'Survived'], xlabel="Survival", title="Sum of fares collected and Survival")
g = sns.catplot(x="Survived", y="Fare", hue="Pclass", data=training_dataset.sort_values(by='Pclass'), kind="bar");

g.set(xticklabels=['Died', 'Survived'], xlabel="Survival", title="Sum of fares collected across the three Passenger Classes and Survival")
g = sns.catplot(y="Fare_range", hue="Survived", data=training_dataset.sort_values(by='Fare'), kind="count")

g.set(ylabel="Fare range", title="Fare ranges and Survival")

new_labels = ['Died', 'Survived']

for t, l in zip(g._legend.texts, new_labels): 

    t.set_text(l)



g.fig.set_figwidth(30)
def passenger_class_filtered_dataset(passenger_class):

    dataset = training_dataset.copy()

    class_filter = dataset['Pclass'] == passenger_class

    return dataset[class_filter]



def draw_passenger_class_chart(passenger_class, title):

    dataset = passenger_class_filtered_dataset(passenger_class)

    g = sns.catplot(y="Fare_range", hue="Survived", data=dataset.sort_values(by='Pclass'), kind="count")

    g.set(ylabel="Fare range", title=title)

    new_labels = ['Died', 'Survived']

    for t, l in zip(g._legend.texts, new_labels): 

        t.set_text(l)



    g.fig.set_figwidth(30)
draw_passenger_class_chart('1st class', "First class passengers and Fare ranges")
draw_passenger_class_chart('2nd class', "Second class passengers and Fare ranges")
draw_passenger_class_chart('3rd class', "Third class passengers and Fare ranges")
zero_fare_filter = training_dataset['Fare'] == 0.0

training_dataset[zero_fare_filter].pivot_table(index=['Pclass', 'Cabin', 'Ticket'])
zero_filter_columns = ['Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived', 'Died', 'Pclass', 'Cabin', 'Ticket', 'Marital_status', 'Age_group', 'Age_ranges']

training_dataset[zero_fare_filter][zero_filter_columns]
training_dataset[zero_fare_filter][zero_filter_columns].describe()
training_dataset[zero_fare_filter].pivot_table(values=['Survived', 'Died'], index=['Pclass', 'Cabin'], aggfunc=np.sum)
zero_fare_pivot_table = training_dataset[zero_fare_filter].pivot_table(values=['Survived', 'Died'], index=['Pclass'], aggfunc=np.sum)

zero_fare_pivot_table
zero_fare_pivot_table.plot(kind='barh')
passenger_class_pivot_table = training_dataset.pivot_table(values=['Survived', 'Died'], index=['Pclass'], aggfunc=np.sum)

passenger_class_pivot_table
passenger_class_pivot_table.plot(kind='barh')
g = sns.catplot(x="Pclass", y="Fare", hue="Survived", data=training_dataset.sort_values(by='Pclass'))

g.set(xlabel="Passenger Class")



new_labels = ['Died', 'Survived']

for t, l in zip(g._legend.texts, new_labels): 

    t.set_text(l)



g.fig.set_figwidth(16)
g = sns.catplot(y="Age", x="Sex", hue="Survived", data=training_dataset)

new_labels = ['Died', 'Survived']

for t, l in zip(g._legend.texts, new_labels): 

    t.set_text(l)

g.fig.set_figwidth(16)
g = sns.catplot(x="Age_group", hue="Survived", data=training_dataset.sort_values(by='Age'), kind='count')

new_labels = ['Died', 'Survived']

for t, l in zip(g._legend.texts, new_labels): 

    t.set_text(l)

g.fig.set_figwidth(16)

g.set(xlabel="Age groups")
g = sns.catplot(col="Sex", x="Survived", hue="Age_group", data=training_dataset.sort_values(by='Age'), kind='count')

g.set(xlabel="Survival", xticklabels=['Died', 'Survived'])

g.fig.set_figwidth(16)
g = sns.catplot(x="Survived", hue="Age_ranges", data=training_dataset.sort_values(by='Age'), kind='count')

g.set(xlabel="Survival", xticklabels=['Died', 'Survived'])

g.set(xlabel="Survival")

g.fig.set_figwidth(16)
sibling_spouse_pivot_table = training_dataset.pivot_table(values=['Survived', 'Died'], index='SibSp', aggfunc=np.sum)

sibling_spouse_pivot_table
sibling_spouse_pivot_table.plot(kind='barh')

plt.ylabel('Sibling/spouse count')
parent_children_pivot_table = training_dataset.pivot_table(values=['Survived', 'Died'], index='Parch', aggfunc=np.sum)

parent_children_pivot_table
parent_children_pivot_table.plot(kind='barh')

plt.ylabel('Parent/children count')
travel_companion_pivot_table = training_dataset.pivot_table(values=['Survived', 'Died'], index='Travel_companion', aggfunc=np.sum)

travel_companion_pivot_table
travel_companion_pivot_table.plot(kind='barh')

plt.ylabel('Travel companion')
training_dataset.pivot_table(values=['Survived', 'Died'], index='Adult_or_minor', aggfunc=np.sum)
adult_or_minor_pivot_table = training_dataset.pivot_table(values=['Survived', 'Died'], index=['Adult_or_minor', 'Sex'], aggfunc=np.sum)

adult_or_minor_pivot_table
g = sns.catplot(x="Adult_or_minor", col='Sex', hue="Survived", kind="count", data=training_dataset.sort_values(by='Age'));

new_labels = ['Died', 'Survived']

for t, l in zip(g._legend.texts, new_labels): 

    t.set_text(l)

g.fig.set_figwidth(16)
unknown_marital_status_filter = training_dataset['Marital_status'] == 'Unknown'

training_dataset[females_filter & unknown_marital_status_filter]
not_legal_marital_status_filter = training_dataset['Marital_status'] == "Not of legal age"

training_dataset[not_legal_marital_status_filter]
female_marital_status_pivot_table = training_dataset[females_filter].pivot_table(

    values=['Survived', 'Died'], index='Marital_status', aggfunc=np.sum

)

female_marital_status_pivot_table
female_marital_status_pivot_table.plot(kind='barh')

plt.ylabel('Marital status')

plt.title('Adult females (Single or married)')
marital_status_pivot_table = training_dataset.pivot_table(values=['Survived', 'Died'], index='Marital_status', aggfunc=np.sum)

marital_status_pivot_table
marital_status_pivot_table.plot(kind='barh')

plt.ylabel('Marital status')

plt.title('All ages')
embarked_pivot_table = training_dataset.pivot_table(values=['Survived', 'Died'], index='Embarked', aggfunc=np.sum)

embarked_pivot_table
embarked_pivot_table.plot(kind='barh')
embarked_passenger_class_pivot_table = training_dataset.pivot_table(

    values=['Survived', 'Died'], index=['Embarked', 'Pclass'], aggfunc=np.sum

)

embarked_passenger_class_pivot_table
g = sns.catplot(col="Embarked", x='Pclass', hue="Survived", kind="count", data=training_dataset.sort_values(by='Pclass'));

new_labels = ['Died', 'Survived']

for t, l in zip(g._legend.texts, new_labels): 

    t.set_text(l)

g.fig.set_figwidth(16)

g.set(xlabel="Passenger Class")
in_cabin_pivot_table = training_dataset.pivot_table(values=['Survived', 'Died'], index='In_Cabin', aggfunc=np.sum)

in_cabin_pivot_table
in_cabin_pivot_table.plot(kind='barh')

plt.ylabel('Cabin')
training_dataset.pivot_table(values=['Survived', 'Died'], index=['In_Cabin', 'Pclass'], aggfunc=np.sum)
g = sns.catplot(col="In_Cabin", x='Pclass', hue="Survived", kind="count", data=training_dataset.sort_values(by='Pclass'));

new_labels = ['Died', 'Survived']

for t, l in zip(g._legend.texts, new_labels): 

    t.set_text(l)

g.fig.set_figwidth(16)

g.set(xlabel="Passenger Class")
training_dataset['Ticket'].describe()
training_dataset['Ticket'].value_counts()
sorted_training_dataset = training_dataset.sort_values(by=['Pclass', 'Ticket', 'Cabin', 'Fare'], ascending=True)

first_class_filter = sorted_training_dataset['Pclass'] == '1st class'

second_class_filter = sorted_training_dataset['Pclass'] == '2nd class'

third_class_filter = sorted_training_dataset['Pclass'] == '3rd class'
first_class_sorted = sorted_training_dataset[first_class_filter]

first_class_pivot_table = first_class_sorted.pivot_table(values=['Fare'], index=['Cabin', 'Ticket'], aggfunc=np.mean)

print("Tickets count:", first_class_sorted.shape[0])

first_class_pivot_table
first_class_sorted['Fare'].describe(percentiles=np.arange(10)/10.0)
plt.figure(figsize=(20,4))

first_class_sorted['Fare'].hist(bins=40)
second_class_sorted = sorted_training_dataset[second_class_filter]

second_class_pivot_table = second_class_sorted.pivot_table(values=['Fare'], index=['Cabin', 'Ticket'], aggfunc=np.mean)

print("Tickets count:", second_class_sorted.shape[0])

second_class_pivot_table
second_class_sorted['Fare'].describe(percentiles=np.arange(10)/10.0)
plt.figure(figsize=(20,4))

second_class_sorted['Fare'].hist(bins=40)
third_class_sorted = sorted_training_dataset[third_class_filter]

third_class_pivot_table = third_class_sorted.pivot_table(values=['Fare'], index=['Cabin', 'Ticket'], aggfunc=np.mean)

print("Tickets count:", third_class_sorted.shape[0])

third_class_pivot_table
third_class_sorted['Fare'].describe(percentiles=np.arange(10)/10.0)
plt.figure(figsize=(20,4))

third_class_sorted['Fare'].hist(bins=20)

test_dataset = pd.read_csv('/kaggle/input/titanic/test.csv')

test_dataset
test_dataset_passengers_count = print_count_of_passengers(test_dataset)
print("Training dataset passengers (count):", training_dataset_passengers_count)

print("Test dataset passengers (count):", test_dataset_passengers_count)

total_passengers = training_dataset_passengers_count + test_dataset_passengers_count

print("Total passengers on board:", total_passengers)

print("Total passengers on board (as per description):", 2224)

print("")

print("~~~ Discrepancy between the above two figures:", abs(total_passengers - 2224), "extra people on board or miscalculation ~~~")