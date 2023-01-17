import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('../input/haberman.csv', names=['age','operation_year','positive_nodes','survival_status'])
data.head(5)
data.shape
data.describe()
data.columns
print("age\n", data.age.unique())
print("operation year\n", data.operation_year.unique())
print("survival status\n", data.survival_status.unique())
print("positive nodes\n", data.positive_nodes.unique())
data.survival_status.value_counts()
survived_data = data.loc[data['survival_status'] == 1]
non_survived_data = data.loc[data['survival_status'] == 2]
data.age.value_counts(ascending=False).head(12)
data.operation_year.value_counts(ascending=False)
data.positive_nodes.value_counts(ascending=False).head(12)
plt.close()
sns.set_style('whitegrid')
sns.pairplot(data, hue='survival_status', vars=["age", "operation_year", 'positive_nodes'], size=4)
plt.show()
sns.set_style('whitegrid')
plot = sns.FacetGrid(data, hue='survival_status', size=4) \
   .map(sns.distplot, 'operation_year', bins=10)\
   .add_legend()
plot.set_axis_labels('operation year', 'density')
sns.boxplot(x='survival_status', y='operation_year', data=data)
sns.violinplot(x='survival_status', y='operation_year', data=data)
operation_year_less_than_60_who_survived = (data.survival_status == 1) & (data.operation_year < 60)
operation_year_greater_than_60_who_survived =  (data.survival_status == 1) & (data.operation_year >= 60)

operation_year_less_than_60_who_survived = data[operation_year_less_than_60_who_survived]
operation_year_greater_than_60_who_survived = data[operation_year_greater_than_60_who_survived]
operation_year_less_than_60_who_did_not_survive = (data.survival_status == 2) & (data.operation_year < 60)
operation_year_greater_than_60_who_did_not_survive =  (data.survival_status == 2) & (data.operation_year >= 60)

operation_year_less_than_60_who_did_not_survive = data[operation_year_less_than_60_who_did_not_survive]
operation_year_greater_than_60_who_did_not_survive = data[operation_year_greater_than_60_who_did_not_survive]
print("The total people who survived for the operation year less than 60 is %d"% operation_year_less_than_60_who_survived.operation_year.count())
print("The total people who survived for the operation year greater than 60 is %d"% operation_year_greater_than_60_who_survived.operation_year.count())
print("The total people who did not survive for the operation year less than 60 is %d"% operation_year_less_than_60_who_did_not_survive.operation_year.count())
print("The total people who did not survive for the operation year greater than 60 is %d"% operation_year_greater_than_60_who_did_not_survive.operation_year.count())
sns.set_style('whitegrid')
plot = sns.FacetGrid(data, hue='survival_status', size=4) \
   .map(sns.distplot, 'age', bins=10)\
   .add_legend()
plot.set_axis_labels('age', 'density')
sns.boxplot(x='survival_status', y='age', data=data)
sns.violinplot(x='survival_status', y='age', data=data)
age_less_than_52_who_survived = (data.survival_status == 1) & (data.age < 52)
age_greater_than_52_who_survived =  (data.survival_status == 1) & (data.age >= 52)

age_less_than_52_who_survived = data[age_less_than_52_who_survived]
age_greater_than_52_who_survived = data[age_greater_than_52_who_survived]
age_less_than_52_who_did_not_survive = (data.survival_status == 2) & (data.age < 52)
age_greater_than_52_who_did_not_survive =  (data.survival_status == 2) & (data.age >= 52)

age_less_than_52_who_did_not_survive = data[age_less_than_52_who_did_not_survive]
age_greater_than_52_who_did_not_survive = data[age_greater_than_52_who_did_not_survive]
print("The total people who survived for the age less than 52 is %d"% age_less_than_52_who_survived.age.count())
print("The total people who survived for the age greater than 52 is %d"% age_greater_than_52_who_survived.age.count())
print("The total people who did not survive for the age less than 52 is %d"% age_less_than_52_who_did_not_survive.age.count())
print("The total people who did not survive for the age greater than 52 is %d"% age_greater_than_52_who_did_not_survive.age.count())
sns.set_style('whitegrid')
plot = sns.FacetGrid(data, hue='survival_status', size=4) \
   .map(sns.distplot, 'positive_nodes', bins=10)\
   .add_legend()
plot.set_axis_labels('positivenodes', 'density')
sns.boxplot(x='survival_status', y='positive_nodes', data=data, width=0.9)
nodes_less_than_4_who_survived = (data.survival_status == 1) & (data.positive_nodes < 4)
nodes_greater_than_4_who_survived =  (data.survival_status == 1) & (data.positive_nodes >= 4)

nodes_less_than_4_who_survived = data[nodes_less_than_4_who_survived]
nodes_greater_than_4_who_survived = data[nodes_greater_than_4_who_survived]
nodes_less_than_4_who_did_not_survive = (data.survival_status == 2) & (data.positive_nodes < 4)
nodes_greater_than_4_who_did_not_survive =  (data.survival_status == 2) & (data.positive_nodes >= 4)

nodes_less_than_4_who_did_not_survive = data[nodes_less_than_4_who_did_not_survive]
nodes_greater_than_4_who_did_not_survive = data[nodes_greater_than_4_who_did_not_survive]
print("The total people who survived for the axillary nodes less than 4 is %d"% nodes_less_than_4_who_survived.positive_nodes.count())
print("The total people who survived for the axillary nodes greater than 4 is %d"% nodes_greater_than_4_who_survived.positive_nodes.count())
print("The total people who did not survive for the axillary nodes less than 4 is %d"% nodes_less_than_4_who_did_not_survive.positive_nodes.count())
print("The total people who did not survive for the axillary nodes greater than 4 is %d"% nodes_greater_than_4_who_did_not_survive.positive_nodes.count())