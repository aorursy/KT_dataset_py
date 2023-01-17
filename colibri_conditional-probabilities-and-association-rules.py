from datetime import timedelta, date

from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import association_rules, apriori

%matplotlib inline

import numpy as np

import pandas as pd

pd.options.display.max_columns = 50

pd.options.display.max_rows = 150

plt.style.use('bmh')
cat_columns = ['sex', 'country', 'region', 'group', 'reason', 'age_bin']
raw_patient = pd.read_csv("/kaggle/input/coronavirusdataset/patient.csv", index_col = 0)
display(raw_patient.head())

display(raw_patient.describe(include = 'all').fillna("-"))

display(pd.DataFrame(raw_patient.isnull().sum()).T)
reason_dict = {

    "contact with patient in Singapore": "contact with patient",

    "pilgrimage to Israel": "visit to other area",

    "contact with patient in Singapore": "contact with patient",

    "residence in Wuhan": "visit to Wuhan",

    "contact with the patient": "contact with patient",

    "visit to Vietnam": "visit to other area",

    "contact with patient in Japan": "contact with patient",

    "visit to China": "visit to other area",

    "visit to Thailand": "visit to other area",

    "ccontact with patient": "contact with patient",

}



patient_df = raw_patient.copy()



today = date.today()

date_cols = patient_df.loc[:, patient_df.columns.str.endswith("_date")].columns



# Sex

patient_df['sex'] = patient_df.sex.str.replace('female ', 'female')



# Date

for col in date_cols:

    patient_df[col] = pd.to_datetime(patient_df[col])



# Age

patient_df["age"] = patient_df["confirmed_date"].dt.year - patient_df["birth_year"]

patient_df["age"] = patient_df["age"].fillna(-1).astype(int)



# Age_bin: [0, 10), [11, 20)...

patient_df["age_bin"] = pd.cut(patient_df["age"], np.arange(0, 100, 10), include_lowest=True, right=False)

patient_df["age_bin"] = patient_df["age_bin"].cat.add_categories("-").fillna("-")



# Endpoint: censored date or closed date

patient_df["endpoint_date"] = patient_df.loc[:, [*date_cols.tolist(), "state"]].apply(

    lambda x: x[1] if x[3] == "released" else x[2] if x[3] == "deceased" else today,

    axis=1

)

# patient_df["endpoint_days"] = patient_df["endpoint_date"] - patient_df["confirmed_date"]

# patient_df["endpoint_days"] = patient_df["endpoint_days"].dt.days.astype(int)



# Integer

patient_df["infection_order"] = patient_df["infection_order"].fillna(-1).astype(int)

patient_df["contact_number"] = patient_df["contact_number"].fillna(0).astype(int)



# Infection reason

patient_df["infection_reason"] = patient_df["infection_reason"].fillna("-")

patient_df["infection_reason"] = patient_df["infection_reason"].replace(reason_dict)



# Fill NAs

patient_df["sex"] = patient_df["sex"].fillna("-")

patient_df["region"] = patient_df["region"].fillna("-")

patient_df["group"] = patient_df["group"].fillna("-")

patient_df["infected_by"] = patient_df["infected_by"].fillna(-1).astype(int)



# State

state_series = patient_df["state"]

patient_df = pd.get_dummies(patient_df, columns=["state"])

state_cols = patient_df.loc[:, patient_df.columns.str.startswith("state_")].columns

patient_df.loc[:, state_cols] = patient_df.loc[:, state_cols].astype(bool)

patient_df["state"] = state_series



# Delete/Rename columns and show the dataframe

patient_df = patient_df.drop(["birth_year"], axis=1)

patient_df = patient_df.rename(

    {"infection_reason": "reason", "infection_order": "order", "infected_by": "by"},

    axis=1

)

patient_df.head()
recovered_died = patient_df.loc[patient_df.state != 'isolated']

pd.crosstab(index = recovered_died['sex'], columns = recovered_died['state'])
pd.crosstab(index = recovered_died['sex'], columns = recovered_died['state'], normalize = 'index')
recovered_died = patient_df.loc[patient_df.state != 'isolated']

fig, ax = plt.subplots(len(cat_columns), figsize = (18, 6*len(cat_columns)))

ax = ax.ravel()

for col in range(len(cat_columns)):

    print(f'Analysis on {cat_columns[col]}')

    tmp = pd.crosstab(index = recovered_died[cat_columns[col]], columns = recovered_died['state'], normalize = 'index')

    tmp.plot.bar(ax = ax[col])

    display(pd.crosstab(index = recovered_died[cat_columns[col]], columns = recovered_died['state']))

    display(tmp)

fig.tight_layout()
antecedents = cat_columns

consequent = 'state'
%%time

basketed = pd.get_dummies(patient_df[antecedents + [consequent]])

frequent_itemsets = apriori(basketed, min_support = 0.1, use_colnames=True)

rules = association_rules(frequent_itemsets, metric = 'lift', 

                          min_threshold = 0.0).sort_values('lift', ascending = False)

rules
rules_state = rules[rules.consequents.apply(lambda x: any(y in set(state_cols) for y in x) and len(x) == 1)]

rules_state
%%time

basketed = pd.get_dummies(recovered_died[antecedents + [consequent]])

frequent_itemsets = apriori(basketed, min_support = 0.1, use_colnames=True)

rules = association_rules(frequent_itemsets, metric = 'lift', 

                          min_threshold = 0.0).sort_values('lift', ascending = False)

rules
rules_state = rules[rules.consequents.apply(lambda x: any(y in set(state_cols) for y in x) and len(x) == 1)]

rules_state
list(rules.antecedents[:20])