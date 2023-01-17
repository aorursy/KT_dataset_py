# Say hey.
print('hello punchcut 👋🏽')
# Split the bill. 
party_size = 8
total_charges = 200
individual_balance = total_charges / party_size
print('You owe ${:,.2f}'.format(individual_balance))
# Proclaim the greatest rappers of all time.
for i in range(5):
    print('{}. Dylan'.format(i + 1))
import numpy as np

vector_a = np.array([5, 5])
vector_b = np.array([5, -5])

# These two vectors form a 90° angle. Dot product should be 0.
print('dot product =', np.dot(vector_a, vector_b))
import pandas as pd

# Convert a COVID case CSV file to a DataFrame.
covid19_cases_df = pd.read_csv('../input/../input/coronavirus-covid19-data-in-the-united-states/us-states.csv')
covid19_cases_df
top_states_df = covid19_cases_df.drop(['date', 'fips'], axis=1)
top_states_df = covid19_cases_df.groupby('state').max().sort_values(by=['cases'], ascending=False).reset_index()
top_states_df.head(5)
state = 'California'
state_cases_df = covid19_cases_df[covid19_cases_df['state'] == state]
state_cases_df.tail()
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(20, 5))
plt.plot(state_cases_df['date'], state_cases_df['cases'])
plt.plot(state_cases_df['date'], state_cases_df['deaths'])
plt.xticks(state_cases_df['date'][::5], rotation=45, ha='right')
plt.ylabel('Cases & Deaths in Thousands')
plt.legend(['Cases', 'Deaths'])
plt.title('Cumulative Cases & Deaths in ' + state)
plt.show()