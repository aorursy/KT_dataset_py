import pandas as pd



all_schools = pd.read_csv('../input/kindergartens.csv')

gen_schools = all_schools[all_schools['Program'] == 'GEN'].copy()

gen_schools.sort_values(by='Total', ascending=False).reset_index()
gen_schools.count()