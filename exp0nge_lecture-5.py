import pandas as pd

class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue', 'John'],
                    'hire_date': [2004, 2008, 2012, 2014, 2015]})
display('df1', 'df2')
df3 = pd.merge(df1, df2)
df3
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
df4
display('df3', 'df4', 'pd.merge(df3, df4)')
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
df5
display('df1', 'df5', "pd.merge(df1, df5)")
df3
display('df1', 'df2', "pd.merge(df1, df2, on='employee')")
pd.merge(df1, df2, on='hire_date')
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
df3
display('df1', 'df3', 'pd.merge(df1, df3, left_on="employee", right_on="name")')
pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)
df1
df2
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
display('df1a', 'df2a')
df1a.loc['Bob']
display('df1a', 'df2a',
        "pd.merge(df1a, df2a, left_index=True, right_index=True)")
display('df1a', 'df2a',
        "pd.merge(df1a, df2a)")
display('df1a', 'df2a', 'df1a.join(df2a)')
display('df1a', 'df3', "pd.merge(df1a, df3, left_index=True, right_on='name')")
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                   columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                   columns=['name', 'drink'])
display('df6', 'df7', 'pd.merge(df6, df7)')
pd.merge(df6, df7, how='inner')
display('df6', 'df7', "pd.merge(df6, df7, how='outer')")
display('df6', 'df7', "pd.merge(df6, df7, how='left')")
display('df6', 'df7', "pd.merge(df6, df7, how='right')")
!curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv
!curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-areas.csv
!curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-abbrevs.csv
!ls
pop = pd.read_csv('state-population.csv')
areas = pd.read_csv('state-areas.csv')
abbrevs = pd.read_csv('state-abbrevs.csv')

display('pop.head()', 'areas.head()', 'abbrevs.head()')
pop
pop.tail()
merged = pd.merge(pop, abbrevs, how='outer',
                  left_on='state/region', right_on='abbreviation')
merged
merged = merged.drop('abbreviation', 1) # drop duplicate info
merged.head()
merged
merged.isnull().any()
merged['population'].isnull()
merged[merged['population'].isnull()].head()
# some of the new state entries are also null, which means that there was no corresponding entry in the abbrevs key
merged.loc[merged['state'].isnull(), 'state/region'].unique()
merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()
final = pd.merge(merged, areas, on='state', how='left')
final.head()
final
final.isnull().any()
final['state'][final['area (sq. mi)'].isnull()].unique()
final.dropna.__doc__

final.dropna(inplace=True)
final
final.isnull().any()
# final[final["year == 2010] & ages == 'total'"]
data2010 = final.query("year == 2010 & ages == 'total'")
data2010
data2010.set_index('state', inplace=True)
data2010
density = data2010['population'] / data2010['area (sq. mi)']
density
density.sort_values(ascending=False, inplace=True)
density.head()
density.tail()
import requests
surveys_content = requests.get("https://ndownloader.figshare.com/files/2292172")
print("HelloWorld")
print("Hello\nWorld")
text = surveys_content.text
text
rows = text.split("\n")
rows[:10]
columns = rows[0].split(",")
columns
len(columns)
import pandas as pd
data = [x.split(",") for x in rows[1:]]
data[:10]
surveys_df = pd.DataFrame(data, columns=columns)
surveys_df
surveys_df.dtypes
surveys_df['record_id'] = pd.to_numeric(surveys_df['record_id'])
surveys_df.dtypes
for numeric in ['month', 'day', 'year', 'plot_id', 'hindfoot_length', 'weight']:
    surveys_df[numeric] = pd.to_numeric(surveys_df[numeric])
surveys_df.dtypes
surveys_df.columns
pd.unique(surveys_df['species_id'])
surveys_df['weight'].describe()
species_content = requests.get("https://ndownloader.figshare.com/files/3299483").text
species_content
species_rows = species_content.split("\n")
species_columns = species_rows[0].split(",")
species_columns
species_df = pd.DataFrame([x.split(",") for x in species_rows[1:]], columns=species_columns)
species_df.head()
# Read in first 10 lines of surveys table
survey_sub = surveys_df.head(10)
# Grab the last 10 rows
survey_sub_last10 = surveys_df.tail(10)
# Reset the index values to the second dataframe appends properly
survey_sub_last10 = survey_sub_last10.reset_index(drop=True)
# drop=True option avoids adding new index column with old index values
# Stack the DataFrames on top of each other
vertical_stack = pd.concat([survey_sub, survey_sub_last10], axis=0)
vertical_stack
# Place the DataFrames side by side
horizontal_stack = pd.concat([survey_sub, survey_sub_last10], axis=1)
horizontal_stack
!ls ../input/
species_sub = pd.read_csv('../input/speciesSubset.csv', keep_default_na=False, na_values=[""])
species_sub
survey_sub = surveys_df.head(10)
survey_sub
species_sub.columns
survey_sub.columns
merged_inner = pd.merge(left=survey_sub, right=species_sub, left_on='species_id', right_on='species_id')
merged_inner
merged_left = pd.merge(left=survey_sub, right=species_sub, how='left', left_on='species_id', right_on='species_id')
merged_left

merged_left[ pd.isnull(merged_left.genus) ]