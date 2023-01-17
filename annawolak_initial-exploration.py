import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
df = pd.read_csv('../input/data.csv', low_memory=False)
df.head()
df['Department Title'].value_counts()
benefits = df[['Department Title', 'Job Class Title', 'Employment Type', 

               'Projected Annual Salary', 'Average Health Cost', 'Average Dental Cost', 

               'Average Basic Life', 'Average Benefit Cost', 'Benefits Plan']]
benefits.head()
benefits['Benefits Plan'].value_counts()
benefits['Benefits Plan'].value_counts().plot(kind='bar')
by_empl_type = benefits.groupby('Employment Type')
by_empl_type['Benefits Plan'].count().plot(kind='bar')
# Remove dollar sign in the strings of all salary and cost data entries

benefits = benefits.replace( '[\$,)]','', regex=True )



# Convert salaries and costs to numeric

benefits[['Projected Annual Salary', 

          'Average Health Cost', 

          'Average Dental Cost', 

          'Average Basic Life',

          'Average Benefit Cost']] = benefits[['Projected Annual Salary', 

                                               'Average Health Cost', 

                                               'Average Dental Cost', 

                                               'Average Basic Life', 

                                               'Average Benefit Cost']].apply(pd.to_numeric)
fire = benefits[benefits['Benefits Plan'] == 'Fire']
fire['Average Benefit Cost'].mean()
by_plan = benefits.groupby('Benefits Plan')
by_plan[['Average Health Cost', 'Average Dental Cost', 

         'Average Basic Life', 'Average Benefit Cost']].mean()
by_plan[['Average Health Cost', 

         'Average Dental Cost', 

         'Average Basic Life']].mean().plot(kind='bar', stacked=True, figsize=(8, 5), 

                                            title='Benefits Cost Breakdown Per Department')
by_plan[['Projected Annual Salary', 

         'Average Benefit Cost']].mean().plot(kind='bar', stacked=True, figsize=(8, 5),

                                             title='Total Compensation Breakdown By Department')