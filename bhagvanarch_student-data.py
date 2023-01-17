import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/student_data.csv',index_col=0)
#print(dfall.columns.tolist())
#print(dfall['paid'])

#df = dfall
sns.kdeplot(df.Grade[df['guardian'] == 'father'], 
            label = 'Father', shade = True)
sns.kdeplot(df.Grade[df['guardian'] == 'mother'], 
            label = 'Mother', shade = True)
sns.kdeplot(df.Grade[df['guardian'] == 'other'], 
            label = 'Other', shade = True)
# Add labeling
plt.xlabel('Grade')
plt.ylabel('Density')
plt.title('Density Plot of Final Grades by Location')

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/student_data.csv',index_col=0)
#print(df)
print('Max', df['age'].max())
print('Min', df['age'].min())

import matplotlib.pyplot as plt
# Histogram of grades
plt.hist(df['Grade'], bins = 14)
plt.xlabel('Grade')
plt.ylabel('Count')
plt.title('Distribution of Final Grades')

plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/student_data.csv',index_col=0)

sns.kdeplot(df.loc[df['address'] == 'U', 'Grade'], 
            label = 'Urban', shade = True)
sns.kdeplot(df.loc[df['address'] == 'R', 'Grade'], 
            label = 'Rural', shade = True)
# Add labeling
plt.xlabel('Grade')
plt.ylabel('Density')
plt.title('Density Plot of Final Grades by Location')

plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../input/student_data.csv",index_col=0)
#output = df.head()

# Select only categorical variables
category_df = df.select_dtypes('object')
#print(category_df)
# One hot encode the variables
dummy_df = pd.get_dummies(category_df)
# Put the grade back in the dataframe
dummy_df['Grade'] = df['Grade']
# Find correlations with grade
output = dummy_df.corr()['Grade'].sort_values()

print(output)

