import pandas as pd

df = pd.DataFrame({'X':[78,85,96,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]});

print(df)
import pandas as pd

import numpy as np

exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura'],

        'score': [12.5, 9, 16.5, 6.0, 9, 20, np.nan, 5.5],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']



df = pd.DataFrame(exam_data , index=labels)

print(df)
import pandas as pd

import numpy as np

exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura'],

        'score': [12.5, 9, 16.5, 6.0, 9, 20, np.nan, 5.5],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']



df = pd.DataFrame(exam_data , index=labels)

print("First three rows of the data frame:")

print(df.iloc[:3])
import pandas as pd

import numpy as np

exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura'],

        'score': [12.5, 9, 16.5, 6.0, 9, 20, np.nan, 5.5],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

df = pd.DataFrame(exam_data , index=labels)

print("Select specific columns and rows:")

print(df.iloc[[1, 3, 5, 6], [1, 3]])
import pandas as pd

import numpy as np

exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura'],

        'score': [12.5, 9, 16.5, 6.0, 9, 20, np.nan, 5.5],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

df = pd.DataFrame(exam_data , index=labels)

print("Rows where score is missing:")

print(df[df['score'].isnull()])