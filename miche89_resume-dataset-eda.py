import os

import pandas as pd
print(os.listdir('../input'))
data = pd.read_csv('../input/resume_dataset.csv')
print(data.columns)
data.shape
data['Category'].value_counts()
def clean_text(resume: str):

    tmp = resume.replace('\\n', '\n')

    return tmp[2:-1]



data['Resume'] = [clean_text(entry) for entry in data['Resume'].values]
data = data.loc[pd.Series([len(resume) for resume in data['Resume'].values]) > 200]
data.shape
pd.Series([len(resume) for resume in data['Resume'].values]).describe()
data['Category'].value_counts().head(10)
mask = (

        (data['Category'] == 'Engineering') |

        (data['Category'] == 'Information Technology') | 

        (data['Category'] == 'Education') | 

        (data['Category'] == 'Health & Fitness') |

        (data['Category'] == 'Managment')

       )

data_selection = data.loc[mask]