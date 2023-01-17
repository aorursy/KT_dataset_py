import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
fields = ['SG_UF_RESIDENCIA', 'Q006', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 

          'NU_NOTA_MT', 'NU_NOTA_REDACAO']

df = pd.read_csv('../input/microdados_enem_2016_coma.csv', encoding='latin-1', 

                 sep=',', usecols=fields)
# Create final grade column

df['GRADE'] = (

    df['NU_NOTA_CN'] + df['NU_NOTA_CH'] + df['NU_NOTA_LC'] + df['NU_NOTA_MT'] + 

    df['NU_NOTA_REDACAO']) / 5.0



# Replace NaN with zeros

df.fillna(0, inplace=True)

    

# Filter grades zero (probabilly the applicant doesn't make the exam...)

df = df[df.GRADE != 0]
df.head()
# Number of rows...

len(df.index)
# Column order to be used in charts

Q006_order=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"]



def plot_violin(input_df, state="All", title=None):

    # Filter by state

    if(state != "All"):

        input_df = input_df[(input_df.SG_UF_RESIDENCIA == state)]

        

    sns.violinplot(

        x="Q006", 

        y="GRADE", 

        data=input_df, 

        order=Q006_order

    )

    plt.title(title)
def plot_violin_state(input_df, title=None):

    sns.violinplot(

        x="SG_UF_RESIDENCIA", 

        y="GRADE", 

        data=input_df

    )

    plt.title(title)
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(11.7, 8.27)



plot_violin(df, title="Brazil")
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(11.7, 8.27)



plot_violin(df, state="SP", title="São Paulo")
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(11.7, 8.27)



plot_violin_state(df, title="Brazil")