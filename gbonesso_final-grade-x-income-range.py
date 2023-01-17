import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/enem2015"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def load_and_preprocess(file_path, state="All"):

    df = pd.read_csv(file_path, encoding='latin-1')

    

    # Create final grade column

    df['GRADE'] = (

        df['NU_NOTA_CN'] + df['NU_NOTA_CH'] + df['NU_NOTA_LC'] + df['NU_NOTA_MT'] + 

        df['NU_NOTA_REDACAO']) / 5.0

    

    # Filter by state

    if(state != "All"):

        df = df[(df.SG_UF_RESIDENCIA == state)]

        

    # Extract just the grade and Q006 (income range) columns

    df = df[['GRADE', 'Q006', 'SG_UF_RESIDENCIA']]

    

    # Replace NaN with zeros

    df.fillna(0, inplace=True)

    

    # Filter grades zero (probabilly the applicant doesn't make the exam...)

    df = df[df.GRADE != 0]

    

    return(df)
# Read all files and concatenate in one Dataframe

df1 = load_and_preprocess("../input/enem2015/ENEM_2015_SMALL_SU.csv")

df2 = df1.append(load_and_preprocess("../input/enem2015/ENEM_2015_SMALL_SE1.csv"), ignore_index=True)

df3 = df2.append(load_and_preprocess("../input/enem2015/ENEM_2015_SMALL_SE2.csv"), ignore_index=True)

df4 = df3.append(load_and_preprocess("../input/enem2015/ENEM_2015_SMALL_NE1.csv"), ignore_index=True)

df5 = df4.append(load_and_preprocess("../input/enem2015/ENEM_2015_SMALL_NE2.csv"), ignore_index=True)

df6 = df5.append(load_and_preprocess("../input/enem2015/ENEM_2015_SMALL_CO.csv"), ignore_index=True)

df  = df6.append(load_and_preprocess("../input/enem2015/ENEM_2015_SMALL_NO.csv"), ignore_index=True)

df.head()
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

    sns.plt.title(title)
plot_violin(df, title="Brazil")
plot_violin(df, state="SP", title="São Paulo")
plot_violin(df, state="RR", title="Roraima")