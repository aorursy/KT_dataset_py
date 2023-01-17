
import pandas as pd

data = pd.read_csv("../input/coronavirus-france-dataset/patient.csv")
data.head()
data.shape
data.columns
data = data.drop(['id', 'group', 'infection_reason', 'infection_order', 'infected_by',
       'contact_number', 'confirmed_date', 'released_date', 'deceased_date',
       'status', 'health', 'source', 'comments'], axis=1)
data.head()
data.describe()
data = data.dropna(axis=0)
data.shape
data.describe()
data['birth_year'].hist()

data['sex'].value_counts().plot.bar()
data.groupby(['sex']).mean()
data['region'].value_counts()