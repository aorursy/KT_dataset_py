import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
import numpy as np
corona_virus_korea_path = '../input/coronavirusdataset/PatientInfo.csv'

corona_virus_korea = pd.read_csv(corona_virus_korea_path, index_col='patient_id' )



print('Terminated')
plt.rcParams['figure.figsize'] = (20,7)



corona_virus_korea_sex = corona_virus_korea.groupby('sex').count()

plt.pie(corona_virus_korea_sex['infection_case'], autopct="%1.1f%%", labels={'female','male'})

plt.show()
plt.rcParams['figure.figsize'] = (20,7)

corona_virus_korea_country = corona_virus_korea.groupby('province').count()

index = np.arange( len(corona_virus_korea_country) )

corona_virus_korea_country = corona_virus_korea_country.reset_index()

plt.barh( corona_virus_korea_country['province'], corona_virus_korea_country['infection_case'] )

plt.show()
##corona_virus_korea.isnull().sum()
## plt.figure(figsize=(20,7))



#corona_virus_korea.reset_index()

corona_virus_korea_scatterplot = corona_virus_korea.loc[:,['confirmed_date', 'birth_year', 'sex']]

corona_virus_korea_scatterplot = corona_virus_korea_scatterplot.reset_index()

corona_virus_korea_scatterplot.isnull().sum()

print('Finished')
sns.scatterplot( x=corona_virus_korea_scatterplot['birth_year'], y=corona_virus_korea_scatterplot.index, hue=corona_virus_korea_scatterplot['sex'])
plt.figure(figsize=(40,14))



corona_virus_korea = corona_virus_korea.reset_index()



corona_virus_korea_barh = corona_virus_korea[['infection_case','patient_id']].groupby('infection_case').count()



plt.barh( corona_virus_korea_barh.index , corona_virus_korea_barh['patient_id'])



print('Finished')