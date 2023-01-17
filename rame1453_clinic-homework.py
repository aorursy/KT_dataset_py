# Firstly we need dowloand libraries which will be need for project



import numpy as np 

import pandas as pd 
# a.Import the data in the file into a DataFrame



df = pd.read_csv('../input/vbo-bootcamp-clinic/clinic_data.csv')

df
# b.Display the names of all doctors

df.doctor
# c.From the 1st and 4th columns, display the 3rd to the 5th rows (inclusive)

df[2:5][['doctor','patients']]
# d.For all records, display the year, months and number of patients



df[['year','month','patients']]
# e.Display the records for visits in November



df.query('month=="November"')
# f.Display the record with the maximum patients. (hint: to find the max, df.col_index.max())



max_value = df[df.patients == df.patients.max()].copy()

max_value
# g.Display records from September of February.



#Records from September or February:

search = ['September','February']



sep_feb = df[(df.month.isin(search))]

sep_feb
# h.Display records after 2014 with fewer than 300 patients.



#Records after 2014 with fewer than 300 patients:

df.query('year > 2014 and patients < 300')
# i.Display records from September or February with over 250 patients.



# Records from September or February with over 250 patients:

sep_feb.query('patients > 250')
# j.Set the first column in the DataFrame as the index



df.set_index(['doctor'],inplace=True)

df
# k.Use loc[] to display the year and patients for doctors Benitz and Carli.



df.loc[["Benitz","Carli"],["year","patients"]]