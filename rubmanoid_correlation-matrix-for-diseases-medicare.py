import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # operating system
import seaborn as sns # For plots
for filename in os.listdir("../input/"):
    print (filename)
    
data = pd.read_csv("../input/medicare-skilled-nursing-facility-snf-provider-aggregate-report-cy-2015.csv",low_memory=False)
print(data.columns.values)
# Delete excess columns
col_list = ['Percent of Beneficiaries with Asthma', 'Percent of Beneficiaries with Cancer', 'Percent of Beneficiaries with CHF', 'Percent of Beneficiaries with Chronic Kidney Disease', 'Percent of Beneficiaries with Diabetes','Percent of Beneficiaries with Hyperlipidemia','Percent of Beneficiaries with Hypertension','Percent of Beneficiaries with IHD','Percent of Beneficiaries with Osteoporosis','Percent of Beneficiaries with RA/OA','Percent of Beneficiaries with Schizophrenia','Percent of Beneficiaries with Stroke']
data=data[col_list]
print(data.shape)
# Delete nan rows for question 17 and 23
data = data.fillna(0)
print(data.shape)
import seaborn as sns
corr = data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)