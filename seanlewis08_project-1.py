# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import csv

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Get the Drug Data from csv to dataframe

drug_data = pd.read_csv('../input/adverse_drug_35.csv')



#Take a sample of the data

#drug_data_sample = drug_data[0:10]



#Convert Patient Drug Column to a list 

patients = drug_data['patient.drug'].tolist()



#Begin Writing data to file.csv

print("Writing data to file....")



with open('patient_drug.csv', 'w', newline='') as outfile:

    writer = csv.writer(outfile)

    

    #Make headers for patient_drug.csv

    writer.writerow(['Name',

                     'Action Drug',

                     'Active Substance',

                     'Drug Additional',

                     'Drug Administration Route',

                     'Drug Authorization Number',

                     'Drug Batch Number',

                     'Drug Characterization',

                     'Drug Cumulative Dosage Number',

                     'Drug Cumulative Dosage Unit',

                     'Drug Dosage Form',

                     'Drug Dosage Text',

                     'Drug End Date',

                     'Drug End Date Format',

                     'Drug Indication',

                     'Drug Interval Dosage Definition',

                     'Drug Interval Dosage Unit Number',

                     'Drug Recurrence Administration',

                     'Drug Recurrence',

                     'Drug Separate Dosage Number',

                     'Drug Start Date',

                     'Drug Start Date Format',

                     'Drug Structure Dosage Number',

                     'Drug Structure Dosage Unit',

                     'drug Treatment Duration',

                     'Drug Treatment Duration Unit',

                     'Medicinal Product',

                     'openfda'

                     ])

    #Counter for current patient

    i = 1

    

    #Counter for current drug

    j = 1

    

    #Iterate through each patient in the list to fill rows with patient.drug info

    for patient in patients:

        

        #Convert list inside string to list()

        drugs = eval(patient)

        

        #Iterate through drugs for specific patient

        for drug in drugs:

            

            #Fill rows for each column header. Default = None

            writer.writerow(["Patient " + str(i) + "\n" + "Drug " + str(j),

                             drug.get('actiondrug'), 

                             drug.get('activesubstance'), 

                             drug.get('drugadditional'), 

                             drug.get('drugadministrationroute'),

                             drug.get('drugauthorizationnumb'),

                             drug.get('drugbatchnumb'),

                             drug.get('drugcharacterization'),

                             drug.get('drugcumulativedosagenumb'),

                             drug.get('drugcumulativedosageunit'),

                             drug.get('drugdosageform'),

                             drug.get('drugdosagetext'),

                             drug.get('drugenddate'),

                             drug.get('drugenddateformat'),

                             drug.get('drugindication'),

                             drug.get('drugintervaldosagedefinition'),

                             drug.get('drugadmintervaldosageunitnumb'),

                             drug.get('drugrecurreadministration'),

                             drug.get('drugrecurrence'),

                             drug.get('drugseparatedosagenumb'),

                             drug.get('drugstartdate'),

                             drug.get('drugstartdateformat'),

                             drug.get('drugstructuredosagenumb'),

                             drug.get('drugstructuredosageunit'),

                             drug.get('drugtreatmentduration'),

                             drug.get('drugtreatmentdurationunit'),

                             drug.get('medicinalproduct'),

                             drug.get('openfda'),

                        ])

            # Increase number of drugs

            j += 1

        #Increase Number of patients

        i += 1

        j = 1

    print("Data written to " + 'test.csv' + " successfully....")

    print("There are " + str(i) + patients)

    outfile.close()
drug_table = pd.read_csv('../input/drug-output/output_data1.csv')



sample = drug_table





sample

i = 0 

for substance in sample['Active Substance']:

    if type(substance) != float:

        

        x = eval(substance)

        sample.iloc[i, 2] = x.get('activesubstancename')

    i += 1
sample
sample = drug_table[['Name', 'Active Substance', 'Drug Characterization', 'Drug Indication', 'Medicinal Product', 'openfda']]
m_name = pd.Series([''])

p_type = pd.Series([''])

rte    = pd.Series([''])

g_name = pd.Series([''])

b_name = pd.Series([''])

s_name = pd.Series([''])

i = 0



for fda in sample['openfda']:

    if type(fda) == str:

        fda = eval(fda)

        if fda.get('manufacturer_name') == None:

            m_name = m_name.append(None)

        else:

            if len(fda.get('manufacturer_name')) > 1:

                m_name = m_name.append(pd.Series([fda.get('manufacturer_name')], index = [i]))

            else:

                m_name = m_name.append(pd.Series(fda.get('manufacturer_name'), index = [i]))

        

        if fda.get('product_type') == None:

            p_type = p_type.append(None)

        else:

            if len(fda.get('product_type')):

                p_type = p_type.append(pd.Series([fda.get('product_type')], index = [i]))

            else:

                p_type = p_type.append(pd.Series(fda.get('product_type'), index = [i]))

        

        if fda.get('route') == None:

            rte = rte.append(None)

        else:

            if len(fda.get('route')) > 1:

                rte = rte.append(pd.Series([fda.get('route')], index = [i]))

            else:

                rte = rte.append(pd.Series(fda.get('route'), index = [i]))

        

        if fda.get('generic_name') == None:

            g_name = g_name.append(None)

        else:

            if len(fda.get('generic_name')) > 1:

                   g_name = g_name.append(pd.Series([fda.get('generic_name')], index = [i]))

            else:

                   g_name = g_name.append(pd.Series(fda.get('generic_name'), index = [i]))

           

        if fda.get('brand_name') == None:

            b_name = b_name.append(None)

        else:

            if len(fda.get('brand_name')) > 1:

                b_name = b_name.append(pd.Series([fda.get('brand_name')], index = [i]))

            else:

                b_name = b_name.append(pd.Series(fda.get('brand_name'), index = [i]))

        

        if fda.get('substance_name') == None:

            s_name = s_name.append(None)

        else:

            if len(fda.get('substance_name')) > 1:

                s_name = s_name.append(pd.Series([fda.get('substance_name')], index = [i]))

            else:

                s_name = s_name.append(pd.Series(fda.get('substance_name'), index = [i]))

    else:

        m_name = m_name.append(None)

        p_type = p_type.append(None)

        rte = rte.append(None)

        g_name = g_name.append(None)

        b_name = b_name.append(None)

        s_name = s_name.append(None)

    i += 1





sample.insert(6, "Manufacturer Name", m_name)

sample.insert(7, "Product Type", p_type)

sample.insert(8, "Route", rte)

sample.insert(9, 'Generic Name', g_name)

sample.insert(10, "Brand Name", b_name)

sample.insert(11, "Substance Name", s_name)



rel_table = sample[['Name', 

                    'Active Substance', 

                    'Drug Characterization', 

                    'Medicinal Product', 

                    "Manufacturer Name", 

                    'Product Type',

                   'Route',

                   'Generic Name',

                   'Brand Name',

                   'Substance Name']]
rel_table
rel_table.to_csv("drug_table.csv")