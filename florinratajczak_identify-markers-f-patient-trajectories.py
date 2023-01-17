import pandas as pd

import numpy as np

import scipy as sp



data = pd.read_csv("../input/uncover/UNCOVER/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv",

                  header = 0)



positives = data[data.sars_cov_2_exam_result.eq("positive")]

positives.shape
missing_pos = pd.DataFrame(positives.isnull().mean() * 100)

pd.set_option('display.max_rows',len(missing_pos))

missing_pos
for rowname, missingness in zip(missing_pos.index, missing_pos.values):

    if  missingness[0] > 95:

        positives.drop(str(rowname), inplace=True, axis = 1)

        print("Dropped " + rowname)
positives.shape
regular = positives[positives.patient_addmited_to_regular_ward_1_yes_0_no.eq("t")]

semi = positives[positives.patient_addmited_to_semi_intensive_unit_1_yes_0_no.eq("t")]

intensive = positives[positives.patient_addmited_to_intensive_care_unit_1_yes_0_no.eq("t")]



home = positives.drop(list(regular.index) + list(semi.index) + list(intensive.index), axis=0, inplace=False)



print(len(regular), len(semi), len(intensive), len(home))
pd.concat([

           pd.DataFrame(home.isnull().mean() * 100).rename(columns={0:"Home"}),

           pd.DataFrame(regular.isnull().mean() * 100).rename(columns={0:"Regular"}),

           pd.DataFrame(semi.isnull().mean() * 100).rename(columns={0:"Semi"}),

           pd.DataFrame(intensive.isnull().mean() * 100).rename(columns={0:"Intensive"}),

          ],

           axis=1)
def select_columns(column):

    # returns a list of lists for the specified column for each stratum

    return [np.array(stratum[column].dropna()) for stratum in [home,regular,semi,intensive]]



from scipy.stats import f_oneway



def analyze(df,blacklist):

    # run a oneway-anova between the for groups for each column and return the results

    res = {}

    for column in df.columns:

        if column not in blacklist:

            print(column + ":")

            try:

                f, p = f_oneway(*select_columns(column))

                print("p-Value: " + str(p))

                res.update({column : (f,p)})

            except ValueError as e:

                print(e)

            

    return res



# we are not interested in the following columns:



blacklist= ['patient_id', 'patient_age_quantile', 'sars_cov_2_exam_result',

       'patient_addmited_to_regular_ward_1_yes_0_no',

       'patient_addmited_to_semi_intensive_unit_1_yes_0_no',

       'patient_addmited_to_intensive_care_unit_1_yes_0_no']
analysis = analyze(positives,blacklist)
def fdr(p_vals):



    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)

    fdr = p_vals * len(p_vals) / ranked_p_values

    fdr[fdr > 1] = 1



    return fdr
p_values = [x[1] for x in list(analysis.values())]

p_values= np.array(p_values)



fdr(p_values)
adj_p_values = fdr(p_values)



print(np.sum(adj_p_values < 0.05))



significant_columns = list(map(list(analysis.keys()).__getitem__,list(np.where(adj_p_values < 0.05)[0])))



print(significant_columns)
def column_generator(columns_of_interest):

    

    for column in columns_of_interest:

        

        values = np.array([stratum[column].dropna() for stratum in [home,regular,semi,intensive]])

        treatments = np.repeat(["Home","Regular","Semi","Intensive"], repeats= [len(x) for x in values])

         

        values = np.hstack(values)

        # Stack the data (and rename columns):



        value_df = pd.DataFrame(values.T,columns=["Values"])

        treatments_df = pd.DataFrame(treatments.T,columns=["Treatments"])



        stacked_data = pd.concat([treatments_df,value_df],axis=1)

        stacked_data.name = column

        

        yield stacked_data

from statsmodels.stats.multicomp import (pairwise_tukeyhsd,

                                         MultiComparison)
# Set up the data for comparison (creates a specialised object)

for stacked_data in column_generator(significant_columns):

    MultiComp = MultiComparison(stacked_data['Values'],

                                stacked_data['Treatments'])



    # Show all pair-wise comparisons:

    

    # Print the comparisons

    print("Variable: " + stacked_data.name)

    print(MultiComp.tukeyhsd(alpha=0.05/len(significant_columns)).summary())
# interim result

pd.DataFrame(list(map(list(analysis.keys()).__getitem__,list(np.where(adj_p_values < 0.05)[0])))).to_csv("submission.csv")