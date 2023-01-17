# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Basics

print("Importing data mgmt basic libraries...")

import pandas as pd

import numpy as np

print("Done!")



print("Importing viz libraries...")

# visualization

import matplotlib.pyplot as plt

plt.style.use('bmh')



print("Done!")



print("Importing os libraries...")

#os

from os import listdir

from os.path import isfile, join

from IPython.display import display, HTML, Javascript

print("Done!")

plt.rcParams['figure.figsize'] = (12,9)

encoding_type = 'latin-1'
# Get all of the files that have a csv in the file name

onlyfiles = pd.DataFrame([f for f in listdir("../input/") if isfile(join("../input/", f)) and f.find('.csv')>0 ])



# Print the descriptor for the datasets in the library

#print("The .csv datasets included in the library are as follows:")



# Print the file names

#for x in onlyfiles[0]:

#    print(x)



# Create a dictionary of table names

alltables = {'varname':[],

            'filename':[],

            'dataset':[]}



# Loop through the file names and import them, adding a table name that reflects the file name

for i, onlyfile in enumerate(onlyfiles[0]):

    varname = onlyfile[0:3] + str(i)

    alltables['varname'].append(varname)

    alltables['filename'].append(onlyfile)

    

    #test = pd.read_csv('../input/test.csv', encoding='latin-1')

    exec(varname + " = pd.read_csv('../input/"+ onlyfile +"',  encoding='"+ encoding_type + "')") 

    exec("alltables['dataset'].append(" + varname + ")")

    #print(test.columns)

    #exec("print("+ varname + ".columns)") 





# Print the list of table variable names

alltables = pd.DataFrame(alltables)
pd.DataFrame(alltables.filter(['varname','filename']))
pd_some = pd.concat([med0, med1, med2, med3], sort=True)

len(pd_some)
mi_providers = pd_some[pd_some["NPPES Provider State"]=="MI"]

mi_providers = mi_providers[mi_providers["Opioid Prescribing Rate"]>0]

len(mi_providers)
mi_providers.head()
mi_providers["Specialty Description"] = mi_providers["Specialty Description"].str.replace('/', '')

mi_providers["Specialty Description"] = mi_providers["Specialty Description"].str.replace(' ', '_')

mi_providers["Specialty Description"] = mi_providers["Specialty Description"].str.replace('(', '_')

mi_providers["Specialty Description"] = mi_providers["Specialty Description"].str.replace(')', '_')
avg_pres_rate = mi_providers.groupby("Specialty Description")[["Opioid Prescribing Rate"]].mean()

std_pres_rate = mi_providers.groupby("Specialty Description")[["Opioid Prescribing Rate"]].std()

max_pres_rate = mi_providers.groupby("Specialty Description")[["Opioid Prescribing Rate"]].max()

min_pres_rate = mi_providers.groupby("Specialty Description")[["Opioid Prescribing Rate"]].min()

total_providers = mi_providers.groupby("Specialty Description")[["NPI"]].count()



newset = pd.merge(total_providers,avg_pres_rate , how='left', on="Specialty Description")

newset = pd.merge(newset, std_pres_rate, how='left', on="Specialty Description")

newset = pd.merge(newset, max_pres_rate, how='left', on="Specialty Description")

newset = pd.merge(newset, min_pres_rate, how='left', on="Specialty Description") 

newset.head()
mi_providers.groupby("Specialty Description")["Opioid Prescribing Rate"].describe()
a = newset[newset["NPI"] >= 100].index

mi_providers2 = mi_providers[mi_providers["Specialty Description"].isin(a)]  
plt.rcParams['figure.figsize'] = (22,9)



# mi_providers2 = mi_providers2.reset_index()



def boxplot_sorted(df, by, column, rot=0):

    # use dict comprehension to create new dataframe from the iterable groupby object

    # each group name becomes a column in the new dataframe

    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})

    # find and sort the median values in this new dataframe

    meds = df2.median().sort_values()

    # use the columns in the dataframe, ordered sorted by median value

    # return axes so changes can be made outside the function

    return df2[meds.index].boxplot(rot=rot, return_type="axes",)



mi_providers2 = mi_providers2.reindex()



mi_providers2 = mi_providers2.loc[~mi_providers2.index.duplicated(keep='first')]



axes = boxplot_sorted(mi_providers2, by=['Specialty Description'], column='Opioid Prescribing Rate', rot=90)

axes.set_title("Provider Opioid Prescribing Rates: All Specialties with >= 100 NPIs")

# axes.ylabel("Proportion of Claims with Opioid Rx")



# ax.set_xlabel('common xlabel')

axes.set_ylabel("% of Claims with Opioid Rx")

axes.set_ylim([0,100]);



list_string = ""



for x in newset.index:



    all_phys = mi_providers[mi_providers["Specialty Description"]==x] [["Opioid Prescribing Rate"]]

   #plt.xlabel("Opioid Prescribing Rate")

   # plt.ylabel("Providers")

   # plt.title("Provider Opioid Prescribing Rate: " + x)

   # plt.hist(all_phys["Opioid Prescribing Rate"], bins=20, range=(0,100), alpha=0.8)

#     plt.plot(all_phys["Opioid Prescribing Rate"], '--')

   # plt.savefig(x + '.png')

   # plt.close()

    

    list_string = list_string + " <option value=" + x + ">" + x + "</option> "
js_getResults = """

</style>

<script>

function test() {

    var value =$('#changeme').val()

    if (value != ""){

        $('#output').empty();

        var text = "<img src='" + value + ".png'>";

        $('#output').append(text);

        };

    };

</script>

<center>Select a specialty and take a look at the specialty's opioid prescribing rate...<br/>(% = pharmacy claims with opioids prescribed / all pharmacy claims)<br/><br/><select id = "changeme" onChange="test()">

    <option value="">Select a Provider Type...</option>""" + list_string + """

</select></center

"""

display(HTML(js_getResults))

display(HTML("<div id='output'></div><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>"))