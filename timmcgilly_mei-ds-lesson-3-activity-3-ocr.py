import pandas as pd

import matplotlib.pyplot as plt



# importing the data

travel_2011_data=pd.read_csv('../input/ocrlds/OCR-lds-travel-2011.csv', thousands=',')



# inspecting the dataset to check that it has imported correctly

travel_2011_data.head()
# check the datatypes

travel_2011_data.dtypes


# use describe for any fields you are going to investigate and filter out or replace any unusable values

travel_2011_data['Bicycle'].describe()
# find the means and standard deviations for different fields grouped by region

regionsDf = travel_2011_data.groupby('Region',as_index=False).agg(["mean","std", "sum"])

regionsDf
transportTypes = ['Underground, tram', 'Train', 'Bus', 'Taxi', 'Motorcycle', 'Driving a car', 'Passenger in a car', 'Bicycle', 'On foot', 'Other']



regionsPercentDF = regionsDf.xs('sum', level=1,axis=1)

regionsPercentDF = regionsPercentDF[transportTypes].div(regionsPercentDF["In employment"], axis=0)

regionsPercentDF = regionsPercentDF[transportTypes].multiply(100)

regionsPercentDF
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages



regionsPercentDF.plot.bar(figsize=(12, 8))

regionsPercentDF.transpose().plot.pie(subplots=True, autopct='%1.1f%%', figsize=(100, 100))

plt.savefig("pie.jpg")
regionsPercentDF.plot.line(figsize=(20, 20))



rowLables = []

for i,v in regionsPercentDF.reset_index().iterrows():

    rowLables.append(v["Region"])

    for t in transportTypes:

        plt.annotate("{:0.2f}%".format(v[t]), xy=(i,v[t]) )

        

plt.xticks(range(0, len(rowLables)), rowLables, rotation=45)
regionsPercentDF.drop(columns='Driving a car').plot.line(figsize=(20, 10))



rowLables = []

for i,v in regionsPercentDF.reset_index().iterrows():

    rowLables.append(v["Region"])

    for t in transportTypes:

        plt.annotate("{:0.2f}%".format(v[t]), xy=(i,v[t]) )



plt.xticks(range(0, len(rowLables)), rowLables, rotation=45)
#regionsDf.droplevel(0,axis=1)

del regionsDf["In employment"]

del regionsDf["Work at home"]

del regionsDf["Not in employment"]

regionsDf.xs('mean', level=1,axis=1).plot.bar(figsize=(12, 8), title="Mean activites for each region")
# create box plots for different fields grouped by region



with PdfPages('tranportPerRegion.pdf') as pdf:

    for t in transportTypes:

        plt.figure()

        fig=travel_2011_data.boxplot(column = [t],by='Region', vert=False,figsize=(12, 8)).get_figure()

        pdf.savefig(fig)
regionsPercentDF['On foot'].sort_values(ascending = False)
# communicate the result